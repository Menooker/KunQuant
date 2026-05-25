#!/usr/bin/env python3
# RUN: %python %s
# REQUIRES: cuda-device
"""End-to-end test for the cs_rank GPU dispatch path.

Cross-sectional rank (`KunQuant.Op.Rank`) is special on the GPU: it
has no kunir representation at all.  CodegenMLIR detects a cs_rank
partition in the Python IR and routes it as an "external kernel"
descriptor straight to the C++ binding, which fabricates a KernelMeta
tagged with `KernelKind::ExtCsRankF*`.  At Executable construction the
runtime loads a pre-compiled, sm_75-baseline PTX bundled inside
libKunCudaRuntime as a second CUmodule and resolves
`kun_cs_rank_f{32,64}` from it.  This test exercises the whole
plumbing end-to-end:

  * Python frontend skips the kunir pipeline for cs_rank partitions,
  * the executor lazy-loads the bundled PTX and resolves the right
    symbol,
  * the launch uses a time-major grid + dynamic shared memory (one
    CTA per timestep, smem = S * sizeof(T)) rather than the default
    stock-major grid the JIT'd kernels use,
  * the result matches the CPU `equal_range`-based reference exactly,
    including NaN passthrough and tie averaging,
  * a graph that mixes cs_rank with regular JIT kernels stitches up
    correctly (cs_rank produces an intermediate consumed by an
    elementwise kernel).
"""

from __future__ import annotations
import argparse
import sys

import numpy as np

from KunQuant.Op import Builder, Input, Output, Rank
from KunQuant.ops import Add
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Stage import Function
from KunQuant.jit import KunMLIR
from KunQuant.jit.cuda import compile_func, CudaCompilerConfig, to_mlir


# ── CPU reference (matches cpp/Kun/Rank.hpp's equal_range formula) ──

def _ref_cs_rank(arr: np.ndarray) -> np.ndarray:
    """Per-row average-rank percentile in (0, 1], NaN preserved.

    Matches cpp/Kun/Rank.hpp exactly:
      sum   = (start + end + 1) * (end - start) / 2
      out   = sum / (end - start) / num_valid
    where [start, end) is the equal-range of the value in the sorted
    non-NaN array.  Algebraically this equals
      (2 * less + equal + 1) / (2 * num_valid)
    which is what the GPU kernel computes.
    """
    T, S = arr.shape
    out = np.full((T, S), np.nan, dtype=arr.dtype)
    for t in range(T):
        row = arr[t]
        valid_mask = ~np.isnan(row)
        v = row[valid_mask]
        nv = len(v)
        if nv == 0:
            continue
        sorted_v = np.sort(v)
        ranks = np.empty(nv, dtype=arr.dtype)
        for i, x in enumerate(v):
            lo = np.searchsorted(sorted_v, x, side='left')
            hi = np.searchsorted(sorted_v, x, side='right')
            # avg rank (1-indexed) within the equal-range, divided by nv
            ranks[i] = (lo + hi + 1) / 2.0 / nv
        out[t][valid_mask] = ranks
    return out


# ── Function builders ────────────────────────────────────────────────

def _build_cs_rank_only() -> Function:
    """r = cs_rank(a) — a single cs_rank partition, no other compute."""
    b = Builder()
    with b:
        Output(Rank(Input('a')), 'r')
    return Function(b.ops, name='cs_rank_only')


def _build_cs_rank_mixed() -> Function:
    """out = a + cs_rank(b) — forces the partitioner to produce
    two kernels: an external cs_rank partition, then a JIT'd
    elementwise Add that consumes its result."""
    b = Builder()
    with b:
        a = Input('a')
        bin_ = Input('b')
        Output(Add(a, Rank(bin_)), 'out')
    return Function(b.ops, name='cs_rank_mixed')


# ── Helpers ──────────────────────────────────────────────────────────

def _dtype_pair(dtype_token: str):
    """Map CudaCompilerConfig.dtype → (numpy dtype, label)."""
    if dtype_token == 'float':  return np.float32, 'float32'
    if dtype_token == 'double': return np.float64, 'float64'
    raise ValueError(dtype_token)


def _run_cs_rank_only(target: str, dtype_token: str, T: int, S: int,
                       *, with_nan: bool, with_ties: bool, seed: int) -> int:
    """Compile r = cs_rank(a), launch, and compare to the CPU
    reference.  Asserts the partition was tagged as external (i.e.
    kernel_names should be a single kernel that doesn't show up as a
    typical compute kernel — but since we can't introspect kind from
    Python directly, we lean on the correctness check to prove the
    external path is wired up)."""
    import cupy as cp

    np_dt, label = _dtype_pair(dtype_token)
    print(f"=== cs_rank ({label}) T={T} S={S} "
           f"nan={with_nan} ties={with_ties} ===")

    f = _build_cs_rank_only()
    kcfg = KunCompilerConfig(input_layout="TS", output_layout="TS",
                              dtype=dtype_token)
    ccfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)
    mod = to_mlir(_build_cs_rank_only(), kcfg, ccfg)
    print("--- mlir ---")
    print(mod.to_string())

    exe = compile_func(f, kcfg, ccfg)
    print(f"  kernel_names={exe.kernel_names}  "
          f"num_buffers={exe.num_buffers}  "
          f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    rng = np.random.default_rng(seed)
    a_h = rng.standard_normal((T, S)).astype(np_dt)
    if with_ties:
        # Force a moderate tie population: quantize ~30% of cells.
        tie_mask = rng.random((T, S)) < 0.3
        a_h[tie_mask] = np.round(a_h[tie_mask] * 2) / 2  # snap to 0.5-grid
    if with_nan:
        # Sprinkle ~10% NaNs.  Also force a row to be all-NaN to test
        # the valid==0 path.
        nan_mask = rng.random((T, S)) < 0.1
        a_h[nan_mask] = np.nan
        a_h[0, :] = np.nan

    a_d = cp.asarray(a_h)
    out_d = cp.zeros((T, S), dtype=np_dt)
    ex = KunMLIR.Executor()
    ex.runGraph(exe, inputs={'a': a_d}, outputs={'r': out_d})
    ex.synchronize()
    out_h = cp.asnumpy(out_d)

    ref = _ref_cs_rank(a_h)
    # NaN cells in the reference must remain NaN on the GPU.
    nan_ref = np.isnan(ref)
    if not np.array_equal(np.isnan(out_h), nan_ref):
        print("  FAIL — NaN pattern mismatch", file=sys.stderr)
        return 1
    # Numeric cells must match exactly modulo a few ulps (the formula
    # is the same algebraic expression on both sides).
    atol = 1e-6 if np_dt == np.float32 else 1e-12
    diff = np.abs(out_h[~nan_ref] - ref[~nan_ref])
    max_abs = float(diff.max()) if diff.size else 0.0
    if max_abs > atol:
        print(f"  FAIL — max |Δ| = {max_abs:.3e} > {atol:.0e}",
                file=sys.stderr)
        return 1
    print(f"  ok — max |Δ| = {max_abs:.3e} (atol={atol:.0e}) "
          f"on {diff.size} numeric cells")
    return 0


def _run_cs_rank_mixed(target: str, T: int, S: int, *, seed: int) -> int:
    """out = a + cs_rank(b) — proves cs_rank's intermediate buffer is
    handed off correctly to a downstream JIT'd kernel.  Forces
    partition_factor=1 so the graph splits into >= 2 kernels."""
    import cupy as cp

    print(f"=== cs_rank-mixed (float32) T={T} S={S} ===")
    f = _build_cs_rank_mixed()
    kcfg = KunCompilerConfig(input_layout="TS", output_layout="TS",
                              dtype='float', partition_factor=1)
    ccfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)
    mod = to_mlir(_build_cs_rank_mixed(), kcfg, ccfg)
    print("--- mlir ---")
    print(mod.to_string())

    exe = compile_func(f, kcfg, ccfg)
    print(f"  kernel_names={exe.kernel_names}  "
          f"num_kernels={exe.num_kernels}  "
          f"num_buffers={exe.num_buffers}  "
          f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    # The whole point: at least 2 kernels (cs_rank + downstream Add),
    # and at least one intermediate slot threading them together.
    assert exe.num_kernels >= 2, exe.num_kernels
    assert exe.peak_intermediate_slots >= 1, exe.peak_intermediate_slots

    rng = np.random.default_rng(seed)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out_d = cp.zeros((T, S), dtype=cp.float32)

    ex = KunMLIR.Executor()
    ex.runGraph(exe,
                inputs={'a': cp.asarray(a_h), 'b': cp.asarray(b_h)},
                outputs={'out': out_d})
    ex.synchronize()
    out_h = cp.asnumpy(out_d)

    ref = a_h + _ref_cs_rank(b_h)
    diff = np.abs(out_h - ref)
    max_abs = float(diff.max())
    if max_abs > 1e-5:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  FAIL — max |Δ| = {max_abs:.3e} at {idx}", file=sys.stderr)
        return 1
    print(f"  ok — max |Δ| = {max_abs:.3e} across {exe.num_kernels} kernels")
    return 0


# ── Entry ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("-T", "--time-length", type=int, default=8)
    ap.add_argument("-S", "--num-stocks", type=int, default=257)
    args = ap.parse_args()

    import cupy as cp
    from KunQuant.jit.env import get_cuda_compute_capability
    args.target = args.target or get_cuda_compute_capability()
    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    rc = 0
    # f32 — golden path
    rc |= _run_cs_rank_only(args.target, 'float',
                              args.time_length, args.num_stocks,
                              with_nan=False, with_ties=False, seed=1)
    print()
    # f32 with ties — exercises the equal-range averaging
    rc |= _run_cs_rank_only(args.target, 'float',
                              args.time_length, args.num_stocks,
                              with_nan=False, with_ties=True, seed=2)
    print()
    # f32 with NaN + ties — exercises every branch
    rc |= _run_cs_rank_only(args.target, 'float',
                              args.time_length, args.num_stocks,
                              with_nan=True, with_ties=True, seed=3)
    print()
    # NOTE: the cs_rank kernel itself is templated and `kun_cs_rank_f64`
    # is built into the embedded PTX, but the rest of the runtime
    # (Runtime.cpp's slot pool, MlirBinding.cpp's CAI typestr check)
    # is still float32-only.  Lifting that requires plumbing dtype
    # through Executable / ExecutableData and is out of scope here.
    # When that lands, this test can re-enable:
    #
    #   rc |= _run_cs_rank_only(args.target, 'double',
    #                            args.time_length, args.num_stocks,
    #                            with_nan=True, with_ties=True, seed=4)
    print()
    # Mixed cs_rank + Add — proves intermediate buffer flow works
    rc |= _run_cs_rank_mixed(args.target,
                              args.time_length, args.num_stocks, seed=5)
    return rc


if __name__ == "__main__":
    sys.exit(main())
