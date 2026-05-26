#!/usr/bin/env python3
# RUN: %python %s
# REQUIRES: cuda-device
"""Negative tests for the KunMLIR launch-time validation path.

The runtime consumes every input/output via DLPack (the protocol
implemented by CuPy / PyTorch / JAX / TensorFlow).  This file
exercises:

  * DLPack field validation — wrong dtype, wrong ndim, non-contiguous
    strided view, host-only ndarray (DLPack CPU device), object that
    implements neither protocol at all.
  * Graph-arg checks — missing kwarg, unknown kwarg, cross-arg shape
    mismatch.
  * cs_rank dynamic-smem cap — pick `num_stocks` exceeding the device's
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN and assert the runtime fails with
    the GPU-aware message instead of letting cuLaunchKernel emit a
    generic CUDA_ERROR_INVALID_VALUE; the at-cap case must still launch.
  * DLPack-only producer — verify it works when the object hides CAI
    behind a wrapper, since DLPack is the path we rely on for non-CuPy
    frameworks.

`_expect_fail` returns 0 if the right error fires, 1 otherwise.
"""

from __future__ import annotations
import argparse
import sys

import numpy as np

from KunQuant.Driver import KunCompilerConfig


_KCFG_TS = KunCompilerConfig(input_layout="TS", output_layout="TS")


# ── Fixture helpers ──────────────────────────────────────────────────

def _build_elemwise_exe(cfg):
    """Add(a, b) → out.  Workhorse for arg-validation tests."""
    from KunQuant.Op import Builder, Input, Output
    from KunQuant.ops import Add
    from KunQuant.Stage import Function
    from KunQuant.jit.cuda import compile_func
    b = Builder()
    with b:
        a = Input("a"); bb = Input("b")
        Output(Add(a, bb), "out")
    f = Function(b.ops, name="addk")
    return compile_func(f, _KCFG_TS, cfg)


def _build_cs_rank_exe(cfg):
    """cs_rank(a) → r.  Used for the smem-cap test."""
    from KunQuant.Op import Builder, Input, Output, Rank
    from KunQuant.Stage import Function
    from KunQuant.jit.cuda import compile_func
    b = Builder()
    with b:
        Output(Rank(Input("a")), "r")
    f = Function(b.ops, name="csr")
    return compile_func(f, _KCFG_TS, cfg)


def _expect_fail(label, fn, needle):
    print(f"  {label} ...", end=" ", flush=True)
    try:
        fn()
    except Exception as e:
        msg = str(e)
        if needle in msg:
            print(f"ok (raised: {msg.splitlines()[0][:100]})")
            return 0
        print(f"FAIL — wrong message: {msg!r}", file=sys.stderr)
        return 1
    print("FAIL — no exception raised", file=sys.stderr)
    return 1


# ── DLPack / arg-validation test set ────────────────────────────────

def run_validation_tests(target):
    import cupy as cp
    from KunQuant.jit import KunMLIR
    from KunQuant.jit.cuda import CudaCompilerConfig

    print("=== DLPack + arg validation ===")
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)
    exe = _build_elemwise_exe(cfg)
    ex  = KunMLIR.Executor()
    T, S = 4, 32

    rc = 0
    a   = cp.zeros((T, S), dtype=cp.float32)
    b   = cp.zeros((T, S), dtype=cp.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    # 1. Object implementing neither CAI nor DLPack (a plain int)
    rc |= _expect_fail(
        "object without __dlpack__ rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": 0xdeadbeef, "b": b},
                            outputs={"out": out}),
        "does not implement __dlpack__")

    # 2. Host numpy array — numpy is a CPU-only producer; it refuses
    #    `stream != None` (our binding always passes the executor's
    #    CUDA stream).  The error comes from numpy, not from us, but
    #    the effect is what we want: host arrays can't sneak into a
    #    GPU launch.
    rc |= _expect_fail(
        "host numpy array rejected (CPU producer)",
        lambda: ex.runGraph(exe,
                            inputs={"a": np.zeros((T, S), dtype=np.float32),
                                    "b": b},
                            outputs={"out": out}),
        "stream")

    # 3. Wrong dtype: float64
    rc |= _expect_fail(
        "f64 dtype rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": cp.zeros((T, S), dtype=cp.float64),
                                    "b": b},
                            outputs={"out": out}),
        "kernel expects float32")

    # 4. Wrong ndim: 1-D
    rc |= _expect_fail(
        "1-D array rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": cp.zeros((T*S,), dtype=cp.float32),
                                    "b": b},
                            outputs={"out": out}),
        "must be 2-D")

    # 5. Wrong ndim: 3-D
    rc |= _expect_fail(
        "3-D array rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": cp.zeros((T, S, 1), dtype=cp.float32),
                                    "b": b},
                            outputs={"out": out}),
        "must be 2-D")

    # 6. Non-contiguous strided view (transpose).  (T, S) and (S, T) are
    #    different shapes, so build matching transposed b/out too.
    a_t = a.T                                          # (S, T) view of (T, S)
    b_t = cp.zeros((S, T), dtype=cp.float32)
    out_t = cp.zeros((S, T), dtype=cp.float32)
    rc |= _expect_fail(
        "non-contiguous transposed view rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": a_t, "b": b_t},
                            outputs={"out": out_t}),
        "not C-contiguous")

    # 7. Missing graph input.  Outputs may be omitted by design: the
    #    binding auto-allocates them and returns the buffer dict.
    rc |= _expect_fail(
        "missing graph_input rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": a},
                            outputs={"out": out}),
        "missing input 'b'")

    # 8. Shape mismatch between args
    rc |= _expect_fail(
        "shape mismatch rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": a,
                                    "b": cp.zeros((T, S+1), dtype=cp.float32)},
                            outputs={"out": out}),
        "expected")

    # 9. Unknown kwarg (the hot-path skip kicks in for size == ordered,
    #    so add a real extra to trip the strict check).
    rc |= _expect_fail(
        "unknown argument rejected",
        lambda: ex.runGraph(exe,
                            inputs={"a": a, "b": b, "bogus": a},
                            outputs={"out": out}),
        "unexpected input 'bogus'")

    # 10. DLPack-only producer — wrap a cupy ndarray and hide every
    #     attribute except __dlpack__ / __dlpack_device__.  Verifies the
    #     binding works for objects that don't quack like CuPy (e.g.
    #     JAX, TF, custom buffers).
    class DLOnly:
        def __init__(self, arr):
            self._arr = arr
        def __dlpack__(self, stream=None):
            return self._arr.__dlpack__(stream=stream)
        def __dlpack_device__(self):
            return self._arr.__dlpack_device__()

    print("  dlpack-only producer happy path ...", end=" ", flush=True)
    try:
        ex.runGraph(exe,
                    inputs={"a": DLOnly(a), "b": DLOnly(b)},
                    outputs={"out": DLOnly(out)})
        ex.synchronize()
        print("ok")
    except Exception as e:
        print(f"FAIL — DLPack-only happy path raised: {e}", file=sys.stderr)
        rc |= 1

    return rc


# ── cs_rank smem-cap test ────────────────────────────────────────────

def run_smem_cap_tests(target):
    import cupy as cp
    from KunQuant.jit import KunMLIR
    from KunQuant.jit.cuda import CudaCompilerConfig

    print("=== cs_rank smem cap ===")
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)
    exe = _build_cs_rank_exe(cfg)
    ex  = KunMLIR.Executor()
    rc = 0

    # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
    dev = cp.cuda.Device(0)
    try:
        max_smem = cp.cuda.runtime.deviceGetAttribute(97, dev.id)
    except Exception:
        max_smem = 49152    # conservative fallback
    too_many = max_smem // 4 + 1   # one stock past the float32 cap
    print(f"  device max_smem={max_smem} bytes; using num_stocks={too_many} "
          f"(needs {too_many*4} bytes)")

    T = 2
    a   = cp.zeros((T, too_many), dtype=cp.float32)
    out = cp.zeros((T, too_many), dtype=cp.float32)
    rc |= _expect_fail(
        "smem cap exceeded → clear error",
        lambda: ex.runGraph(exe, inputs={"a": a}, outputs={"r": out}),
        "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")

    # At-cap case must still launch (off-by-one regression guard).
    at_limit = max_smem // 4
    a2   = cp.zeros((T, at_limit), dtype=cp.float32)
    out2 = cp.zeros((T, at_limit), dtype=cp.float32)
    print(f"  at-cap launch (num_stocks={at_limit}) ...", end=" ", flush=True)
    try:
        ex.runGraph(exe, inputs={"a": a2}, outputs={"r": out2})
        ex.synchronize()
        print("ok")
    except Exception as e:
        print(f"FAIL — at-cap should succeed but got: {e}", file=sys.stderr)
        rc |= 1
    return rc


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    args = ap.parse_args()

    import cupy as cp
    from KunQuant.jit.env import get_cuda_compute_capability
    args.target = args.target or get_cuda_compute_capability()
    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    rc = 0
    rc |= run_validation_tests(args.target)
    print()
    rc |= run_smem_cap_tests(args.target)
    print()
    print("=== all tests done ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
