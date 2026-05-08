#!/usr/bin/env python3
"""End-to-end test for the KunQuant Python-IR → MLIR → CUDA path.

Builds a KunQuant Function with the high-level Op API, runs the same
Driver.optimize() pipeline the CPU compileit uses, then compiles to a
CUDA Executable via kun_mlir and validates against numpy.

Two factors are exercised:
  * elemwise: out = (a + b) * a - b * b
  * windowed: ws  = WindowedSum(a + b, N)   (decomposes into
                ForeachBackWindow + ReduceAdd inside the optimizer pass)

Note: ops that lower to math.absf / math.log / math.copysign aren't
exercised here yet — the kunir-to-LLVM pipeline doesn't link libdevice,
so those would end up as unresolved __nv_* externals.  Once the math →
LLVM intrinsic / libdevice lowering is wired up, swap in Abs/Log/Sign.
"""

from __future__ import annotations
import argparse
import sys

import numpy as np

from KunQuant.Op import Builder, Input, Output
from KunQuant.ops import Add, Sub, Mul, WindowedSum
from KunQuant.Stage import Function
from KunQuant.jit.cuda import compileit, CudaCompilerConfig, to_mlir


def build_func_elemwise() -> Function:
    """out = (a + b) * a - b * b"""
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        v = Sub(Mul(Add(a, bin_), a), Mul(bin_, bin_))
        Output(v, "out")
    return Function(builder.ops, name="elemwise_kernel")


def build_func_windowed(N: int) -> Function:
    """ws = WindowedSum(a + b, N)"""
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        s = WindowedSum(Add(a, bin_), N)
        Output(s, "ws")
    return Function(builder.ops, name="windowed_kernel")


def run_elemwise(target: str, T: int, S: int) -> int:
    print("=== elemwise: out = (a+b)*a - b*b ===")
    f = build_func_elemwise()
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    # Show the IR for sanity — same passes + translator, no compile.
    mod = to_mlir(build_func_elemwise(), cfg)
    print("--- mlir ---")
    print(mod.to_string())

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(0)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    exe.launch({"a": cp.asarray(a_h), "b": cp.asarray(b_h), "out": out})
    cp.cuda.runtime.deviceSynchronize()
    out_h = cp.asnumpy(out)

    expected = (a_h + b_h) * a_h - b_h * b_h
    if not np.allclose(out_h, expected, atol=1e-5):
        diff = np.abs(out_h - expected)
        print(f"  FAIL — max abs diff {diff.max()}", file=sys.stderr)
        return 1
    print(f"  ok — output matches (a+b)*a - b*b on every cell ({T*S} cells)")
    return 0


def run_windowed(target: str, T: int, S: int, N: int) -> int:
    print(f"=== windowed: ws = WindowedSum(a + b, N={N}) ===")
    f = build_func_windowed(N)
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    mod = to_mlir(build_func_windowed(N), cfg)
    print("--- mlir ---")
    print(mod.to_string())

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(1)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    exe.launch({"a": cp.asarray(a_h), "b": cp.asarray(b_h), "ws": out})
    cp.cuda.runtime.deviceSynchronize()
    out_h = cp.asnumpy(out)

    c = a_h + b_h
    cumsum = np.cumsum(c, axis=0, dtype=np.float64)
    expected = np.empty((T, S), dtype=np.float32)
    expected[:N - 1] = np.nan
    expected[N - 1] = cumsum[N - 1]
    if T > N:
        expected[N:] = (cumsum[N:] - cumsum[:-N]).astype(np.float32)

    diff = np.abs(out_h[N - 1:] - expected[N - 1:])
    max_abs = float(diff.max())
    atol = max(1e-3, 5e-7 * N)
    if max_abs > atol:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  FAIL — max |Δ| = {max_abs:.3e} > {atol:.0e} at {idx}",
                file=sys.stderr)
        return 1
    print(f"  ok — max |Δ| = {max_abs:.3e} (atol={atol:.0e})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sm_120")
    ap.add_argument("-T", "--time-length", type=int, default=64)
    ap.add_argument("-S", "--num-stocks", type=int, default=2048)
    ap.add_argument("-N", "--window", type=int, default=5)
    args = ap.parse_args()

    import cupy as cp
    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    rc = 0
    rc |= run_elemwise(args.target, args.time_length, args.num_stocks)
    print()
    rc |= run_windowed(args.target, args.time_length, args.num_stocks, args.window)
    return rc


if __name__ == "__main__":
    sys.exit(main())
