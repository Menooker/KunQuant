#!/usr/bin/env python3
"""End-to-end test for the KunQuant Python-IR → MLIR → CUDA path.

Builds a KunQuant Function with the high-level Op API, runs the same
Driver.optimize() pipeline the CPU compileit uses, then compiles to a
CUDA Executable via kun_mlir and validates against numpy.

Three factors are exercised:
  * elemwise:   out = (a + b) * a - b * b           (binary elemwise only;
                  doesn't touch libdevice)
  * libdevice:  out = log(abs(a)) * sign(b - a)     (Abs / Log / Sign all
                  lower to math.* ops that emit __nv_* libdevice externs;
                  the upstream `gpu-module-to-binary` pass links libdevice
                  for us, so this works end-to-end)
  * windowed:   ws  = WindowedSum(a + b, N)         (decomposes into
                  ForeachBackWindow + ReduceAdd inside the optimizer pass)

The runtime auto-discovers the CUDA toolkit (CUDA_HOME / CUDA_PATH /
CUDA_TOOLKIT_PATH / CUDA_ROOT or standard install paths).  Override
with `CudaCompilerConfig(toolkit_path=...)` if needed.
"""

from __future__ import annotations
import argparse
import sys

import numpy as np

from KunQuant.Op import Builder, Input, Output
from KunQuant.ops import Add, Sub, Mul, Abs, Log, Sign, WindowedSum
from KunQuant.ops.MiscOp import BackRef, FastWindowedSum
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


def build_func_libdevice() -> Function:
    """out = log(abs(a)) * sign(b - a) — exercises Abs/Log/Sign, all of
    which lower to math.* ops that need libdevice."""
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        v = Mul(Log(Abs(a)), Sign(Sub(bin_, a)))
        Output(v, "out")
    return Function(builder.ops, name="libdevice_kernel")


def build_func_windowed(N: int) -> Function:
    """ws = WindowedSum(a + b, N)"""
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        s = WindowedSum(Add(a, bin_), N)
        Output(s, "ws")
    return Function(builder.ops, name="windowed_kernel")


def build_func_backref(N: int) -> Function:
    """out = BackRef(a + b, N) — value of (a+b) at time t-N"""
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        Output(BackRef(Add(a, bin_), N), "out")
    return Function(builder.ops, name="backref_kernel")


def build_func_fastwindowedsum(N: int) -> Function:
    """ws = FastWindowedSum(a + b, N) — same windowed-sum semantics as
    WindowedSum, but uses the stateful Kahan-corrected algorithm from
    cpp/Kun/Ops.hpp."""
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        Output(FastWindowedSum(Add(a, bin_), N), "ws")
    return Function(builder.ops, name="fastwindowedsum_kernel")


def _run_one(label: str, build_fn, expected_fn, target: str, T: int, S: int,
              atol: float = 1e-5) -> int:
    """Compile a Function, launch it, validate against numpy."""
    print(f"=== {label} ===")
    f = build_fn()
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    mod = to_mlir(build_fn(), cfg)
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

    expected = expected_fn(a_h, b_h)
    if not np.allclose(out_h, expected, atol=atol, equal_nan=True):
        diff = np.abs(out_h - expected)
        print(f"  FAIL — max abs diff {np.nanmax(diff)}", file=sys.stderr)
        return 1
    print(f"  ok — output matches reference on every cell ({T*S} cells)")
    return 0


def run_elemwise(target: str, T: int, S: int) -> int:
    return _run_one("elemwise: out = (a+b)*a - b*b",
                     build_func_elemwise,
                     lambda a, b: (a + b) * a - b * b,
                     target, T, S)


def run_libdevice(target: str, T: int, S: int) -> int:
    # `sign` differs between MLIR (math.copysign — keeps sign bit incl 0)
    # and numpy.sign (returns 0 at 0).  With Gaussian inputs the
    # sign-of-zero case has measure zero, so equality holds.
    return _run_one("libdevice: out = log(abs(a)) * sign(b - a)",
                     build_func_libdevice,
                     lambda a, b: np.log(np.abs(a)) * np.sign(b - a),
                     target, T, S, atol=1e-4)


def run_backref(target: str, T: int, S: int, N: int) -> int:
    print(f"=== backref: out = (a+b)[t - {N}] ===")
    f = build_func_backref(N)
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    mod = to_mlir(build_func_backref(N), cfg)
    print("--- mlir ---")
    print(mod.to_string())

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(2)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    exe.launch({"a": cp.asarray(a_h), "b": cp.asarray(b_h), "out": out})
    cp.cuda.runtime.deviceSynchronize()
    out_h = cp.asnumpy(out)

    # Reference: out[t] = (a+b)[t-N] for t >= N; undefined for t < N.
    c = a_h + b_h
    diff = np.abs(out_h[N:] - c[: T - N])
    max_abs = float(diff.max())
    if max_abs > 1e-5:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  FAIL — max abs diff {max_abs} at {idx}", file=sys.stderr)
        return 1
    print(f"  ok — max |Δ| = {max_abs:.3e} on the {T - N} valid time steps")
    return 0


def run_fastwindowedsum(target: str, T: int, S: int, N: int) -> int:
    print(f"=== fast_windowed_sum: ws = FastWindowedSum(a + b, N={N}) ===")
    f = build_func_fastwindowedsum(N)
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    mod = to_mlir(build_func_fastwindowedsum(N), cfg)
    print("--- mlir ---")
    print(mod.to_string())

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(3)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    exe.launch({"a": cp.asarray(a_h), "b": cp.asarray(b_h), "ws": out})
    cp.cuda.runtime.deviceSynchronize()
    out_h = cp.asnumpy(out)

    # Reference matches WindowedSum (same window, no NaN inputs).
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
    rc |= run_libdevice(args.target, args.time_length, args.num_stocks)
    print()
    rc |= run_windowed(args.target, args.time_length, args.num_stocks, args.window)
    print()
    rc |= run_backref(args.target, args.time_length, args.num_stocks, args.window)
    print()
    rc |= run_fastwindowedsum(args.target, args.time_length, args.num_stocks,
                                args.window)
    return rc


if __name__ == "__main__":
    sys.exit(main())
