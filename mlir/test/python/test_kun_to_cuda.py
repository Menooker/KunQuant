#!/usr/bin/env python3
"""End-to-end test for the KunQuant Python-IR → MLIR → CUDA path.

Builds a KunQuant Function with the high-level Op API, runs the same
Driver.optimize() pipeline the CPU compileit uses, then compiles to a
CUDA Executable via KunMLIR and validates against numpy.

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

from KunQuant.Op import (
    Builder, Input, Output, ConstantOp,
    WindowedTempOutput, ForeachBackWindow, IterValue,
)
from KunQuant.ops import Add, Sub, Mul, Abs, Log, Sign, WindowedSum, ReduceMax
from KunQuant.ops.ElewiseOp import (
    GreaterThan, GreaterEqual, LessThan, LessEqual, Equals,
    And, Or, Not, Select,
)
from KunQuant.ops.MiscOp import (
    BackRef, FastWindowedSum,
    Accumulator, SetAccumulator, ReturnFirstValue,
)
from KunQuant.Stage import Function
from KunQuant.jit import KunMLIR
from KunQuant.jit.cuda import compileit, CudaCompilerConfig


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
    """Two outputs over c = a + b:
       ws        = WindowedSum(c, N)
       ws_maxabs = max_{k in [0..N-1]} |c[t-k] - c[t]|

    `ws_maxabs` is a hand-built ForeachBackWindow whose body reads BOTH:
      - the block-arg  (= c[t-k], the iter value)
      - the outer ts c (= c[t], current time step)
    and reduces |·| via ReduceMax.  This exercises the kunir-to-kungpu
    inner-scope inheritance of the outer scalarMap/tsMap: `c` is computed
    outside the loop but used inside.
    """
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        c = Add(a, bin_)
        Output(WindowedSum(c, N), "ws")

        wtemp = WindowedTempOutput(c, N)
        loop  = ForeachBackWindow(wtemp, N)
        builder.set_loop(loop)
        diff = Sub(IterValue(loop, wtemp), c)
        a_diff = Abs(diff)
        builder.set_loop(None)
        Output(ReduceMax(a_diff), "ws_maxabs")
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


def build_func_accumulator() -> Function:
    """Running count of timesteps where a > 0:

       cnt[t] = cnt[t-1] + (a[t] > 0 ? 1 : 0)            (cnt[-1] = 0)

    Built directly with Accumulator + SetAccumulator + ReturnFirstValue.
    `is_whole_time_required=True` propagates `unreliable_count = -1` into
    kunir.func; the runtime treats that as a hard "single chunk" signal.
    """
    builder = Builder()
    with builder:
        a = Input("a")
        cnt = Accumulator(a, "cnt", is_whole_time_required=True)
        mask = GreaterThan(a, ConstantOp(0))
        new_cnt = Select(mask, Add(cnt, ConstantOp(1)), cnt)
        sa = SetAccumulator(cnt, mask, new_cnt)
        Output(ReturnFirstValue([new_cnt, sa]), "cnt_out")
    return Function(builder.ops, name="accumulator_kernel")


def build_func_cmp_logical() -> Function:
    """Single-graph multi-output factor that exercises every kunir cmp,
    logical, and select op in one shot:

      gt_out  = a > b  ? a : b              # element-wise max
      lt_out  = a < b  ? a : b              # element-wise min
      ge_out  = a >= b ? a : b              # max (tiebreaks to a)
      le_out  = a <= b ? a : b              # min (tiebreaks to a)
      eq_out  = a == b ? a : b              # always a where they match
      and_out = (a > 0)  & (b > 0)  ? a : b # gt + and
      or_out  = (a > 0)  | (b > 0)  ? a : b # gt + or
      not_out = !(a > b) ? a : b            # = (a <= b) ? a : b
    """
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        zero = ConstantOp(0)
        Output(Select(GreaterThan(a, bin_), a, bin_), "gt_out")
        Output(Select(LessThan(a, bin_),    a, bin_), "lt_out")
        Output(Select(GreaterEqual(a, bin_), a, bin_), "ge_out")
        Output(Select(LessEqual(a, bin_),    a, bin_), "le_out")
        Output(Select(Equals(a, bin_),       a, bin_), "eq_out")
        Output(Select(And(GreaterThan(a, zero), GreaterThan(bin_, zero)),
                       a, bin_), "and_out")
        Output(Select(Or(GreaterThan(a, zero), GreaterThan(bin_, zero)),
                       a, bin_), "or_out")
        Output(Select(Not(GreaterThan(a, bin_)), a, bin_), "not_out")
    return Function(builder.ops, name="cmp_logical_kernel")


def build_func_multipartition() -> Function:
    """A graph with three independent outputs.  Combined with
    `partition_factor=1` this drives `do_partition` to split into
    multiple sub-Functions, each becoming its own kunir.func — the
    primary thing this test exercises."""
    builder = Builder()
    with builder:
        a = Input("a")
        bin_ = Input("b")
        Output(Add(a, bin_), "add_out")
        Output(Mul(a, bin_), "mul_out")
        Output(Sub(a, bin_), "sub_out")
    return Function(builder.ops, name="multi")


def _compare_post_warmup(out_h: np.ndarray, expected: np.ndarray,
                            valid_start: int, atol: float) -> int:
    """Validate kernel output against the reference on rows
    `[valid_start:]`.  Fails loudly on **any** NaN in the kernel
    output past the warmup region — the naive `np.abs(NaN-x).max() >
    atol` form silently returns False because NaN comparisons are
    False, which would let a multi-chunk regression slip through.
    """
    tail = out_h[valid_start:]
    if np.isnan(tail).any():
        nrows = int(np.unique(np.where(np.isnan(tail))[0]).size)
        print(f"  FAIL — {nrows} of {tail.shape[0]} validated rows "
               f"contain NaN past row {valid_start}", file=sys.stderr)
        return 1
    diff = np.abs(tail - expected[valid_start:])
    max_abs = float(diff.max())
    if max_abs > atol:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  FAIL — max |Δ| = {max_abs:.3e} > {atol:.0e} at "
               f"row {valid_start + idx[0]}, col {idx[1]}", file=sys.stderr)
        return 1
    print(f"  ok — max |Δ| = {max_abs:.3e} (atol={atol:.0e})")
    return 0


def _run_one(label: str, build_fn, expected_fn, target: str, T: int, S: int,
              atol: float = 1e-5) -> int:
    """Compile a Function, launch it, validate against numpy."""
    print(f"=== {label} ===")
    f = build_fn()
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(0)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": cp.asarray(a_h),
                              "b": cp.asarray(b_h), "out": out})
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

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(2)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": cp.asarray(a_h),
                              "b": cp.asarray(b_h), "out": out})
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

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(3)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": cp.asarray(a_h),
                              "b": cp.asarray(b_h), "ws": out})
    out_h = cp.asnumpy(out)

    # Reference matches WindowedSum (same window, no NaN inputs).
    c = a_h + b_h
    cumsum = np.cumsum(c, axis=0, dtype=np.float64)
    expected = np.empty((T, S), dtype=np.float32)
    expected[:N - 1] = np.nan
    expected[N - 1] = cumsum[N - 1]
    if T > N:
        expected[N:] = (cumsum[N:] - cumsum[:-N]).astype(np.float32)

    return _compare_post_warmup(out_h, expected, valid_start=N - 1,
                                  atol=max(1e-3, 5e-7 * N))


def run_multipartition(target: str, T: int, S: int) -> int:
    """End-to-end test of the do_partition + post_optimize path:
    three independent outputs forced into separate partitions by
    `partition_factor=1`, each becoming a sibling kunir.func in the
    generated gpu.module."""
    print("=== multipartition: 3 outputs (add/mul/sub) split via "
           "partition_factor=1 ===")
    f = build_func_multipartition()
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4,
                              partition_factor=1)

    exe = compileit(f, cfg)
    print(f"  kernel_names           = {exe.kernel_names}")
    print(f"  num_kernels            = {exe.num_kernels}")
    print(f"  launch_order           = {exe.launch_order}")
    print(f"  num_buffers            = {exe.num_buffers}")
    print(f"  peak_intermediate_slots= {exe.peak_intermediate_slots}")

    # The point of the test: the partitioner actually produced more
    # than one kunir.func.  No intermediates because the three outputs
    # are independent (each consumes only graph inputs).
    assert exe.num_kernels >= 2, exe.num_kernels
    assert exe.peak_intermediate_slots == 0, exe.peak_intermediate_slots
    assert set(exe.input_names)  == {"a", "b"}
    assert set(exe.output_names) == {"add_out", "mul_out", "sub_out"}

    import cupy as cp
    rng = np.random.default_rng(7)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    add_out = cp.zeros((T, S), dtype=cp.float32)
    mul_out = cp.zeros((T, S), dtype=cp.float32)
    sub_out = cp.zeros((T, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": cp.asarray(a_h), "b": cp.asarray(b_h),
                              "add_out": add_out,
                              "mul_out": mul_out,
                              "sub_out": sub_out})

    add_h = cp.asnumpy(add_out)
    mul_h = cp.asnumpy(mul_out)
    sub_h = cp.asnumpy(sub_out)
    if not (np.allclose(add_h, a_h + b_h, atol=1e-5)
            and np.allclose(mul_h, a_h * b_h, atol=1e-5)
            and np.allclose(sub_h, a_h - b_h, atol=1e-5)):
        print(f"  FAIL — at least one of add/mul/sub mismatch",
                file=sys.stderr)
        return 1
    print(f"  ok — all 3 outputs match across {exe.num_kernels} kernels")
    return 0


def run_accumulator(target: str, T: int, S: int) -> int:
    """End-to-end correctness of Accumulator + SetAccumulator +
    ReturnFirstValue: cnt[t] = cnt[t-1] + (a[t] > 0 ? 1 : 0).

    With default `sm_fill_factor` the runtime would normally split this
    T-sized job into many chunks; the `is_whole_time_required=True` flag
    on the Accumulator propagates `unreliable_count = -1` through the
    kunir.func attr, and computeChunkPlan collapses to a single chunk.
    A failure here means the sentinel path is broken — multi-chunk
    accumulators silently reset across chunk boundaries."""
    print(f"=== accumulator: cnt[t] = cnt[t-1] + (a[t] > 0)  "
           f"(whole-time sentinel) ===")
    f = build_func_accumulator()
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)
    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(13)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    out = cp.zeros((T, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": cp.asarray(a_h), "cnt_out": out})
    out_h = cp.asnumpy(out)

    expected = np.cumsum((a_h > 0).astype(np.float32), axis=0)
    return _compare_post_warmup(out_h, expected, valid_start=0, atol=1e-5)


def run_cmp_logical(target: str, T: int, S: int) -> int:
    """End-to-end test for kunir.gt/ge/lt/le/eq + and/or/not + select.

    Verifies a single graph with eight outputs against the obvious numpy
    reference.  Exercises both bool-producing ops (cmp) and bool-consuming
    ops (and/or/not/select) plus the i1 ts type round-tripping through the
    kunir → kungpu lowering.
    """
    print("=== cmp/logical/select: 8 outputs exercising kunir bool ops ===")
    f = build_func_cmp_logical()
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(11)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)

    out_names = ["gt_out", "lt_out", "ge_out", "le_out",
                  "eq_out", "and_out", "or_out", "not_out"]
    outs = {n: cp.zeros((T, S), dtype=cp.float32) for n in out_names}

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": cp.asarray(a_h), "b": cp.asarray(b_h), **outs})

    def ref(cond: np.ndarray) -> np.ndarray:
        return np.where(cond, a_h, b_h)

    zero = np.zeros_like(a_h)
    expected = {
        "gt_out":  ref(a_h >  b_h),
        "lt_out":  ref(a_h <  b_h),
        "ge_out":  ref(a_h >= b_h),
        "le_out":  ref(a_h <= b_h),
        "eq_out":  ref(a_h == b_h),
        "and_out": ref((a_h > zero) & (b_h > zero)),
        "or_out":  ref((a_h > zero) | (b_h > zero)),
        "not_out": ref(~(a_h > b_h)),
    }

    rc = 0
    for n in out_names:
        out_h = cp.asnumpy(outs[n])
        if not np.allclose(out_h, expected[n], atol=1e-5):
            diff = np.abs(out_h - expected[n])
            idx  = np.unravel_index(int(np.nanargmax(diff)), diff.shape)
            print(f"  FAIL {n} — max |Δ|={float(diff.max()):.3e} at {idx}",
                    file=sys.stderr)
            rc = 1
        else:
            print(f"  ok {n}")
    if rc == 0:
        print(f"  ok — all 8 outputs match across {T*S} cells")
    return rc


def build_windowed(target: str, N: int):
    """Compile `build_func_windowed(N)` once.  The returned executable
    can be reused across multiple `test_windowed` invocations with
    different T / S / mask (anything that doesn't change the graph
    topology or window size N)."""
    f = build_func_windowed(N)
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)
    exe = compileit(f, cfg)
    print(f"  [build windowed N={N}] kernels={exe.kernel_names}  "
           f"num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")
    return exe


def test_windowed(exe, T: int, S: int, N: int, mask: int = 0) -> int:
    """Correctness check against numpy for the two outputs of
    `build_func_windowed`:
       ws        = WindowedSum(c, N)             — stateful fast_windowed_sum
       ws_maxabs = max_k |c[t-k] - c[t]|         — hand-built ForeachBackWindow
                                                    body that reads BOTH the
                                                    block-arg (c[t-k]) AND the
                                                    outer ts c (c[t]).
       (c = a + b, k in [0..N-1])

    With `mask > 0` the output time dim shrinks by `mask` and the kernel
    runs with that mask — exercises the multi-chunk + mask path
    (chunk-local `t - loop_lb >= window` guard) for both outputs.

    `exe` must have been compiled with the matching `N`.
    """
    assert 0 <= mask < T
    mask_tag = f", mask={mask}" if mask else ""
    print(f"=== windowed: ws = WindowedSum(a + b, N={N}){mask_tag}; "
           f"ws_maxabs = max_k |c[t-k] - c[t]|  (c = a+b) ===")

    import cupy as cp
    rng = np.random.default_rng(1)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    out_T      = T - mask
    ws_out     = cp.zeros((out_T, S), dtype=cp.float32)
    maxabs_out = cp.zeros((out_T, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    inputs = {"a": cp.asarray(a_h), "b": cp.asarray(b_h),
              "ws": ws_out, "ws_maxabs": maxabs_out}
    if mask:
        executor.runGraph(exe, inputs, mask=mask)
    else:
        executor.runGraph(exe, inputs)
    ws_h     = cp.asnumpy(ws_out)
    maxabs_h = cp.asnumpy(maxabs_out)

    # Build full-T references, then slice from `mask` onward (no-op when
    # mask == 0).  Output row i ↔ input time i+mask; reliable when
    # i + mask >= N - 1.
    c = a_h + b_h
    cumsum = np.cumsum(c, axis=0, dtype=np.float64)
    ws_full = np.empty((T, S), dtype=np.float32)
    ws_full[:N - 1] = np.nan
    ws_full[N - 1] = cumsum[N - 1]
    if T > N:
        ws_full[N:] = (cumsum[N:] - cumsum[:-N]).astype(np.float32)
    ws_expected = ws_full[mask:]

    maxabs_full = np.empty((T, S), dtype=np.float32)
    maxabs_full[:N - 1] = np.nan
    for t in range(N - 1, T):
        window = c[t - N + 1 : t + 1]                     # (N, S)
        maxabs_full[t] = np.max(np.abs(window - c[t]), axis=0)
    maxabs_expected = maxabs_full[mask:]

    valid_start = max(0, N - 1 - mask)
    rc = 0
    rc |= _compare_post_warmup(ws_h, ws_expected,
                                  valid_start=valid_start,
                                  atol=max(1e-3, 5e-7 * N))
    rc |= _compare_post_warmup(maxabs_h, maxabs_expected,
                                  valid_start=valid_start, atol=1e-5)
    return rc


def run_backref_with_mask(target: str, T: int, S: int, N: int,
                              mask: int) -> int:
    """Same BackRef(a+b, N) graph as `run_backref`, but driven with a
    non-zero `mask`.  Picked over `WindowedSum` for the mask test
    BackRef is stateless along the time axis (each output is a gmem
    load at offset -N), so this case isolates the mask/warmup
    interaction from any rolling-state concerns.  The windowed sum
    counterpart below covers the stateful path.
    """
    print(f"=== backref + mask: out = (a+b)[t - {N}], mask={mask} ===")
    assert 0 < mask < T, "test requires 0 < mask < T"
    f = build_func_backref(N)
    cfg = CudaCompilerConfig(gpu_arch=target, warps_per_cta=4)

    exe = compileit(f, cfg)
    print(f"  kernels={exe.kernel_names}  num_buffers={exe.num_buffers}  "
           f"peak_intermediate_slots={exe.peak_intermediate_slots}")

    import cupy as cp
    rng = np.random.default_rng(4)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    # Output time dim shrinks by mask.
    out = cp.zeros((T - mask, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": cp.asarray(a_h),
                              "b": cp.asarray(b_h), "out": out},
                       mask=mask)
    out_h = cp.asnumpy(out)

    # Reference: out_full[t] = (a+b)[t-N] for t ≥ N; undefined for t < N.
    # With mask, out_full[mask + i] lands at out_h[i].  Reliable when
    # mask + i ≥ N, i.e., i ≥ max(0, N - mask).
    c = a_h + b_h
    valid_start = max(0, N - mask)
    # Build a full-(T-mask) expected so _compare_post_warmup can validate
    # the post-warmup tail uniformly (matches the windowed test below).
    expected = np.empty((T - mask, S), dtype=np.float32)
    expected[:valid_start] = np.nan
    if valid_start < T - mask:
        in_time = np.arange(mask + valid_start, T)
        expected[valid_start:] = c[in_time - N]
    return _compare_post_warmup(out_h, expected,
                                  valid_start=valid_start, atol=1e-5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sm_120")
    # Defaults sized to comfortably trigger multi-chunk: T=128 with
    # warmup=5 (N) gives `cap_warmup = 128/(4*5) = 6` chunks; S=1024
    # gives `stock_tiles = 1024/(4*32) = 8`, so even on a small GPU
    # the sm-fill target ≥ 2 — well inside the multi-chunk regime.
    ap.add_argument("-T", "--time-length", type=int, default=128)
    ap.add_argument("-S", "--num-stocks", type=int, default=1024)
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
    # Build once for N=args.window, reuse across the mask=0 and mask=3
    # validations (graph topology + window size are the same; only T/S/mask
    # differ at run time).
    windowed_exe = build_windowed(args.target, args.window)
    rc |= test_windowed(windowed_exe, args.time_length, args.num_stocks,
                          args.window)
    print()
    rc |= run_backref(args.target, args.time_length, args.num_stocks, args.window)
    print()
    # Mask smaller than the window, so the post-mask output still
    # contains unreliable rows — exercises both warmup overlap (chunks
    # ≥ 1 prime by reading back `unreliable_count` steps) AND the
    # mask-skip-vs-warmup-skip distinction on chunk 0.  Two graphs:
    # stateless BackRef and stateful WindowedSum / fast_windowed_sum.
    rc |= run_backref_with_mask(args.target, args.time_length, args.num_stocks,
                                  args.window, mask=3)
    print()
    rc |= test_windowed(windowed_exe, args.time_length, args.num_stocks,
                          args.window, mask=3)
    print()
    rc |= run_fastwindowedsum(args.target, args.time_length, args.num_stocks,
                                args.window)
    print()
    # Single-chunk fallback corner case: warmup so large relative to T
    # that `cap_warmup = T/(K*N) = 64/(4*20) = 0` clamps num_chunks to 1.
    # Exercises the multi-chunk kernel binary in its degenerate
    # grid_y=1 launch configuration — guards against regressions in
    # time_lb / time_ub / write-gating when `chunk_size = T`.  Different
    # N → fresh build.
    windowed_exe_n20 = build_windowed(args.target, N=20)
    rc |= test_windowed(windowed_exe_n20, T=64, S=args.num_stocks,
                          N=20, mask=1)
    print()
    rc |= run_multipartition(args.target, args.time_length, args.num_stocks)
    print()
    rc |= run_accumulator(args.target, args.time_length, args.num_stocks)
    print()
    rc |= run_cmp_logical(args.target, args.time_length, args.num_stocks)
    return rc


if __name__ == "__main__":
    sys.exit(main())
