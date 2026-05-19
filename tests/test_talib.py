'''
Tests for KunQuant.predefined.talib indicators against the ta-lib reference.

ta-lib is installed via `conda install -c conda-forge ta-lib` or
`pip install ta-lib`.
'''
import argparse
import time

import numpy as np
import talib

from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Input, Output, Builder
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.predefined.talib import TRANGE, ATR, SAR
from KunQuant.runner import KunRunner as kr


def _build_module():
    builder = Builder()
    with builder:
        high = Input("high")
        low = Input("low")
        close = Input("close")
        Output(TRANGE(high, low, close), "trange")
        Output(ATR(high, low, close, 14), "atr")
        Output(SAR(high, low), "sar")
    return Function(builder.ops)


def _make_data(T: int, S: int, seed: int = 13):
    np.random.seed(seed)
    close = (np.cumsum(np.random.randn(T, S), axis=0) + 100.0).astype("float64")
    noise = np.abs(np.random.randn(T, S)).astype("float64")
    high = close + noise
    low = close - noise
    return high, low, close


def test_talib():
    f = _build_module()
    cfg = KunCompilerConfig(input_layout="TS", output_layout="TS", dtype="double",
                            options={"no_fast_stat": "no_warn"})
    lib = cfake.compileit([("test_talib", f, cfg)], "test_talib_lib",
                          cfake.CppCompilerConfig(machine=cfake.NativeCPUFlags()))
    modu = lib.getModule("test_talib")
    executor = kr.createSingleThreadExecutor()

    T, S = 200, 16
    high, low, close = _make_data(T, S)
    out = kr.runGraph(executor, modu,
                      {"high": high, "low": low, "close": close}, 0, T)

    expected_tr = np.column_stack([talib.TRANGE(high[:, j], low[:, j], close[:, j])
                                   for j in range(S)])
    np.testing.assert_allclose(out["trange"], expected_tr,
                               rtol=0, atol=1e-12, equal_nan=True)

    expected_atr = np.column_stack([talib.ATR(high[:, j], low[:, j], close[:, j], timeperiod=14)
                                    for j in range(S)])
    np.testing.assert_allclose(out["atr"], expected_atr,
                               rtol=1e-10, atol=1e-10, equal_nan=True)

    expected_sar = np.column_stack([talib.SAR(high[:, j], low[:, j])
                                    for j in range(S)])
    np.testing.assert_allclose(out["sar"], expected_sar,
                               rtol=1e-10, atol=1e-10, equal_nan=True)


def _bench_compile(name: str, build_func, T: int):
    builder = Builder()
    with builder:
        build_func(builder)
    f = Function(builder.ops)
    cfg = KunCompilerConfig(input_layout="TS", output_layout="TS", dtype="double",
                            options={"no_fast_stat": "no_warn"})
    lib = cfake.compileit([(name, f, cfg)], name + "_lib",
                          cfake.CppCompilerConfig(machine=cfake.NativeCPUFlags()))
    return lib.getModule(name)


def _bench_run(label: str, kun_fn, talib_fn, T: int, S: int, runs: int, warmup: int):
    high, low, close = _make_data(T, S, seed=42)

    for _ in range(warmup):
        kun_fn(high, low, close)
    elapsed = []
    for _ in range(runs):
        t0 = time.perf_counter()
        kun_fn(high, low, close)
        elapsed.append(time.perf_counter() - t0)
    kun_avg_ms = sum(elapsed) / runs * 1000.0

    for _ in range(warmup):
        talib_fn(high, low, close)
    elapsed = []
    for _ in range(runs):
        t0 = time.perf_counter()
        talib_fn(high, low, close)
        elapsed.append(time.perf_counter() - t0)
    tl_avg_ms = sum(elapsed) / runs * 1000.0

    print(f"[benchmark] {label:<3} T={T} S={S} single-thread runs={runs} warmup={warmup}  "
          f"KunQuant avg={kun_avg_ms:.3f} ms   ta-lib avg={tl_avg_ms:.3f} ms")


def benchmark_atr(T: int = 10000, S: int = 64, runs: int = 10, warmup: int = 1):
    def build(b):
        Output(ATR(Input("high"), Input("low"), Input("close"), 14), "atr")
    modu = _bench_compile("bench_atr", build, T)
    executor = kr.createSingleThreadExecutor()

    def kun_fn(h, l, c):
        kr.runGraph(executor, modu, {"high": h, "low": l, "close": c}, 0, T)

    def tl_fn(h, l, c):
        for j in range(S):
            talib.ATR(h[:, j], l[:, j], c[:, j], timeperiod=14)

    _bench_run("ATR", kun_fn, tl_fn, T, S, runs, warmup)


def benchmark_sar(T: int = 10000, S: int = 64, runs: int = 10, warmup: int = 1):
    def build(b):
        Output(SAR(Input("high"), Input("low")), "sar")
    modu = _bench_compile("bench_sar", build, T)
    executor = kr.createSingleThreadExecutor()

    def kun_fn(h, l, c):
        kr.runGraph(executor, modu, {"high": h, "low": l}, 0, T)

    def tl_fn(h, l, c):
        for j in range(S):
            talib.SAR(h[:, j], l[:, j])

    _bench_run("SAR", kun_fn, tl_fn, T, S, runs, warmup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true",
                        help="Run single-thread ATR/SAR benchmarks after the correctness test.")
    args = parser.parse_args()

    test_talib()
    if args.bench:
        benchmark_atr()
        benchmark_sar()
    print("done")
