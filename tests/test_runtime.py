from KunQuant.Driver import KunCompilerConfig
import numpy as np
import pandas as pd
import sys
import warnings
import os
from KunQuant.jit import cfake
from KunQuant.Op import Input, Output, Builder
from KunQuant.Stage import Function
from KunQuant.Op import *
from KunQuant.ops import *
from KunQuant.predefined.Alpha101 import *
from  KunQuant.runner import KunRunner as kr
import sys
from KunQuant.jit.env import cpu_arch

def get_compiler_flags():
    if os.environ.get("KUN_TEST_NO_AVX2", "0") != "0":
        print("building AVX without AVX2")
        return cfake.X64CPUFlags(avx2=False, fma=False)
    else:
        return cfake.NativeCPUFlags()

def test_generic_cross_sectional():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        inp3 = Input("c")
        Output(DiffWithWeightedSum(inp1 - inp2, inp3), "out")
    f = Function(builder.ops)
    lib = cfake.compileit([("test1", f, cfake.KunCompilerConfig(input_layout="TS", output_layout="TS"))],
                          "cfaketest", cfake.CppCompilerConfig(machine=get_compiler_flags()))

    a = np.random.rand(24, 16).astype("float32")
    b = np.random.rand(24, 16).astype("float32")
    c = np.random.rand(24, 16).astype("float32")
    ret = np.random.rand(24, 16).astype("float32")
    executor = kr.createSingleThreadExecutor()
    kr.runGraph(executor, lib.getModule("test1"), {"a": a, "b": b, "c": c}, 0, 24, {"out": ret})

    v1 = a - b
    # compute row-wise dot of v1 an c
    v2 = np.sum(v1 * c, axis=1)
    expected = v1 - v2.reshape((-1, 1))
    np.testing.assert_allclose(expected, ret, atol=1e-6, rtol=1e-4, equal_nan=True)

def test_corrwith():
    a = np.random.rand(24, 20).astype("float32")
    b = np.random.rand(24, 20).astype("float32")
    ret = np.random.rand(24, 20).astype("float32")
    out1 = np.empty((24), dtype="float32")
    out2 = np.empty((24), dtype="float32")
    executor = kr.createSingleThreadExecutor()
    kr.corrWith(executor, [a,b], ret, [out1, out2], "TS")
    ex1 = pd.DataFrame(a).corrwith(pd.DataFrame(ret), axis=1)
    np.testing.assert_allclose(out1, ex1, atol=1e-6, rtol=1e-4, equal_nan=True)
    ex1 = pd.DataFrame(b).corrwith(pd.DataFrame(ret), axis=1)
    np.testing.assert_allclose(out2, ex1, atol=1e-6, rtol=1e-4, equal_nan=True)

    a2 =  pd.DataFrame(a).rank(axis=1, pct=True).astype("float32")
    b2 =  pd.DataFrame(b).rank(axis=1, pct=True).astype("float32")
    ret2 =  pd.DataFrame(b).rank(axis=1, pct=True).astype("float32")
    kr.corrWith(executor, [a2.to_numpy(),b2.to_numpy()], ret2.to_numpy(), [out1, out2], "TS", rank_inputs = True)
    ex1 = a2.corrwith(ret2, axis=1)
    np.testing.assert_allclose(out1, ex1, atol=1e-6, rtol=1e-4, equal_nan=True)
    ex1 = b2.corrwith(ret2, axis=1)
    np.testing.assert_allclose(out2, ex1, atol=1e-6, rtol=1e-4, equal_nan=True)

def test_cfake():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        Output(inp1 * inp2 + 10, "out")
    f = Function(builder.ops)
    lib = cfake.compileit([("test1", f, cfake.KunCompilerConfig(input_layout="TS", output_layout="TS"))],
        "cfaketest", cfake.CppCompilerConfig(machine=get_compiler_flags()))
    mod = lib.getModule("test1")
    inp = np.random.rand(10, 24).astype("float32")
    inp2 = np.random.rand(10, 24).astype("float32")
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, mod, {"a": inp, "b": inp2}, 0, 10)
    np.testing.assert_allclose(inp * inp2 + 10, out["out"])

def test_runtime(libpath):
    # test error handling
    try:
        print("Load non-exist library. It may print some error message below. It is OK.")
        lib1 = kr.Library.load(libpath+"notexist")
        raise ValueError("Should not reach here")
    except RuntimeError as e:
        pass
    lib2 = kr.Library.load(libpath)
    inp = np.random.rand(3, 10, 8).astype("float32")
    modu = lib2.getModule("testRuntimeModule")
    try:
        lib2.getModule("testRuntimeModule2")
        raise ValueError("Should not reach here")
    except RuntimeError as e:
        pass
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"input": inp}, 0, 10)

    expected = inp + inp * 2
    if not np.allclose(expected, out["output"]):
        raise RuntimeError("")

def ST_ST8t(data: np.ndarray = 8, is_double = False) -> np.ndarray:
    blocking = 4 if cpu_arch == "aarch64" else 8
    if is_double:
        blocking = blocking // 2

    return np.ascontiguousarray(data.reshape((-1, blocking, data.shape[1])).transpose((0, 2, 1)))


def ST8t_ST(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.transpose((0, 2, 1)).reshape((-1, data.shape[1])))

def build_avg_and_stddev():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        v1 = WindowedAvg(inp1, 10)
        v2 = WindowedStddev(inp1, 10)
        out1 = Output(v1, "ou1")
        out2 = Output(v2, "ou2")
    return Function(builder.ops)

def check_1():
    return "avg_and_stddev", build_avg_and_stddev(), KunCompilerConfig()

def test_avg_stddev(lib):
    modu = lib.getModule("avg_and_stddev")
    assert(modu)
    inp = np.random.rand(24, 20).astype("float32")
    df = pd.DataFrame(inp.transpose())
    expected_mean = df.rolling(10).mean().to_numpy().transpose()
    expected_stddev = df.rolling(10).std().to_numpy().transpose()
    blocked = ST_ST8t(inp)
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": blocked}, 0, 20)
    outmean = ST8t_ST(out["ou1"])
    outstd = ST8t_ST(out["ou2"])
    np.testing.assert_allclose(outmean, expected_mean, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(outstd, expected_stddev, rtol=1e-6, equal_nan=True)

####################################

def check_TS():
    return "avg_and_stddev_TS", build_avg_and_stddev(), KunCompilerConfig(dtype="double", input_layout="TS", output_layout="TS", options={"no_fast_stat": 'no_warn'})

def test_avg_stddev_TS(lib):
    modu = lib.getModule("avg_and_stddev_TS")
    assert(modu)
    inp = np.random.rand(24, 20)
    df = pd.DataFrame(inp.transpose())
    expected_mean = df.rolling(10).mean().to_numpy().transpose()
    expected_stddev = df.rolling(10).std().to_numpy().transpose()
    blocked = np.ascontiguousarray(inp.transpose())
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": blocked}, 0, 20)
    outmean = out["ou1"].transpose()
    outstd = out["ou2"].transpose()
    np.testing.assert_allclose(outmean, expected_mean, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(outstd, expected_stddev, rtol=1e-6, equal_nan=True)
####################################

def check_covar():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        out1 = Output(WindowedCovariance(inp1, 10, inp2), "ou1")
        out2 = Output(WindowedCorrelation(inp1, 10, inp2), "ou2")
    f = Function(builder.ops)
    return "test_covar", f, KunCompilerConfig(input_layout="TS", output_layout="TS", dtype="double", options={"no_fast_stat": 'no_warn'})

def test_covar(lib):
    modu = lib.getModule("test_covar")
    assert(modu)
    inp = np.random.rand(200, 24).astype("float64")
    inp2 = np.random.rand(200, 24).astype("float64")
    df = pd.DataFrame(inp)
    df2 = pd.DataFrame(inp2)
    expected_covar = df.rolling(10).cov(df2).to_numpy()
    expected_corr = df.rolling(10).corr(df2).to_numpy()
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp, "b": inp2}, 0, 200)
    outcovar = out["ou1"]
    outcorr = out["ou2"]
    np.testing.assert_allclose(outcovar, expected_covar, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(outcorr, expected_corr, rtol=1e-6, equal_nan=True)

####################################

def check_quantile():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out1 = Output(WindowedQuantile(inp1, 10, 0.49), "ou1")
    f = Function(builder.ops)
    return "test_quantile", f, KunCompilerConfig(input_layout="TS", output_layout="TS", dtype="double", options={"no_fast_stat": 'no_warn'})

def test_quantile(lib):
    modu = lib.getModule("test_quantile")
    assert(modu)
    inp = np.random.rand(200, 24).astype("float64")
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp}, 0, 200)
    outquantile = out["ou1"]
    df = pd.DataFrame(inp)
    expected_quantile = df.rolling(10).quantile(0.49, interpolation='linear').to_numpy()
    np.testing.assert_allclose(outquantile, expected_quantile, rtol=1e-6, equal_nan=True)

####################################

def check_large_rank():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out1 = Output(TsRank(inp1, 200), "ou1")
        out2 = Output(WindowedMin(inp1, 200), "min")
        out3 = Output(WindowedMax(inp1, 200), "max")
        out4 = Output(TsArgMin(inp1, 200), "argmin")
        out5 = Output(TsArgMax(inp1, 200), "argmax")
    f = Function(builder.ops)
    return "test_large_rank", f, KunCompilerConfig(input_layout="TS", output_layout="TS", dtype="double", options={"no_fast_stat": 'no_warn'})

def test_large_rank(lib):
    modu = lib.getModule("test_large_rank")
    assert(modu)
    inp = np.random.rand(2000, 24).astype("float64")
    # test with duplicates
    inp[400:410,:] = -1
    # inp[1400:1410,:] = 10
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp}, 0, 2000)
    outrank = out["ou1"]
    df = pd.DataFrame(inp)
    expected_rank = df.rolling(200).rank().to_numpy()
    np.testing.assert_allclose(outrank, expected_rank, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(out["min"], df.rolling(200).min().to_numpy(), rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(out["max"], df.rolling(200).max().to_numpy(), rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(out["argmin"], df.rolling(200).apply(np.argmin)+1, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(out["argmax"], df.rolling(200).apply(np.argmax)+1, rtol=1e-6, equal_nan=True)

####################################

def RefExpMovingAvg(v: pd.DataFrame):
    return v.ewm(span=5, adjust=False, ignore_na=True).mean()

def check_ema():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(ExpMovingAvg(inp1, 5), "ou2")
        Output(ExpMovingAvg(ExpMovingAvg(ExpMovingAvg(BackRef(inp1, 1), 5), 5), 5), "gh_issue_26")
    f = Function(builder.ops)
    return "test_ema", f, KunCompilerConfig(input_layout="TS", output_layout="TS")

def test_ema(lib):
    modu = lib.getModule("test_ema")
    assert(modu)
    inp = np.random.rand(20, 24).astype("float32")
    inp[5,:] = np.nan
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp}, 0, 20)
    output = out["ou2"]
    df = pd.DataFrame(inp)
    expected = RefExpMovingAvg(df)
    expected2 = RefExpMovingAvg(RefExpMovingAvg(RefExpMovingAvg(df.shift(1))))
    np.testing.assert_allclose(output, expected, rtol=1e-6, equal_nan=True)    
    np.testing.assert_allclose(out["gh_issue_26"], expected2, rtol=1e-6, equal_nan=True)

####################################

def check_ema_init():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        init = Input("__init_1")
        init.attrs["single_value"] = True
        out2 = Output(ExpMovingAvg(inp1, 5, init), "ou2")
    f = Function(builder.ops)
    return "test_ema_init", f, KunCompilerConfig(input_layout="TS", output_layout="TS")

def test_ema_init(lib):
    modu = lib.getModule("test_ema_init")
    assert(modu)
    inp = np.random.rand(21, 24).astype("float32")
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp, '__init_1': inp[0]}, 1, 20)
    output = out["ou2"]
    df = pd.DataFrame(inp)
    expected = RefExpMovingAvg(df)
    np.testing.assert_allclose(output, expected[1:], rtol=1e-6, equal_nan=True)


####################################

def check_argmin():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(TsArgMin(inp1, 5), "ou2")
        Output(WindowedMin(inp1, 5), "tsmin")
        Output(TsRank(inp1, 5), "tsrank")
    f = Function(builder.ops)
    return "test_argmin", f, KunCompilerConfig(input_layout="TS", output_layout="TS")

def test_argmin_issue19(lib):
    #https://github.com/Menooker/KunQuant/issues/19
    modu = lib.getModule("test_argmin")
    assert(modu)
    inp = np.empty((6, 8),"float32")
    data = [ 0.6898481863442985, 0.6992020600574415, 0.6992020600574417, 0.6968635916291558, 0.6968635916291558, 0.6968635916291558 ]
    for i in range(6):
        inp[i, :] = data[i]
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp}, 0, 6)
    df = pd.DataFrame(inp)
    expected =df.rolling(5, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
    output = out["ou2"][4:]
    np.testing.assert_allclose(output, expected[4:], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(out["tsmin"], df.rolling(5).min(), rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(out["tsrank"], df.rolling(5).rank(), rtol=1e-6, equal_nan=True)

####################################

def check_rank():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(Rank(inp1), "ou2")
    f = Function(builder.ops)
    return "test_rank", f, KunCompilerConfig()

def test_rank(lib):
    modu = lib.getModule("test_rank")
    assert(modu)
    def check(inp, timelen):
        df = pd.DataFrame(inp.transpose())
        # print(df)
        expected = df.rank(pct=True, axis = 1).to_numpy().transpose()
        blocked = ST_ST8t(inp)
        executor = kr.createSingleThreadExecutor()
        out = kr.runGraph(executor, modu, {"a": blocked}, 0, timelen)
        output = ST8t_ST(out["ou2"])
        # print(expected[:,0])
        # print(output[:,0])
        np.testing.assert_allclose(output, expected, rtol=1e-6, equal_nan=True)
    inp = np.random.rand(24, 20).astype("float32")
    check(inp, 20)
    for i in range(20):
        inp[i,:] = i//2
    check(inp, 20)
    inp[10,:] = np.nan
    check(inp, 20)

####################################

def check_rank2():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        v1 = Add(inp1, inp1)
        v2 = Rank(v1)
        v3 = Add(v2, v1)
        Output(v3, "out")
    f = Function(builder.ops)
    return "test_rank2", f, KunCompilerConfig(dtype="double", input_layout="TS", output_layout="TS", options={'no_fast_stat': 'no_warn'})

def test_rank2(lib):
    modu = lib.getModule("test_rank2")
    assert(modu)
    def compute(stocks):
        inp = np.random.rand(stocks, 200).astype("float64")
        df = pd.DataFrame(inp.transpose())
        # print(df)
        df = df + df
        expected = (df.rank(pct=True, axis = 1) + df).to_numpy().transpose()
        blocked = np.ascontiguousarray(inp.transpose())
        executor = kr.createSingleThreadExecutor()
        out = kr.runGraph(executor, modu, {"a": blocked}, 0, 200)
        output = out["out"].transpose()
        # print(expected[:,0])
        # print(output[:,0])
        np.testing.assert_allclose(output, expected, rtol=0, atol=0, equal_nan=True)
    compute(24)
    # test unaligned
    compute(20)

####################################

def check_rank_alpha029():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inner = WindowedSum(Rank(Rank(-1 * Rank(inp1))), 5)
        v = Rank(inner)
        Output(inner, "ou1")
        Output(v, "ou2")
    f = Function(builder.ops)
    allow_unaligned = cpu_arch != "aarch64"
    return "test_rank_alpha029", f, KunCompilerConfig(input_layout="TS", output_layout="TS", allow_unaligned = allow_unaligned, dtype="double", options={"opt_reduce": True, "fast_log": True, 'no_fast_stat': 'no_warn'})

def test_rank029(lib):
    modu = lib.getModule("test_rank_alpha029")
    assert(modu)
    def rank(df):
        """
        Cross sectional rank
        :param df: a pandas DataFrame.
        :return: a pandas DataFrame with rank along columns.
        """
        #return df.rank(axis=1, pct=True)
        return df.rank(axis=1, pct=True)
    def compute(stocks):
        inp = np.random.rand(stocks, 300).astype("float64")
        df = pd.DataFrame(inp.transpose())
        inner = rank(rank(-1 * rank(df))).rolling(5).sum()
        expected = rank(inner)
        inner = inner.to_numpy().transpose()
        expected = expected.to_numpy().transpose()
        blocked = np.ascontiguousarray(inp.transpose())
        executor = kr.createSingleThreadExecutor()
        out = kr.runGraph(executor, modu, {"a": blocked}, 0, 300)
        output1 = out["ou1"].transpose()
        output2 = out["ou2"].transpose()
        np.set_printoptions(precision=60)
        # check that they are exactly the same
        np.testing.assert_allclose(output1, inner, rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(output2, expected, rtol=0, atol=0, equal_nan=True)
    compute(20)

####################################

def check_log(dtype, name):
    builder = Builder()
    with builder:
        inp1 = Input("a")
        Output(Log(inp1), "outlog")
    f = Function(builder.ops)
    return (f"test_log{name}", f, KunCompilerConfig(dtype=dtype, options={'no_fast_stat': 'no_warn'}))

def test_log(lib, dtype, name):
    modu = lib.getModule(f"test_log{name}")
    inp = np.zeros(shape=(24, 20), dtype=dtype)
    for i in range(24):
        inp[i,:] = pow(10, i-10)
    inp[0,:] = -10
    inp[-1,:] = 0
    inp[1,:] = np.nan
    # print(inp)
    blocked = ST_ST8t(inp, is_double=(dtype=="float64"))
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": blocked}, 0, 20)
    output = ST8t_ST(out["outlog"])
    # print(expected[:,0])
    # print(output[:,0])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'(divide by zero encountered)|(invalid value encountered)')
        np.testing.assert_allclose(output, np.log(inp), rtol=1e-5, atol=1e-5, equal_nan=True)

####################################

def check_pow():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        Output(Pow(inp1, ConstantOp(0.5)), "sqr")
        Output(Pow(inp1, ConstantOp(2)), "pow2")
        Output(Pow(inp1, ConstantOp(5)), "pow5")
        Output(Pow(inp1, ConstantOp(1.2)), "pow1_2")
        Output(Pow(inp1, inp2), "powa_b")
        Output(Pow(ConstantOp(1.2), inp2), "pow12_b")
    f = Function(builder.ops)
    return ("test_pow", f, KunCompilerConfig())

def test_pow(lib):
    modu = lib.getModule("test_pow")
    base = np.zeros(shape=(16, 20), dtype="float32")
    for i in range(16):
        base[i,:] = pow(10, i-8)
    base[-1,:] = 0
    base[1,:] = np.nan
    
    expo = np.zeros(shape=(16, 20), dtype="float32")
    for i in range(16):
        expo[i,:] = pow(10, i/8-1)
    expo[-1,:] = 0
    expo[1,:] = np.nan
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": ST_ST8t(base), "b": ST_ST8t(expo)}, 0, 20)
    # print(out.keys())
    # print(expected[:,0])
    # print(output[:,0])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'(divide by zero encountered)|(invalid value encountered)')
        np.testing.assert_allclose(ST8t_ST(out["sqr"]), np.power(base, 0.5), rtol=1e-5, atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(ST8t_ST(out["pow2"]), np.power(base, 2), rtol=1e-5, atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(ST8t_ST(out["pow5"]), np.power(base, 5), rtol=1e-5, atol=1e-5, equal_nan=True)
        # np.testing.assert_allclose(ST8t_ST(out["pow1_2"]), np.power(base, 1.2), rtol=1e-5, atol=1e-5, equal_nan=True)
        # np.testing.assert_allclose(ST8t_ST(out["powa_b"]), np.power(base, expo), rtol=1e-5, atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(ST8t_ST(out["pow12_b"]), np.power(1.2, expo), rtol=1e-5, atol=1e-5, equal_nan=True)

####################################

def check_aligned():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(Rank(inp1) *2, "ou2")
    f = Function(builder.ops)
    return ("test_aligned", f, KunCompilerConfig(input_layout="TS", output_layout="TS", allow_unaligned = False))

def test_aligned(lib):
    modu = lib.getModule("test_aligned")
    assert(modu)
    def check(inp, timelen):
        df = pd.DataFrame(inp.transpose())
        expected = df.rank(pct=True, axis = 1).to_numpy().transpose()
        expected *= 2
        blocked = np.ascontiguousarray(inp.transpose())
        executor = kr.createSingleThreadExecutor()
        out = kr.runGraph(executor, modu, {"a": blocked}, 0, timelen)
        output = out["ou2"].transpose()
        np.testing.assert_allclose(output, expected, rtol=1e-6, equal_nan=True)
        blocked = np.ascontiguousarray(blocked[:,:-3])
        try:
            # check that the shape must be SIMD aligned in aligned mode
            out = kr.runGraph(executor, modu, {"a": blocked}, 0, timelen)
            # expecting a runtime error
            raise ValueError()
        except RuntimeError as e:
            assert(str(e) == "Bad shape at a")
    inp = np.random.rand(24, 20).astype("float32")
    check(inp, 20)

####################################

def check_skew_kurt():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(WindowedSkew(inp1, 5), "ou2")
        out2 = Output(WindowedKurt(inp1, 5), "ou3")
        # test both fast and slow mode
        skew2 = WindowedSkew(inp1, 5)
        kurt2 = WindowedKurt(inp1, 5)
        skew2.attrs['no_fast_stat'] = True
        kurt2.attrs['no_fast_stat'] = True
        Output(skew2, "ou4")
        Output(kurt2, "ou5")
    f = Function(builder.ops)
    return "test_skew", f, KunCompilerConfig(input_layout="TS", output_layout="TS", dtype="double", options={'no_fast_stat': 'no_warn'})

def test_skew_kurt():
    modu = lib.getModule("test_skew")
    assert(modu)
    inp = np.random.rand(20, 24)
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp}, 0, 20)
    output = out["ou2"]
    df = pd.DataFrame(inp)
    expected = df.rolling(5).skew()
    np.testing.assert_allclose(output, expected, equal_nan=True)
    np.testing.assert_allclose(out["ou4"], expected, equal_nan=True)

    
    output = out["ou3"]
    expected = df.rolling(5).kurt()
    np.testing.assert_allclose(output, expected, equal_nan=True)
    np.testing.assert_allclose(out["ou5"], expected, equal_nan=True)

######################################################

def create_stream_gh_issue_41():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(inp1 * inp1, "ou2")
    f = Function(builder.ops)
    lib = cfake.compileit([("test_stream", f, KunCompilerConfig(input_layout="STREAM", output_layout="STREAM"))],
        "test2", cfake.CppCompilerConfig(machine=get_compiler_flags()))
    modu = lib.getModule("test_stream")
    executor = kr.createSingleThreadExecutor()
    return kr.StreamContext(executor, modu, 8)

def test_stream_lifetime_gh_issue_41():
    # This causes segfault when calling methods before issue 41
    stream = create_stream_gh_issue_41()
    stream.pushData(0, np.array([0] * 8, dtype="float32"))

####################################


def create_stream_double():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(inp1 * inp1, "ou2")
    f = Function(builder.ops)
    return "test_stream", f, KunCompilerConfig(input_layout="STREAM", output_layout="STREAM", dtype="double")

def test_stream_double():
    modu = lib.getModule("test_stream")
    executor = kr.createSingleThreadExecutor()
    stream = kr.StreamContext(executor, modu, 8)
    input_data = np.array([4] * 8, dtype="double")
    stream.pushData(stream.queryBufferHandle("a"), input_data)
    stream.run()
    value: np.ndarray = stream.getCurrentBuffer(stream.queryBufferHandle("ou2"))[:]
    np.testing.assert_allclose(value, input_data * input_data, equal_nan=False)


def create_loop_index():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        Output(WindowedMaxDrawdown(inp1, 5), "out")
    f = Function(builder.ops)
    return "test_max_drawdown", f, KunCompilerConfig(input_layout="TS", output_layout="TS")


def test_loop_index():
    modu = lib.getModule("test_max_drawdown")
    assert(modu)
    inp = np.random.rand(20, 24).astype("float32")
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": inp}, 0, 20)
    output = out["out"]
    
    # reference implementation, from https://stackoverflow.com/a/21059308. Modified for our version of maxdd
    def windowed_view(x, window_size):
        from numpy.lib.stride_tricks import as_strided
        y = as_strided(x, shape=(x.size - window_size + 1, window_size),
                    strides=(x.strides[0], x.strides[0]))
        return y

    def rolling_max_dd(x, window_size, min_periods=1):
        if min_periods < window_size:
            pad = np.empty(window_size - min_periods)
            pad.fill(x[0])
            x = np.concatenate((pad, x))
        y = windowed_view(x, window_size).copy()
        maxv = y.max(axis=1)
        argmax_pos = np.argmax(y, axis=1)
        for idx, pos in enumerate(argmax_pos):
            y[idx, :pos] = np.inf
        minv = y.min(axis=1)
        return (maxv - minv) / maxv
    expected = np.empty_like(output)
    expected.fill(np.nan)
    for i in range(inp.shape[1]):
        expected[:,i] = rolling_max_dd(inp[:,i], 5, min_periods=1)
    np.testing.assert_allclose(output[5:], expected[5:], equal_nan=True, atol=1e-7, rtol=1e-7)

test_stream_lifetime_gh_issue_41()
test_corrwith()
funclist = [
    check_1(),
    check_TS(),
    check_rank(),
    check_log("float", ""),
    check_pow(),
    # check_alpha101_double(),
    check_ema(),
    check_ema_init(),
    check_argmin(),
    create_loop_index(),
    check_log("double", "64"),
    create_stream_double(),
    check_rank2(),
    check_skew_kurt(),
    check_aligned(),
    check_rank_alpha029(),
    check_covar(),
    check_quantile(),
    check_large_rank(),
    ]
lib = cfake.compileit(funclist, "test", cfake.CppCompilerConfig(machine=get_compiler_flags()))

test_cfake()
test_avg_stddev_TS(lib)
kun_test_dll = os.path.join(cfake.get_runtime_path(), "KunTest.dll" if cfake.is_windows() else "libKunTest.so")
if os.path.exists(kun_test_dll):
    test_runtime(kun_test_dll)
test_avg_stddev(lib)
test_rank(lib)
test_log(lib, "float32", "")
test_pow(lib)
test_ema(lib)
test_ema_init(lib)
test_argmin_issue19(lib)
test_generic_cross_sectional()
test_stream_double()
test_log(lib, "float64", "64")
test_rank2(lib)
test_rank029(lib)
test_skew_kurt()
test_aligned(lib)
test_loop_index()
test_covar(lib)
test_quantile(lib)
test_large_rank(lib)
print("done")
