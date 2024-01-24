import numpy as np
import pandas as pd
import sys
import warnings
import os

base_dir = "./build/Release/projects" if os.name == "nt" else "./build/projects"
base_dir2 = "./build/Release/" if os.name == "nt" else "./build/"
sys.path.append(base_dir2)
import KunRunner as kr


# inp = np.ndarray((3, 100, 8), dtype="float32")

lib = kr.Library.load(base_dir+"/Test.dll" if os.name == "nt" else base_dir+"/libTest.so")
print(lib)


def test_runtime():
    lib2 = kr.Library.load(base_dir2+"/KunTest.dll" if os.name == "nt" else base_dir2+"/libKunTest.so")
    inp = np.random.rand(3, 10, 8).astype("float32")
    modu = lib2.getModule("testRuntimeModule")
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"input": inp}, 0, 10)

    expected = inp + inp * 2
    if not np.allclose(expected, out["output"]):
        raise RuntimeError("")


def ST_ST8t(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.reshape((-1, 8, data.shape[1])).transpose((0, 2, 1)))


def ST8t_ST(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.transpose((0, 2, 1)).reshape((-1, data.shape[1])))

def test_avg_stddev():
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


def test_rank():
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

def test_rank2():
    modu = lib.getModule("test_rank2")
    assert(modu)
    inp = np.random.rand(24, 20).astype("float32")
    df = pd.DataFrame(inp.transpose())
    # print(df)
    df = df + df
    expected = (df.rank(pct=True, axis = 1) + df).to_numpy().transpose()
    blocked = ST_ST8t(inp)
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": blocked}, 0, 20)
    output = ST8t_ST(out["out"])
    # print(expected[:,0])
    # print(output[:,0])
    np.testing.assert_allclose(output, expected, rtol=1e-6, equal_nan=True)

def test_log():
    modu = lib.getModule("test_log")
    inp = np.zeros(shape=(24, 20), dtype="float32")
    for i in range(24):
        inp[i,:] = pow(10, i-10)
    inp[0,:] = -10
    inp[-1,:] = 0
    inp[1,:] = np.nan
    # print(inp)
    blocked = ST_ST8t(inp)
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, {"a": blocked}, 0, 20)
    output = ST8t_ST(out["outlog"])
    # print(expected[:,0])
    # print(output[:,0])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'(divide by zero encountered)|(invalid value encountered)')
        np.testing.assert_allclose(output, np.log(inp), rtol=1e-5, atol=1e-5, equal_nan=True)

def test_pow():
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
    print(out.keys())
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

test_runtime()
test_avg_stddev()
test_rank()
test_rank2()
test_log()
test_pow()
print("done")
