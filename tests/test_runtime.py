import numpy as np
import pandas as pd
import sys

sys.path.append("./build/")
import KunRunner as kr


# inp = np.ndarray((3, 100, 8), dtype="float32")

lib = kr.Library.load("./build/libKunTest.so")
print(lib)


def test_runtime():
    inp = np.random.rand(3, 10, 8).astype("float32")
    modu = lib.getModule("testRuntimeModule")
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

test_runtime()
test_avg_stddev()
test_rank()
test_rank2()
