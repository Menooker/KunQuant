import numpy as np
import pandas as pd
from KunQuant.jit import cfake
from KunQuant.Op import Input, Output, Builder
from KunQuant.Stage import Function
from KunQuant.Op import *
from KunQuant.ops import *
from KunQuant.runner import KunRunner as kr


def test_stream():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        Output(WindowedQuantile(inp1, 10, 0.49), "quantile")
        Output(ExpMovingAvg(inp1, 10), "ema")
        Output(WindowedLinearRegressionSlope(inp1, 10), "slope")
    f = Function(builder.ops)
    lib = cfake.compileit([("stream_test", f, cfake.KunCompilerConfig(dtype="double", input_layout="STREAM", output_layout="STREAM"))],
                          "stream_test", cfake.CppCompilerConfig())

    executor = kr.createSingleThreadExecutor()
    stream = kr.StreamContext(executor, lib.getModule("stream_test"), 24)
    a = np.random.rand(100, 24)
    handle_a = stream.queryBufferHandle("a")
    handle_quantile = stream.queryBufferHandle("quantile")
    handle_ema = stream.queryBufferHandle("ema")
    handle_slope = stream.queryBufferHandle("slope")
    out = np.empty((100, 24))
    ema = np.empty((100, 24))
    slope = np.empty((100, 24))
    for i in range(100):
        stream.pushData(handle_a, a[i])
        stream.run()
        out[i] = stream.getCurrentBuffer(handle_quantile)
        ema[i] = stream.getCurrentBuffer(handle_ema)
        slope[i] = stream.getCurrentBuffer(handle_slope)
    df = pd.DataFrame(a)
    expected_quantile = df.rolling(10).quantile(0.49, interpolation='linear').to_numpy()
    expected_ema = df.ewm(span=10, adjust=False, ignore_na=True).mean().to_numpy()
    expected_slope = df.rolling(10).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]).to_numpy()
    np.testing.assert_allclose(out, expected_quantile, atol=1e-6, rtol=1e-4, equal_nan=True)
    np.testing.assert_allclose(ema, expected_ema, atol=1e-6, rtol=1e-4, equal_nan=True)
    np.testing.assert_allclose(slope[10:], expected_slope[10:], atol=1e-6, rtol=1e-4, equal_nan=True)

test_stream()
print("test_stream passed")