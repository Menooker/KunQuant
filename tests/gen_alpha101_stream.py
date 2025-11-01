from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.predefined.Alpha101 import AllData, all_alpha
from KunQuant.runner import KunRunner as kr
import sys

def check_alpha101_stream():
    builder = Builder()
    cnt = 0
    with builder:
        all_data = AllData(low=Input("low"),high=Input("high"),close=Input("close"),open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        for f in all_alpha:
            out = f(all_data)
            Output(out, f.__name__)
            cnt += 1
    f = Function(builder.ops)
    return "alpha_101_stream", f, KunCompilerConfig(partition_factor=8, output_layout="STREAM", options={"opt_reduce": False, "fast_log": True})


cfake.compileit([check_alpha101_stream()], "alpha101_stream", cfake.CppCompilerConfig(), tempdir=sys.argv[1], keep_files=True)