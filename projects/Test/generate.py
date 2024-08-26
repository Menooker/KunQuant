from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
import KunQuant.passes
from KunQuant.passes import *
from KunQuant.Driver import *
from KunQuant.predefined.Alpha101 import *
import os
import sys

paths = []
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
    f = build_avg_and_stddev()
    src = compileit(f, "avg_and_stddev")
    paths.append(sys.argv[1]+"/AvgAndStddev.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_TS():
    f = build_avg_and_stddev()
    src = compileit(f, "avg_and_stddev_TS", input_layout="TS", output_layout="TS")
    paths.append(sys.argv[1]+"/AvgAndStddevTS.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_ema():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(ExpMovingAvg(inp1, 5), "ou2")
    f = Function(builder.ops)
    src = compileit(f, "test_ema", input_layout="TS", output_layout="TS")
    paths.append(sys.argv[1]+"/TestEMA.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_argmin():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(TsArgMin(inp1, 5), "ou2")
        Output(WindowedMin(inp1, 5), "tsmin")
        Output(TsRank(inp1, 5), "tsrank")
    f = Function(builder.ops)
    src = compileit(f, "test_argmin", input_layout="TS", output_layout="TS")
    paths.append(sys.argv[1]+"/TestArgMin.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_rank():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(Rank(inp1), "ou2")
    f = Function(builder.ops)
    src = compileit(f, "test_rank")
    paths.append(sys.argv[1]+"/TestRank.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_rank2():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        v1 = Add(inp1, inp1)
        v2 = Rank(v1)
        v3 = Add(v2, v1)
        Output(v3, "out")
    f = Function(builder.ops)
    src = compileit(f, "test_rank2", input_layout="TS", output_layout="TS")
    paths.append(sys.argv[1]+"/TestRank2.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_log(dtype, name):
    builder = Builder()
    with builder:
        inp1 = Input("a")
        Output(Log(inp1), "outlog")
    f = Function(builder.ops)
    src = compileit(f, f"test_log{name}", dtype=dtype)
    paths.append(sys.argv[1]+f"/TestLog{name}.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

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
    src = compileit(f, "test_pow")
    paths.append(sys.argv[1]+"/TestPow.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_alpha101_double():
    builder = Builder()
    with builder:
        all_data = AllData(low=Input("low"),high=Input("high"),close=Input("close"),open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        for f in all_alpha:
            # if f.__name__ != "alpha043" and f.__name__ != "alpha039":
            #     continue
            out = f(all_data)
            Output(out, f.__name__)
    simd_len = 8 if sys.argv[2] == "avx512" else 4
    f = Function(builder.ops)
    src = compileit(f, "alpha_101", blocking_len=simd_len, input_layout="TS", output_layout="TS", dtype="double", options={"opt_reduce": True, "fast_log": True})
    os.makedirs(sys.argv[1], exist_ok=True)
    with open(sys.argv[1]+"/Alpha101.cpp", 'w') as f:
        f.write(src)

def check_aligned():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(Rank(inp1) *2, "ou2")
    f = Function(builder.ops)
    src = compileit(f, "test_aligned", input_layout="TS", output_layout="TS", allow_unaligned = False)
    paths.append(sys.argv[1]+"/TestAligned.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)


os.makedirs(sys.argv[1], exist_ok=True)
check_1()
check_TS()
check_rank()
check_rank2()
check_log("float", "")
check_log("double", "64")
check_pow()
check_alpha101_double()
check_ema()
check_argmin()
check_aligned()

# with open(sys.argv[1]+"/generated.txt", 'w') as f:
#     f.write(";".join(paths))
# print(";".join(paths))
# check_alpha101()