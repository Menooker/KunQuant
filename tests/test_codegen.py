from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
import KunQuant.passes
from KunQuant.passes import *
from KunQuant.Driver import *
from KunQuant.predefined.Alpha101 import *

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
    with open("./tests/cpp/generated/AvgAndStddev.cpp", 'w') as f:
        f.write(src)

def check_rank():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        out2 = Output(Rank(inp1), "ou2")
    f = Function(builder.ops)
    src = compileit(f, "test_rank")
    with open("./tests/cpp/generated/TestRank.cpp", 'w') as f:
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
    src = compileit(f, "test_rank2")
    with open("./tests/cpp/generated/TestRank2.cpp", 'w') as f:
        f.write(src)

def check_log():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        Output(Log(inp1), "outlog")
    f = Function(builder.ops)
    src = compileit(f, "test_log")
    with open("./tests/cpp/generated/TestLog.cpp", 'w') as f:
        f.write(src)

def check_alpha101():
    builder = Builder()
    with builder:
        all_data = AllData(low=Input("low"),high=Input("high"),close=Input("close"),open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        for f in all_alpha:
            f(all_data)
    f = Function(builder.ops)
    src = compileit(f, "alpha_101", output_layout="TS", options={"opt_reduce": True, "fast_log": True})
    with open("./tests/cpp/generated/Alpha101.cpp", 'w') as f:
        f.write(src)

#check_1()
#check_rank()
#check_rank2()
#check_log()
check_alpha101()