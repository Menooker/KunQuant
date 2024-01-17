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
    src = compileit(f, "test_rank2")
    paths.append(sys.argv[1]+"/TestRank2.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

def check_log():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        Output(Log(inp1), "outlog")
    f = Function(builder.ops)
    src = compileit(f, "test_log")
    paths.append(sys.argv[1]+"/TestLog.cpp")
    with open(paths[-1], 'w') as f:
        f.write(src)

os.makedirs(sys.argv[1], exist_ok=True)
check_1()
check_rank()
check_rank2()
check_log()

# with open(sys.argv[1]+"/generated.txt", 'w') as f:
#     f.write(";".join(paths))
# print(";".join(paths))
# check_alpha101()