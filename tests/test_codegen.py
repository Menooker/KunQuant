from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
import KunQuant.passes
from KunQuant.passes import *
from KunQuant.Driver import *

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
    compileit(f, 8, 8, ['a', "ou2", "ou1"])


check_1()