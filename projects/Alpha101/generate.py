from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
import KunQuant.passes
from KunQuant.passes import *
from KunQuant.Driver import *
from KunQuant.predefined.Alpha101 import *
import sys
import os

def check_alpha101():
    builder = Builder()
    with builder:
        all_data = AllData(low=Input("low"),high=Input("high"),close=Input("close"),open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        for f in all_alpha:
            # if f.__name__ != "alpha043" and f.__name__ != "alpha039":
            #     continue
            out = f(all_data)
            Output(out, f.__name__)
    simd_len = 16 if sys.argv[2] == "avx512" else 8
    f = Function(builder.ops)
    src = compileit(f, "alpha_101", blocking_len=simd_len, output_layout="TS", options={"opt_reduce": True, "fast_log": True})
    os.makedirs(sys.argv[1], exist_ok=True)
    with open(sys.argv[1]+"/Alpha101.cpp", 'w') as f:
        f.write(src)

check_alpha101()
# with open(sys.argv[1]+"/generated.txt", 'w') as f:
#     f.write(sys.argv[1]+"/Alpha101.cpp")
# print(sys.argv[1]+"/Alpha101.cpp")