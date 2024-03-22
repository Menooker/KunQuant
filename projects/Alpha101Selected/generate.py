from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
from KunQuant.passes import *
from KunQuant.Driver import *
from KunQuant.predefined.Alpha101 import *
import sys
import os


def check_alpha101():
    builder = Builder()
    cnt = 0
    with builder:
        all_data = AllData(low=Input("low"),high=Input("high"),close=Input("close"),open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        for f in all_alpha:
            out = f(all_data)
            Output(out, f.__name__)
            cnt += 1
    print("Total", cnt)
    f = Function(builder.ops)
    src = compileit(f, "alpha_101_selected", partition_factor=8, output_layout="STREAM", options={"opt_reduce": False, "fast_log": True})
    os.makedirs(sys.argv[1], exist_ok=True)
    with open(sys.argv[1]+"/Alpha101Selected.cpp", 'w') as f:
        f.write(src)

check_alpha101()