from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
from KunQuant.passes import *
from KunQuant.Driver import *
from KunQuant.predefined.Alpha158 import AllData as AllData158
import sys
import os


def check_alpha158():
    builder = Builder()
    with builder:
        pack_158 = AllData158(low=Input("low"),high=Input("high"),close=Input("close"),open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        alpha158, names = pack_158.build({
            'kbar': {}, # whether to use some hard-code kbar features
            "price": {
                "windows": [0],
                "feature": [("OPEN",pack_158.open), ("HIGH",pack_158.high), ("LOW",pack_158.low), ("VWAP",pack_158.vwap)],
            },
            # 'volume': { # whether to use raw volume features
            #     'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            # },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                #if include is None we will use default operators
                # 'exclude': ['RANK'], # rolling operator not to use
            }
        })
        for v,k in zip(alpha158, names):
            Output(v, k)
    print("Total names: ", len(names))
    f = Function(builder.ops)
    simd_len = 8 if sys.argv[2] == "avx512" else 4
    src = compileit(f, "alpha158", dtype='double', blocking_len=simd_len, partition_factor=4, output_layout="TS", input_layout="TS", options={"opt_reduce": True, "fast_log": True})
    os.makedirs(sys.argv[1], exist_ok=True)
    with open(sys.argv[1]+"/Alpha158.cpp", 'w') as f:
        f.write(src)

check_alpha158()