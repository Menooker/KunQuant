from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
from KunQuant.passes import *
from KunQuant.Driver import *
from KunQuant.predefined.Alpha101 import *
import sys
import os

alphas = [alpha001, alpha002, alpha003, alpha004, alpha005, alpha006, alpha008, alpha009, alpha010, alpha011, alpha012,
          alpha013, alpha014, alpha015, alpha016, alpha017, alpha018, alpha020, alpha021, alpha022, alpha023, alpha025,
          alpha026, alpha027, alpha028, alpha029, alpha030, alpha031, alpha033, alpha034, alpha035, alpha038, alpha040,
          alpha041, alpha042, alpha043, alpha044, alpha045, alpha046, alpha047, alpha049, alpha050, alpha051, alpha053,
          alpha054, alpha055, alpha057, alpha060, alpha066, alpha068, alpha073, alpha077, alpha083, alpha084, alpha085,
          alpha101]
scale_alphas = {alpha009:1000,alpha010:1000,alpha012:1000,alpha023:1000,alpha035:10000,alpha041:1000,
                alpha042:10, alpha046:1000,alpha047:10, alpha043:1000,alpha049:1000,alpha051:1000,
                alpha053:1000, alpha084: 1000, alpha083:1000}
clip_alphas = {alpha053,alpha084, alpha083}

def check_alpha101():
    builder = Builder()
    with builder:
        all_data = AllData(low=Input("low"),high=Input("high"),close=Input("close"),open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        for f in alphas:
            out = f(all_data)
            if f in scale_alphas:
                out = out / scale_alphas[f]
            if f in clip_alphas:
                out = Clip(out, 10)
            Output(out, f.__name__)
        avg8 = WindowedAvg(all_data.close, 8)
        prev8 = BackRef(all_data.close, 8)
        Output(Select(prev8<avg8, ConstantOp(1), ConstantOp(0)), "cmpavg8")

    f = Function(builder.ops)
    src = compileit(f, "alpha_101_selected", output_layout="TS", options={"opt_reduce": True, "fast_log": True})
    os.makedirs(sys.argv[1], exist_ok=True)
    with open(sys.argv[1]+"/Alpha101Selected.cpp", 'w') as f:
        f.write(src)

check_alpha101()