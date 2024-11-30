from KunQuant.Driver import KunCompilerConfig
import numpy as np
import sys
import time
import os
import argparse
from typing import Dict
from KunQuant.jit import cfake
from KunQuant.runner import KunRunner as kr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.predefined.Alpha158 import AllData


def check_alpha158(avx512):
    builder = Builder()
    with builder:
        pack_158 = AllData(low=Input("low"), high=Input("high"), close=Input(
            "close"), open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        alpha158, names = pack_158.build({
            'kbar': {},  # whether to use some hard-code kbar features
            "price": {
                "windows": [0],
                "feature": [("OPEN", pack_158.open), ("HIGH", pack_158.high), ("LOW", pack_158.low), ("VWAP", pack_158.vwap)],
            },
            # 'volume': { # whether to use raw volume features
            #     'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            # },
            'rolling': {  # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60],  # rolling windows size
                # if include is None we will use default operators
                # 'exclude': ['RANK'], # rolling operator not to use
            }
        })
        for v, k in zip(alpha158, names):
            Output(v, k)
    print("Total names: ", len(names))
    f = Function(builder.ops)
    simd_len = 8 if avx512 else 4
    target = [("alpha158", f, KunCompilerConfig(dtype='double', blocking_len=simd_len, partition_factor=4,
               output_layout="TS", input_layout="TS", options={"opt_reduce": True, "fast_log": True}))]
    return cfake.compileit(target, "test", cfake.CppCompilerConfig())


num_stock = 8
num_time = 260


def load(inputs, ref):
    return dict(np.load(inputs)), dict(np.load(ref))


def ST_TS(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.transpose()).astype('float64')


def test(lib: kr.Library, inputs: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]):
    rtol = 1e-4
    atol = 1e-5
    modu = lib.getModule("alpha158")
    start_window = modu.getOutputUnreliableCount()
    num_stock = 8
    num_time = 260
    outnames = modu.getOutputNames()
    print("Total num alphas", len(outnames))
    executor = kr.createMultiThreadExecutor(8)
    my_input = {"high": ST_TS(inputs['dhigh']), "low": ST_TS(inputs['dlow']), "close": ST_TS(inputs['dclose']),
                "open": ST_TS(inputs['dopen']), "volume": ST_TS(inputs['dvol']), "amount": ST_TS(inputs['damount'])}
    outbuffers = dict()
    # Factors, Time, Stock
    sharedbuf = np.empty((len(outnames), num_time, num_stock), dtype="float64")
    sharedbuf[:] = np.nan
    for idx, name in enumerate(outnames):
        outbuffers[name] = sharedbuf[idx]
    start = time.time()
    out = kr.runGraph(executor, modu, my_input, 0, num_time, outbuffers)
    end = time.time()
    print(f"Exec takes: {end-start:.6f} seconds")
    for k, v in outbuffers.items():
        s = start_window[k]
        if not np.allclose(v[s:], ref[k][s:], rtol=rtol, atol=atol, equal_nan=True):
            print("Correctness check failed at " + k)
            for sid in range(num_stock):
                print("Check stock", sid)
                myout = v.transpose()[sid, s:]
                refv = ref[k].transpose()[sid, s:]
                if not np.allclose(myout, refv, rtol=rtol, atol=atol, equal_nan=True):
                    for j in range(num_time-s):
                        if not np.allclose(myout[j], refv[j], rtol=rtol, atol=atol, equal_nan=True):
                            print("j", j, myout[j], refv[j])
                    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run and check alpha158 again pre-computed result")
    parser.add_argument("--inputs", required=True, type=str,
                        help="The path to the input npz file")
    parser.add_argument("--ref", required=True, type=str,
                        help="The path to the reference output npz file")
    parser.add_argument("--avx512", action="store_true", default=False,
                        help="The path to the reference output npz file")
    args = parser.parse_args()
    lib = check_alpha158(args.avx512)
    inp, ref = load(args.inputs, args.ref)
    test(lib, inp, ref)
    print("done")
