import numpy as np
import sys
import time
import os
import argparse
from typing import Dict

sys.path.append("./build/Release" if os.name == "nt" else "./build")
import KunRunner as kr

num_stock = 8
num_time = 260


def load(inputs, ref):
    return dict(np.load(inputs)), dict(np.load(ref))


def ST_TS(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.transpose()).astype('float64')


def test(inputs: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]):
    rtol = 1e-4
    atol = 1e-5
    lib = kr.Library.load("./build/projects/Release/Alpha158.dll" if os.name ==
                          "nt" else "./build/projects/libAlpha158.so")
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
                            print("j",j,myout[j], refv[j])
                    exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run and check alpha158 again pre-computed result")
    parser.add_argument("--inputs", required=True, type=str,
                        help="The path to the input npz file")
    parser.add_argument("--ref", required=True, type=str,
                        help="The path to the reference output npz file")
    args = parser.parse_args()
    inp, ref = load(args.inputs, args.ref)
    test(inp, ref)
    print("done")
