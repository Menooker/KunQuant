from KunTestUtil import ref_alpha101
import numpy as np
import pandas as pd
import sys
import time
import os

sys.path.append("./build/Release" if os.name == "nt" else "./build")
import KunRunner as kr

def count_unmatched_elements(arr1: np.ndarray, arr2: np.ndarray, atol=1e-8, rtol=1e-5, equal_nan=False):
    # Check if arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")

    # Mask for NaN equality
    nan_equal_mask = np.isnan(arr1) & np.isnan(arr2)

    # Check absolute and relative differences
    absolute_diff = np.abs(arr1 - arr2)
    tol = np.maximum(np.abs(arr1), np.abs(arr2))* rtol + atol

    # Mask for elements that meet the allclose criteria
    close_mask = (absolute_diff <= tol) | (nan_equal_mask if equal_nan else False)

    # Count unmatched elements
    unmatched_count = np.sum(~close_mask)

    return unmatched_count, close_mask

def rand_float(stocks, low = 0.9, high = 1.11):
    return np.random.uniform(low, high, size= stocks)

def gen_stock_data2(low, high, stocks, num_time, stddev):
    xopen = np.random.uniform(low, high, size = (stocks, 1)).astype("float32")
    # xvol = np.random.uniform(5, 5.2, size = (stocks, 1)).astype("float32")

    chopen = np.random.normal(1, stddev, size = (stocks, num_time)).astype("float32")
    chopen = np.cumprod(chopen, axis=1, dtype="float32")
    outopen = xopen * chopen

    chopen = np.random.uniform(0.95, 1.05, size = (stocks, num_time)).astype("float32")
    outclose = outopen * chopen

    chopen = np.random.uniform(0.995, 1.12, size = (stocks, num_time)).astype("float32")
    outhigh = outopen * chopen
    outhigh = np.maximum.reduce([outopen, outhigh, outclose])

    chopen = np.random.uniform(0.9, 1.005, size = (stocks, num_time)).astype("float32")
    outlow = outopen * chopen
    outlow = np.minimum.reduce([outopen, outhigh, outclose, outlow])
    
    # chopen = np.random.normal(1, stddev, size = (stocks, num_time)).astype("float32")
    # chopen = np.cumprod(chopen, axis=1, dtype="float32")
    outvol = np.random.uniform(5, 10, size = (stocks, num_time)).astype("float32")

    outamount = outvol * outopen * np.random.uniform(0.99, 1.01, size = (stocks, num_time)).astype("float32")
    return outopen, outclose, outhigh, outlow, outvol, outamount

def gen_stock_data(low, high, stocks, num_time, stddev):
    xopen = np.random.uniform(low, high, size = stocks).astype("float32")
    xvol = np.random.uniform(5, 10, size = stocks).astype("float32")
    outopen = np.empty((stocks, num_time), dtype="float32")
    outclose = np.empty((stocks, num_time), dtype="float32")
    outhigh = np.empty((stocks, num_time), dtype="float32")
    outlow = np.empty((stocks, num_time), dtype="float32")
    outvol = np.empty((stocks, num_time), dtype="float32")
    outamount = np.empty((stocks, num_time), dtype="float32")
    for i in range(num_time):
        xopen *= np.random.normal(1, stddev, size = (stocks)).astype("float32")
        xclose = xopen * rand_float(stocks, 0.99, 1.01)
        xhigh = np.maximum.reduce([xopen * rand_float(stocks, 0.99, 1.03), xopen, xclose])
        xlow = np.minimum.reduce([xopen * rand_float(stocks, 0.97, 1.01), xopen, xclose, xhigh])
        xvol *= np.random.normal(1, stddev, size = (stocks)).astype("float32")
        xamount = xvol * xopen * rand_float(stocks)
        outopen[:,i] = xopen
        outclose[:,i] = xclose
        outhigh[:,i] = xhigh
        outlow[:,i] = xlow
        outvol[:,i] = xvol
        outamount[:,i] = xamount
    return outopen, outclose, outhigh, outlow, outvol, outamount




def ST_ST8t(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.reshape((-1, 8, data.shape[1])).transpose((0, 2, 1)))


def ST8t_ST(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.transpose((0, 2, 1)).reshape((-1, data.shape[1])))

def TS_ST(data: np.ndarray) -> np.ndarray:
    return data.transpose()

def make_data_and_ref(num_stock, num_time, ischeck):
    rng = np.random.get_state()
    start = time.time()
    dopen, dclose, dhigh, dlow, dvol, damount = gen_stock_data2(0.5, 100, num_stock, num_time, 0.03 if num_time > 1000 else 0.05)
    end = time.time()
    print(f"DataGen takes: {end-start:.6f} seconds")
    my_input = {"high": ST_ST8t(dhigh), "low": ST_ST8t(dlow), "close": ST_ST8t(dclose), "open": ST_ST8t(dopen), "volume": ST_ST8t(dvol), "amount": ST_ST8t(damount)}
    ref = None
    if ischeck:
        df_dclose = pd.DataFrame(dclose.transpose())
        df_dopen = pd.DataFrame(dopen.transpose())
        df_vol = pd.DataFrame(dvol.transpose())
        df_low = pd.DataFrame(dlow.transpose())
        df_high = pd.DataFrame(dhigh.transpose())
        df_amount = pd.DataFrame(damount.transpose())
        start = time.time()
        ref = ref_alpha101.get_alpha({"S_DQ_HIGH": df_high, "S_DQ_LOW": df_low, "S_DQ_CLOSE": df_dclose, 'S_DQ_OPEN': df_dopen, "S_DQ_VOLUME": df_vol, "S_DQ_AMOUNT": df_amount})
        end = time.time()
        print(f"Ref takes: {end-start:.6f} seconds")
    return my_input, ref

tolerance = {
    "atol" : {
        # alpha013 has rank(cov(rank(X), rank(Y))). Output of cov seems to have very similar results
        # like 1e-6 and 0. Thus the rank result will be different
        "alpha003": 1e-4,
        "alpha013": 1e-4,
        "alpha013": 0.2,
        "alpha016": 0.2,
        "alpha015": 1,
        "alpha005": 0.2,
        "alpha002": 0.2,
        "alpha031": 0.1,
        "alpha034": 0.05,
        "alpha043": 1,
        "alpha044": 3e-5,
        # alpha045 has "correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)". It is 1/-1 in our synthetic data
        # because "close" is generated by normal distribution. We are using float32, so may +/-0.999X are created and ranks
        # are not stable 
        "alpha045": 0.3,
        "alpha050": 0.15,
        "alpha055": 0.1,
        "alpha071": 0.51,
        "alpha078": 0.1,
        "alpha085": 0.1,
        "alpha083": 0.05,
    },
    "rtol" : {
        "alpha013": 0.1,
        "alpha017": 0.1,
        "alpha018": 0.05,
        "alpha014": 1e-2,
        "alpha016": 0.1,
        "alpha027": 0.1,
        "alpha036": 2e-2,
        "alpha039": 0.1,
        "alpha043": 0.1,
        "alpha045": 0.1,
        "alpha050": 1e-1,
        "alpha072": 2e-1,
        "alpha077": 1e-1,
        "alpha078": 1e-1,
    },
    "bad_count": {
        "alpha008": 0.001,
        "alpha022": 0.001,
        "alpha027": 0.08,
        "alpha021": 0.001,
        "alpha045": 0.001,
        "alpha045": 0.07,
        "alpha050": 0.003,
        # hard selecting numbers >0
        "alpha053": 0.001,
        "alpha061": 0.001,
        "alpha065": 0.002,
        "alpha066": 0.002,
        # corr on rank, will produce NAN
        "alpha071": 0.04,
        "alpha072": 0.025,
        "alpha074": 0.04,
        "alpha075": 0.09,
        "alpha077": 0.001,
        "alpha078": 0.015,
        "alpha081": 0.27,
        "alpha085": 0.005,
    }
}

def test(modu, executor, start_window, num_stock, num_time, my_input, ref, ischeck, start_time):
    rtol=0.01
    atol=1e-5
    # prepare outputs
    outnames = modu.getOutputNames()
    layout = modu.output_layout
    outbuffers = dict()
    print(layout)
    if layout == "TS":
        # Factors, Time, Stock
        sharedbuf = np.empty((len(outnames), num_time-start_time, num_stock), dtype="float32")
        sharedbuf[:] = np.nan
        for idx, name in enumerate(outnames):
            outbuffers[name] = sharedbuf[idx]
    # print(ref.alpha001())
    # blocked = ST_ST8t(inp)
    
    start = time.time()
    out = kr.runGraph(executor, modu, my_input, start_time, num_time-start_time, outbuffers)
    end = time.time()
    print(f"Exec takes: {end-start:.6f} seconds")
    if not ischeck:
        return
    # print(out)
    for k in list(out.keys()):
        if layout == "TS":
            out[k] = TS_ST(out[k])
        else:
            out[k] = ST8t_ST(out[k])
    done = True
    for k in outnames:
        cur_rtol = tolerance["rtol"].get(k, rtol)
        cur_atol = tolerance["atol"].get(k, atol)
        check_start = 0
        if start_time:
            check_start = start_window[k] + start_time
        v = out[k][:,check_start-start_time:]
        refv = ref[k][check_start:].to_numpy().transpose()
        if k == "alpha031":
            pass
            # print(refv[0])
            # print(v[9, 40:50])
            # print(refv[9, 40:50])
        bad_count, result = count_unmatched_elements(v, refv, rtol=cur_rtol, atol=cur_atol, equal_nan=True)
        bad_rate = bad_count/ (result.size if result.size else 1)
        if bad_count:
            print(k)
            print(f"Unmatched bad_count = {bad_count}/{result.size} ({bad_rate*100:.4f}%) atol={cur_atol} rtol={cur_rtol}")
            if bad_rate < tolerance["bad_count"].get(k, 0.0001):
                print("bad count meets the tolerance, skipping")
                continue
            done = False
            # print(rng)
            for i in range(num_stock):
                if not np.allclose(v[i], refv[i], rtol=cur_rtol, atol=cur_atol, equal_nan=True):
                    print("Bad stock", i)
                    print("Our output", v[i])
                    print("Ref", refv[i])
                    for j in range(num_time-check_start):
                        if not np.allclose(v[i,j], refv[i,j], rtol=cur_rtol, atol=cur_atol, equal_nan=True):
                            print("j",j,v[i,j], refv[i,j])
                    break
    return done

def main():
    lib = kr.Library.load("./build/Release/projects/Alpha101.dll" if os.name == "nt" else "./build/projects/libAlpha101.so")
    print(lib)
    modu = lib.getModule("alpha_101")
    start_window = modu.getOutputUnreliableCount()
    print(start_window)
    num_stock = 64
    num_time = 260
    is_check = True
    my_input, pd_ref = make_data_and_ref(num_stock, num_time, is_check)
    executor = kr.createSingleThreadExecutor()
    done = True
    done = done & test(modu, executor, start_window, num_stock, num_time, my_input, pd_ref, is_check, 0)
    done = done & test(modu, executor, start_window, num_stock, num_time, my_input, pd_ref, is_check, 50)
    executor = kr.createMultiThreadExecutor(4)
    done = done & test(modu, executor, start_window, num_stock, num_time, my_input, pd_ref, is_check, 0)
    print("OK", done)

main()