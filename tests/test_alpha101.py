from KunTestUtil import ref_alpha101
import numpy as np
import pandas as pd
import sys
import time

sys.path.append("./build/")
import KunRunner as kr

def rand_float(stocks, low = 0.9, high = 1.11):
    return np.random.uniform(low, high, size= stocks)

def gen_stock_data2(low, high, stocks, num_time, stddev):
    xopen = np.random.uniform(low, high, size = (stocks, 1)).astype("float32")
    xvol = np.random.uniform(5, 10, size = (stocks, 1)).astype("float32")

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
    
    chopen = np.random.normal(1, stddev, size = (stocks, num_time)).astype("float32")
    chopen = np.cumprod(chopen, axis=1, dtype="float32")
    outvol = xvol * chopen

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

def test(executor, num_stock, num_time, ischeck):
    rtol=6e-5
    atol=1e-5
    lib = kr.Library.load("./build/libKunTest.so")
    print(lib)
    modu = lib.getModule("alpha_101")
    rng = np.random.get_state()
    start = time.time()
    dopen, dclose, dhigh, dlow, dvol, damount = gen_stock_data2(0.5, 100, num_stock, num_time, 0.01 if num_time > 1000 else 0.05)
    end = time.time()
    print(f"DataGen takes: {end-start:.6f} seconds")
    my_input = {"high": ST_ST8t(dhigh), "low": ST_ST8t(dlow), "close": ST_ST8t(dclose), "open": ST_ST8t(dopen), "volume": ST_ST8t(dvol), "amount": ST_ST8t(damount)}

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

    # prepare outputs
    outnames = modu.getOutputNames()
    layout = modu.output_layout
    outbuffers = dict()
    print(layout)
    if layout == "TS":
        # Factors, Time, Stock
        sharedbuf = np.empty((len(outnames), num_time, num_stock), dtype="float32")
        sharedbuf[:] = np.nan
        for idx, name in enumerate(outnames):
            outbuffers[name] = sharedbuf[idx]
    # print(ref.alpha001())
    # blocked = ST_ST8t(inp)
    
    start = time.time()
    out = kr.runGraph(executor, modu, my_input, 0, num_time, outbuffers)
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

    for k in outnames:
        print(k)
        cur_rtol = rtol
        cur_atol = atol
        if k in ["alpha013", "alpha016"]:
            # alpha013 has rank(cov(rank(X), rank(Y))). Output of cov seems to have very similar results
            # like 1e-6 and 0. Thus the rank result will be different
            cur_atol = 0.49
        elif k in ["alpha015"]:
            cur_atol = 2
        elif k in ["alpha005"]:
            cur_atol = 0.02
        elif k in ["alpha002"]:
            cur_atol = 0.02
        v = out[k]
        refv = ref[k].to_numpy().transpose()
        if k == "alpha101":
            print(df_dclose)
            print(v[9, 40:50])
            print(refv[9, 40:50])
        try:
            np.testing.assert_allclose(v, refv, rtol=cur_rtol, atol=cur_atol, equal_nan=True)
        except Exception as e:
            print(e)
            # print(rng)
            for i in range(num_stock):
                if not np.allclose(v[i], refv[i], rtol=cur_rtol, atol=cur_atol, equal_nan=True):
                    print("Bad stock", i)
                    print("Our output", v[i])
                    print("Ref", refv[i])
                    for j in range(num_time):
                        if not np.allclose(v[i,j], refv[i,j], rtol=cur_rtol, atol=cur_atol, equal_nan=True):
                            print("j",j,v[i,j], refv[i,j])
                    break


    # output = ST8t_ST(out["out"])
    # # print(expected[:,0])
    # # print(output[:,0])
    # np.testing.assert_allclose(output, expected, rtol=1e-6, equal_nan=True)

# executor = kr.createMultiThreadExecutor(16)
executor = kr.createSingleThreadExecutor()
test(executor, 64, 100, True)
executor = kr.createMultiThreadExecutor(16)
test(executor, 64, 100, True)