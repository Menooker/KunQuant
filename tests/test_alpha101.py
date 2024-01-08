from KunTestUtil import ref_alpha101
import numpy as np
import pandas as pd
import sys

sys.path.append("./build/")
import KunRunner as kr

def ST_ST8t(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.reshape((-1, 8, data.shape[1])).transpose((0, 2, 1)))


def ST8t_ST(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.transpose((0, 2, 1)).reshape((-1, data.shape[1])))

def test(num_time):
    lib = kr.Library.load("./build/libKunTest.so")
    print(lib)
    modu = lib.getModule("alpha_101")
    dclose = np.random.rand(16, num_time).astype("float32")
    dopen = np.random.rand(16, num_time).astype("float32")
    dvol = np.random.rand(16, num_time).astype("float32")
    damount = np.random.rand(16, num_time).astype("float32")
    dlow = np.random.rand(16, num_time).astype("float32")
    my_input = {"low": ST_ST8t(dlow), "close": ST_ST8t(dclose), "open": ST_ST8t(dopen), "volume": ST_ST8t(dvol), "amount": ST_ST8t(damount)}

    df_dclose = pd.DataFrame(dclose.transpose())
    df_dopen = pd.DataFrame(dopen.transpose())
    df_vol = pd.DataFrame(dvol.transpose())
    df_low = pd.DataFrame(dlow.transpose())
    df_amount = pd.DataFrame(damount.transpose())
    ref = ref_alpha101.get_alpha({"S_DQ_LOW": df_low, "S_DQ_CLOSE": df_dclose, 'S_DQ_OPEN': df_dopen, "S_DQ_VOLUME": df_vol, "S_DQ_AMOUNT": df_amount})

    # print(ref.alpha001())
    # blocked = ST_ST8t(inp)
    executor = kr.createSingleThreadExecutor()
    out = kr.runGraph(executor, modu, my_input, 0, num_time)
    # print(out)
    for k in list(out.keys()):
        out[k] = ST8t_ST(out[k])
    
    for k, v in out.items():
        print(k)
        refv = ref[k].to_numpy().transpose()
        if k == "alpha107":
            print(dlow[0])
            print(v[0])
            print(refv[0])
        np.testing.assert_allclose(v, refv, rtol=6e-5, atol=1e-6, equal_nan=True)

    # output = ST8t_ST(out["out"])
    # # print(expected[:,0])
    # # print(output[:,0])
    # np.testing.assert_allclose(output, expected, rtol=1e-6, equal_nan=True)

test(100)