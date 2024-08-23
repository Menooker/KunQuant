import numpy as np

def aligned_copy(v: np.ndarray) -> np.ndarray:
    alloc_size = v.nbytes
    pointer = v.__array_interface__['data'][0]
    rem = pointer % 64
    if rem == 0:
        return v
    newbase = np.empty((alloc_size + 64, 0), dtype=np.uint8)
    pointer = newbase.__array_interface__['data'][0]
    rem = pointer % 64
    if rem == 0:
        offset = 0
    else:
        offset = 64  -  rem
    ret = newbase[offset:offset+alloc_size].view(v.dtype).reshape(v.shape)
    ret[:] = v
    return ret



def gen_stock_data2(low, high, stocks, num_time, stddev, dtype):
    xopen = np.random.uniform(low, high, size = (stocks, 1)).astype(dtype)
    # xvol = np.random.uniform(5, 5.2, size = (stocks, 1)).astype(dtype)

    chopen = np.random.normal(1, stddev, size = (stocks, num_time)).astype(dtype)
    chopen = np.cumprod(chopen, axis=1, dtype=dtype)
    outopen = xopen * chopen

    chopen = np.random.uniform(0.95, 1.05, size = (stocks, num_time)).astype(dtype)
    outclose = outopen * chopen

    chopen = np.random.uniform(0.995, 1.12, size = (stocks, num_time)).astype(dtype)
    outhigh = outopen * chopen
    outhigh = np.maximum.reduce([outopen, outhigh, outclose])

    chopen = np.random.uniform(0.9, 1.005, size = (stocks, num_time)).astype(dtype)
    outlow = outopen * chopen
    outlow = np.minimum.reduce([outopen, outhigh, outclose, outlow])
    
    # chopen = np.random.normal(1, stddev, size = (stocks, num_time)).astype(dtype)
    # chopen = np.cumprod(chopen, axis=1, dtype=dtype)
    outvol = np.random.uniform(5, 10, size = (stocks, num_time)).astype(dtype)

    outamount = outvol * outopen * np.random.uniform(0.99, 1.01, size = (stocks, num_time)).astype(dtype)
    return outopen, outclose, outhigh, outlow, outvol, outamount