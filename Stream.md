# Streaming mode

If you use KunQuant in online services, when the data for each tick are received one by one, you may need the "streaming" mode.

## Building Streaming mode Factor libraries

It is almost the same as the steps in [Customize.md](./Customize.md) and [Readme.md](./Readme.md). The main difference is that you need to specify `output_layout="STREAM"` in `generate.py` of your Factor library generator. `project/Alpha101Stream` is an example of Alpha101 in streaming mode. You can check the difference of `projects/Alpha101/generate.py` and `project/Alpha101Stream/generate.py`. Except the difference in the names, the only difference is at the line

```python
src = compileit(f, "alpha_101_stream", partition_factor=8, output_layout="STREAM", options={"opt_reduce": False, "fast_log": True})
```

We specified a different `partition_factor` for performance and we turn off the `opt_reduce` optimization. We also tell KunQuant that it should compile in streaming mode by `output_layout="STREAM"`.

We can build it via the commands

```shell
cmake --build . --target Alpha101Stream
```

For more info of building customized factors, see [Customize.md](./Customize.md)

## Running streaming mode factors in Python

Load the compiled factor library as usual:

```python
import KunRunner as kr
lib = kr.Library.load("./projects/libAlpha101Stream.so")
modu = lib.getModule("alpha_101_stream")
```

Create the executor (mult-thread executor is also supported). Assume we have 16 stocks (the number of stocks must be a multiple of 8). And we create a streaming context:

```python
num_stock = 16
executor = kr.createSingleThreadExecutor()
stream = kr.StreamContext(executor, modu, num_stock)
```

Query the buffer handles. You need to cache the handles

```python
buffer_name_to_id = dict()
for name in ["high","low","close","open","volume","amount"]:
    buffer_name_to_id[name] = stream.queryBufferHandle(name)
for name in ["alpha001", "alpha002"]: #and other factors
    buffer_name_to_id[name] = stream.queryBufferHandle(name)
```

We define a function to process each tick of new data. The data should contain "high","low","close","open","volume","amount" for each of the stocks. Each parameter should be a numpy array of length `num_stock` for the data of each stocks.

```python
def on_tick(high, low, close, open, volume, amount):
    # high should be an ndarray of shape (16,)
    stream.pushData(buffer_name_to_id["high"], high)
    stream.pushData(buffer_name_to_id["low"], low)
    stream.pushData(buffer_name_to_id["close"], close)
    stream.pushData(buffer_name_to_id["open"], open)
    stream.pushData(buffer_name_to_id["volume"], volume)
    stream.pushData(buffer_name_to_id["amount"], amount)

    stream.run()

    alpha001: np.ndarray = stream.getCurrentBuffer(buffer_name_to_id["alpha001"])[:]
    alpha002: np.ndarray = stream.getCurrentBuffer(buffer_name_to_id["alpha002"])[:]

    # do with alpha001 and alpha002
```

Basically, you need to call `pushData` for each input. Then call `run()` to let the stream step forward. Finally, use `getCurrentBuffer` to get the result. A very important note is that the `ndarray` returned by `getCurrentBuffer` is valid only 
 * before next call of `pushData` or `run` on the same stream
 * before the stream context object is deleted

That's why in the above code, we immediately copy the `ndarray` returned by `getCurrentBuffer` with `[:]`.

## C-API for Streaming mode

The logic is similar to the Python API above. For details, see `tests/capi/test_c.cpp` and `cpp/Kun/CApi.h`.