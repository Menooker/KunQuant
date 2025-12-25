# Utility functions

KunQuant provides C++ implemented utility functions to compute Row-to-row correlation and Aggregation. They are accelerated by SIMD and can be parallelized by multi-threading. 

## Row-to-row correlation (for IC/IR calculation)

```python
KunQuant.runner.KunRunner.corrWith(executor, inputs, corr_with, outputs, layout = layout, rank_inputs = rank_inputs)
```

Compute row-to-row corr values of a list of matrices `inputs` with a single matrix `corr_with`. And write the results to the pre-allocated matrices in `outputs`. That is `outputs[i] = corr(inputs[i], corr_with)`. Each element of list `inputs` should be a matrix of `TS` or `STs` layout. Let the time-dimension of `inputs[i]` be `T`. `outputs[i]` should be a buffer (e.g. allocated by `np.empty(...)`) of 1D shape of `[T]`. `corr_with` should have the same time-dimension as `T`. Each element in `outputs[i]` should be the correlation of a row in `inputs[i]` and `corr_with`.

Example:

```python
from KunQuant.runner import KunRunner as kr
data1 = ... # np.ndarray of shape [time*stocks]. For example, a factor's results
data2 = ... # np.ndarray of shape [time*stocks]. For example, a factor's results
valid_in = {"alpha1": data1, "alpha2": data2}
returns = ... # np.ndarray of shape [time*stocks]. For example, the rank of returns
valid_corr = {"alpha1": np.empty((time,), dtype="float32"), "alpha2": np.empty((time,), dtype="float32")}
kr.corrWith(executor, valid_in, returns, valid_corr, layout = "TS", rank_inputs = True)
# outputs in valid_corr
alpha1_ic = valid_corr["alpha1"].mean()
```

The parameter `rank_inputs=True` will first compute rank in the first input array (e.g. `valid_in` above) and compute the correlation with the second input (e.g. `returns` above). It will not compute the rank of the second input.

## Aggregation Functions

KunQuant provides utility functions for aggregration, including min, max, first, last, count, sum, mean. Users can specify an matrix to aggregate (of shape `[num_time x num_stocks]`) and a 1D vector as the label (of shape `[num_time]`, with same `num_time` of the matrix). The aggregration is performed in the `time` dimension. The labels should be in the same datatype of the matrix and should be monotinically incresing. The rows of the matrix with the same label (indexed by the row-id) will be aggregrated together.

For example, aggregating by the day, like pandas `a_df.groupby(a_df.index.date).sum()`:

```python
from KunQuant.runner import KunRunner as kr
# randomly generate time stamps between 2026-01-01 to 2026-01-07
dates = pd.date_range(start="2026-01-01", end="2026-01-07", freq="min")
# select 240 time stamps
labels = pd.DataFrame(sorted(np.random.choice(dates, size=240)))
# time=240 stocks=16
a = np.random.rand(240, 16).astype(dtype)
a_df = pd.DataFrame(a, index=labels[0])
# labels are the day-in-month, 1 to 6. len(labels) == 240
labels = a_df.index.day.to_numpy().astype(dtype)
# 6 days in total 
out_length = 6
out_a = {"sum": np.empty((length, 16), dtype=dtype), "min": np.empty((length, 16), dtype=dtype),
            "max": np.empty((length, 16), dtype=dtype)}
executor = kr.createMultiThreadExecutor(3)
kr.aggregrate(executor, [a], [labels], [out_a])
# reference result in pandas
np.testing.assert_allclose(out_a["sum"], a_df.groupby(a_df.index.date).sum())
np.testing.assert_allclose(out_a["min"], a_df.groupby(a_df.index.date).min())
np.testing.assert_allclose(out_a["max"], a_df.groupby(a_df.index.date).max())
```

You can also pass a list of matrices, labels and outputs to utilize multithreading in `kr.aggregrate`:

```python
a = np.random.rand(...).astype(dtype)
b = np.random.rand(...).astype(dtype)
c = np.random.rand(...).astype(dtype)
labels_a = ...
labels_b = ...
labels_c = ...
out_a = {"sum": ..., "min": ...}
out_b = {"min": ..., "min": ...}
out_c = {"mean": ...}
executor = kr.createMultiThreadExecutor(3)
kr.aggregrate(executor, [a,b,c], [labels_a, labels_b, labels_c], [out_a, out_b, out_c])
```

⚠️⚠️⚠️ **Note** that the output buffers should be large enough in `time` dimension. KunQuant will not check if it is large enough to hold the result. If it is not large enough, undefined behavior will occur.