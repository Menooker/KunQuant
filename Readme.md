# KunQuant

![Kun](https://github.com/Menooker/KunQuant/assets/10137875/cb67b6fb-2bd3-41dd-921f-581c4c8d34d6)

KunQuant is a optimizer, code generator and executor for financial expressions and factors, e.g. `(close - open) /((high - low) + 0.001)`. The initial aim of it is to generate efficient implementation code for [Alpha101](https://arxiv.org/pdf/1601.00991) of WorldQuant and [Alpha158](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) of Qlib. Some existing implementations of Alpha101 is straightforward but too simple. Hence we are developing KunQuant to provide optimizated code on a batch of general customized factors.

This project has mainly two parts: `KunQuant` and `KunRunner`. KunQuant is an optimizer & code generator written in Python. It takes a batch of financial expressions as the input and it generates highly optimized C++ code for computing these expressions. KunRunner is a supporting runtime library and Python wrapper to load and run the generated C++ code from KunQuant. Startring from version `0.1.0`, KunQuant no longer depends on `cmake` to run the generated factor code. Users can use pure Python interfaces to build and run factors.


Experiments show that KunQuant-generated code can be more than 170x faster than naive implementation based on Pandas. We ran Alpha001~Alpha101 with [Pandas-based code](https://github.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py) and our optimized code. See results below:

| Datatype | Pandas-based  |  KunQuant 1-thread  |  KunQuant  4-threads |
|---|---|---|---|
| Single precision (STs layout) | 6.138s |  0.083s  |  0.027s  |
| Double precision (TS layout) | 6.332s |  0.120s  |  0.031s  |

The data was collected on 4-core Intel i7-7700HQ CPU, running synthetic data of 64 stocks with 260 rows of data. Environment:

```
OS=Ubuntu 22.04.3 on WSL2 on Windows 10
python=3.10.2
pandas=2.1.4
numpy=1.26.3
g++=11.4.0
```

## Supported features of KunQuant

 * Batch mode and stream mode for the input
 * Double and single precision float point data type
 * TS or STs memory layout as input/output in batch mode
 * Python/C/C++ interfaces to call the factor computation functions
 * Only x86 CPU are supported

**Important node**: For better performance compared with Pandas, KunQuant suggests to use a multiple of `{blocking_len}` as the number of stocks in inputs. For single-precision float type and AVX2 instruction set, `blocking_len=8`. That is, you are suggested to input 8, 16, 24, ..., etc. stocks in a batch, if your code is compiled with AVX2 (without AVX512) and `float` datatype. Other numbers of stocks **are supported**, with lower execution performance.

## Why KunQuant is fast

 * KunQuant parallelizes the computation for factors and uses SIMD (AVX2) to vectorize them.
 * Redundant computation among factors are eliminated: Think what we can do with `sum(x)`, `avg(x)`, `stddev(x)`? The result of `sum(x)` is needed by all these factors. KunQuant also automatically finds if a internal result of a factor is used by other factors and try to reuse the results.
 * Temp buffers are minimized by operator-fusion. For a factor like `(a+b)/2`, pandas and numpy will first compute the result of `(a+b)` and collect all the result in a buffer. Then, `/2` opeator is applied on each element of the temp buffer of `(a+b)`. This will result in large memory usage and bandwidth. KunQuant will generate C++ code to compute `(a[i]+b[i])/2` in the same loop, to avoid the need to access and allocate temp memory.

## Sponsor this project!

Sponsor the author [@Menooker](https://github.com/sponsors/Menooker)

## Installing KunQuant

Install a released version:

`pip install KunQuant`

Or install the latest version on `main` branch

`pip install -i https://testpypi.python.org/pypi KunQuant`

KunQuant supports Windows (MSVC needs to be installed) and Linux (g++ or clang needs to be installed). Please make sure a working C++ compiler with C++11 support is properly installed and configured in your system

## Example: Build & Run Alpha101

This section serves as am example for compiling an existing factor library: Alpha101 and running it. Building and running your own factors will be similar. If you are only interested in how you can run Alpha101 factors, this section is all you need.

First, import KunQuant and necessary modules

```python
from KunQuant.jit import cfake
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.predefined import Alpha101
from KunQuant.runner import KunRunner as kr
```

Then build a `Function` object and generete predefined factor `alpha001` in Alpha101:

```python
builder = Builder()
with builder:
    vclose = Input("close")
    low = Input("low")
    high = Input("high")
    vopen = Input("open")
    amount = Input("amount")
    vol = Input("volume")
    all_data = Alpha101.AllData(low=low,high=high,close=vclose,open=vopen, amount=amount, volume=vol)
    Output(Alpha101.alpha001(all_data), "alpha001")
f = Function(builder.ops)
```

You can review the `alpha001` expression by `print(f)`. And you will get output

```
v0 = Input@{name:close}()
v2 = Div@(v0,v1)
v3 = SubConst@{value:1.0}(v2)
v4 = LessThanConst@{value:0.0}(v3)
v5 = WindowedStddev@{window:20}(v3)
v6 = Select@(v4,v5,v0)
v7 = Mul@(v6,v6)
v8 = TsArgMax@{window:5}(v7)
v9 = Rank@(v8)
v10 = Output@{name:alpha001}(v9)
```

Then compile it into an executable object (it may takes a few seconds to compile. If you encounter an subprocess error, please make sure MSVC or g++ is installed).

```python
lib = cfake.compileit([("alpha101", f, KunCompilerConfig(input_layout="TS", output_layout="TS"))], "out_first_lib", cfake.CppCompilerConfig())
modu = lib.getModule("alpha101")
```

We will explain the function `cfake.compileit` in [Customize.md](./Customize.md). Let's continue to see how to use the compiled `lib`.

Load your stock data. In this example, load from local pandas files. We assume the open, close, high, low, volumn and amount data for different stocks are stored in different files.

```python
import pandas as pd

# we need a multiple of 8 number of stocks
watch_list = ["000002", "000063", ...]
num_stocks = len(watch_list)
df = []

for stockid in watch_list:
    d = pd.read_hdf(f"{stockid}.hdf5")
    df.append(d)

print(df[0])

cols = df[0].columns.values
col2idx = dict(zip(cols, range(len(cols))))
print("columns to index", col2idx)
num_time = len(df[0])
print("dimension in time", num_time)
```

Here we printed the data frame of the first stock and the column-index mapping, it should look like:

```
                 open       high        low      close       volume        amount
date                                                                             
2020-01-02  32.799999  33.599998  32.509998  32.560001  101213040.0  3.342374e+09
2020-01-03  32.709999  32.810001  31.780001  32.049999   80553632.0  2.584310e+09
2020-01-06  31.750000  31.760000  31.250000  31.510000   87684056.0  2.761449e+09
...               ...        ...        ...        ...          ...           ...
2024-01-30  10.000000  10.050000   9.790000   9.790000   79792704.0  7.903654e+08
2024-01-31   9.770000   9.850000   9.560000   9.600000   67478864.0  6.527274e+08
2024-02-01   9.530000   9.660000   9.420000   9.440000   62786032.0  5.980486e+08

[993 rows x 6 columns]
columns to index {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4, 'amount': 5}
dimension in time 993
```


Transform your pandas data to numpy array of shape `[features, stocks, time]`. Feature here means the columns for open, close, high, low, volumn and amount.

```python
import numpy as np

# [features, stocks, time]
collected = np.empty((len(col2idx), num_stocks, len(df[0])), dtype="float32")
for stockidx, data in enumerate(df):
    for colname, colidx in col2idx.items():
        mat = data[colname].to_numpy()
        collected[colidx, stockidx, :] = mat
```

Transpose the matrix to `[features, time, stocks]`
```python
# [features, stocks, time] => [features, time, stocks]
transposed = collected.transpose((0, 2, 1))
transposed = np.ascontiguousarray(transposed)
```

Now fill the input data in a dict of `{"open": matrix_open, "close": ...}`

```python
input_dict = dict()
for colname, colidx in col2idx.items():
    input_dict[colname] = transposed[colidx]
```

Create an executor and compute the factors!

```python
# using 4 threads
executor = kr.createMultiThreadExecutor(4)
out = kr.runGraph(executor, modu, input_dict, 0, num_time)
print("Result of alpha101", out["alpha001"])
print("Shape of alpha101", out["alpha001"].shape)
```

Each output factors are computed in an array of shape `[time, stocks]`. The output of above code can be:

```
Result of alpha001 [[   nan    nan    nan ...    nan    nan    nan]
 [   nan    nan    nan ...    nan    nan    nan]
 [   nan    nan    nan ...    nan    nan    nan]
 ...
 [0.6875 0.1875 0.1875 ... 0.6875 0.6875 0.6875]
 [0.6875 0.1875 0.1875 ... 0.6875 0.6875 0.6875]
 [0.4375 1.     0.875  ... 0.4375 0.4375 0.4375]]
Shape of alpha001 (993, 8)
```

By default, runGraph will allocate an numpy array for each of the output factor. However, you can preallocate a numpy array and tell KunRunner to fill in this array instead of creating new ones.

```python
outnames = modu.getOutputNames()
out_dict = dict()
# [Factors, Time, Stock]
sharedbuf = np.empty((len(outnames), num_time, num_stocks), dtype="float32")
for idx, name in enumerate(outnames):
    out_dict[name] = sharedbuf[idx]
out = kr.runGraph(executor, modu, input_dict, 0, num_time, out_dict)
# results are in "out" and "sharedbuf"
```

Note that the executors are reusable. A multithread executor is actually a thread pool inside. If you want to run on multiple batches of data, you don’t need to create new executors for each batch.


## Customized factors

KunQuant is a tool for general expressions. You can further read [Customize.md](./Customize.md) for how you can compile your own customized factors. This document also provides infomation on
 * building and keeping the compilation result for later use
 * Loading existing compiled factor library
 * enabling AVX512
 * select data types (float/double)
 * Memory layout


## Build from source and developing tips

This section is for developer who would like to build KunQuant from source, instead of installing via pip.

### Dependency

* pybind11 (automatically cloned via git as a submodule)
* Python (3.7+ with f-string and dataclass support)
* cmake
* A working C++ compiler with C++11 support (e.g. clang, g++, msvc)
* x86-64 CPU with at least AVX2-FMA instruction set
* Optionally requires AVX512 on CPU for better performance

### Build and install

```shell
git clone https://github.com/Menooker/KunQuant --recursive
cd KunQuant
pip install .
```
### Build in develop mode

If you would like to install KunQuant and edit it. You can use `editable` mode of python library.

Linux:

```shell
# in the root directory of KunQuant
KUN_BUILD_TESTS=1 pip install -e . 
```

Windows powershell:

```shell
# in the root directory of KunQuant
$env:KUN_BUILD_TESTS=1
pip install -e . 
```

You can also set environment variable `KUN_BUILD_TYPE=Debug` before `pip install -e .` to enable debug build of KunQuant. It will provide debug info of the KunQuant runtime but also slow down the execution.

### Useful environment variables

 * `KUN_DEBUG=1` Print the internal results of each compiler pass
 * `KUN_DEBUG_JIT=1` Print the C++ compilation internals, including command lines, temp results and etc. 

## Streaming mode

KunQuant can be configured to generate factor libraries for streaming, when the data arrive one at a time. See [Stream.md](./Stream.md)

## Row-to-row correlation (for IC/IR calculation)

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

## Using C-style APIs

KunQuant provides C-style APIs to call the generated factor code in shared libraries. See [CAPI.md](./CAPI.md)


## Operator definitions

See [Operators.md](./Operators.md)

To add new operators, see [NewOperators.md](./NewOperators.md)

## Testing and validation

Unit tests for some of the internal IR transformations:

```
python tests/test.py
python tests/test2.py
```

Unit tests for C++ runtime:

```
python tests/test_runtime.py
```

To run the runtime UTs, you need to make sure you have built the cmake target `KunTest` by

```bash
cmake --build . --target KunTest
```

Correctness test of Alpha101

```bash
# current dir should be at the base directory of KunQuant
python tests/test_alpha101.py
```

The input data are randomly genereted data and the results are checked against a modified (corrected) version of [Pandas-based code](https://github.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py). Note that some of the factors like `alpha013` are very sensitive to numerical changes in the intermeidate results, because `rank` operators are used. The result may be very different after `rank` even if the input is very close. Hence, the tolerance of these factors will be high to avoid false positives.

To test Alpha158, you need first download the input data and reference result files: [alpha158.npz](https://github.com/Menooker/KunQuant/releases/download/alpha158/alpha158.npz) and [input.npz](https://github.com/Menooker/KunQuant/releases/download/alpha158/input.npz).

Then run

```bash
# current dir should be at the base directory of KunQuant
python tests/test_alpha158.py --inputs /PATH/TO/input.npz --ref /PATH/TO/alpha158.npz 
```

This script runs alpha158 with double precision mode in KunQuant. It feeds the library with predefined values from `input.npz` and check against the result with `alpha158.npz`, which is computed by `qlib`.

To generate another Alpha158 result with another randomly generated input, you can run

```bash
# current dir should be at the base directory of KunQuant
python ./tests/gen_alpha158.py --tmp /tmp/a158 --qlib /path/to/source/of/qlib --out /tmp
```

It will create the random input at `/tmp/input.npz` and result at `/tmp/alpha158.npz`


## Acknowledgement

The implementation and testing code for Alpha101 is based on https://github.com/yli188/WorldQuant_alpha101_code

The implementation code for Alpha158 is based on https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py. Licensed under the MIT License.

The AVX vector operators at `cpp/KunSIMD/cpu` was developed based on [x86simd](https://github.com/uxlfoundation/oneDNN/tree/2eb3dd1082db767fab171e934c551c609008289a/src/graph/backend/graph_compiler/core/src/runtime/kernel_include/x86simd) as a component of GraphCompiler, a backend of oneDNN Graph API. Licensed under the Apache License, Version 2.0 (the "License").

The MSVC environment configuration was originated from cupy, Licensed under the MIT License: https://github.com/cupy/cupy/blob/main/cupy/cuda/compiler.py