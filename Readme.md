# KunQuant

![Kun](https://github.com/Menooker/KunQuant/assets/10137875/cb67b6fb-2bd3-41dd-921f-581c4c8d34d6)

KunQuant is a optimizer, code generator and executor for financial expressions and factors, e.g. `(close - open) /((high - low) + 0.001)`. The initial aim of it is to generate efficient implementation code for [Alpha101](https://arxiv.org/pdf/1601.00991) of WorldQuant and [Alpha158](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) of Qlib. Some existing implementations of Alpha101 is straightforward but too simple. Hence we are developing KunQuant to provide optimizated code on a batch of general customized factors.

This project has mainly two parts: `KunQuant` and `KunRunner`. KunQuant is an optimizer & code generator written in Python. It takes a batch of financial expressions as the input and it generates highly optimized C++ code for computing these expressions. KunRunner is a supporting runtime library and Python wrapper to load and run the generated C++ code from KunQuant.

A typical workload of designing and running financial factors with KunQuant will be

1. Write the factors with `KunQuant` Python library
2. Use `KunQuant` to optimize the factors and transform them into C++ source code
3. Use `cmake` to compile the generated code
4. Load the genereted binary via `KunRunner` in Python code

Experiments show that KunQuant-generated code can be more than 170x faster than naive implementation based on Pandas. We ran Alpha001~Alpha101 with [Pandas-based code](https://github.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py) and our optimized code. See results below:

| Pandas-based  |  KunQuant 1-thread  |  KunQuant  4-threads |
|---|---|---|
| 6.138s |  0.115s  |  0.035s  |

The data was collected on 4-core Intel i7-7700HQ CPU, running synthetic data of 64 stocks with 260 rows of data. Environment:

```
OS=Ubuntu 22.04.3 on WSL2 on Windows 10
python=3.10.2
pandas=2.1.4
numpy=1.26.3
g++=11.4.0
```

## Why KunQuant is fast

 * KunQuant parallelizes the computation for factors and uses SIMD (AVX2) to vectorize them.
 * Redundant computation among factors are eliminated: Think what we can do with `sum(x)`, `avg(x)`, `stddev(x)`? The result of `sum(x)` is needed by all these factors. KunQuant also automatically finds if a internal result of a factor is used by other factors and try to reuse the results.
 * Temp buffers are minimized by operator-fusion. For a factor like `(a+b)/2`, pandas and numpy will first compute the result of `(a+b)` and collect all the result in a buffer. Then, `/2` opeator is applied on each element of the temp buffer of `(a+b)`. This will result in large memory usage and bandwidth. KunQuant will generate C++ code to compute `(a[i]+b[i])/2` in the same loop, to avoid the need to access and allocate temp memory.

## Dependency

* pybind11 (automatically cloned via git as a submodule)
* Python (3.7+ with f-string and dataclass support)
* cmake
* A working C++ compiler with C++11 support (e.g. clang, g++, msvc)
* x86-64 CPU with at least AVX2-FMA instruction set

**Important node**: Currently KunQuant only supports a multiple of 8 as the number of stocks as inputs. That is, you can only input 8, 16, 24, ..., etc. stocks in a batch.

## Compiling and running Alpha101

This section serves as am example for compiling an existing factor library: Alpha101 and running it. Building and running your own factors will be similar. If you are only interested in how you can run Alpha101 factors, this section is all you need.
First, clone the KunQuant repo and make a new directory named `build`:

```shell
git clone https://github.com/Menooker/KunQuant --recursive
cd KunQuant
mkdir build
cd build
```

Then run cmake to configure the build:

```shell
cmake ..
```

If you want to use a non-default binary of Python executable, instead of the above command, run

```shell
cmake .. -DPYTHON_EXECUTABLE="PATH/TO/PYTHON/EXECUTABLE"
```

Build the code with cmake:

```shell
cmake --build . -- -j4
```


If the build is successful, you should be able to see in the terminal:

```
...
[100%] Built target Alpha101
```

You can find `KunRunner.cpython-??-{x86_64-linux-gnu.so, amd64.pyd, darwin.so}` and `projects/{libAlpha101.so, libAlpha101.dylib}` (on Linux/macOS) or `projects/Release/Alpha101.dll` (on Windows) in your build directory.

`libAlpha101.so`, `Alpha101.dll` or `libAlpha101.dylib` is the compiled code for Alpha101 factors on Linux, Windows or macOS. KunRunner is a Cpp extension for Python with helps to load the generated factor libraries. It also contains some supportive functions for the loaded libraries.

Before running Python, set the environment variable of `PYTHONPATH`:

On linux

```bash
export PYTHONPATH=$PYTHONPATH:/PATH/TO/KunQuant/build
```

On windows powershell

```powershell
$env:PYTHONPATH+=";x:\PATH\TO\KunQuant\build\Release"
```

Note that `/PATH/TO/KunQuant/build` or `x:\PATH\TO\KunQuant\build\Release` should be the directory containing `KunRunner.cpython-...{pyd,so}`

Then in Python, import KunRunner and load the Alpha101 library:

```python
import KunRunner as kr
lib = kr.Library.load("./projects/libAlpha101.so")
modu = lib.getModule("alpha_101")
```

Note that you need to give KunRunner a relative or absolute path of the factor library by replacing "./projects/libAlpha101.so" above.

Load your stock data. In this example, load from local pandas files. We assume the open, close, high, low, volumn and amount data for different stocks are stored in different files.

```python
import pandas as pd

# we need a multiple of 8 number of stocks
watch_list = ["000002", "000063", ...]
num_stocks = len(watch_list)
assert(num_stocks % 8 == 0)
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

Then an important step is to transpose the numpy array to shape `[features, stocks//8, time, 8]`. We split the axis of stocks into two axis `[stocks//8, 8]`. This step makes the memory layout of the numpy array match the SIMD length of AVX2, so that KunQuant can process the data in parallel in a single SIMD instruction.

```python
# [features, stocks, time] => [features, stocks//8, 8, time] => [features, stocks//8, time, 8]
transposed = collected.reshape((collected.shape[0], -1, 8, collected.shape[2])).transpose((0, 1, 3, 2))
transposed = np.ascontiguousarray(transposed)
```

Now fill the input data in a dict

```python
input_dict = dict()
for colname, colidx in col2idx.items():
    input_dict[colname] = transposed[colidx]
```

Create an executor and compute the factors!

```
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

KunQuant is a tool for general expressions. You can further read [Customize.md](./Customize.md) for how you can compile your own customized factors.

## Streaming mode

KunQuant can be configured to generate factor libraries for streaming, when the data arrive one at a time. See [Stream.md](./Stream.md)

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

```
python tests/test_alpha101.py
```

The input data are randomly genereted data and the results are checked against a modified (corrected) version of [Pandas-based code](https://github.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py). Note that some of the factors like `alpha013` are very sensitive to numerical changes in the intermeidate results, because `rank` operators are used. The result may be very different after `rank` even if the input is very close. Hence, the tolerance of these factors will be high to avoid false positives.

## Using C-style APIs

KunQuant provides C-style APIs to call the generated factor code in shared libraries. See [CAPI.md](./CAPI.md)

## Acknowledgement

The implementation and testing code for Alpha101 is based on https://github.com/yli188/WorldQuant_alpha101_code

The implementation code for Alpha158 is based on https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py. Licensed under the MIT License.

The AVX vector operators at `cpp/KunSIMD/cpu` was developed based on [x86simd](https://github.com/oneapi-src/oneDNN/tree/main/src/graph/backend/graph_compiler/core/src/runtime/kernel_include/x86simd) as a component of GraphCompiler, a backend of oneDNN Graph API. Licensed under the Apache License, Version 2.0 (the "License").
