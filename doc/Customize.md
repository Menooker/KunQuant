# Customizing your factors

This document describes how you can build your own factors.

## Generating C++ source code and shared library for financial expressions

You can invoke KunQuant as a Python library to generate high performance C++ source code for your own factors. KunQuant also provides predefined factors of Alpha101, at the Python module KunQuant.predefined.Alpha101.

First, you need to install KunQuant. See [Readme.md](../Readme.md).

Then in Python code, import the needed classes and functions.

```python
from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
```

An expression in KunQuant is composed of operators `ops`. An Op means an operation on the data, or a source of the data. Ops can fall into some typical categories, like
* elementwise (like add, sub, sqrt), where the output of the operation depends only on the newest input
* windowed (like sum, stddev), where the output of the operation depends on several history values near the current input. These Ops correspond to the operations on `rolling()` in pandas.
* cross sectional operator (like rank and scale), whose output is the computed for the current stock in all stocks at the same time
* inputs and ouputs: these Ops reads or writes the user input/output buffers

you need to first make an instance of KunQuant.Ops.Builder. It will automatically record the expressions you made within a “with” block. A program to build simple expressions to compute the mean and average of `close` stock data can be:

```python
builder = Builder()
with builder:
    inp1 = Input("a")
    v1 = WindowedAvg(inp1, 10)
    v2 = WindowedStddev(inp1, 10)
    out1 = Output(v1, "ou1")
    out2 = Output(v2, "ou2")
```


If you have several different factors, remember to write them all in the same `with` block of the builder to build them in the same function. This can let different expressions potentially share the intermediate results, if possible. You can also call the predefined factors of Alpha101 in the builder block:

```python
from KunQuant.predefined.Alpha101 import alpha001, Alldata
builder = Builder()
with builder:
    inp1 = Input("close")
    v1 = WindowedAvg(inp1, 10)
    v2 = WindowedStddev(inp1, 10)
    out1 = Output(v1, "avg_close")
    out2 = Output(v2, "std_close")
    all_data = AllData(low=Input("low"),high=Input("high"),close=inp1,open=Input("open"), amount=Input("amount"), volume=Input("volume"))
    Output(alpha001(all_data), "alpha001")
```

Next step, create a `Function` to hold the expressions:

```python
builder = Builder()
with builder:
    # code omitted
    ...
f = Function(builder.ops)
```

A function can be viewed as a collection of Ops. A single function may contain several factors.

Then generate the C++ source and build the library with “compileit” function!

```python
from KunQuant.jit import cfake
from KunQuant.Driver import KunCompilerConfig
lib = cfake.compileit([("my_library_name", f, KunCompilerConfig(input_layout="TS", output_layout="TS"))], "my_library_name", cfake.CppCompilerConfig())
modu = lib.getModule("my_library_name")
```

The `lib` variable has type `KunQuant.runner.KunRunner.Library`. It is a container of multiple `modules` (in the above example, only one module is in the library). The variable `modu` has type `KunQuant.runner.KunRunner.Module`. It is the entry-point of a factor library.

Note that `"my_library_name"` corresponds to `my_library_name` in the line `cfake.compileit(...)` in our Python script.

More reading on operators provided by KunQuant: See [Operators.md](./Operators.md)

## Save the compilation result as a shared library

Like the example above, and by default, the compiled factor library is stored in a temp dir and will be automatically cleaned up. You can choose to keep the compilation result files (C++ source code, object files and the shared library), if
 * your factors does not change and you want to save the compilation time by caching the factor library
 * or, you want to use the compilation result in another machine/ programming language (like C/Go/Rust)

In the above alpha101 example, you can run 

```python
cfake.compileit([("my_library_name", f, KunCompilerConfig(input_layout="TS", output_layout="TS"))], "your_lib_name", cfake.CppCompilerConfig(), tempdir="/path/to/a/dir", keep_files=True, load=False)
```

This will create a directory `/path/to/a/dir/your_lib_name`, and the generated C++ file will be at `your_lib_name.cpp` and the shared library file will be at `your_lib_name.{so,dll}` in the directory.

In another process, you can load the library and get the module via

```python
from KunQuant.runner import KunRunner as kr
lib = kr.Library.load("/path/to/a/dir/your_lib_name/your_lib_name.so")
modu = lib.getModule("my_library_name")
```

And use the `modu` object just like in the example in [Readme](../Readme.md).

## Compiler options

The key function of KunQuant is `cfake.compileit`. Its signature is

```python
def compileit(func: List[Tuple[str, Function, KunCompilerConfig]], libname: str, compiler_config: CppCompilerConfig, tempdir: str | None = None, keep_files: bool = False, load: bool = True) -> KunQuant.runner.KunRunner.Library | str
```

This function compiles a list of tuples `(module_name, function, config)`. By default, KunQuant will use multi-threading to compile this list of modules in parallel. The compiled modules (in C++ object files) will be linked into a shared library named by `libname`. If parameter `load` is true, the function returns the loaded library of the compilation result. Otherwise, it returns the path of the library.

Each module has a `KunCompilerConfig` of configurations like `layout`, `datatype`, SIMD length (will discuss below):

```python
@dataclass
class KunCompilerConfig:
    partition_factor : int = 3
    dtype:str = "float"
    blocking_len: int = None
    input_layout:str = "STs"
    output_layout:str = "STs"
    allow_unaligned: Union[bool, None] = None
    split_source: int = 0
    options: dict = field(default_factory=dict)
```

If `split_source` is not 0, KunQuant will split the generated C++ source code into several files, each with at most `split_source` functions. This is useful to parallelize the compilation.

The `CppCompilerConfig` controls how KunQuant calls the C++ compiler. To choose the non-default compiler, you can pass `CppCompilerConfig(compiler="/path/to/your/C++/compiler")` to `cfake.compileit`. You can also enable/disable AVX512 by this config class.

`options` is a dict to specify the flags and configurations of KunQuant internal IR transforms and passes. Supported keys/values of `options`

| Key | Comments  |  Value type  |  Default value |
|---|---|---|---|
| opt_reduce | optimize WindowedSum by rolling sum algorithm |  bool  |  If in stream mode, False. Otherwise, True  |
| fast_log | Use KunQuant's implementation of math log function instead of `std::log` |  bool  |  True  |
| no_fast_stat | Disable fast rolling algorithm for statistics functions like stddev/corr/etc. Setting this flag to True may help to get better precision with the cost of performance. KunQuant will warn the precision issue if `options['no_fast_stat']==False`. To disable the warning and set no_fast_stat to False, set `options['no_fast_stat']=='no_warn'` |  bool or Literal\["no_warn"\]  |  If dtype is float or in stream mode, True. Otherwise, False |

## Specifing Memory layouts and data types and enabling AVX512

### Enabling AVX512 and choosing blocking_len

This project by default turns off AVX512, since this intruction set is not yet well adopted. If you are sure your CPU has AVX512, you can turn it on by passing `machine = cfake.X64CPUFlags(avx512=True)` when creating `cfake.CppCompilerConfig(machine=...)`. This will enable AVX512 features when compiling the KunQuant generated code. Some speed-up over `AVX2` mode are expected.

In your customized project, you need to specify `blocking_len` parameter of in `KunCompilerConfig` to enable AVX512. Please note that `blocking_len` will affect the `STs` format (see below section). For example, if your datatype is `float`, the `blocking_len` should be 16 to enable AVX512.

There are some other CPU instruction sets that is optional for KunQuant. You can turn on `AVX512DQ` and `AVX512VL` to accelerate some parts of KunQuant-generated code. To enable them, add `avx512dq=True`, `avx512vl=True` in `cfake.X64CPUFlags(...)` respectively.

To see if your CPU supports AVX512 (and `AVX512DQ` and `AVX512VL`), you can run command `lscpu` in Linux and check the outputs.

Enabling AVX512 will slightly improve the performance, if it is supported by the CPU. Experiments only shows ~1% performance gain for 16-threads of AVX512 on Icelake, testing on double-precision Alpha101, with 128 stocks and time length of 12000. A single thread running the same task shows 5% performance gain on AVX512.

### Memory layouts

The developers can choose the memory layout when compiling KunQuant factor libraries. The memory layout decribes how the input/output matrix is organized. Currently, KunQuant supports `TS`, `STs` and `STREAM` as the memory layout. In `TS` layout, the input and output data is in plain `[num_time, num_stocks]` 2D matrix. In `STs` with `blocking_len = 8`, the data should be transformed to `[num_stocks//8, num_time, 8]` for better performance. The `STREAM` layout is for the streaming mode. You can choose the input/output layout independently in `KunCompilerConfig`, by the parameters `KunCompilerConfig(..., input_layout="TS", output_layout="STs")` for example. By default, the input layout is `STs` and the output layout is `TS`.

For the alpha101 example above, to use `STs` for input, replace the compilation code with

```python
lib = cfake.compileit([("alpha101", f, KunCompilerConfig(input_layout="STs", output_layout="TS"))], "out_first_lib", cfake.CppCompilerConfig())
```

And you need to transpose the numpy array to shape `[features, stocks//8, time, 8]`, we split the axis of stocks into two axis `[stocks//8, 8]`. This step makes the memory layout of the numpy array match the SIMD length of AVX2, so that KunQuant can process the data in parallel in a single SIMD instruction. Notes:
 * the number `8` here is the `blocking_num` of the compiled code. It is decided by the SIMD lanes of the data type and the instruction set (AVX2 or AVX512). By default, the example code of `Alpha101` generates `float` dtype with AVX2. The register size of AVX2 is 256 bits, so the SIMD lanes of `float` should be 8.

```python
# [features, stocks, time] => [features, stocks//8, 8, time] => [features, stocks//8, time, 8]
transposed = collected.reshape((collected.shape[0], -1, 8, collected.shape[2])).transpose((0, 1, 3, 2))
transposed = np.ascontiguousarray(transposed)
```

### Specifing data types

KunQuant supports `float` and `double` data types. It can be selected by the `dtype` parameter of `KunCompilerConfig(...)`.

If AVX512 `ON` (by default is `OFF`), the `blocking_len` for `dtype='float'` can be 8 or 16, and for `dtype='double'` can be 4 or 8. If `AVX512` is `OFF`, the `blocking_len` for `dtype='float'` should only be 8, and for `dtype='double'` should be 4.


## Performance tuning

There are some configurable options of function `compileit(...)` above that may improve the performance (and maybe at the cost of accuracy).

 * Input and output memory layout: `compileit(input_layout=?, output_layout=?)`. This affects how data are arranged in memory. Usually `STs` layout is faster than `TS` but may require some additional memory movement when you call the factor library.
 * Partition factor: `compileit(partition_factor=some_int)`. A larger Partition factor will put more computations in a single generated function in C++. Enlarging Partition factor may reduce the overhead of thread-scheduling and eliminate some of the temp buffers. However, if the factor is too high, the generated C++ code will suffer from register-spilling.
 * Blocking len: `compileit(blocking_len=some_int)`. It selects AVX2 or AVX512 instruction sets. Using AVX512 might have some slight performance gain over AVX2.
 * Unaligned stock number: `compileit(allow_unaligned=some_bool)`. By default `True`. When `allow_unaligned` is set to false, the generated C++ code will assume the number of stocks to be aligned with the SIMD length (e.g., 8 float32 on AVX2). This will slightly improve the performance.