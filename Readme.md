# KunQuant

KunQuant is a optimizer, code generator and executor for financial expressions and factors. The initial aim of it is to generate efficient implementation code for [Alpha101](https://arxiv.org/pdf/1601.00991) of WorldQuant. Some existing implementations of Alpha101 is straightforward but too simple. Hence we are developing KunQuant to provide optimizated code on a batch of general factors.

This project has mainly two parts: `KunQuant` and `KunRunner`. KunQuant is an optimizer & code generator written in Python. It takes a batch of financial expressions as the input and it generates highly optimized C++ code for computing these expressions. KunRunner is a supporting runtime library and Python wrapper to load and run the generated C++ code from KunQuant.

Experiments show that KunQuant-generated code can be more than 100x faster than naive implementation based on Pandas. We ran Alpha001~Alpha020 with [Pandas-based code](https://github.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py) and our optimized code. See results below:

| Pandas-based  |  KunQuant 1-core  |  KunQuant  4-cores |
|---|---|---|
| 3.26s |  0.10s  |  0.029s  |

The data was collected on 4-core i7-7700HQ, running synthetic data of 64 stocks with 1000 days.

## Why KunQuant is fast

 * KunQuant parallelized the computation factors and we use SIMD (AVX2) to vectorize them.
 * Redundant computation among factors are eliminated: Think what we can do with `sum(x)`, `avg(x)`, `stddev(x)`? The result of `sum(x)` is needed by all these factors. KunQuant also automatically finds if a internal result of a factor is used by other factors.
 * Temp buffers are minimized by operator-fusion. For a factor like `(a+b)/2`, pandas and numpy will first compute the result of `(a+b)` and collect all the result in a buffer. Then, `/2` opeator is applied on each element of the temp buffer of `(a+b)`. This will result in large memory usage and bandwidth. KunQuant will generate C++ code to compute `(a[i]+b[i])/2` in the same loop, to avoid the need to access and allocate temp memory.

