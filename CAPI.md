# Using C-style APIs

You can call generated factor code in your C code via our C-style APIs. For other programming languages like Go and Rust, you can also utilize the C-style APIs of KunQuant to call it in your language. Just search something like "go call c library" in Google. :)

## Build Necessary dependencies

You need to build the target `KunRuntime` for the core runtime library of KunQuant (or you can find it in `KunQuant/runner/` in the KunQuant install directory). And you may need to build your factor as a shared library. See `Save the compilation result as a shared library` in [Customize.md](./Customize.md). In your C-language program (or whatever else language), you need to link to `libKunRuntime.so` (in Linux. Other OS may have different names like `KunRuntime.dll` or `libKunRuntime.dylib`). You don't need to directly link to the factor library (like `libAlpha101.so`).

## C language example

`tests/capi/test_c.cpp` is the test code for CAPI, which is also an example for how you can use the C-style APIs. It is designed to run and check the `KunTest` factor library. `KunTest` computes a single factor `output = input + input * 2`. The output buffer name is "output" and the input buffer name is "input". The implementation of `KunTest` factor is in `tests/cpp/TestRuntime.cpp`.

First, include the necessary headers:

```C
#include <Kun/CApi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
```

Note that `<Kun/CApi.h>` is the header for the C language. You can check the comments in the header for the details of each C interfaces. The implementation of the APIs in this header is in the library `KunRuntime`

Allocate input and output buffer:

```C
    const size_t num_stocks = 24;
    const size_t num_time = 10;
    // dimension: [3 x 10 x 8]
    float *inputs = new float[num_stocks/8*10*8];
    for(size_t i=0;i<num_stocks*num_time; i++) {
        inputs[i] = float(rand()) / RAND_MAX;
    }
    float *outputs = new float[num_stocks/8*10*8];
```

The above code randomly fills the input buffer. Note that in real world workloads, you need to transpose the **input** data to `[stocks//8, time, 8]`. For example, if your original input data is in `[stocks, time]`, you can transpose it by

```C
void transpose_ST8s(float* in, float* out, int stocks, int time) {
    for(int i = 0; i < stocks ; i++) {
        for(int j = 0; j < time; j++) {
            float inval = in[i * time + j];
            out[i / 8 * time * 8 + j * 8 + i % 8 ] = inval;
        }
    }
}
```

Again, you need to make sure the number of stocks is a multiple of 8.

Then, similar to the Python example in [Readme.md](./Readme.md), we need to create the executor, load library and get the module:

```C
    KunExecutorHandle exec = kunCreateSingleThreadExecutor();
    KunLibraryHandle lib = kunLoadLibrary("/path/to/build/libKunTest.so");
    KunModuleHandle modu = kunGetModuleFromLibrary(lib, "testRuntimeModule");
```

`"/path/to/build/libKunTest.so"` can be replaced by the path to the generated factor library of you own. `"testRuntimeModule"` should be the name specified in the code `src = compileit(f, "LibNameHere", ...)` in `generate.py`.

Note that `KunExecutorHandle`, `KunLibraryHandle` and other handle types are opaque pointer types to the underlying KunRuntime objects. When you use `KunRuntime` in language other than C, you can treat them as `void*`.

Next, tell the `KunRuntime` the pointers of the buffers. In `KunTest`, we only have one input buffer named "input" and an output buffer named "output":

```C
    KunBufferNameMapHandle bufs = kunCreateBufferNameMap();
    kunSetBufferNameMap(bufs, "input", inputs);
    kunSetBufferNameMap(bufs, "output", outputs);
```

Execute the computataion. For more details of `kunRunGraph`, please check the comments in `<Kun/CApi.h>`.

```C
    kunRunGraph(exec, modu, bufs, num_stocks, num_time, 0, num_time);
```

The output data should be filled in the C buffer `outputs`. It is OK to reuse the same `KunBufferNameMap`, `KunExecutor`, `KunModule` in multiple calls to `kunRunGraph`. Finally, remember to release the resources:

```C++
    delete []inputs;
    delete []outputs;
    kunDestoryBufferNameMap(bufs);
    kunUnloadLibrary(lib);
    kunDestoryExecutor(exec);
```

Note that you cannot and do not need to release `KunModuleHandle`.

## C-API for Streaming mode

The logic is similar to the Python API example in [Stream.md](./Stream.md). For details, see `tests/capi/test_c.cpp` and `cpp/Kun/CApi.h`.