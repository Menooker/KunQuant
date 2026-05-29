# Running KunQuantMLIR on NVIDIA GPUs

This document explains how to install and use the optional `KunQuant-MLIR`
CUDA backend. The regular `KunQuant` package provides the Python IR, optimizer,
and CPU runtime. The `KunQuant-MLIR` package adds the MLIR/CUDA compiler and
runtime used to run KunQuant factor graphs on NVIDIA GPUs. Its Python API is
intentionally close to the CPU KunQuant API, so users who already compile and
run factors with KunQuant can switch to KunQuant-MLIR with only small changes
to the compiler and executor setup.

## Installation

Install the released packages from pip:

```bash
python -m pip install KunQuant KunQuant-MLIR
```

The examples below use CuPy for convenient CUDA arrays. Install the CuPy package
that matches your local CUDA runtime, for example:

```bash
python -m pip install cupy-cuda12x
```

or, for CUDA 13.x:

```bash
python -m pip install cupy-cuda13x
```

`KunQuant-MLIR` itself does not directly depend on PyTorch or CuPy. The GPU
runtime consumes CUDA arrays through the DLPack protocol, so inputs and
preallocated outputs may be CuPy arrays, PyTorch CUDA tensors, or any other
CUDA array object that implements `__dlpack__`. The only component in this
document that requires CuPy is `KunQuantMLIR.OverlapRunner`, because it uses
CuPy streams, events, pinned memory, and async copies.

Runtime compilation also needs an NVIDIA driver and a CUDA toolkit. The toolkit
must contain:

```text
<cuda-root>/bin/ptxas
<cuda-root>/nvvm/libdevice/libdevice.10.bc
```

## CUDA Toolkit Discovery

The GPU compiler locates CUDA through
`KunQuant.jit.cuda.find_cuda_toolkit()`. The search order is:

1. `CudaCompilerConfig(toolkit_path="...")`
2. `$CUDA_HOME`
3. `$CUDA_PATH`
4. `$CUDA_TOOLKIT_PATH`
5. `$CUDA_ROOT`
6. common install locations: `/usr/local/cuda`, `/opt/cuda`, `/opt/nvidia/cuda`

If CUDA is not installed in a default location, set the environment explicitly:

```bash
export CUDA_PATH=/usr/local/cuda-13.2
```

You can also pass the toolkit path from Python:

```python
from KunQuant.jit.cuda import CudaCompilerConfig

ccfg = CudaCompilerConfig(
    gpu_arch="sm_90",
    toolkit_path="/usr/local/cuda-13.2",
)
```

`gpu_arch` is the CUDA target architecture, for example `sm_80`, `sm_90`, or
`sm_120`. If CuPy or PyTorch is installed and can see the current CUDA device,
KunQuant can query it:

```python
from KunQuant.jit.env import get_cuda_compute_capability

gpu_arch = get_cuda_compute_capability()
```

## Alpha101 Example

The GPU backend currently uses `TS` layout only: every input and output is a
2-D array with shape `[time, stocks]`. The following example compiles and runs
`alpha001` from Alpha101. It uses CuPy arrays, but the same `runGraph` call can
consume any CUDA array that supports DLPack. The GPU version of running alpha001
is very similar to the CPU example in the main Readme document of KunQuant.

```python
import numpy as np
import cupy as cp

from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.predefined.Alpha101 import AllData, alpha001
from KunQuant.jit import KunMLIR
from KunQuant.jit.cuda import CudaCompilerConfig, compileit
from KunQuant.jit.env import get_cuda_compute_capability


cp.cuda.Device(0).use()

builder = Builder()
with builder:
    all_data = AllData(
        low=Input("low"),
        high=Input("high"),
        close=Input("close"),
        open=Input("open"),
        amount=Input("amount"),
        volume=Input("volume"),
    )
    Output(alpha001(all_data), "alpha001")

f = Function(builder.ops)

kcfg = KunCompilerConfig(
    input_layout="TS",
    output_layout="TS",
    blocking_len=1,
    partition_factor=2,
    options={"opt_reduce": True, "fast_log": True},
)
ccfg = CudaCompilerConfig(gpu_arch=get_cuda_compute_capability())

lib = compileit([("alpha001", f, kcfg)], "alpha001_cuda", ccfg)
exe = lib.getModule("alpha001")

num_time = 260
num_stocks = 64
rng = np.random.default_rng(0)

inputs = {
    name: cp.asarray(
        rng.random((num_time, num_stocks), dtype=np.float32) + np.float32(1.0)
    )
    for name in exe.input_names
}

executor = KunMLIR.Executor()
outputs = {
    "alpha001": cp.empty((num_time, num_stocks), dtype=cp.float32),
}

ret = executor.runGraph(
    exe,
    inputs,
    outputs=outputs,
    use_cuda_graph=True,
)

# runGraph is asynchronous. Synchronize before reading the result.
executor.synchronize()
out = ret["alpha001"]
print(out.shape, out.dtype)
```

`Executor.runGraph(...)` only enqueues work on the executor's CUDA stream. It
does not wait for completion. Before reading results on the host, either call:

```python
executor.synchronize()
```

or copy through a framework API that synchronizes internally, such as:

```python
host_out = cp.asnumpy(ret["alpha001"])
```

`cp.asnumpy` waits for the relevant CUDA work before returning the NumPy array.

If `outputs` is omitted, `runGraph` allocates CUDA output buffers and returns
DLPack-compatible array objects. With CuPy, convert them like this:

```python
ret = executor.runGraph(exe, inputs, use_cuda_graph=True)
out = cp.from_dlpack(ret["alpha001"])
executor.synchronize()
```

PyTorch CUDA tensors can be passed in the same way:

```python
import torch

inputs = {
    "close": torch.empty((num_time, num_stocks), device="cuda", dtype=torch.float32),
    # Fill the other required input names in exe.input_names.
}
```

All input and output arrays must be CUDA-resident, 2-D, C-contiguous, and have
dtype `float32` or `float64`.

## Use CUDA Graph Mode

For production runs, prefer CUDA Graph mode:

```python
ret = executor.runGraph(exe, inputs, outputs=outputs, use_cuda_graph=True)
executor.synchronize()
```

CUDA Graph mode records the kernel dependency graph and intermediate buffer
allocation/free nodes into a CUDA Graph. This reduces launch overhead for
factor graphs such as Alpha101, which are compiled into many partitions and
kernels.

You can reproduce the benchmark with:

```bash
python tests/test_alpha101.py --benchmode --gpu-arch sm_120 --time 2600 --num-stocks 1024
python tests/test_alpha101.py --benchmode --gpu-arch sm_120 --time 2600 --num-stocks 1024 --use-cuda-graph
```

On the local sm_120 GPU used for this document, `tests/test_alpha101.py` printed
the following `Exec takes` values. Each value is the average of 20 timed runs:

| Mode | f32 Alpha101 | f64 Alpha101 |
|---|---:|---:|
| normal launches | 0.262146 s | 3.502938 s |
| CUDA Graph | 0.171881 s | 2.606657 s |
| speedup | 1.53x | 1.34x |

## CPU Input/Output and OverlapRunner

If inputs and outputs live in CPU memory and the time dimension is large, a
simple pipeline is often inefficient:

1. copy all inputs from host to device
2. run `runGraph`
3. copy all outputs from device to host

If these steps all run serially on one stream, H2D and D2H copies cannot
overlap with compute. The overlap GPU runner in the current Python API is
`KunQuantMLIR.OverlapRunner.OverlapRunner`. It uses three non-blocking streams:

1. an H2D stream for copying the next input chunk
2. the executor stream for running the current chunk
3. a D2H stream for copying the previous output chunk

This can hide a significant part of copy overhead when the time axis can be
split into chunks. The runner also reuses per-slot GPU input buffers, GPU
output buffers, and pinned host output buffers.

`OverlapRunner` depends on CuPy:

```bash
python -m pip install cupy-cuda12x
```

The following example assumes the compiled executable has input and output of name:
 `exe.input_names == ["a"]` and
`exe.output_names == ["out"]`. It uses one CPU input array named `"a"` with
shape `[5000, 1024]`. It submits five time slices of length 1000 first, then
waits for the five `PendingResult` objects and puts the output chunks in a
list. Every `submit` returns immediately after queuing H2D copy, compute, and
D2H copy. Do not read the output arrays until `wait()` returns.

```python
import numpy as np
import cupy as cp

from KunQuant.jit import KunMLIR
from KunQuantMLIR.OverlapRunner import OverlapRunner


cp.cuda.Device(0).use()

compute_stream = cp.cuda.Stream(non_blocking=True)
executor = KunMLIR.Executor(compute_stream)
runner = OverlapRunner(exe, executor, num_slots=3)

host_a = np.arange(5000 * 1024, dtype=np.float32).reshape(5000, 1024)

# First loop: submit five CPU slices.
pending = []
for start in range(0, 5000, 1000):
    chunk = host_a[start:start + 1000]
    pending.append(
        runner.submit(
            {"a": chunk},
            length=1000,
            use_cuda_graph=True,
        )
    )

# Second loop: wait for results and collect output chunks.
output_chunks = []
for result in pending:
    chunk_outputs = result.wait()
    output_chunks.append(chunk_outputs["out"])

runner.synchronize()
# output_chunks is an array of numpy.array of size [1000*1024]
```

For rolling/windowed factors, non-first chunks must include enough previous
time steps to rebuild the rolling state at the beginning of the chunk. Those
overlapped rows are warmup rows: they are required for correctness, but their
outputs should not be consumed as final results. For example,
`WindowedSum(x, 10)` needs 9 previous values, so the first 9 rows of a fresh
run are unreliable and should be dropped.

If the full time length is 5000 and you want 1000 valid output rows per
submitted chunk, the input slices should overlap like this:

```text
valid rows 0..999:     submit host_a[0:1000]
valid rows 1000..1999: submit host_a[1000 - 9:2000]
valid rows 2000..2999: submit host_a[2000 - 9:3000]
...
```

Compared with the simple example above, only a few lines need to change.
Define the warmup before the submit loop:

```python
warmup = 9
```

Inside the submit loop, replace `chunk = host_a[start:start + 1000]` with:

```python
chunk = host_a[max(0, start - warmup):start + 1000]
```

When collecting results, drop the overlapped prefix:

```python
for i, result in enumerate(pending):
    drop_prefix = 0 if i == 0 else warmup
    chunk_outputs = result.wait()
    output_chunks.append(chunk_outputs["out"][drop_prefix:])
```

Use `exe.getOutputUnreliableCount()` to query how many leading rows are
unreliable for a compiled executable. If an executable has multiple outputs,
use the relevant output's value, or conservatively use the maximum value across
outputs when chunking a shared input batch. The same API is described below in
the `Executable` section.

The tradeoff of `OverlapRunner` is memory: `OverlapRunner` increases both pinned CPU memory and
GPU memory usage, roughly proportional to `num_slots`, the number of inputs,
the number of outputs, and the chunk size. It is useful when copy cost is
visible and data starts or ends on the CPU. If data is already on the GPU,
calling `executor.runGraph` directly is simpler.

## Core APIs

### CudaCompilerConfig

`CudaCompilerConfig` controls GPU compile-time settings:

```python
from KunQuant.jit.cuda import CudaCompilerConfig

ccfg = CudaCompilerConfig(
    gpu_arch="sm_120",
    occupancy=1,
    warps_per_cta=4,
    smem_size=49152,
    opt_level=3,
    toolkit_path="",
)
```

| Field | Meaning |
|---|---|
| `gpu_arch` | CUDA target architecture, such as `sm_80`, `sm_90`, or `sm_120` |
| `occupancy` | target occupancy used by shared-memory planning |
| `warps_per_cta` | number of warps per CTA; affects block size |
| `smem_size` | shared-memory budget used by KunGpu window-buffer planning |
| `opt_level` | LLVM/NVVM optimization level |
| `toolkit_path` | CUDA toolkit root; empty means use the discovery rules above |

Each factor graph still uses the regular `KunQuant.Driver.KunCompilerConfig`.
The GPU backend requires:

```python
KunCompilerConfig(input_layout="TS", output_layout="TS")
```

`dtype` supports `"float"` and `"double"`. On the GPU path, `blocking_len` maps
to `vector_size`; `1` is the usual choice. The GPU backend forces
`no_skip_list=True` because the MLIR code generator does not lower the CPU
runtime's `SkipList*` operators.

### Compile Entry Points

```python
from typing import List, Tuple

from KunQuant.Driver import KunCompilerConfig
from KunQuant.Stage import Function
from KunQuant.jit import KunMLIR
from KunQuant.jit.cuda import CudaCompilerConfig, Library
from KunQuant.jit.cuda import compile_func, compileit, to_mlir


def compile_func(
    f: Function,
    kcfg: KunCompilerConfig,
    ccfg: CudaCompilerConfig,
) -> KunMLIR.Executable: ...


def compileit(
    funclist: List[Tuple[str, Function, KunCompilerConfig]],
    libname: str,
    compiler_config: CudaCompilerConfig,
) -> Library: ...


def to_mlir(
    f: Function,
    kcfg: KunCompilerConfig,
    ccfg: CudaCompilerConfig,
) -> KunMLIR.ModuleOp: ...
```

Usage:

```python
exe = compile_func(f, kcfg, ccfg)
lib = compileit(
    [("module_name", f, kcfg)],
    "library_name",
    ccfg,
)
exe = lib.getModule("module_name")
mod = to_mlir(f, kcfg, ccfg)
```

`compile_func(f, kcfg, ccfg)` compiles one `KunQuant.Stage.Function` and
returns a `KunMLIR.Executable`. It runs the same KunQuant optimization and
partitioning pipeline used by the GPU path, lowers the result to KunIR/MLIR,
invokes the CUDA backend, and loads the resulting cubin into the CUDA driver.
The input `Function` is optimized in place, so create a fresh `Function` if you
need to compile the same graph with different settings.

`compileit(funclist, libname, compiler_config)` mirrors the CPU backend's
multi-function API shape. `funclist` is a list of
`(module_name, function, KunCompilerConfig)` tuples. It compiles every function
with the same `CudaCompilerConfig` and returns a lightweight `Library`
container. Use `lib.getModule(module_name)` to retrieve each
`KunMLIR.Executable`. The `libname` is kept for API symmetry and diagnostics;
the GPU path does not produce a CPU-style shared library.

`to_mlir(f, kcfg, ccfg)` runs the frontend optimization, partitioning, and
KunIR translation stages, then returns the intermediate `KunMLIR.ModuleOp`
without invoking PTX/CUBIN generation. It is intended for debugging generated
IR. Like `compile_func`, it mutates the input `Function`. External runtime
kernels such as cross-sectional `Rank`/`Scale` are represented as descriptors
for `KunMLIR.compile`, so they do not appear as regular KunIR functions in the
returned module.

### Executable

`KunMLIR.Executable` is a compiled and CUDA-driver-loaded factor graph.

| API | Meaning |
|---|---|
| `input_names` | graph-level input names |
| `output_names` / `getOutputNames()` | graph-level output names |
| `getOutputUnreliableCount()` | number of leading time steps to ignore for each output because of rolling warmup |
| `kernel_names` | internal kernel names |
| `launch_order` | runtime topological kernel launch order |
| `num_kernels` | number of kernels |
| `peak_intermediate_slots` | peak number of runtime intermediate buffer slots |
| `num_buffers` | total graph runtime buffers |
| `clone()` | create an executable with independent launch state while sharing immutable compiled data |
| `save_to_files(dir, name)` | save metadata JSON and cubin files |
| `Executable.load_from_files(dir, name)` | load metadata JSON and cubin files |

Use `exe.clone()` if the same compiled graph must run concurrently on multiple
streams or executors. A single `Executable` owns mutable launch state and an
intermediate buffer pool, so it should not be driven concurrently by multiple
executors.

### Executor.runGraph

```python
executor = KunMLIR.Executor()
ret = executor.runGraph(
    exe,
    inputs,
    cur_time=0,
    length=0,
    outputs=None,
    mask=0,
    min_chunk_warmup_factor=4,
    sm_fill_factor=1.5,
    use_cuda_graph=False,
)
```

| Parameter | Meaning |
|---|---|
| `exe` | `KunMLIR.Executable` |
| `inputs` | `{name: cuda_array}`; keys must match `exe.input_names` |
| `cur_time` | currently only `0` is supported on GPU |
| `length` | time dimension; `0` means infer it from input shape |
| `outputs` | optional `{name: cuda_array}`; missing outputs are allocated by the runtime |
| `mask` | leave the first `mask` output rows unwritten; useful for chunked warmup |
| `min_chunk_warmup_factor` | lower bound for internal time chunk size relative to warmup |
| `sm_fill_factor` | heuristic target for internal chunk count relative to SM count |
| `use_cuda_graph` | launch with CUDA Graph mode |

`KunMLIR.Executor(stream=None)` uses the CUDA default stream. You can also pass
a CuPy stream or a raw stream pointer:

```python
stream = cp.cuda.Stream(non_blocking=True)
executor = KunMLIR.Executor(stream)
```

`runGraph` is asynchronous. Before reading results on the CPU, call
`executor.synchronize()`, wait on the stream yourself, or use a framework copy
such as `cp.asnumpy` that synchronizes internally.

### OverlapRunner

```python
from KunQuantMLIR.OverlapRunner import OverlapRunner

runner = OverlapRunner(exe, executor, num_slots=3)
pending = runner.submit(
    host_inputs,
    cur_time=0,
    length=0,
    mask=0,
    min_chunk_warmup_factor=4,
    sm_fill_factor=1.5,
    use_cuda_graph=True,
)
chunk_outputs = pending.wait()
runner.synchronize()
```

`submit` accepts a CPU NumPy input dict, asynchronously queues H2D copy,
`runGraph`, and D2H copy, and returns a `PendingResult`. `PendingResult.wait()`
waits for D2H completion and returns `{output_name: numpy.ndarray}`.

## Building KunQuantMLIR from Source

Source builds require:

* Python 3.9+
* CMake 3.18+
* Ninja (or GNU make. We use Ninja in the tutorial below)
* a C++17 compiler
* CUDA toolkit
* LLVM/MLIR built with the NVPTX target

The LLVM/MLIR version used by this KunQuant checkout is recorded in:

```bash
cat mlir/llvm_commit.txt
```

### Download Prebuilt LLVM/MLIR

You can download a matching LLVM/MLIR build from the KunQuant GitHub releases.
The archive URL format is:

```text
https://github.com/Menooker/KunQuant/releases/download/llvm-mlir-${LLVM_TAG}/llvm-mlir-install-${LLVM_LINK}-${LLVM_TAG}.tar.gz
```

`LLVM_LINK` must be either `static` or `dynamic`.

Choose `static` for a PyPI-wheel-like build. This is the mode used by uploaded
`KunQuant-MLIR` wheels, and it does not require LLVM/MLIR libraries on
`LD_LIBRARY_PATH` at runtime. Choose `dynamic` for local development: linking
is faster and outputs are smaller, but running tests or importing the locally
built extension requires LLVM/MLIR libraries on `LD_LIBRARY_PATH`.

```bash
LLVM_TAG="$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' mlir/llvm_commit.txt | head -n1 | tr -d '[:space:]')"

# Choose before running this block:
#   export LLVM_LINK=static
#   export LLVM_LINK=dynamic
LLVM_LINK="???"

LLVM_PREFIX=/tmp/llvm-mlir
mkdir -p "$LLVM_PREFIX"
curl -fL --retry 3 \
  "https://github.com/Menooker/KunQuant/releases/download/llvm-mlir-${LLVM_TAG}/llvm-mlir-install-${LLVM_LINK}-${LLVM_TAG}.tar.gz" \
  -o /tmp/llvm-mlir.tar.gz
tar -xzf /tmp/llvm-mlir.tar.gz -C "$LLVM_PREFIX" --strip-components=1
```

Only for the `dynamic` variant, set `LD_LIBRARY_PATH` at runtime before
importing/running KunMLIR python package:

```bash
export LD_LIBRARY_PATH=$LLVM_PREFIX/lib:${LD_LIBRARY_PATH:-}
```

### Manual CMake Build and Tests

```bash
# Only needed if CUDA is not in a default location.
# export CUDA_PATH=/usr/local/cuda-13.2

export LLVM_PREFIX=/tmp/llvm-mlir
export LLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm
export MLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir

python -m pip install lit

cmake -S . -B build/mlir-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DKUN_BUILD_MLIR=ON \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DLLVM_EXTERNAL_LIT="$(command -v lit)"

cmake --build build/mlir-build --target check-kun-mlir --parallel 8
```

If CMake cannot find CUDA from the default paths or `$CUDA_PATH`, add
`-DCUDAToolkit_ROOT="$CUDA_PATH"` to the configure command. If it still cannot
find `nvcc`, add `-DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc"`.

`check-kun-mlir` runs the MLIR/CUDA backend tests. Tests that exercise the CUDA
runtime need access to a real CUDA device.

### Build and Install with pip

The `python/kunquant_mlir` package can also drive CMake for you:

```bash
# Only needed if CUDA is not in a default location.
# export CUDA_PATH=/usr/local/cuda-13.2

export LLVM_PREFIX=/tmp/llvm-mlir
export LLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm
export MLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir
export LLVM_EXTERNAL_LIT="$(command -v lit)"

python -m pip install -e .
python -m pip install -e python/kunquant_mlir -v
```

`python/kunquant_mlir/setup.py` forwards `LLVM_DIR`, `MLIR_DIR`,
`CUDAToolkit_ROOT`, `CMAKE_CUDA_COMPILER`, and `LLVM_EXTERNAL_LIT` from the
environment into CMake. Usually `LLVM_DIR`, `MLIR_DIR`, and
`LLVM_EXTERNAL_LIT` are enough. Set `CUDAToolkit_ROOT=$CUDA_PATH`, and only if
needed `CMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc`, when CMake cannot discover
your CUDA toolkit automatically.

## Current GPU Backend Limitations

* `Executor.runGraph(..., cur_time=...)` currently supports only
  `cur_time=0`. The nonzero start-time behavior available in the CPU runtime is
  not implemented in the GPU runtime yet.
* The GPU backend supports only `TS` layout. Inputs and outputs must be 2-D
  `[time, stocks]` arrays.
* Inputs and outputs must be CUDA-resident C-contiguous DLPack arrays with dtype
  `float32` or `float64`.
* `WindowedQuantile` / quantile is not supported yet. It currently only has a
  skip-list decomposition path, and the GPU backend does not support
  `SkipListQuantile`.
* `WindowedMin`, `WindowedMax`, `TsArgMin`, `TsArgMax`, `TsRank`, and composite
  operators that lower through them currently use linear scans over the rolling
  window. Their cost is approximately
  `O(window_size * time_length)`, so very large windows can be
  slow.
* Cross-sectional `Rank` is implemented as an external CUDA kernel. It loads
  one time step's cross-section into dynamic shared memory and scans the row
  for every stock. Its cost is approximately `O(num_stocks^2)`.
* Cross-sectional `Rank` is limited by per-block dynamic shared memory. On the
  local sm_120 device used for this document, smem size is 101376 bytes,
  which corresponds to roughly 25344 `float32` stocks or 12672 `float64` stocks. Exceeding this
  limit raises a runtime shared-memory error. Cross-sectional `Scale` uses the
  same external-kernel path and needs `(num_stocks + 1) * sizeof(T)` bytes of
  dynamic shared memory.
* Not every KunQuant operator has MLIR lowering yet. The current GPU path mainly
  covers the elementwise, rolling-reduction, and cross-sectional `Rank`/`Scale`
  paths needed by the Alpha101 tests. Unsupported operators fail at compile time
  with `NotImplementedError`.
