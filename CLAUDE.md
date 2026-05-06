# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

KunQuant is a compiler, optimizer, and code generator for financial factor expressions (e.g., WorldQuant Alpha101, Qlib Alpha158). It takes Python-defined financial expressions, applies optimization passes, generates C++ code with SIMD and parallelism, and executes it via a Python binding (KunRunner). Achieves 170x+ speedup over naive Pandas.

## Build & Install

```bash
# Standard install
pip install .

# Editable install with C++ tests
KUN_BUILD_TESTS=1 pip install -e .
```

Build environment variables:
- `KUN_BUILD_TYPE=Debug|Release` — default: Release
- `KUN_BUILD_TESTS=1` — enables `KunTest` and `KunCApiTest` targets
- `KUN_NO_AVX2=1` — disable AVX2/FMA (for older CPUs)
- `KUN_DEBUG=1` — print internal compiler pass output
- `KUN_DEBUG_JIT=1` — print JIT C++ compilation details

## Running Tests

```bash
# Python IR transformation tests (no build required)
python tests/test.py
python tests/test2.py

# Streaming mode tests
python tests/test_stream.py

# C++ runtime tests (requires KUN_BUILD_TESTS=1)
python tests/test_runtime.py

# Alpha101 correctness (random data)
python tests/test_alpha101.py

# Integration tests
bash tests/tests.sh
```

## Architecture

The pipeline: **Python expression graph → optimization passes → C++ code generation → JIT compile → shared library → KunRunner execution**

### Python Compiler Layer (`KunQuant/`)

- **`Op.py`** — Core IR. All operations inherit from `OpBase`. `Builder` is a thread-local context manager that records ops as they're constructed. Key traits: `WindowedOp`, `ReductionOp`, `SinkOpTrait`, `CrossSectionalOp`.
- **`Stage.py`** — `Function` holds the op graph; `OpInfo` tracks use counts. Provides topological sort and dead-op elimination.
- **`Driver.py`** — Orchestrates compilation. `KunCompilerConfig` holds config (dtype, layout, streaming). `compileit()` is the main entry point; `optimize()` runs the pass pipeline.
- **`ops/`** — Concrete op types: `ElewiseOp.py` (Add, Mul, etc.), `ReduceOp.py` (ReduceAdd, ReduceRank, etc.), `CompOp.py` (Greater, Less, etc.), `MiscOp.py`.
- **`passes/`** — All optimization passes and code generation:
  - `InferWindow.py` — Infers time-window sizes
  - `SpecialOpt.py` — Domain-specific rewrites (stddev, rank)
  - `Decompose.py` — Expands windowed ops into `ForeachBackWindow` loops + reductions
  - `ExprFold.py` — Constant folding and algebraic simplification
  - `TempWindowElim.py` — Eliminates intermediate window buffers
  - `MergeLoops.py` — Fuses compatible loops
  - `Partitioner.py` — Partitions ops into parallel execution blocks
  - `CodegenCpp.py` — Emits C++ source from the final IR
- **`jit/cfake.py`** — Invokes MSVC/GCC/Clang to compile generated C++ to a shared library.
- **`predefined/`** — Ready-to-use factor libraries: `Alpha101.py` (101 factors), `Alpha158.py`.
- **`runner/`** — Python bindings via pybind11; `KunRunner` loads shared libs, creates executors, runs graphs.

### Optimization Pass Order (in `Driver.optimize()`)

1. InferWindow → SpecialOpt → Decompose → ExprFold → SpecialOpt → ExprFold → DecomposeRank → MoveDupRankOutput → TempWindowElim

Post-compile (`post_optimize()`): TempWindowElim → InferInputWindow → MergeLoops

### C++ Runtime (`cpp/`)

- **`cpp/Kun/`** — Core runtime: `Runtime.cpp` (execution engine), `Executor.cpp`, `Module.cpp`, `CApi.cpp`, `Ops.hpp` (operator implementations), `Rank.hpp`, `Scale.hpp`, `SkipList.cpp` (sorted stream state).
- **`cpp/KunSIMD/`** — SIMD vector ops for x86 (AVX2/AVX512) and ARM (NEON).
- **`cpp/Python/`** — pybind11 bindings.

### Memory Layouts

- `TS` (Time-Stock): time is outer dimension — default for batch mode
- `STs` (Stock-Time-blocked): stocks are outer, time is inner with blocking — better for streaming

Recommended blocking: 8 stocks (float + AVX2), 4 stocks (double + AVX2).

## Typical Usage Pattern

```python
from KunQuant.Op import Builder, Input, Output
from KunQuant.ops import *
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.Driver import KunCompilerConfig

with Builder() as b:
    close = Input("close")
    # ... define factor expressions ...
    Output(some_expr, "factor_name")

f = Function(b.ops)
lib = cfake.compileit([("mylib", f, KunCompilerConfig())], "out_lib", cfake.CppCompilerConfig())
modu = lib.getModule("mylib")

# Execute
from KunQuant.runner import KunRunner as kr
executor = kr.createMultiThreadExecutor(num_threads)
result = kr.runGraph(executor, modu, input_dict, start_time, num_time)
```
