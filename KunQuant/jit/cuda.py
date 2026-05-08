"""GPU JIT entry point for KunQuant.

Mirror of `KunQuant.jit.cfake.compileit` but targets a CUDA backend
through the KunMLIR / kunir pipeline.  Reuses the existing Driver pass
list (`Driver.optimize`) so any IR rewrites the CPU path benefits from
also apply here — only the codegen layer is replaced.

User entry point::

    from KunQuant.jit import KunMLIR
    from KunQuant.jit.cuda import compileit, CudaCompilerConfig

    exe = compileit(f, CudaCompilerConfig(gpu_arch="sm_80"))
    executor = KunMLIR.Executor()                       # default stream
    executor.runGraph(exe, {"a": cp_a, "b": cp_b, "out": cp_out})
    executor.synchronize()

Scope (v0):
  * Single Function in, single kunir.func out.  Multi-Function /
    auto-partition support is future work.
  * dtype = "float" only (kunir lowers f32 today).
  * Layout is implicit: kunir uses the TS-major layout exposed by the
    runtime (see KunCuda/Runtime.h).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from KunQuant.jit import KunMLIR

from KunQuant.Driver import optimize, post_optimize
from KunQuant.Op import Input, Output
from KunQuant.passes import do_partition
from KunQuant.Stage import Function
from KunQuant.passes.CodegenMLIR import TargetSpec, translate_function


# Standard locations searched when CudaCompilerConfig.toolkit_path is left
# empty.  A toolkit dir must contain `nvvm/libdevice/libdevice.10.bc` (the
# upstream `gpu-module-to-binary` pass links libdevice into the LLVM
# module) and `bin/ptxas` (PTX → cubin).
_TOOLKIT_ENV_VARS  = ("CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH",
                       "CUDA_ROOT")
_TOOLKIT_FALLBACKS = ("/usr/local/cuda", "/opt/cuda", "/opt/nvidia/cuda")


def _is_toolkit_dir(path: str) -> bool:
    return (path
            and os.path.isfile(os.path.join(path, "nvvm", "libdevice",
                                              "libdevice.10.bc"))
            and os.path.isfile(os.path.join(path, "bin", "ptxas")))


def find_cuda_toolkit(override: str = "") -> str:
    """Locate a CUDA toolkit root suitable for `gpu-module-to-binary`.

    Search order:
      1. `override` (if non-empty and looks like a toolkit dir)
      2. $CUDA_HOME / $CUDA_PATH / $CUDA_TOOLKIT_PATH / $CUDA_ROOT
      3. Standard install paths (/usr/local/cuda, /opt/cuda, …)

    Raises FileNotFoundError if nothing usable is found — the message
    lists every location consulted so the caller can fix the env.
    """
    tried = []
    if override:
        tried.append(f"override={override!r}")
        if _is_toolkit_dir(override):
            return override
    for env in _TOOLKIT_ENV_VARS:
        val = os.environ.get(env, "")
        if val:
            tried.append(f"${env}={val!r}")
            if _is_toolkit_dir(val):
                return val
    for fallback in _TOOLKIT_FALLBACKS:
        tried.append(f"fallback={fallback!r}")
        if _is_toolkit_dir(fallback):
            return fallback
    raise FileNotFoundError(
        "Could not locate a CUDA toolkit (need "
        "<root>/nvvm/libdevice/libdevice.10.bc and <root>/bin/ptxas). "
        "Searched: " + ", ".join(tried) +
        ". Set CUDA_PATH or pass toolkit_path explicitly.")


@dataclass
class CudaCompilerConfig:
    """Mirrors the parts of KunCompilerConfig that matter for GPU.

    `dtype`, `gpu_arch`, and the kunir target_spec fields are the only
    knobs we actually expose.  The CPU-only fields (blocking_len,
    input_layout, etc.) deliberately do not appear here — they're not
    meaningful for the GPU path.
    """
    gpu_arch:    str = "sm_80"
    dtype:         str = "float"   # only "float" supported in v0

    # kunir.target_spec — graph-wide for v0.
    occupancy:     int = 1
    warps_per_cta: int = 4
    smem_size:     int = 49152
    vector_size:   int = 1

    # LLVM optimization level (forwarded to #nvvm.target<O = ...>).
    opt_level:     int  = 3
    # Path to the CUDA toolkit (where libdevice.10.bc + ptxas live).
    # Empty → upstream search: CUDA_HOME / CUDA_PATH / standard locations.
    toolkit_path:  str  = ""

    # Forwarded to `do_partition` — same default as KunCompilerConfig.
    # Larger factor ⇒ coarser partitions (fewer, bigger kernels).  After
    # partition each sub-Function becomes one kunir.func inside the
    # generated gpu.module; intermediate buffers between them are
    # auto-managed by the runtime's slot pool.
    partition_factor: int = 3

    # Pass-list options forwarded to optimize().  We seed reasonable GPU
    # defaults; user-supplied keys override.
    options:       Optional[dict] = None


def _gpu_pass_options(cfg: CudaCompilerConfig) -> dict:
    """Defaults for `Driver.optimize`'s `options` dict on the GPU path.

    The CPU compileit() does the same kind of seeding — we replicate the
    bits that affect graph rewriting.  `blocking_len` is needed by some
    decompose paths (skip-list cutoff in WindowedMin/Max); we feed it
    `warps_per_cta * 32 * vector_size`, which matches the GPU's
    stocks-per-block.
    """
    opts: dict = {
        "blocking_len":   cfg.warps_per_cta * 32 * cfg.vector_size,
        # Fast-stat tricks rely on running stats / FMA orderings that
        # don't map cleanly onto the GPU primitives we lower today.
        # Keep it off until the corresponding kunir lowerings exist.
        "no_fast_stat":   True,
    }
    if cfg.options:
        opts.update(cfg.options)
    return opts


def _to_dtype_token(dtype: str) -> str:
    if dtype == "float":  return "f32"
    if dtype == "double": return "f64"
    raise ValueError(f"compile_to_cuda: unsupported dtype '{dtype}' "
                       f"(supported: float, double — kunir today only "
                       f"lowers float on GPU)")


def _graph_io_names(f: Function):
    """User-facing graph inputs/outputs.  Captured BEFORE optimize +
    do_partition because those passes mutate `f` and may scatter the
    Input/Output ops across multiple sub-Functions (some of which then
    look like 'TEMP' from the partition's POV but stay user-visible at
    the graph boundary)."""
    ins  = [op.attrs["name"] for op in f.ops if isinstance(op, Input)]
    outs = [op.attrs["name"] for op in f.ops if isinstance(op, Output)]
    if not ins:
        raise ValueError("CudaCompilerConfig: function has no Input ops")
    if not outs:
        raise ValueError("CudaCompilerConfig: function has no Output ops")
    return ins, outs


def _run_full_pipeline(f: Function, cfg: CudaCompilerConfig):
    """Same pass pipeline the CPU `compileit` runs:

        optimize  →  do_partition  →  post_optimize

    Returns the list of post-partition Functions that the translator
    should walk (one kunir.func per Function).  Mutates `f` in place.
    """
    options = _gpu_pass_options(cfg)
    optimize(f, options)
    _mainf, impl = do_partition(f, cfg.partition_factor, options)
    post_optimize(impl, options)
    return impl


def _translate_partitions(impl, cfg: CudaCompilerConfig) -> KunMLIR.ModuleOp:
    """Emit one kunir.func per partitioned Function into a single
    KunMLIR module (single `gpu.module` with N siblings).  Cross-
    partition buffers stitch up automatically because each impl's
    Input/Output names match the producing/consuming partition's
    Output/Input names."""
    target = TargetSpec(occupancy=cfg.occupancy,
                          warps_per_cta=cfg.warps_per_cta,
                          smem_size=cfg.smem_size,
                          vector_size=cfg.vector_size)
    ir = KunMLIR.IRBuilder()
    dtype = _to_dtype_token(cfg.dtype)
    for sub in impl:
        translate_function(sub, target, ir, dtype=dtype)
    return ir.finish()


def compileit(f: Function, cfg: CudaCompilerConfig) -> KunMLIR.Executable:
    """Compile a KunQuant Function to a GPU `KunMLIR.Executable`.

    Pipeline mirrors `KunQuant.jit.cfake.compileit` on the CPU path:

      1. Capture user-facing Input/Output names (graph_inputs/outputs).
      2. Run Driver.optimize on `f` in place.
      3. do_partition splits `f` into one or more sub-Functions.
      4. post_optimize per sub-Function (TempWindowElim + MergeLoops + …).
      5. Translate each sub-Function into a kunir.func (siblings in one
         gpu.module).
      6. Hand off to KunMLIR.compile, which generates the cubin and
         resolves cross-kernel data flow via I/O names.
    """
    if cfg.dtype not in ("float", "double"):
        raise ValueError(
            f"CudaCompilerConfig.dtype must be 'float' or 'double', got "
            f"{cfg.dtype!r}")

    toolkit_path = find_cuda_toolkit(cfg.toolkit_path)

    graph_inputs, graph_outputs = _graph_io_names(f)
    impl = _run_full_pipeline(f, cfg)
    mod  = _translate_partitions(impl, cfg)

    return KunMLIR.compile(
        mod,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        gpu_arch=cfg.gpu_arch,
        opt_level=cfg.opt_level,
        toolkit_path=toolkit_path,
    )


def to_mlir(f: Function, cfg: CudaCompilerConfig) -> KunMLIR.ModuleOp:
    """Run the same passes + translator as `compileit`, but return the
    KunMLIR module before PTX/CUBIN.  Useful for debugging the IR.
    Mutates `f` in place (same as `compileit`)."""
    _graph_io_names(f)              # raises if no Input / Output ops
    impl = _run_full_pipeline(f, cfg)
    return _translate_partitions(impl, cfg)
