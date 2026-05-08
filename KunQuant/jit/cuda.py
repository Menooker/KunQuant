"""GPU JIT entry point for KunQuant.

Mirror of `KunQuant.jit.cfake.compileit` but targets a CUDA backend
through the KunMLIR / kunir pipeline.  Reuses the existing Driver pass
list (`Driver.optimize`) so any IR rewrites the CPU path benefits from
also apply here — only the codegen layer is replaced.

User entry point::

    from KunQuant.jit.cuda import compileit, CudaCompilerConfig

    exe = compileit(f, CudaCompilerConfig(gpu_arch="sm_80"))
    exe.launch({"a": cp_a, "b": cp_b, "out": cp_out})

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

from KunQuant.Driver import optimize
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


def compileit(f: Function, cfg: CudaCompilerConfig) -> KunMLIR.Executable:
    """Compile a single KunQuant Function to a GPU `KunMLIR.Executable`.

    The Function is mutated in place by Driver.optimize() (same as the
    CPU path).  Inputs/Outputs declared via `Input(name)` / `Output(...,
    name)` become the resulting Executable's graph_inputs / graph_outputs.
    """
    if cfg.dtype not in ("float", "double"):
        raise ValueError(
            f"CudaCompilerConfig.dtype must be 'float' or 'double', got "
            f"{cfg.dtype!r}")

    # Resolve the CUDA toolkit before invoking C++.  Auto-search if the
    # user didn't pass an explicit path.  Failure here gives a useful
    # message; failure later (in ptxas / libdevice link) is opaque.
    toolkit_path = find_cuda_toolkit(cfg.toolkit_path)

    # 1.  Same optimizer pipeline the CPU path runs.  This is where
    #     WindowedSum etc. decompose into ForeachBackWindow + Reduce.
    options = _gpu_pass_options(cfg)
    optimize(f, options)

    # 2.  Translate the post-optimize IR to a KunMLIR module.
    target = TargetSpec(occupancy=cfg.occupancy,
                          warps_per_cta=cfg.warps_per_cta,
                          smem_size=cfg.smem_size,
                          vector_size=cfg.vector_size)
    ir = KunMLIR.IRBuilder()
    in_names, out_names = translate_function(
        f, target, ir, dtype=_to_dtype_token(cfg.dtype))
    mod = ir.finish()

    # 3.  Hand off to the KunMLIR compile pipeline.
    return KunMLIR.compile(
        mod,
        graph_inputs=in_names,
        graph_outputs=out_names,
        gpu_arch=cfg.gpu_arch,
        opt_level=cfg.opt_level,
        toolkit_path=toolkit_path,
    )


def to_mlir(f: Function, cfg: CudaCompilerConfig) -> KunMLIR.ModuleOp:
    """Run the same passes + translator as `compileit`, but return the
    KunMLIR module before PTX/CUBIN.  Useful for debugging the IR."""
    options = _gpu_pass_options(cfg)
    optimize(f, options)
    target = TargetSpec(occupancy=cfg.occupancy,
                          warps_per_cta=cfg.warps_per_cta,
                          smem_size=cfg.smem_size,
                          vector_size=cfg.vector_size)
    ir = KunMLIR.IRBuilder()
    translate_function(f, target, ir, dtype=_to_dtype_token(cfg.dtype))
    return ir.finish()
