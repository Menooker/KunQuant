"""GPU JIT entry point for KunQuant.

Mirror of `KunQuant.jit.cfake.compileit` but targets a CUDA backend
through the kun_mlir / kunir pipeline.  Reuses the existing Driver pass
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
from dataclasses import dataclass
from typing import Optional

import kun_mlir

from KunQuant.Driver import optimize
from KunQuant.Stage import Function
from KunQuant.passes.CodegenMLIR import TargetSpec, translate_function


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

    # ptx → cubin
    opt_level:     int  = 3
    ptxas_path:    str  = ""

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
        # opt_reduce rewrites WindowedSum etc. into the stateful
        # FastWindowedSum op, which kunir doesn't have a counterpart
        # for yet — keep the canonical ForeachBackWindow + Reduce shape.
        "opt_reduce":     False,
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


def compileit(f: Function, cfg: CudaCompilerConfig) -> kun_mlir.Executable:
    """Compile a single KunQuant Function to a GPU `kun_mlir.Executable`.

    The Function is mutated in place by Driver.optimize() (same as the
    CPU path).  Inputs/Outputs declared via `Input(name)` / `Output(...,
    name)` become the resulting Executable's graph_inputs / graph_outputs.
    """
    if cfg.dtype not in ("float", "double"):
        raise ValueError(
            f"CudaCompilerConfig.dtype must be 'float' or 'double', got "
            f"{cfg.dtype!r}")

    # 1.  Same optimizer pipeline the CPU path runs.  This is where
    #     WindowedSum etc. decompose into ForeachBackWindow + Reduce.
    options = _gpu_pass_options(cfg)
    optimize(f, options)

    # 2.  Translate the post-optimize IR to a kun_mlir module.
    target = TargetSpec(occupancy=cfg.occupancy,
                          warps_per_cta=cfg.warps_per_cta,
                          smem_size=cfg.smem_size,
                          vector_size=cfg.vector_size)
    ir = kun_mlir.IRBuilder()
    in_names, out_names = translate_function(
        f, target, ir, dtype=_to_dtype_token(cfg.dtype))
    mod = ir.finish()

    # 3.  Hand off to the kun_mlir compile pipeline.
    return kun_mlir.compile(
        mod,
        graph_inputs=in_names,
        graph_outputs=out_names,
        gpu_arch=cfg.gpu_arch,
        opt_level=cfg.opt_level,
        ptxas_path=cfg.ptxas_path,
    )


def to_mlir(f: Function, cfg: CudaCompilerConfig) -> kun_mlir.ModuleOp:
    """Run the same passes + translator as `compileit`, but return the
    kun_mlir module before PTX/CUBIN.  Useful for debugging the IR."""
    options = _gpu_pass_options(cfg)
    optimize(f, options)
    target = TargetSpec(occupancy=cfg.occupancy,
                          warps_per_cta=cfg.warps_per_cta,
                          smem_size=cfg.smem_size,
                          vector_size=cfg.vector_size)
    ir = kun_mlir.IRBuilder()
    translate_function(f, target, ir, dtype=_to_dtype_token(cfg.dtype))
    return ir.finish()
