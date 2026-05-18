"""GPU JIT entry point for KunQuant.

Mirror of `KunQuant.jit.cfake.compileit` but targets a CUDA backend
through the KunMLIR / kunir pipeline.  Reuses the existing Driver pass
list (`Driver.optimize`) so any IR rewrites the CPU path benefits from
also apply here — only the codegen layer is replaced.

Two-tier config split matches the CPU path:

  * Per-Function knobs live in `KunCompilerConfig` (the CPU-shared
    dataclass): `dtype`, `blocking_len`, `partition_factor`,
    `input_layout` / `output_layout` (TS only on GPU), `options`.
  * Compile-/link-time knobs live in `CudaCompilerConfig`: `gpu_arch`,
    `warps_per_cta`, `smem_size`, `occupancy`, `opt_level`,
    `toolkit_path`.  Shared across every Function in a `Library`.

Single-Function compile::

    from KunQuant.jit import KunMLIR
    from KunQuant.jit.cuda import compile_func, CudaCompilerConfig
    from KunQuant.Driver import KunCompilerConfig

    exe = compile_func(f,
                        KunCompilerConfig(input_layout="TS",
                                            output_layout="TS"),
                        CudaCompilerConfig(gpu_arch="sm_80"))
    executor = KunMLIR.Executor()                       # default stream
    out = executor.runGraph(exe, {"a": cp_a, "b": cp_b})  # length auto-inferred
    executor.synchronize()

Multi-Function compile (CPU `cfake.compileit` shape)::

    from KunQuant.jit.cuda import compileit, CudaCompilerConfig
    from KunQuant.Driver import KunCompilerConfig

    kcfg = KunCompilerConfig(input_layout="TS", output_layout="TS")
    ccfg = CudaCompilerConfig(gpu_arch="sm_80")
    lib = compileit([("mod1", f1, kcfg), ("mod2", f2, kcfg)],
                     "my_lib", ccfg)
    exe = lib.getModule("mod1")
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple

from KunQuant.jit import KunMLIR

from KunQuant.Driver import KunCompilerConfig, optimize, post_optimize
from KunQuant.Op import Input, Output, MayRequireWholeTime
from KunQuant.passes import do_partition
from KunQuant.passes.InferWindow import infer_window
from KunQuant.Stage import Function
from KunQuant.passes.CodegenMLIR import TargetSpec, translate_function


# Sentinel passed via kunir.func's `unreliable_count` attribute to mean
# "this partition needs the full time history; the runtime must launch
# it as a single chunk".  Kept in sync with the kunir verifier (which
# only allows -1 or non-negative) and the CUDA runtime's `computeChunkPlan`.
_WHOLE_TIME_UNRELIABLE = -1


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
    """Compile- / link-time knobs that are shared across every Function
    in a `Library`.  Per-Function graph-rewriting knobs (dtype,
    blocking_len, partition_factor, layout, pass options) live in
    `KunQuant.Driver.KunCompilerConfig` instead — the same dataclass
    the CPU path uses.
    """
    gpu_arch:    str = "sm_80"

    # kunir.target_spec — graph-wide for v0.  `vector_size` is taken
    # from the per-Function `KunCompilerConfig.blocking_len` at compile
    # time (the two are the same concept on GPU).
    occupancy:     int = 1
    warps_per_cta: int = 4
    smem_size:     int = 49152

    # LLVM optimization level (forwarded to #nvvm.target<O = ...>).
    opt_level:     int  = 3
    # Path to the CUDA toolkit (where libdevice.10.bc + ptxas live).
    # Empty → upstream search: CUDA_HOME / CUDA_PATH / standard locations.
    toolkit_path:  str  = ""


def _resolve_vector_size(kcfg: KunCompilerConfig) -> int:
    """On GPU `vector_size` (kunir target_spec) is the same as
    `blocking_len` from the per-Function config.  Default to 1 (scalar
    kunir) if the user didn't specify."""
    return 1 if kcfg.blocking_len is None else int(kcfg.blocking_len)


def _gpu_pass_options(kcfg: KunCompilerConfig) -> dict:
    """`Driver.optimize`'s `options` dict for the GPU path.

    `blocking_len` is needed by some decompose paths (it's also the
    skip-list / naive cost-model knob).  `kcfg.options` flows through
    verbatim — including `no_fast_stat`, `opt_reduce`, `fast_log`,
    all of which the GPU lowering now supports.

    `no_skip_list=True` is forced unconditionally and overrides any
    user-provided value: the kunir codegen has no lowering for
    `SkipList*` ops, so the naive `ForeachBackWindow + Reduce*` path
    is the only one that lowers on GPU.
    """
    opts: dict = {"blocking_len": _resolve_vector_size(kcfg)}
    if kcfg.options:
        opts.update(kcfg.options)
    opts["no_skip_list"] = True
    # Pipeline lowering doesn't know about ExpMovingAvg or the
    # WindowedLinearRegression* family — turn on the Accumulator-based
    # expansion pass instead.
    opts["experimental_expand"] = True
    return opts


def _to_dtype_token(dtype: str) -> str:
    if dtype == "float":  return "f32"
    if dtype == "double": return "f64"
    raise ValueError(f"compile_func: unsupported dtype '{dtype}' "
                       f"(supported: float, double — kunir today only "
                       f"lowers float on GPU)")


def _validate_kun_cfg(kcfg: KunCompilerConfig) -> None:
    """GPU path only supports TS layout on both input and output (kunir
    runtime is TS-major).  dtype must be a kunir-supported token."""
    if kcfg.input_layout != "TS":
        raise ValueError(
            f"GPU backend only supports input_layout='TS', got "
            f"{kcfg.input_layout!r}")
    if kcfg.output_layout != "TS":
        raise ValueError(
            f"GPU backend only supports output_layout='TS', got "
            f"{kcfg.output_layout!r}")
    if kcfg.dtype not in ("float", "double"):
        raise ValueError(
            f"KunCompilerConfig.dtype must be 'float' or 'double', got "
            f"{kcfg.dtype!r}")


def _graph_io_names(f: Function):
    """User-facing graph inputs/outputs.  Captured BEFORE optimize +
    do_partition because those passes mutate `f` and may scatter the
    Input/Output ops across multiple sub-Functions (some of which then
    look like 'TEMP' from the partition's POV but stay user-visible at
    the graph boundary)."""
    ins  = [op.attrs["name"] for op in f.ops if isinstance(op, Input)]
    outs = [op.attrs["name"] for op in f.ops if isinstance(op, Output)]
    if not ins:
        raise ValueError("compile_func: function has no Input ops")
    if not outs:
        raise ValueError("compile_func: function has no Output ops")
    return ins, outs


def _run_full_pipeline(f: Function, kcfg: KunCompilerConfig):
    """Same pass pipeline the CPU `compileit` runs:

        optimize  →  do_partition  →  post_optimize

    Returns the list of post-partition Functions that the translator
    should walk (one kunir.func per Function).  Mutates `f` in place.
    """
    options = _gpu_pass_options(kcfg)
    optimize(f, options)
    _mainf, impl = do_partition(f, kcfg.partition_factor, options)
    post_optimize(impl, options)
    return impl


def _translate_partitions(impl, kcfg: KunCompilerConfig,
                            ccfg: CudaCompilerConfig):
    """Emit one kunir.func per partitioned Function into a single
    KunMLIR module (single `gpu.module` with N siblings).  Cross-
    partition buffers stitch up automatically because each impl's
    Input/Output names match the producing/consuming partition's
    Output/Input names.

    Cross-sectional partitions (currently: cs_rank) bypass the kunir
    pipeline entirely — `translate_function` returns a descriptor and
    we collect those into `external_kernels`, which the C++ side
    appends to the executable's kernel list without ever generating
    LLVM IR / PTX for them.

    Returns (ModuleOp, list[dict]) — the second element is the list
    of external-kernel descriptors to forward to KunMLIR.compile.
    """
    target = TargetSpec(occupancy=ccfg.occupancy,
                          warps_per_cta=ccfg.warps_per_cta,
                          smem_size=ccfg.smem_size,
                          vector_size=_resolve_vector_size(kcfg))
    ir = KunMLIR.IRBuilder()
    dtype = _to_dtype_token(kcfg.dtype)
    externals = []
    for sub in impl:
        # Per-partition warmup: max windowed-chain depth from any input
        # to any output of THIS partition.  Earlier partitions have already
        # written their (post-warmup) values into the shared device buffers
        # by the time this kernel runs, so we don't accumulate their
        # unreliable counts here.  infer_window walks back to Input ops
        # of the partition; cross-partition deps stop at those Inputs.
        # If any op in this partition requires the whole time history,
        # override the inferred warmup with the sentinel so the runtime
        # collapses this kernel to a single chunk.
        if any(isinstance(op, MayRequireWholeTime)
                and op.is_whole_time_required()
                for op in sub.ops):
            per_kernel_unreliable = _WHOLE_TIME_UNRELIABLE
        else:
            per_kernel_unreliable = max(infer_window(sub).values(), default=0)
        ext = translate_function(sub, target, ir, dtype=dtype,
                                   unreliable_count=per_kernel_unreliable)
        if ext is not None:
            externals.append(ext)
    return ir.finish(), externals


def compile_func(f: Function, kcfg: KunCompilerConfig,
                   ccfg: CudaCompilerConfig) -> KunMLIR.Executable:
    """Compile a single KunQuant Function to a GPU `KunMLIR.Executable`.

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
    _validate_kun_cfg(kcfg)

    toolkit_path = find_cuda_toolkit(ccfg.toolkit_path)

    graph_inputs, graph_outputs = _graph_io_names(f)
    impl = _run_full_pipeline(f, kcfg)
    mod, externals = _translate_partitions(impl, kcfg, ccfg)

    return KunMLIR.compile(
        mod,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        gpu_arch=ccfg.gpu_arch,
        opt_level=ccfg.opt_level,
        toolkit_path=toolkit_path,
        external_kernels=externals,
        # Forwarded for the no-JIT-kernel case: when every partition
        # is external (e.g. a graph that is just `cs_rank(a)`), the
        # MLIR module is empty and `data.warpsPerCta` would otherwise
        # default to 1 — but the cs_rank launch uses it to size
        # blockDim, so feed the config value through.
        warps_per_cta=ccfg.warps_per_cta,
    )


class Library:
    """Bag of named `KunMLIR.Executable`s, mirroring the CPU `kr.Library`
    shape so callers can compile multiple Functions in one go and look
    them up by name.  Returned by the multi-Function `compileit` below.
    """
    def __init__(self, libname: str = "") -> None:
        self.libname = libname
        self._modules: dict = {}

    def getModule(self, name: str) -> KunMLIR.Executable:
        if name not in self._modules:
            raise RuntimeError(
                f"Library.getModule: no module named '{name}' "
                f"(have: {sorted(self._modules)})")
        return self._modules[name]

    @property
    def names(self):
        """All compiled module names in registration order."""
        return list(self._modules.keys())

    def _add(self, name: str, exe: KunMLIR.Executable) -> None:
        if name in self._modules:
            raise RuntimeError(
                f"Library: duplicate module name '{name}'")
        self._modules[name] = exe


def compileit(
    funclist: List[Tuple[str, Function, KunCompilerConfig]],
    libname: str,
    compiler_config: CudaCompilerConfig,
) -> Library:
    """Compile a list of `(name, Function, KunCompilerConfig)` tuples
    into a `Library`, mirroring the shape of
    `KunQuant.jit.cfake.compileit(func, libname, compiler_config)`.

    Each entry's third element is the per-Function `KunCompilerConfig`
    (dtype / blocking_len / partition_factor / layout / pass options);
    `compiler_config` is the GPU-wide `CudaCompilerConfig` applied to
    every entry.  cfake's other arguments (`tempdir`, `keep_files`,
    `load`) don't apply to the GPU path and are intentionally absent.

    Returns a `Library` keyed by the tuple's `name`; look up individual
    kernels via `lib.getModule(name)`.
    """
    lib = Library(libname=libname)
    for name, f, kcfg in funclist:
        lib._add(name, compile_func(f, kcfg, compiler_config))
    return lib


def to_mlir(f: Function, kcfg: KunCompilerConfig,
              ccfg: CudaCompilerConfig) -> KunMLIR.ModuleOp:
    """Run the same passes + translator as `compile_func`, but return
    the KunMLIR module before PTX/CUBIN.  External (cs_rank) partitions
    are absent from the returned module — they never become kunir
    ops.  Useful for debugging the IR.  Mutates `f` in place (same
    as `compile_func`)."""
    _validate_kun_cfg(kcfg)
    _graph_io_names(f)              # raises if no Input / Output ops
    impl = _run_full_pipeline(f, kcfg)
    mod, _externals = _translate_partitions(impl, kcfg, ccfg)
    return mod
