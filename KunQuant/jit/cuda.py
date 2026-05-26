"""Compatibility shim for the optional KunQuant-MLIR CUDA backend."""

try:
    from KunQuantMLIR.jit_cuda import *  # noqa: F401,F403
except ModuleNotFoundError as e:
    if e.name and e.name.startswith("KunQuantMLIR"):
        raise ImportError(
            "KunQuant MLIR/CUDA backend is not installed. "
            "Install KunQuant-MLIR to use KunQuant.jit.cuda."
        ) from e
    raise
