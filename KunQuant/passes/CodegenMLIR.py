"""Compatibility shim for the optional KunQuant-MLIR codegen backend."""

try:
    from KunQuantMLIR.codegen_mlir import *  # noqa: F401,F403
except ModuleNotFoundError as e:
    if e.name and e.name.startswith("KunQuantMLIR"):
        raise ImportError(
            "KunQuant MLIR codegen backend is not installed. "
            "Install KunQuant-MLIR to use KunQuant.passes.CodegenMLIR."
        ) from e
    raise
