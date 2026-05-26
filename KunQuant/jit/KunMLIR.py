"""Compatibility shim for the optional KunQuant-MLIR extension module."""

from importlib import import_module as _import_module
import sys as _sys

try:
    _KunMLIR = _import_module("KunQuantMLIR.KunMLIR")
except ModuleNotFoundError as e:
    if e.name and e.name.startswith("KunQuantMLIR"):
        raise ImportError(
            "KunQuant MLIR extension is not installed. "
            "Install KunQuant-MLIR to use KunQuant.jit.KunMLIR."
        ) from e
    raise

_sys.modules[__name__] = _KunMLIR
