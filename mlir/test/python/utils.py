from __future__ import annotations

from typing import Optional, Tuple

from KunQuant.jit.env import get_cuda_compute_capability


def resolve_cuda_compute_capability(explicit_target: Optional[str] = None,
                                    fallback: str = "sm_80"
                                    ) -> Tuple[str, bool]:
    """Return `(gpu_arch, has_device)` for MLIR Python tests.

    Tests that can still cover compile-only behavior without a visible GPU use
    this helper to select a conservative fallback architecture and decide
    whether to skip runtime `Executable` / `runGraph` checks.
    """
    try:
        detected = get_cuda_compute_capability()
        return explicit_target or detected, True
    except RuntimeError:
        return explicit_target or fallback, False
