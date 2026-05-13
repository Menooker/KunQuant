#!/usr/bin/env python3
"""End-to-end test for the windowed_temp lowering across both placements
the memory-planning pass can choose:

    sum_window(a, b, N)[t][s] = sum_{i=0}^{N-1} ( a[t-i][s] + b[t-i][s] )

is compiled twice — once with a small N (fits in shared memory) and once
with a large N (spills to local memory) — and each run is checked
against a numpy reference.  Memory-planning's per-block budget for our
target_spec is

    bytes / windowed_temp = N * (warps_per_cta * 32) * vector_size * 4
                          = N * 128 * 4         (warps_per_cta = 4)
                          = 512 * N

so with smem_size = 49152 the cutoff is N ≤ 96 → smem, N > 96 → local.
We pick N = 5 and N = 200 to bracket that.
"""

from __future__ import annotations
import argparse
import sys
import textwrap

import numpy as np


def build_ir(N: int, warps_per_cta: int = 4, smem_size: int = 49152) -> str:
    """A minimal kunir program that computes a rolling sum of (a + b)."""
    return textwrap.dedent(f"""
gpu.module @kungpu_kernels {{
  kunir.func @sum_window(%a: !kunir.ts<f32, inf>, %b: !kunir.ts<f32, inf>)
      inputs {{%a = "a", %b = "b"}}
      outputs {{"out"}}
      target {{occupancy = 1, warps_per_cta = {warps_per_cta}, smem_size = {smem_size}, vector_size = 1}} unreliable_count = 0
      -> !kunir.ts<f32, 1> {{
    %c = kunir.add %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
    %w = kunir.windowed_output %c [length = {N}] : !kunir.ts<f32, 1> -> !kunir.ts<f32, {N}>
    %total = kunir.for_each_back_window
        (%w : !kunir.ts<f32, {N}>) [window = {N}]
        (%cur : !kunir.ts<f32, 1>)
        -> (!kunir.ts<f32, 1>) {{
      %s = kunir.reduce_add %cur : !kunir.ts<f32, 1>
      kunir.yield %s : !kunir.ts<f32, 1>
    }}
    kunir.return %total : !kunir.ts<f32, 1>
  }}
}}
""").strip()


def reference_sum_window(a: np.ndarray, b: np.ndarray, N: int) -> np.ndarray:
    """CPU rolling-sum of (a + b) with window length N along axis 0.
    Output for t < N-1 is undefined; we fill nan there and skip it
    when comparing.
    """
    c = a + b
    T, S = c.shape
    out = np.empty((T, S), dtype=np.float32)
    out[:N - 1] = np.nan
    cumsum = np.cumsum(c, axis=0, dtype=np.float64)  # higher-precision ref
    out[N - 1] = cumsum[N - 1]
    if T > N:
        out[N:] = (cumsum[N:] - cumsum[:-N])
    return out


def assert_planning(N: int, warps_per_cta: int, smem_size: int,
                     expected: str) -> None:
    """Sanity-check our N choices against the memory-planning formula
    so the test self-documents which placement it exercises."""
    bytes_per_buf = N * warps_per_cta * 32 * 1 * 4   # vector_size=1, f32
    fits_smem = bytes_per_buf <= smem_size
    actual = "smem" if fits_smem else "local"
    if actual != expected:
        raise AssertionError(
            f"N={N} ({bytes_per_buf} bytes) would land in '{actual}', "
            f"but the test wanted '{expected}' (smem budget {smem_size}).")


def run_one(N: int, expected_placement: str, target: str,
              warps_per_cta: int = 4, smem_size: int = 49152,
              T: int = 64, S: int = 2048) -> int:
    from KunQuant.jit import KunMLIR
    import cupy as cp
    from KunQuant.jit.cuda import find_cuda_toolkit

    print(f"=== N = {N}  ({expected_placement} temp buffer) ===")
    assert_planning(N, warps_per_cta, smem_size, expected_placement)

    ir = build_ir(N, warps_per_cta=warps_per_cta, smem_size=smem_size)
    mod = KunMLIR.parse(ir)
    exe = KunMLIR.compile(mod,
                            graph_inputs=["a", "b"],
                            graph_outputs=["out"],
                            gpu_arch=target, opt_level=3,
                            toolkit_path=find_cuda_toolkit())
    print(f"  kernels={exe.kernel_names}  warps_per_cta={exe.warps_per_cta}  "
           f"vector_size={exe.vector_size}  cubin={len(exe.cubin)} bytes")

    # Random input.  T must be > N so we have at least one valid window.
    if T <= N:
        T = N + 32
    rng = np.random.default_rng(0)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    a   = cp.asarray(a_h)
    b   = cp.asarray(b_h)
    out = cp.zeros((T, S), dtype=cp.float32)

    executor = KunMLIR.Executor()
    executor.runGraph(exe, {"a": a, "b": b, "out": out})
    out_h = cp.asnumpy(out)            # implicitly waits via stream 0

    expected = reference_sum_window(a_h, b_h, N)

    # Only the t >= N-1 region is well-defined.
    diff = np.abs(out_h[N - 1:] - expected[N - 1:])
    max_abs = float(diff.max())
    # Tolerance scales with N: each output is a sum of N IID N(0,1)
    # samples, so its magnitude is ~sqrt(N), and float32 ULP-style error
    # accumulates roughly like N * eps.
    atol = max(1e-3, 5e-7 * N)
    if max_abs > atol:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  FAIL: max |Δ| = {max_abs:.3e} > {atol:.0e} at "
               f"{idx} (out_h={out_h[N-1:][idx]:.6g} vs "
               f"expected={expected[N-1:][idx]:.6g})", file=sys.stderr)
        return 1
    print(f"  ok — max |Δ| = {max_abs:.3e} (atol={atol:.0e}, "
           f"shape={(T - N + 1, S)} validated cells)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="sm_120")
    ap.add_argument("-T", "--time-length", type=int, default=64)
    ap.add_argument("-S", "--num-stocks", type=int, default=2048)
    args = ap.parse_args()

    import cupy as cp
    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    rc = 0
    rc |= run_one(N=5,   expected_placement="smem",
                   target=args.target, T=args.time_length, S=args.num_stocks)
    print()
    rc |= run_one(N=200, expected_placement="local",
                   target=args.target,
                   T=max(args.time_length, 256), S=args.num_stocks)
    return rc


if __name__ == "__main__":
    sys.exit(main())
