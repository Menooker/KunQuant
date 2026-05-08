#!/usr/bin/env python3
"""End-to-end test for the v0 multi-kernel pipeline.

Builds a graph with two kernels chained through one intermediate buffer:

    add_kernel:    tmp = a + b
    scale_kernel:  out = tmp * c

graph_inputs  = ["a", "b", "c"]
graph_outputs = ["out"]
intermediate  = "tmp"  → 1 slot expected

Verifies the compile-time topology / slot plan, then runs the kernels and
checks the result against numpy.
"""

from __future__ import annotations
import argparse
import sys
import textwrap

import numpy as np


SAMPLE_KUNIR = textwrap.dedent("""
gpu.module @kungpu_kernels {
  kunir.func @add_kernel(%a: !kunir.ts<f32, inf>, %b: !kunir.ts<f32, inf>)
      inputs {%a = "a", %b = "b"}
      outputs {"tmp"}
      target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1}
      -> !kunir.ts<f32, 1> {
    %s = kunir.add %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
    kunir.return %s : !kunir.ts<f32, 1>
  }

  kunir.func @scale_kernel(%t: !kunir.ts<f32, inf>, %c: !kunir.ts<f32, inf>)
      inputs {%t = "tmp", %c = "c"}
      outputs {"out"}
      target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1}
      -> !kunir.ts<f32, 1> {
    %s = kunir.mul %t, %c : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
    kunir.return %s : !kunir.ts<f32, 1>
  }
}
""").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sm_120")
    ap.add_argument("-T", "--time-length", type=int, default=64)
    ap.add_argument("-S", "--num-stocks", type=int, default=2048)
    args = ap.parse_args()

    from KunQuant.jit import KunMLIR
    import cupy as cp
    from KunQuant.jit.cuda import find_cuda_toolkit

    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    print("=== compile two-kernel graph ===")
    mod = KunMLIR.parse(SAMPLE_KUNIR)
    exe = KunMLIR.compile(mod,
                            graph_inputs=["a", "b", "c"],
                            graph_outputs=["out"],
                            gpu_arch=args.target, opt_level=3,
                            toolkit_path=find_cuda_toolkit())

    print(f"  kernel_names           = {exe.kernel_names}")
    print(f"  num_kernels            = {exe.num_kernels}")
    print(f"  launch_order           = {exe.launch_order}")
    print(f"  num_buffers            = {exe.num_buffers}")
    print(f"  peak_intermediate_slots= {exe.peak_intermediate_slots}")
    print(f"  input_names            = {exe.input_names}")
    print(f"  output_names           = {exe.output_names}")

    # Topology checks.
    assert exe.num_kernels == 2, exe.num_kernels
    assert set(exe.kernel_names) == {"add_kernel", "scale_kernel"}, exe.kernel_names
    # Producer (add) must come before consumer (scale).
    add_pos = exe.launch_order.index(exe.kernel_names.index("add_kernel"))
    scl_pos = exe.launch_order.index(exe.kernel_names.index("scale_kernel"))
    assert add_pos < scl_pos, (exe.kernel_names, exe.launch_order)
    # 3 graph inputs + 1 graph output + 1 intermediate.
    assert exe.num_buffers == 5, exe.num_buffers
    # One intermediate ("tmp") → exactly one slot.
    assert exe.peak_intermediate_slots == 1, exe.peak_intermediate_slots
    assert exe.input_names  == ["a", "b", "c"]
    assert exe.output_names == ["out"]

    # === launch ===
    T, S = args.time_length, args.num_stocks
    rng = np.random.default_rng(0)
    a_h = rng.standard_normal((T, S), dtype=np.float32)
    b_h = rng.standard_normal((T, S), dtype=np.float32)
    c_h = rng.standard_normal((T, S), dtype=np.float32)
    a = cp.asarray(a_h)
    b = cp.asarray(b_h)
    c = cp.asarray(c_h)
    out = cp.zeros((T, S), dtype=cp.float32)

    print()
    print(f"=== launch ({T} × {S}) ===")
    exe.launch({"a": a, "b": b, "c": c, "out": out})
    cp.cuda.runtime.deviceSynchronize()
    out_h = cp.asnumpy(out)

    expected = (a_h + b_h) * c_h
    if not np.allclose(out_h, expected, atol=1e-5):
        diff = np.abs(out_h - expected)
        print(f"  FAIL — max abs diff {diff.max()}, "
                f"argmax @ {np.unravel_index(diff.argmax(), diff.shape)}",
                file=sys.stderr)
        return 1

    print(f"  ok — output matches (a+b)*c on every (t, s) cell ({T*S} cells)")

    # === second launch with different shape — exercises slot pool re-alloc ===
    T2, S2 = T // 2, S + 64
    a2 = cp.asarray(rng.standard_normal((T2, S2), dtype=np.float32))
    b2 = cp.asarray(rng.standard_normal((T2, S2), dtype=np.float32))
    c2 = cp.asarray(rng.standard_normal((T2, S2), dtype=np.float32))
    out2 = cp.zeros((T2, S2), dtype=cp.float32)

    print()
    print(f"=== launch ({T2} × {S2}) — different shape, slot pool re-alloc ===")
    exe.launch({"a": a2, "b": b2, "c": c2, "out": out2})
    cp.cuda.runtime.deviceSynchronize()
    out2_h = cp.asnumpy(out2)
    expected2 = (cp.asnumpy(a2) + cp.asnumpy(b2)) * cp.asnumpy(c2)
    if not np.allclose(out2_h, expected2, atol=1e-5):
        diff = np.abs(out2_h - expected2)
        print(f"  FAIL — max abs diff {diff.max()}", file=sys.stderr)
        return 1
    print(f"  ok — re-launched with new shape, output matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
