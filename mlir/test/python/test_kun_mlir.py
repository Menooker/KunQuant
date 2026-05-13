#!/usr/bin/env python3
"""End-to-end test for the `KunMLIR` Python bindings.

  parse → to_string → lower_to_ptx (debug only) → compile → launch

Usage:
    PATH=$CUDA_BIN:$PATH PYTHONPATH=<build>/mlir/lib/Python \
        kun python test_kun_mlir.py [--target sm_120]
"""

from __future__ import annotations
import argparse
import sys
import textwrap


SAMPLE_KUNIR = textwrap.dedent("""
gpu.module @kungpu_kernels {
  kunir.func @test_addsum(%a: !kunir.ts<f32, inf>, %b: !kunir.ts<f32, inf>)
      inputs {%a = "a", %b = "b"}
      outputs {"sum"}
      target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} unreliable_count = 0
      -> !kunir.ts<f32, 1> {
    %s = kunir.add %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
    kunir.return %s : !kunir.ts<f32, 1>
  }
}
""").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sm_120",
                     help="GPU compute capability (e.g. sm_120, sm_90, sm_80)")
    ap.add_argument("-T", "--time-length", type=int, default=64)
    ap.add_argument("-S", "--num-stocks", type=int, default=2048)
    args = ap.parse_args()

    from KunQuant.jit import KunMLIR
    import cupy as cp
    import numpy as np
    from KunQuant.jit.cuda import find_cuda_toolkit

    # Force-initialise the CUDA driver + create the primary context now,
    # so subsequent KunMLIR.compile() / Executor.runGraph() find one.
    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    print(f"=== parse + to_string ===")
    mod = KunMLIR.parse(SAMPLE_KUNIR)
    text = mod.to_string()
    assert "kunir.func @test_addsum" in text, "module text missing kunir.func"
    print("ok — module round-trips through parse/to_string")

    toolkit = find_cuda_toolkit()

    print()
    print(f"=== lower_to_ptx (target={args.target}, O3, debug only) ===")
    # Debug entry point — same lowering pipeline as compile() but stops
    # at PTX text via gpu-module-to-binary{format=isa}.  Mutates `mod`
    # (replaces the gpu.module with a gpu.binary), so we re-parse for
    # the main compile step below.
    ptx = KunMLIR.lower_to_ptx(mod, gpu_arch=args.target, opt_level=3,
                                  toolkit_path=toolkit)
    assert "test_addsum" in ptx
    print(f"ok — produced {len(ptx)} bytes of PTX text")

    print()
    print(f"=== compile (all-in-one) ===")
    mod2 = KunMLIR.parse(SAMPLE_KUNIR)
    exe = KunMLIR.compile(mod2,
                            graph_inputs=["a", "b"],
                            graph_outputs=["sum"],
                            gpu_arch=args.target, opt_level=3,
                            toolkit_path=toolkit)
    print(f"  kernel_names           = {exe.kernel_names}")
    print(f"  num_kernels            = {exe.num_kernels}")
    print(f"  launch_order           = {exe.launch_order}")
    print(f"  num_buffers            = {exe.num_buffers}")
    print(f"  peak_intermediate_slots= {exe.peak_intermediate_slots}")
    print(f"  input_names            = {exe.input_names}")
    print(f"  output_names           = {exe.output_names}")
    print(f"  warps_per_cta          = {exe.warps_per_cta}")
    print(f"  vector_size            = {exe.vector_size}")
    print(f"  cubin bytes            = {len(exe.cubin)}")
    assert exe.kernel_names == ["test_addsum"]
    assert exe.num_kernels == 1
    assert exe.launch_order == [0]
    assert exe.num_buffers == 3      # a, b, sum
    assert exe.peak_intermediate_slots == 0  # no intermediates
    assert exe.input_names  == ["a", "b"]
    assert exe.output_names == ["sum"]
    assert exe.warps_per_cta == 4
    assert exe.vector_size   == 1

    # Run the kernel for two num_stocks values:
    #  - one that's a multiple of (warps_per_cta * 32 * vector_size) — no
    #    tail block;
    #  - one that isn't — exercises the active-thread guard inserted by
    #    convert-kungpu-to-llvm phase 1.
    block_x = exe.warps_per_cta * 32 * exe.vector_size
    rng = np.random.default_rng(0)
    rc = 0
    for label, S in [("aligned", args.num_stocks),
                      ("unaligned (tail block)",
                       args.num_stocks + (block_x // 2 + 7))]:
        T = args.time_length
        print()
        is_aligned = (S % block_x == 0)
        print(f"=== launch ({T} × {S}) — {label}, "
               f"S % {block_x} = {S % block_x}, "
               f"aligned={is_aligned} ===")
        a_h = rng.standard_normal((T, S), dtype=np.float32)
        b_h = rng.standard_normal((T, S), dtype=np.float32)
        a   = cp.asarray(a_h)
        b   = cp.asarray(b_h)
        out = cp.zeros((T, S), dtype=cp.float32)
        executor = KunMLIR.Executor()
        executor.runGraph(exe, {"a": a, "b": b, "sum": out})
        # No explicit synchronize: default-stream Executor + cupy's
        # default stream → cp.asnumpy's D2H memcpy goes on the same
        # stream and waits for our kernels.  See test_multi_kernel.py
        # for the case where sync IS required (non-blocking user stream).
        out_h = cp.asnumpy(out)
        expected = a_h + b_h
        if not np.allclose(out_h, expected, atol=1e-5):
            diff = np.abs(out_h - expected)
            print(f"  FAIL — max abs diff {diff.max()}, "
                    f"argmax @ {np.unravel_index(diff.argmax(), diff.shape)}",
                    file=sys.stderr)
            rc = 1
        else:
            print(f"  ok — output matches a + b on every (t, s) cell "
                   f"({T*S} cells)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
