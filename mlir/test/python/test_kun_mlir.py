#!/usr/bin/env python3
"""End-to-end test for the `kun_mlir` Python bindings.

  parse → to_string → lower_to_ptx → ptx_to_cubin → compile → launch

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
      target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1}
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

    import kun_mlir
    import cupy as cp
    import numpy as np

    # Force-initialise the CUDA driver + create the primary context now,
    # so subsequent kun_mlir.compile() / Executable.launch() find one.
    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    print(f"=== parse + to_string ===")
    mod = kun_mlir.parse(SAMPLE_KUNIR)
    text = mod.to_string()
    assert "kunir.func @test_addsum" in text, "module text missing kunir.func"
    print("ok — module round-trips through parse/to_string")

    print()
    print(f"=== lower_to_ptx (target={args.target}, O3) ===")
    ptx = kun_mlir.lower_to_ptx(mod, target_cpu=args.target, opt_level=3)
    assert "test_addsum" in ptx
    print(f"ok — produced {len(ptx)} bytes of PTX text")

    print()
    print(f"=== ptx_to_cubin ({args.target}) ===")
    cubin = kun_mlir.ptx_to_cubin(ptx, gpu_arch=args.target)
    assert isinstance(cubin, bytes) and cubin[:4] == b"\x7fELF"
    print(f"ok — produced {len(cubin)} bytes of CUBIN (ELF magic verified)")

    print()
    print(f"=== compile (all-in-one) ===")
    # `mod` was already mutated by lower_to_ptx above; re-parse so compile()
    # gets a fresh kunir.func module.
    mod2 = kun_mlir.parse(SAMPLE_KUNIR)
    exe = kun_mlir.compile(mod2, target_cpu=args.target, opt_level=3)
    print(f"  kernel_name   = {exe.kernel_name}")
    print(f"  input_names   = {exe.input_names}")
    print(f"  output_names  = {exe.output_names}")
    print(f"  warps_per_cta = {exe.warps_per_cta}")
    print(f"  vector_size   = {exe.vector_size}")
    print(f"  cubin bytes   = {len(exe.cubin)}")
    assert exe.kernel_name == "test_addsum"
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
        exe.launch({"a": a, "b": b, "sum": out})
        cp.cuda.runtime.deviceSynchronize()
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
