// RUN: %kun-opt --kunir-to-llvm %s | %FileCheck %s
//
// End-to-end smoke test for the kunir-to-llvm pipeline:
//   kunir-to-kungpu → memory-planning → convert-kungpu-to-llvm
//   → LICM → canonicalize → cse → scf-to-cf
//   → convert-gpu-to-nvvm (indexBitwidth=32)
//   → index/arith/cf/func to-llvm → reconcile-unrealized-casts.
//
// We verify that no kunir/kungpu/scf/gpu *ops* survive in the function
// body and that the original kunir.func becomes an llvm.func with
// nvvm.kernel tagging, inside the same gpu.module.

// CHECK-NOT: kunir.{{[a-z_.]+ }}
// CHECK-NOT: kungpu.{{[a-z_.]+ }}
// CHECK-NOT: scf.{{[a-z_]+}}
// CHECK-NOT: gpu.{{[a-z_]+ }}

// CHECK:       gpu.module @kungpu_kernels

// llvm.func with the (i32 time_len, i32 num_stocks, i32 mask, i32 chunk_size,
// i32 warmup, ptr...) signature, tagged as a kernel by convert-gpu-to-nvvm.
// CHECK-LABEL: llvm.func @test_addsum
// CHECK-SAME:    i32
// CHECK-SAME:    i32
// CHECK-SAME:    i32
// CHECK-SAME:    i32
// CHECK-SAME:    i32
// CHECK-SAME:    !llvm.ptr
// CHECK-SAME:    !llvm.ptr
// CHECK-SAME:    !llvm.ptr
//
// kunir-func metadata preserved as discardable attributes:
// CHECK-SAME:    kungpu.input_names = ["a", "b"]
// CHECK-SAME:    kungpu.output_names = ["sum"]
// CHECK-SAME:    kungpu.target_spec = #kunir<target_spec{
// CHECK-SAME:    nvvm.kernel

// gpu.thread_id / block_id / block_dim are now NVVM intrinsics.
// CHECK:       nvvm.read.ptx.sreg.tid.x
// CHECK:       nvvm.read.ptx.sreg.ctaid.x
// CHECK:       nvvm.read.ptx.sreg.ntid.x

// Branch-based control flow from scf-to-cf:
// CHECK-DAG:   llvm.br
// CHECK-DAG:   llvm.cond_br

// Lowered arithmetic + load/store from gmem:
// CHECK-DAG:   llvm.fadd
// CHECK-DAG:   llvm.getelementptr
// CHECK-DAG:   llvm.load
// CHECK-DAG:   llvm.store
// CHECK:       llvm.return

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
