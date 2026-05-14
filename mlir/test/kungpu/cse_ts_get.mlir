// RUN: %kun-opt --pass-pipeline='builtin.module(gpu.module(kunir.func(kunir-to-kungpu,cse)))' %s | %FileCheck %s
//
// Verify that the CSE pass placed between `kunir-to-kungpu` and
// `windowed-temp-memory-planning` deduplicates identical kungpu.ts.get
// loads.  kungpu.ts.get is marked Pure, so CSE collapses any pair of
// reads with the same (handle, offset) operands.
//
// Two distinct kunir.back_ref ops on the same windowed_output at the
// same window lower to two ts.get %wt[%c5_i32] inside the outer time
// loop; after CSE only one survives.

gpu.module @kungpu_kernels {
  // CHECK-LABEL: kunir.func @two_back_refs
  kunir.func @two_back_refs(%a: !kunir.ts<f32, inf>)
      inputs {%a = "a"}
      outputs {"out"}
      target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 5
      -> !kunir.ts<f32, 1> {
    %r1 = kunir.back_ref %a [window = 5] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 1>
    %r2 = kunir.back_ref %a [window = 5] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 1>
    %sum = kunir.add %r1, %r2 : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
    kunir.return %sum : !kunir.ts<f32, 1>
  }
}

// One load of %a at offset 0 for the windowed_output fill (outer loop),
// and exactly ONE load of the windowed_temp at offset 5 (the two
// back_refs collapsed via CSE) — for a total of two ts.get ops.
//
// CHECK:       kungpu.ts.get
// CHECK-NOT:   kungpu.ts.get
