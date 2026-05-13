// RUN: %kun-opt --kungpu-memory-planning %s | %FileCheck %s
//
// All three functions share the same target_spec:
//   smem_size = 49152 bytes (per-SM total), occupancy = 1
//   → per-block budget = 49152 / 1 = 49152 bytes
//   warps_per_cta = 1  →  num_threads = 32
//   vector_size = 1
//
// Buffer cost (f32 = 4 bytes): bytes = N * 32 * 1 * 4 = N * 128
//   N=3   →   384 bytes
//   N=5   →   640 bytes
//   N=10  →  1280 bytes
//   N=400 → 51200 bytes  (> 49152)
//   N=500 → 64000 bytes  (> 49152)
//
// Case 1 – all smem:   N=3 (384) + N=5 (640) + N=10 (1280) = 2304 ≤ 49152
// Case 2 – mixed:      N=5 (640) → smem; N=400 (51200) → 640+51200 > 49152 → local
// Case 3 – all local:  N=400 (51200) > 49152 → local; N=500 → local
//
// The pass sorts ops by ascending N before assigning, so declaration order
// in the IR does not affect the assignment.

// -----------------------------------------------------------------------
// Case 1: all three buffers fit in shared memory
// -----------------------------------------------------------------------

// CHECK-LABEL: kunir.func @test_all_smem
kunir.func @test_all_smem(%in: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%in = "in"}
    outputs {%out = "out"}
    target {occupancy = 1, warps_per_cta = 1, smem_size = 49152, vector_size = 1} unreliable_count = 0 {
  // Declared in reverse order to verify sort-by-N behaviour.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 10> {kungpu.smem = true}
  %c = kungpu.windowed_temp : !kunir.ts<f32, 10>
  // CHECK-DAG: kungpu.windowed_temp : <f32, 3> {kungpu.smem = true}
  %a = kungpu.windowed_temp : !kunir.ts<f32, 3>
  // CHECK-DAG: kungpu.windowed_temp : <f32, 5> {kungpu.smem = true}
  %b = kungpu.windowed_temp : !kunir.ts<f32, 5>
  kunir.return
}

// -----------------------------------------------------------------------
// Case 2: small buffer goes to smem, large buffer spills to local memory
// -----------------------------------------------------------------------

// CHECK-LABEL: kunir.func @test_mixed
kunir.func @test_mixed(%in: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%in = "in"}
    outputs {%out = "out"}
    target {occupancy = 1, warps_per_cta = 1, smem_size = 49152, vector_size = 1} unreliable_count = 0 {
  // N=400 (51200 bytes) is declared first but sorted after N=5 (640 bytes).
  // N=5 takes 640 bytes; N=400 would need 51200 more, exceeding 48512 remaining.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 400> {kungpu.smem = false}
  %big = kungpu.windowed_temp : !kunir.ts<f32, 400>
  // CHECK-DAG: kungpu.windowed_temp : <f32, 5> {kungpu.smem = true}
  %small = kungpu.windowed_temp : !kunir.ts<f32, 5>
  kunir.return
}

// -----------------------------------------------------------------------
// Case 3: every buffer exceeds the budget on its own → all local memory
// -----------------------------------------------------------------------

// CHECK-LABEL: kunir.func @test_all_local
kunir.func @test_all_local(%in: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%in = "in"}
    outputs {%out = "out"}
    target {occupancy = 1, warps_per_cta = 1, smem_size = 49152, vector_size = 1} unreliable_count = 0 {
  // N=400 → 51200 bytes > 49152, smem=false.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 400> {kungpu.smem = false}
  %a = kungpu.windowed_temp : !kunir.ts<f32, 400>
  // N=500 → 64000 bytes > 49152, smem=false.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 500> {kungpu.smem = false}
  %b = kungpu.windowed_temp : !kunir.ts<f32, 500>
  kunir.return
}
