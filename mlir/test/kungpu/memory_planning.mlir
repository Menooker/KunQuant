// RUN: %kun-opt --kungpu-memory-planning %s | %FileCheck %s
//
// Default pass parameters: total_smem=49152 bytes, occupancy=1,
//                          num_threads=32, vec=1 → budget=49152 bytes
//
// Buffer cost (f32=4 bytes): bytes = N * 32 * 1 * 4 = N * 128
//   N=3   →   384 bytes
//   N=5   →   640 bytes
//   N=10  →  1280 bytes
//   N=400 → 51200 bytes  (> 49152)
//   N=500 → 64000 bytes  (> 49152)
//
// Case 1 – all smem:   N=3  (384) + N=5  (640) + N=10  (1280) = 2304  ≤ budget
// Case 2 – mixed:      N=5  (640) → smem; N=400 (51200) → 640+51200 > budget → local
// Case 3 – all local:  N=400 (51200) > budget → local; N=500 (64000) > budget → local
//
// The pass sorts ops by ascending N before assigning, so declaration order
// in the IR does not affect the assignment.

// -----------------------------------------------------------------------
// Case 1: all three buffers fit in shared memory
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @test_all_smem
func.func @test_all_smem() {
  // Declared in reverse order to verify sort-by-N behaviour.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 10> {kungpu.smem = true}
  %c = kungpu.windowed_temp : !kunir.ts<f32, 10>
  // CHECK-DAG: kungpu.windowed_temp : <f32, 3> {kungpu.smem = true}
  %a = kungpu.windowed_temp : !kunir.ts<f32, 3>
  // CHECK-DAG: kungpu.windowed_temp : <f32, 5> {kungpu.smem = true}
  %b = kungpu.windowed_temp : !kunir.ts<f32, 5>
  return
}

// -----------------------------------------------------------------------
// Case 2: small buffer goes to smem, large buffer spills to local memory
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @test_mixed
func.func @test_mixed() {
  // N=400 (51200 bytes) is declared first but sorted after N=5 (640 bytes).
  // N=5 takes 640 bytes of the 49152-byte budget; N=400 would need 51200
  // more, which exceeds the remaining 48512 bytes → local.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 400> {kungpu.smem = false}
  %big = kungpu.windowed_temp : !kunir.ts<f32, 400>
  // CHECK-DAG: kungpu.windowed_temp : <f32, 5> {kungpu.smem = true}
  %small = kungpu.windowed_temp : !kunir.ts<f32, 5>
  return
}

// -----------------------------------------------------------------------
// Case 3: every buffer exceeds the budget on its own → all local memory
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @test_all_local
func.func @test_all_local() {
  // N=400 → 51200 bytes > 49152 (budget), so smem=false.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 400> {kungpu.smem = false}
  %a = kungpu.windowed_temp : !kunir.ts<f32, 400>
  // N=500 → 64000 bytes > 49152 (budget), so smem=false.
  // CHECK-DAG: kungpu.windowed_temp : <f32, 500> {kungpu.smem = false}
  %b = kungpu.windowed_temp : !kunir.ts<f32, 500>
  return
}
