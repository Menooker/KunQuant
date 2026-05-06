// RUN: %kun-opt %s | %FileCheck %s
// RUN: %kun-opt %s | %kun-opt | %FileCheck %s

// CHECK-LABEL: kunir.func @test_non_void
// CHECK-SAME: (%[[A:.*]]: !kunir.ts<f32, inf>, %[[B:.*]]: !kunir.ts<f32, inf>)
// CHECK:      inputs {%[[A]] = "close", %[[B]] = "vol"}
// CHECK:      outputs {"alpha"}
// CHECK:      target {occupancy = 2, warps_per_cta = 4, smem_size = 49152, vector_size = 1}
// CHECK:      -> !kunir.ts<f32, 1>
kunir.func @test_non_void(%close: !kunir.ts<f32, inf>, %vol: !kunir.ts<f32, inf>)
    inputs {%close = "close", %vol = "vol"}
    outputs {"alpha"}
    target {occupancy = 2, warps_per_cta = 4, smem_size = 49152, vector_size = 1}
    -> !kunir.ts<f32, 1> {
  %sum = kunir.add %close, %vol : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  kunir.return %sum : !kunir.ts<f32, 1>
}

// Void form: one input, one output — both are function args.
// CHECK-LABEL: kunir.func @test_void
// CHECK-SAME: (%[[IN:.*]]: !kunir.ts<f32, inf>, %[[OUT:.*]]: !kunir.ts<f32, 1>)
// CHECK:      inputs {%[[IN]] = "close"}
// CHECK:      outputs {%[[OUT]] = "alpha"}
// CHECK:      target {occupancy = 1, warps_per_cta = 2, smem_size = 0, vector_size = 1}
// CHECK-NOT:  ->
kunir.func @test_void(%close: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%close = "close"}
    outputs {%out = "alpha"}
    target {occupancy = 1, warps_per_cta = 2, smem_size = 0, vector_size = 1} {
  kunir.return
}

// Void form: two inputs, two outputs — all four are function args.
// CHECK-LABEL: kunir.func @test_void_multi_output
// CHECK-SAME: (%[[I0:.*]]: !kunir.ts<f32, inf>, %[[I1:.*]]: !kunir.ts<f32, inf>, %[[O0:.*]]: !kunir.ts<f32, 1>, %[[O1:.*]]: !kunir.ts<f32, 1>)
// CHECK:      inputs {%[[I0]] = "close", %[[I1]] = "vol"}
// CHECK:      outputs {%[[O0]] = "alpha1", %[[O1]] = "alpha2"}
// CHECK:      target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1}
// CHECK-NOT:  ->
kunir.func @test_void_multi_output(
    %close: !kunir.ts<f32, inf>, %vol: !kunir.ts<f32, inf>,
    %out1: !kunir.ts<f32, 1>, %out2: !kunir.ts<f32, 1>)
    inputs {%close = "close", %vol = "vol"}
    outputs {%out1 = "alpha1", %out2 = "alpha2"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} {
  kunir.return
}

// Non-void multi-result.
// CHECK-LABEL: kunir.func @test_multi_result
kunir.func @test_multi_result(%input: !kunir.ts<f64, inf>)
    inputs {%input = "input"}
    outputs {"sum", "maxval"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 16384, vector_size = 1}
    -> (!kunir.ts<f64, 1>, !kunir.ts<f64, 1>) {
  %w = kunir.windowed_output %input [length = 10] : !kunir.ts<f64, inf> -> !kunir.ts<f64, 10>
  %s, %m = kunir.for_each_back_window
      (%w : !kunir.ts<f64, 10>) [window = 10]
      (%val : !kunir.ts<f64, 1>)
      -> (!kunir.ts<f64, 1>, !kunir.ts<f64, 1>) {
    %radd = kunir.reduce_add %val : !kunir.ts<f64, 1>
    %rmax = kunir.reduce_max %val : !kunir.ts<f64, 1>
    kunir.yield %radd, %rmax : !kunir.ts<f64, 1>, !kunir.ts<f64, 1>
  }
  kunir.return %s, %m : !kunir.ts<f64, 1>, !kunir.ts<f64, 1>
}
