// RUN: %kun-opt --kunir-to-kungpu %s | %FileCheck %s

// CHECK-LABEL: kunir.func @test_binary_lower
// Pure ts args at this stage; the runtime scalars (time_length / num_stocks /
// mask / chunk_size / warmup) are prepended later by convert-kungpu-to-llvm.
// CHECK-SAME: !kunir.ts<f32, inf>
// CHECK-SAME: !kunir.ts<f32, inf>
// Graph output buffers are full TS arrays; the per-op result window has
// already been materialized into the loop body.
// CHECK-SAME: !kunir.ts<f32, inf>
// CHECK-NOT: -> !kunir.ts
kunir.func @test_binary_lower(%a: !kunir.ts<f32, inf>, %b: !kunir.ts<f32, inf>)
    inputs {%a = "a", %b = "b"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 1> {
  // Outer for bounds come from the per-chunk lb/ub ops, not [0, T).
  // Both are operandless — they pull chunk_size / warmup / time_length
  // from gpu.func args at the kungpu-to-llvm stage.
  // CHECK:      %[[LB:.*]] = kungpu.time_lb
  // CHECK:      %[[UB:.*]] = kungpu.time_ub
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C1:.*]] = arith.constant 1 : index
  // outer-loop offset = 0 (i32) used by every gmem ts.get/put
  // CHECK:      %[[OFF:.*]] = arith.constant 0 : i32
  // CHECK:      scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[C1]]
  // CHECK:        kungpu.ts.get %{{.*}}[%[[OFF]]]
  // CHECK:        kungpu.ts.get %{{.*}}[%[[OFF]]]
  // CHECK:        arith.addf
  // CHECK:        kungpu.ts.put
  // CHECK-NOT:    kungpu.ts.put %{{.*}}[
  // CHECK-NOT:    kungpu.time_length
  %sum = kunir.add %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  kunir.return %sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_unary_lower
kunir.func @test_unary_lower(%x: !kunir.ts<f32, inf>)
    inputs {%x = "x"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 1> {
  // CHECK: math.absf
  %a = kunir.abs %x : !kunir.ts<f32, inf>
  kunir.return %a : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_windowed_sum
kunir.func @test_windowed_sum(%close: !kunir.ts<f32, inf>)
    inputs {%close = "close"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 1> {
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C1:.*]] = arith.constant 1 : index
  // CHECK:      %[[OFF0:.*]] = arith.constant 0 : i32
  // CHECK:      %[[WT:.*]] = kungpu.windowed_temp : <f32, 5>
  // CHECK:      scf.for %[[T:.*]] =
  // CHECK:        kungpu.ts.get %{{.*}}[%[[OFF0]]]
  // outer-loop ts.put has no offset operand
  // CHECK:        kungpu.ts.put %[[WT]], %{{[^[]+}} : <f32, 5>, f32
  // CHECK:        %[[WIN:.*]] = arith.constant 5 : index
  // window-loop offset = (window-1) - w  (oldest first)
  // CHECK:        %[[WM1:.*]] = arith.constant 4 : i32
  // CHECK:        scf.for %[[W:.*]] = %[[C0]] to %[[WIN]] step %[[C1]] iter_args
  // CHECK:          %[[WI:.*]] = arith.index_cast %[[W]] : index to i32
  // CHECK:          %[[OFFW:.*]] = arith.subi %[[WM1]], %[[WI]] : i32
  // CHECK:          kungpu.ts.get %[[WT]][%[[OFFW]]]
  // CHECK:          arith.addf
  %w = kunir.windowed_output %close [length = 5] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 5>
  %sum = kunir.for_each_back_window
      (%w : !kunir.ts<f32, 5>) [window = 5]
      (%cur : !kunir.ts<f32, 1>)
      -> (!kunir.ts<f32, 1>) {
    %s = kunir.reduce_add %cur : !kunir.ts<f32, 1>
    kunir.yield %s : !kunir.ts<f32, 1>
  }
  kunir.return %sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_computed_reduce
kunir.func @test_computed_reduce(%x: !kunir.ts<f32, inf>, %y: !kunir.ts<f32, inf>)
    inputs {%x = "x", %y = "y"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 1> {
  // CHECK:      %[[WX:.*]] = kungpu.windowed_temp : <f32, 3>
  // CHECK:      %[[WY:.*]] = kungpu.windowed_temp : <f32, 3>
  // CHECK:      scf.for
  // CHECK:        scf.for {{.*}} iter_args
  // CHECK:          %[[A:.*]] = kungpu.ts.get %[[WX]][%{{.*}}]
  // CHECK:          %[[B:.*]] = kungpu.ts.get %[[WY]][%{{.*}}]
  // CHECK:          %[[P:.*]] = arith.mulf %[[A]], %[[B]]
  // CHECK:          arith.addf {{.*}}, %[[P]]
  %wx = kunir.windowed_output %x [length = 3] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 3>
  %wy = kunir.windowed_output %y [length = 3] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 3>
  %sum = kunir.for_each_back_window
      (%wx : !kunir.ts<f32, 3>, %wy : !kunir.ts<f32, 3>) [window = 3]
      (%a : !kunir.ts<f32, 1>, %b : !kunir.ts<f32, 1>)
      -> (!kunir.ts<f32, 1>) {
    %prod = kunir.mul %a, %b : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
    %s = kunir.reduce_add %prod : !kunir.ts<f32, 1>
    kunir.yield %s : !kunir.ts<f32, 1>
  }
  kunir.return %sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_multi_reduce
// CHECK-SAME: (%[[IN:.*]]: !kunir.ts<f64, inf>, %[[OUT0:.*]]: !kunir.ts<f64, inf>, %[[OUT1:.*]]: !kunir.ts<f64, inf>)
kunir.func @test_multi_reduce(%input: !kunir.ts<f64, inf>)
    inputs {%input = "input"}
    outputs {"sum", "maxval"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} unreliable_count = 0
    -> (!kunir.ts<f64, 1>, !kunir.ts<f64, 1>) {
  // CHECK:      %[[WT:.*]] = kungpu.windowed_temp : <f64, 10>
  // CHECK:      scf.for %[[T:.*]] =
  // CHECK:        kungpu.ts.get %[[IN]][%{{.*}}]
  // CHECK:        kungpu.ts.put %[[WT]], %{{[^[]+}} : <f64, 10>, f64
  // CHECK:        %[[CST0:.*]] = arith.constant 0.0{{.*}} : f64
  // CHECK:        %[[NEGINF:.*]] = arith.constant 0xFFF0000000000000 : f64
  // CHECK:        %[[R:.*]]:2 = scf.for {{.*}} iter_args(%{{.*}} = %[[CST0]], %{{.*}} = %[[NEGINF]]) -> (f64, f64)
  // CHECK:          kungpu.ts.get %[[WT]][%{{.*}}]
  // CHECK:          arith.addf
  // CHECK:          arith.maximumf
  // CHECK:          scf.yield {{.*}}, {{.*}} : f64, f64
  // CHECK:        kungpu.ts.put %[[OUT0]], %[[R]]#0 : <f64, inf>, f64
  // CHECK:        kungpu.ts.put %[[OUT1]], %[[R]]#1 : <f64, inf>, f64
  %w = kunir.windowed_output %input [length = 10] : !kunir.ts<f64, inf> -> !kunir.ts<f64, 10>
  %sum, %max = kunir.for_each_back_window
      (%w : !kunir.ts<f64, 10>) [window = 10]
      (%val : !kunir.ts<f64, 1>)
      -> (!kunir.ts<f64, 1>, !kunir.ts<f64, 1>) {
    %s = kunir.reduce_add %val : !kunir.ts<f64, 1>
    %m = kunir.reduce_max %val : !kunir.ts<f64, 1>
    kunir.yield %s, %m : !kunir.ts<f64, 1>, !kunir.ts<f64, 1>
  }
  kunir.return %sum, %max : !kunir.ts<f64, 1>, !kunir.ts<f64, 1>
}
