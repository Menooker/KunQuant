// RUN: %kun-opt --kunir-to-kungpu %s | %FileCheck %s

// CHECK-LABEL: func.func @test_binary_lower
// CHECK-SAME: !kunir.ts<f32, inf>
// CHECK-SAME: !kunir.ts<f32, inf>
// CHECK-SAME: !kunir.ts<f32, 1>
// CHECK-NOT: -> !kunir.ts
func.func @test_binary_lower(
    %a: !kunir.ts<f32, inf>,
    %b: !kunir.ts<f32, inf>
) -> !kunir.ts<f32, 1> {
  // CHECK:      %[[TL:.*]] = kungpu.time_length
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C1:.*]] = arith.constant 1 : index
  // CHECK:      scf.for %{{.*}} = %[[C0]] to %[[TL]] step %[[C1]]
  // CHECK:        kungpu.ts.get
  // CHECK:        kungpu.ts.get
  // CHECK:        arith.addf
  // CHECK:        kungpu.ts.put
  %sum = kunir.add %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  return %sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_unary_lower
func.func @test_unary_lower(%x: !kunir.ts<f32, inf>) -> !kunir.ts<f32, 1> {
  // CHECK: math.absf
  %a = kunir.abs %x : !kunir.ts<f32, inf>
  return %a : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_windowed_sum
func.func @test_windowed_sum(%close: !kunir.ts<f32, inf>) -> !kunir.ts<f32, 1> {
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[C1:.*]] = arith.constant 1 : index
  // CHECK:      %[[WT:.*]] = kungpu.windowed_temp : <f32, 5>
  // CHECK:      scf.for %[[T:.*]] =
  // CHECK:        kungpu.ts.get %arg0[%[[T]]]
  // CHECK:        kungpu.ts.put %[[WT]][%[[T]]]
  // CHECK:        %[[WIN:.*]] = arith.constant 5 : index
  // CHECK:        scf.for %{{.*}} = %[[C0]] to %[[WIN]] step %[[C1]] iter_args
  // CHECK:          kungpu.ts.get %[[WT]]
  // CHECK:          arith.addf
  %w = kunir.windowed_output %close [length = 5] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 5>
  %sum = kunir.for_each_back_window
      (%w : !kunir.ts<f32, 5>) [window = 5]
      (%cur : !kunir.ts<f32, 1>)
      -> (!kunir.ts<f32, 1>) {
    %s = kunir.reduce_add %cur : !kunir.ts<f32, 1>
    kunir.yield %s : !kunir.ts<f32, 1>
  }
  return %sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_computed_reduce
func.func @test_computed_reduce(
    %x: !kunir.ts<f32, inf>,
    %y: !kunir.ts<f32, inf>
) -> !kunir.ts<f32, 1> {
  // CHECK:      %[[WX:.*]] = kungpu.windowed_temp : <f32, 3>
  // CHECK:      %[[WY:.*]] = kungpu.windowed_temp : <f32, 3>
  // CHECK:      scf.for
  // CHECK:        scf.for {{.*}} iter_args
  // CHECK:          %[[A:.*]] = kungpu.ts.get %[[WX]]
  // CHECK:          %[[B:.*]] = kungpu.ts.get %[[WY]]
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
  return %sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_multi_reduce
// CHECK-SAME: (%[[IN:.*]]: !kunir.ts<f64, inf>, %[[OUT0:.*]]: !kunir.ts<f64, 1>, %[[OUT1:.*]]: !kunir.ts<f64, 1>)
func.func @test_multi_reduce(%input: !kunir.ts<f64, inf>) -> (!kunir.ts<f64, 1>, !kunir.ts<f64, 1>) {
  // CHECK:      %[[WT:.*]] = kungpu.windowed_temp : <f64, 10>
  // CHECK:      scf.for %[[T:.*]] =
  // CHECK:        kungpu.ts.get %[[IN]][%[[T]]]
  // CHECK:        kungpu.ts.put %[[WT]][%[[T]]]
  // CHECK:        %[[CST0:.*]] = arith.constant 0.0{{.*}} : f64
  // CHECK:        %[[NEGINF:.*]] = arith.constant 0xFFF0000000000000 : f64
  // CHECK:        %[[R:.*]]:2 = scf.for {{.*}} iter_args(%{{.*}} = %[[CST0]], %{{.*}} = %[[NEGINF]]) -> (f64, f64)
  // CHECK:          kungpu.ts.get %[[WT]]
  // CHECK:          arith.addf
  // CHECK:          arith.maximumf
  // CHECK:          scf.yield {{.*}}, {{.*}} : f64, f64
  // CHECK:        kungpu.ts.put %[[OUT0]][%[[T]]], %[[R]]#0 : <f64, 1>, f64
  // CHECK:        kungpu.ts.put %[[OUT1]][%[[T]]], %[[R]]#1 : <f64, 1>, f64
  %w = kunir.windowed_output %input [length = 10] : !kunir.ts<f64, inf> -> !kunir.ts<f64, 10>
  %sum, %max = kunir.for_each_back_window
      (%w : !kunir.ts<f64, 10>) [window = 10]
      (%val : !kunir.ts<f64, 1>)
      -> (!kunir.ts<f64, 1>, !kunir.ts<f64, 1>) {
    %s = kunir.reduce_add %val : !kunir.ts<f64, 1>
    %m = kunir.reduce_max %val : !kunir.ts<f64, 1>
    kunir.yield %s, %m : !kunir.ts<f64, 1>, !kunir.ts<f64, 1>
  }
  return %sum, %max : !kunir.ts<f64, 1>, !kunir.ts<f64, 1>
}
