// RUN: %kun-opt %s | %FileCheck %s
// RUN: %kun-opt %s | %kun-opt | %FileCheck %s

// Verify the kunir dialect type and ops parse and round-trip.

// CHECK-LABEL: func.func @test_ts_lookback_type
func.func @test_ts_lookback_type(
    // CHECK-SAME: !kunir.ts<f32, inf>
    %a: !kunir.ts<f32, inf>,
    // CHECK-SAME: !kunir.ts<f32, 1>
    %b: !kunir.ts<f32, 1>,
    // CHECK-SAME: !kunir.ts<f64, 10>
    %c: !kunir.ts<f64, 10>
) -> !kunir.ts<f32, 1> {
  return %b : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_binary_mismatched_lookbacks
func.func @test_binary_mismatched_lookbacks(
    %a: !kunir.ts<f32, 5>,
    %b: !kunir.ts<f32, 10>
) -> !kunir.ts<f32, 1> {
  // CHECK: kunir.add
  // CHECK-SAME: <f32, 5>, <f32, 10>
  %sum = kunir.add %a, %b : !kunir.ts<f32, 5>, !kunir.ts<f32, 10>
  // CHECK: kunir.sub
  %diff = kunir.sub %a, %b : !kunir.ts<f32, 5>, !kunir.ts<f32, 10>
  // CHECK: kunir.mul
  %prod = kunir.mul %sum, %diff : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
  return %prod : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_unary
func.func @test_unary(%x: !kunir.ts<f32, inf>) -> !kunir.ts<f32, 1> {
  // CHECK: kunir.abs
  %a = kunir.abs %x : !kunir.ts<f32, inf>
  // CHECK: kunir.sign
  %s = kunir.sign %a : !kunir.ts<f32, 1>
  return %s : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_windowed_output
func.func @test_windowed_output(%input: !kunir.ts<f32, inf>) -> !kunir.ts<f32, 10> {
  // CHECK: kunir.windowed_output
  // CHECK-SAME: length = 10
  %out = kunir.windowed_output %input [length = 10] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 10>
  return %out : !kunir.ts<f32, 10>
}

// CHECK-LABEL: func.func @test_cs_rank
func.func @test_cs_rank(%input: !kunir.ts<f32, inf>) -> !kunir.ts<f32, 1> {
  // CHECK: kunir.cs_rank
  %ranked = kunir.cs_rank %input : !kunir.ts<f32, inf>
  return %ranked : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_for_each_back_window_single
// Single input, single result.
func.func @test_for_each_back_window_single(%close: !kunir.ts<f32, 10>)
    -> !kunir.ts<f32, 1> {
  // CHECK: kunir.for_each_back_window
  // CHECK-SAME: [window = 5]
  %ts_sum = kunir.for_each_back_window
      (%close : !kunir.ts<f32, 10>) [window = 5]
      (%close_cur : !kunir.ts<f32, 1>)
      -> (!kunir.ts<f32, 1>) {
    // CHECK: kunir.reduce_add
    %s = kunir.reduce_add %close_cur : !kunir.ts<f32, 1>
    kunir.yield %s : !kunir.ts<f32, 1>
  }
  return %ts_sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_for_each_back_window_multi_input
// Two inputs, two results (one reduce per input).
func.func @test_for_each_back_window_multi_input(
    %close: !kunir.ts<f32, 20>,
    %vol:   !kunir.ts<f32, 20>)
    -> (!kunir.ts<f32, 1>, !kunir.ts<f32, 1>) {
  // CHECK: kunir.for_each_back_window
  %sum_c, %sum_v = kunir.for_each_back_window
      (%close : !kunir.ts<f32, 20>, %vol : !kunir.ts<f32, 20>) [window = 10]
      (%cc : !kunir.ts<f32, 1>, %vc : !kunir.ts<f32, 1>)
      -> (!kunir.ts<f32, 1>, !kunir.ts<f32, 1>) {
    // CHECK: kunir.reduce_add
    %sc = kunir.reduce_add %cc : !kunir.ts<f32, 1>
    // CHECK: kunir.reduce_add
    %sv = kunir.reduce_add %vc : !kunir.ts<f32, 1>
    kunir.yield %sc, %sv : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
  }
  return %sum_c, %sum_v : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_for_each_back_window_multi_reduce
// Single input, multiple reductions → multiple results.
func.func @test_for_each_back_window_multi_reduce(%input: !kunir.ts<f32, 20>)
    -> (!kunir.ts<f32, 1>, !kunir.ts<f32, 1>) {
  %sum_ts, %max_ts = kunir.for_each_back_window
      (%input : !kunir.ts<f32, 20>) [window = 10]
      (%val : !kunir.ts<f32, 1>)
      -> (!kunir.ts<f32, 1>, !kunir.ts<f32, 1>) {
    // CHECK: kunir.reduce_add
    %s = kunir.reduce_add %val : !kunir.ts<f32, 1>
    // CHECK: kunir.reduce_max
    %m = kunir.reduce_max %val : !kunir.ts<f32, 1>
    kunir.yield %s, %m : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
  }
  return %sum_ts, %max_ts : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
}

// CHECK-LABEL: func.func @test_for_each_back_window_inf
// inf lookback satisfies any window size.
func.func @test_for_each_back_window_inf(%input: !kunir.ts<f64, inf>)
    -> !kunir.ts<f64, 1> {
  %result = kunir.for_each_back_window
      (%input : !kunir.ts<f64, inf>) [window = 100]
      (%val : !kunir.ts<f64, 1>)
      -> (!kunir.ts<f64, 1>) {
    %s = kunir.reduce_add %val : !kunir.ts<f64, 1>
    kunir.yield %s : !kunir.ts<f64, 1>
  }
  return %result : !kunir.ts<f64, 1>
}

// CHECK-LABEL: func.func @test_f64_binary
func.func @test_f64_binary(%a: !kunir.ts<f64, inf>, %b: !kunir.ts<f64, inf>)
    -> !kunir.ts<f64, 1> {
  // CHECK: !kunir.ts<f64
  %result = kunir.max %a, %b : !kunir.ts<f64, inf>, !kunir.ts<f64, inf>
  return %result : !kunir.ts<f64, 1>
}
