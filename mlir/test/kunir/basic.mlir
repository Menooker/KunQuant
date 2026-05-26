// RUN: %kun-opt %s | %FileCheck %s
// RUN: %kun-opt %s | %kun-opt | %FileCheck %s

// Verify the kunir dialect types and ops parse and round-trip inside kunir.func.

// CHECK-LABEL: kunir.func @test_ts_lookback_type
// CHECK-SAME: !kunir.ts<f32, inf>
// CHECK-SAME: !kunir.ts<f32, 1>
// CHECK-SAME: !kunir.ts<f64, 10>
kunir.func @test_ts_lookback_type(
    %a: !kunir.ts<f32, inf>,
    %b: !kunir.ts<f32, 1>,
    %c: !kunir.ts<f64, 10>)
    inputs {%a = "a", %b = "b", %c = "c"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 1> {
  kunir.return %b : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_binary_mismatched_lookbacks
kunir.func @test_binary_mismatched_lookbacks(%a: !kunir.ts<f32, 5>, %b: !kunir.ts<f32, 10>)
    inputs {%a = "a", %b = "b"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 1> {
  // CHECK: kunir.add
  // CHECK-SAME: <f32, 5>, <f32, 10>
  %sum = kunir.add %a, %b : !kunir.ts<f32, 5>, !kunir.ts<f32, 10>
  // CHECK: kunir.sub
  %diff = kunir.sub %a, %b : !kunir.ts<f32, 5>, !kunir.ts<f32, 10>
  // CHECK: kunir.mul
  %prod = kunir.mul %sum, %diff : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
  kunir.return %prod : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_unary
kunir.func @test_unary(%x: !kunir.ts<f32, inf>)
    inputs {%x = "x"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 1> {
  // CHECK: kunir.abs
  %a = kunir.abs %x : !kunir.ts<f32, inf>
  // CHECK: kunir.sign
  %s = kunir.sign %a : !kunir.ts<f32, 1>
  kunir.return %s : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_windowed_output
kunir.func @test_windowed_output(%input: !kunir.ts<f32, inf>)
    inputs {%input = "input"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f32, 10> {
  // CHECK: kunir.windowed_output
  // CHECK-SAME: length = 10
  %out = kunir.windowed_output %input [length = 10] : !kunir.ts<f32, inf> -> !kunir.ts<f32, 10>
  kunir.return %out : !kunir.ts<f32, 10>
}

// CHECK-LABEL: kunir.func @test_for_each_back_window_single
kunir.func @test_for_each_back_window_single(%close: !kunir.ts<f32, 10>)
    inputs {%close = "close"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
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
  kunir.return %ts_sum : !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_for_each_back_window_multi_input
kunir.func @test_for_each_back_window_multi_input(
    %close: !kunir.ts<f32, 20>,
    %vol:   !kunir.ts<f32, 20>)
    inputs {%close = "close", %vol = "vol"}
    outputs {"sum_close", "sum_vol"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
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
  kunir.return %sum_c, %sum_v : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_for_each_back_window_multi_reduce
kunir.func @test_for_each_back_window_multi_reduce(%input: !kunir.ts<f32, 20>)
    inputs {%input = "input"}
    outputs {"sum", "max"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
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
  kunir.return %sum_ts, %max_ts : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
}

// CHECK-LABEL: kunir.func @test_for_each_back_window_inf
kunir.func @test_for_each_back_window_inf(%input: !kunir.ts<f64, inf>)
    inputs {%input = "input"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f64, 1> {
  %result = kunir.for_each_back_window
      (%input : !kunir.ts<f64, inf>) [window = 100]
      (%val : !kunir.ts<f64, 1>)
      -> (!kunir.ts<f64, 1>) {
    %s = kunir.reduce_add %val : !kunir.ts<f64, 1>
    kunir.yield %s : !kunir.ts<f64, 1>
  }
  kunir.return %result : !kunir.ts<f64, 1>
}

// CHECK-LABEL: kunir.func @test_f64_binary
kunir.func @test_f64_binary(%a: !kunir.ts<f64, inf>, %b: !kunir.ts<f64, inf>)
    inputs {%a = "a", %b = "b"}
    outputs {"result"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
    -> !kunir.ts<f64, 1> {
  // CHECK: !kunir.ts<f64
  %result = kunir.max %a, %b : !kunir.ts<f64, inf>, !kunir.ts<f64, inf>
  kunir.return %result : !kunir.ts<f64, 1>
}

// CHECK-LABEL: kunir.func @test_cmp_logical_select
kunir.func @test_cmp_logical_select(%a: !kunir.ts<f32, inf>, %b: !kunir.ts<f32, inf>)
    inputs {%a = "a", %b = "b"}
    outputs {"gt_out", "lt_out", "eq_out", "and_out", "or_out", "not_out"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} unreliable_count = 0
    -> (!kunir.ts<f32, 1>, !kunir.ts<f32, 1>, !kunir.ts<f32, 1>,
        !kunir.ts<f32, 1>, !kunir.ts<f32, 1>, !kunir.ts<f32, 1>) {
  // CHECK: kunir.gt
  %gt = kunir.gt %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  // CHECK: kunir.lt
  %lt = kunir.lt %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  // CHECK: kunir.ge
  %ge = kunir.ge %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  // CHECK: kunir.le
  %le = kunir.le %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  // CHECK: kunir.eq
  %eq = kunir.eq %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  // CHECK: kunir.and
  %and = kunir.and %gt, %lt : !kunir.ts<i1, 1>, !kunir.ts<i1, 1>
  // CHECK: kunir.or
  %or  = kunir.or  %ge, %le : !kunir.ts<i1, 1>, !kunir.ts<i1, 1>
  // CHECK: kunir.not
  %nt  = kunir.not %lt : !kunir.ts<i1, 1>
  // CHECK: kunir.select
  %s_gt  = kunir.select %gt,  %a, %b : !kunir.ts<i1, 1>, !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  %s_lt  = kunir.select %lt,  %a, %b : !kunir.ts<i1, 1>, !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  %s_eq  = kunir.select %eq,  %a, %b : !kunir.ts<i1, 1>, !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  %s_and = kunir.select %and, %a, %b : !kunir.ts<i1, 1>, !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  %s_or  = kunir.select %or,  %a, %b : !kunir.ts<i1, 1>, !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  %s_nt  = kunir.select %nt,  %a, %b : !kunir.ts<i1, 1>, !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
  kunir.return %s_gt, %s_lt, %s_eq, %s_and, %s_or, %s_nt
    : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>, !kunir.ts<f32, 1>,
      !kunir.ts<f32, 1>, !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
}
