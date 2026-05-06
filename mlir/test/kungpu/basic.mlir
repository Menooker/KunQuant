// RUN: %kun-opt %s | %FileCheck %s
// RUN: %kun-opt %s | %kun-opt | %FileCheck %s

// CHECK-LABEL: func.func @test_stock_id
func.func @test_stock_id() -> index {
  // CHECK: kungpu.stock_id
  %id = kungpu.stock_id
  return %id : index
}

// CHECK-LABEL: func.func @test_block_stock_count
func.func @test_block_stock_count() -> index {
  // CHECK: kungpu.block_stock_count
  %n = kungpu.block_stock_count
  return %n : index
}

// CHECK-LABEL: func.func @test_time_length
func.func @test_time_length() -> index {
  // CHECK: kungpu.time_length
  %len = kungpu.time_length
  return %len : index
}

// CHECK-LABEL: func.func @test_ts_get_put
func.func @test_ts_get_put(%ts_in: !kunir.ts<f32, inf>, %ts_out: !kunir.ts<f32, 1>) {
  %c0 = arith.constant 0 : index
  // CHECK: kungpu.ts.get
  // CHECK-SAME: <f32, inf> -> f32
  %v = kungpu.ts.get %ts_in[%c0] : !kunir.ts<f32, inf> -> f32
  // CHECK: kungpu.ts.put
  kungpu.ts.put %ts_out[%c0], %v : !kunir.ts<f32, 1>, f32
  return
}

// CHECK-LABEL: func.func @test_windowed_temp
func.func @test_windowed_temp() -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[WT:.*]] = kungpu.windowed_temp : <f32, 5>
  %wt = kungpu.windowed_temp : !kunir.ts<f32, 5>
  // CHECK: kungpu.ts.get %[[WT]]
  %v = kungpu.ts.get %wt[%c0] : !kunir.ts<f32, 5> -> f32
  return %v : f32
}
