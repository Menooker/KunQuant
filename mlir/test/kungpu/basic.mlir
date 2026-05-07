// RUN: %kun-opt %s | %FileCheck %s
// RUN: %kun-opt %s | %kun-opt | %FileCheck %s

// CHECK-LABEL: kunir.func @test_stock_id
kunir.func @test_stock_id()
    inputs {} outputs {"id"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1}
    -> index {
  // CHECK: kungpu.stock_id
  %id = kungpu.stock_id
  kunir.return %id : index
}

// CHECK-LABEL: kunir.func @test_block_stock_count
kunir.func @test_block_stock_count()
    inputs {} outputs {"n"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1}
    -> index {
  // CHECK: kungpu.block_stock_count
  %n = kungpu.block_stock_count
  kunir.return %n : index
}

// CHECK-LABEL: kunir.func @test_time_length
kunir.func @test_time_length()
    inputs {} outputs {"len"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1}
    -> index {
  // CHECK: kungpu.time_length
  %len = kungpu.time_length
  kunir.return %len : index
}

// CHECK-LABEL: kunir.func @test_ts_get_put
kunir.func @test_ts_get_put(%ts_in: !kunir.ts<f32, inf>, %ts_out: !kunir.ts<f32, 1>)
    inputs {%ts_in = "ts_in"}
    outputs {%ts_out = "ts_out"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1} {
  %off = arith.constant 0 : i32
  // CHECK: kungpu.ts.get
  // CHECK-SAME: <f32, inf> -> f32
  %v = kungpu.ts.get %ts_in[%off] : !kunir.ts<f32, inf> -> f32
  // CHECK: kungpu.ts.put
  kungpu.ts.put %ts_out, %v : !kunir.ts<f32, 1>, f32
  kunir.return
}

// CHECK-LABEL: kunir.func @test_windowed_temp
kunir.func @test_windowed_temp()
    inputs {} outputs {"v"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 0, vector_size = 1}
    -> f32 {
  %off = arith.constant 0 : i32
  // CHECK: %[[WT:.*]] = kungpu.windowed_temp : <f32, 5>
  %wt = kungpu.windowed_temp : !kunir.ts<f32, 5>
  // CHECK: kungpu.ts.get %[[WT]]
  %v = kungpu.ts.get %wt[%off] : !kunir.ts<f32, 5> -> f32
  kunir.return %v : f32
}
