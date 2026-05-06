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
