// RUN: %kun-opt --convert-kungpu-to-llvm %s | %FileCheck %s
//
// All kernels live in a single gpu.module — convert-kungpu-to-llvm rewrites
// each kunir.func to a gpu.func (kernel) inside that gpu.module, with the
// signature prepended by (i32 time_len, i32 num_stocks).

gpu.module @kungpu_kernels {

// =====================================================================
// Smem global emitted by `test_windowed_smem` lands inside gpu.module.
// =====================================================================
// CHECK:       gpu.module @kungpu_kernels {
// CHECK:         llvm.mlir.global internal @[[SMEM:__smem_test_windowed_smem_[0-9]+]]()
// CHECK-SAME:    {addr_space = 3 : i32}
// CHECK-SAME:    !llvm.array<{{[0-9]+}} x f32>


// =====================================================================
// Case 1 — gmem-only: signature change, time_length lowering, TxS GEPs.
// =====================================================================
//
// CHECK-LABEL: gpu.func @test_copy(
// CHECK-SAME:    %[[TL:[^:]+]]: i32,
// CHECK-SAME:    %[[NS:[^:]+]]: i32,
// CHECK-SAME:    %[[IN:[^:]+]]: !llvm.ptr,
// CHECK-SAME:    %[[OUT:[^:]+]]: !llvm.ptr
// kernel attribute is set, kunir-func metadata preserved as discardables:
// CHECK-SAME:    kernel
// CHECK-SAME:    kungpu.input_names = ["in"]
// CHECK-SAME:    kungpu.output_names = ["out"]
// CHECK-SAME:    kungpu.target_spec = #kunir<target_spec{
//
// ── Active-thread guard prologue ──────────────────────────────────────
// Computes stock_id = bid*bdim + tid, compares with %num_stocks, then
// wraps the original kernel body in scf.if so threads with
// stock_id ≥ num_stocks fall straight through to gpu.return.
// CHECK:       %[[TID:.*]]  = gpu.thread_id  x
// CHECK:       %[[BID:.*]]  = gpu.block_id   x
// CHECK:       %[[BDIM:.*]] = gpu.block_dim  x
// CHECK:       %[[BTB:.*]]  = arith.muli %[[BID]], %[[BDIM]]
// CHECK:       %[[SID:.*]]  = arith.addi %[[BTB]], %[[TID]]
// CHECK:       %[[SIDI:.*]] = arith.index_cast %[[SID]] : index to i32
// CHECK:       %[[ACTIVE:.*]] = arith.cmpi slt, %[[SIDI]], %[[NS]] : i32
// CHECK:       scf.if %[[ACTIVE]] {
//
// time_length → arith.index_cast of arg0 (i32 → index)
// CHECK:         %[[TLIDX:.*]] = arith.index_cast %[[TL]] : i32 to index
// CHECK:         %[[OFFCST:.*]] = arith.constant 0 : i32
//
// CHECK:         scf.for %[[T:.*]] = %{{.*}} to %[[TLIDX]] step %{{.*}}
//
// ── ts.get on global %in at offset 0 ───────────────────────────────────
// effective time = t − 0; stock_id = bid*bdim + tid; lin = effT*ns + sid.
// num_stocks (i32 arg[1]) is sign-extended to i64 for the linear index.
// CHECK:         %[[OFFI:.*]] = arith.index_cast %[[OFFCST]] : i32 to index
// CHECK:         %[[NS64:.*]] = arith.extsi %[[NS]] : i32 to i64
// CHECK:         %[[EFFT:.*]] = arith.subi %[[T]], %[[OFFI]] : index
// CHECK:         %[[EFFT64:.*]] = arith.index_cast %[[EFFT]] : index to i64
// CHECK:         %[[TID:.*]] = gpu.thread_id  x
// CHECK:         %[[BID:.*]] = gpu.block_id   x
// CHECK:         %[[BDIM:.*]] = gpu.block_dim  x
// CHECK:         %[[BTB:.*]] = arith.muli %[[BID]], %[[BDIM]]
// CHECK:         %[[SID:.*]] = arith.addi %[[BTB]], %[[TID]]
// CHECK:         %[[SIDI:.*]] = arith.index_cast %[[SID]] : index to i64
// CHECK:         %[[ROW:.*]] = arith.muli %[[EFFT64]], %[[NS64]] : i64
// CHECK:         %[[LIN:.*]] = arith.addi %[[ROW]], %[[SIDI]] : i64
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[IN]][%[[LIN]]] {{.*}} -> !llvm.ptr, f32
// CHECK:         %[[V:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> f32
//
// ── ts.put on global %out (no offset; writes at current iv) ───────────
// CHECK:         %[[NS64B:.*]] = arith.extsi %[[NS]] : i32 to i64
// CHECK:         %[[T64:.*]] = arith.index_cast %[[T]] : index to i64
// CHECK:         %[[ROW2:.*]] = arith.muli %[[T64]], %[[NS64B]] : i64
// CHECK:         %[[LIN2:.*]] = arith.addi %[[ROW2]],
// CHECK:         %[[GEP2:.*]] = llvm.getelementptr %[[OUT]][%[[LIN2]]]
// CHECK:         llvm.store %[[V]], %[[GEP2]]
// scf.if + gpu.return: inactive threads (sid ≥ ns) skip the body and
// arrive at gpu.return directly.
// CHECK:       gpu.return
kunir.func @test_copy(%in: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%in = "in"}
    outputs {%out = "out"}
    target {occupancy = 1, warps_per_cta = 1, smem_size = 0, vector_size = 1} {
  %tl = kungpu.time_length
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %off = arith.constant 0 : i32
  scf.for %t = %c0 to %tl step %c1 {
    %v = kungpu.ts.get %in[%off] : !kunir.ts<f32, inf> -> f32
    kungpu.ts.put %out, %v : !kunir.ts<f32, 1>, f32
  }
  kunir.return
}


// =====================================================================
// Case 2 — windowed_temp in local memory: alloca buffer + i32 pos cell,
//          circular put/get (no modulo).
// =====================================================================
//
// CHECK-LABEL: gpu.func @test_windowed_local
// CHECK-SAME:  i32
// CHECK-SAME:  i32
// CHECK-SAME:  !llvm.ptr
// CHECK-SAME:  !llvm.ptr
//
// ── windowed_temp lowering — buf alloca + 1×i32 pos cell init to 0 ────
// CHECK:       %[[NCST:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:       %[[BUF:.*]] = llvm.alloca %[[NCST]] x f32 : (i32) -> !llvm.ptr
// CHECK:       %[[ONE32A:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:       %[[POS:.*]] = llvm.alloca %[[ONE32A]] x i32 : (i32) -> !llvm.ptr
// CHECK:       %[[ZERO32:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       llvm.store %[[ZERO32]], %[[POS]] : i32, !llvm.ptr
//
// CHECK:       scf.for %[[T:.*]] =
//
// ── ts.put %wt, %v (circular write):  buf[pos] = v; pos = (pos+1>=N)?0:pos+1
// (GEP index is i32 — no sext, since LLVM accepts any int type for indices.)
// CHECK:         %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr -> f32
// CHECK:         %[[P:.*]] = llvm.load %[[POS]] : !llvm.ptr -> i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[BUF]][%[[P]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK:         llvm.store %[[V]], %[[GEP]] : f32, !llvm.ptr
// CHECK:         %[[ONE32:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[N32:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:         %[[Z32:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[PP1:.*]] = llvm.add %[[P]], %[[ONE32]] : i32
// CHECK:         %[[CMP:.*]] = llvm.icmp "uge" %[[PP1]], %[[N32]] : i32
// CHECK:         %[[NEW:.*]] = llvm.select %[[CMP]], %[[Z32]], %[[PP1]] : i1, i32
// CHECK:         llvm.store %[[NEW]], %[[POS]] : i32, !llvm.ptr
//
// ── ts.get %wt[off] (circular read):
//      adj=off+1; idx = pos>=adj ? pos-adj : pos+N-adj; return buf[idx]
// CHECK:         %[[OF:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK:         %[[P2:.*]] = llvm.load %[[POS]] : !llvm.ptr -> i32
// CHECK:         %[[ONE32B:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[N32B:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:         %[[ADJ:.*]] = llvm.add %[[OF]], %[[ONE32B]] : i32
// CHECK:         %[[GE:.*]] = llvm.icmp "uge" %[[P2]], %[[ADJ]] : i32
// CHECK:         %[[PMA:.*]] = llvm.sub %[[P2]], %[[ADJ]] : i32
// CHECK:         %[[PPN:.*]] = llvm.add %[[P2]], %[[N32B]] : i32
// CHECK:         %[[WR:.*]] = llvm.sub %[[PPN]], %[[ADJ]] : i32
// CHECK:         %[[IDX:.*]] = llvm.select %[[GE]], %[[PMA]], %[[WR]] : i1, i32
// CHECK:         %[[GGEP:.*]] = llvm.getelementptr %[[BUF]][%[[IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK:         llvm.load %[[GGEP]] : !llvm.ptr -> f32
kunir.func @test_windowed_local(%in: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%in = "in"}
    outputs {%out = "out"}
    target {occupancy = 1, warps_per_cta = 1, smem_size = 0, vector_size = 1} {
  %wt = kungpu.windowed_temp : !kunir.ts<f32, 5> {kungpu.smem = false}
  %tl = kungpu.time_length
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %off0 = arith.constant 0 : i32
  scf.for %t = %c0 to %tl step %c1 {
    %v  = kungpu.ts.get %in[%off0] : !kunir.ts<f32, inf> -> f32
    kungpu.ts.put %wt, %v : !kunir.ts<f32, 5>, f32
    %off_idx = arith.subi %t, %c0 : index
    %off_i32 = arith.index_cast %off_idx : index to i32
    %w  = kungpu.ts.get %wt[%off_i32] : !kunir.ts<f32, 5> -> f32
    kungpu.ts.put %out, %w : !kunir.ts<f32, 1>, f32
  }
  kunir.return
}


// =====================================================================
// Case 3 — windowed_temp in shared memory: slot-major layout.
//
//   layout:        smem[ slot*K + tid ]   (K = threads_per_block)
//   global size:   N * K elements
//   per-thread base:  bufPtr = smem + tid           (1 GEP at allocation)
//   per-access stride: K  (smem[bufPtr + idx*K] in ts.put / ts.get)
// =====================================================================
//
// Global has 5*128 = 640 elements (N=5, warps_per_cta=4 → K=128).
//
// CHECK-LABEL: gpu.func @test_windowed_smem
// CHECK:       %[[RAW:.*]] = llvm.mlir.addressof @[[SMEM]] : !llvm.ptr<3>
// CHECK:       %[[GEN:.*]] = llvm.addrspacecast %[[RAW]] : !llvm.ptr<3> to !llvm.ptr
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[TIDI:.*]] = arith.index_cast %[[TID]] : index to i32
// bufPtr = smem + tid  (no per-allocation N multiply)
// CHECK:       %[[BUF3:.*]] = llvm.getelementptr %[[GEN]][%[[TIDI]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// pos cell still alloca'd (i32)
// CHECK:       llvm.alloca {{.*}} x i32
//
// Inside the loop — ts.put: stride-K multiply before the GEP.
// CHECK:         %[[POSV:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:         %[[K:.*]] = llvm.mlir.constant(128 : i32) : i32
// CHECK:         %[[OFFP:.*]] = llvm.mul %[[POSV]], %[[K]] : i32
// CHECK:         %[[GEPP:.*]] = llvm.getelementptr %[[BUF3]][%[[OFFP]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK:         llvm.store %{{.*}}, %[[GEPP]]
//
// ts.get: same stride-K pattern.
// CHECK:         %[[K2:.*]] = llvm.mlir.constant(128 : i32) : i32
// CHECK:         %[[OFFG:.*]] = llvm.mul %{{.*}}, %[[K2]] : i32
// CHECK:         %[[GEPG:.*]] = llvm.getelementptr %[[BUF3]][%[[OFFG]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK:         llvm.load %[[GEPG]]
kunir.func @test_windowed_smem(%in: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%in = "in"}
    outputs {%out = "out"}
    target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} {
  %wt = kungpu.windowed_temp : !kunir.ts<f32, 5> {kungpu.smem = true}
  %tl = kungpu.time_length
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %off0 = arith.constant 0 : i32
  scf.for %t = %c0 to %tl step %c1 {
    %v  = kungpu.ts.get %in[%off0] : !kunir.ts<f32, inf> -> f32
    kungpu.ts.put %wt, %v : !kunir.ts<f32, 5>, f32
    %w  = kungpu.ts.get %wt[%off0] : !kunir.ts<f32, 5> -> f32
    kungpu.ts.put %out, %w : !kunir.ts<f32, 1>, f32
  }
  kunir.return
}


// =====================================================================
// Case 4 — stock_id and block_stock_count lowering.
// =====================================================================
//
// CHECK-LABEL: gpu.func @test_indexing
// CHECK:       gpu.thread_id  x
// CHECK-NEXT:  gpu.block_id   x
// CHECK-NEXT:  gpu.block_dim  x
// CHECK-NEXT:  arith.muli
// CHECK-NEXT:  arith.addi
// CHECK:       gpu.block_dim  x
kunir.func @test_indexing(%in: !kunir.ts<f32, inf>, %out: !kunir.ts<f32, 1>)
    inputs {%in = "in"}
    outputs {%out = "out"}
    target {occupancy = 1, warps_per_cta = 1, smem_size = 0, vector_size = 1} {
  %sid = kungpu.stock_id
  %bsc = kungpu.block_stock_count
  %sum = arith.addi %sid, %bsc : index
  kunir.return
}

}  // gpu.module
