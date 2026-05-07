#ifndef KUNGPU_PASSES_TD
#define KUNGPU_PASSES_TD

include "mlir/Pass/PassBase.td"

def WindowedTempMemoryPlanning
    : Pass<"kungpu-memory-planning", "::kunir::FuncOp"> {
  let summary = "Assign shared/local memory to kungpu.windowed_temp ops";
  let description = [{
    Greedy memory planning pass for kungpu.windowed_temp allocations.

    Sorts windowed_temp ops by ascending window size and assigns shared memory
    to as many as fit within the per-block budget.  All parameters are read
    from the enclosing kunir.func's target_spec attribute:

      budget_per_block = target_spec.smem_size / target_spec.occupancy
      num_threads      = target_spec.warps_per_cta * 32
      bytes_per_buffer = N * num_threads * target_spec.vector_size * elem_bytes

    The result is written as a discardable boolean attribute "kungpu.smem" on
    each windowed_temp op.  The pass does not change IR structure; address-space
    selection is deferred to the to-LLVM lowering.
  }];
  let constructor = "::kungpu::createWindowedTempMemoryPlanningPass()";
  let dependentDialects = ["::kungpu::KunGpuDialect"];
}

def ConvertKunGpuToLLVM
    : Pass<"convert-kungpu-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower kungpu ops and kunir.func to func.func + LLVM/GPU dialects";
  let description = [{
    Converts the KunGpu dialect to a mix of func, LLVM, GPU, arith, and scf dialects.

    Type conversion:
      !kunir.ts<T,N> → !llvm.ptr

    Function signature:
      kunir.func @f(%a: !kunir.ts<T,N>, ...) → func.func @f(i64, i64, !llvm.ptr, ...)
      The two prepended i64 arguments are time_len and num_stocks.

    Memory ops:
      kungpu.windowed_temp → llvm.alloca circular buffer + head-state alloca
      kungpu.ts.get        → GEP + llvm.load
      kungpu.ts.put        → advance circular head + GEP + llvm.store
      Global ts (function arg): TxS layout — element(t,s) = base + t*num_stocks + s
      windowed_temp: circular index without modulo (compare + conditional subtract)

    Shared memory (kungpu.smem = true):
      A module-level llvm.mlir.global with addr_space=3 is emitted once per
      windowed_temp; each thread accesses its own N-element slice at
      offset threadIdx.x * N.

    Thread indexing:
      kungpu.time_length       → arith.index_cast %time_len : i64 to index
      kungpu.stock_id          → blockIdx.x * blockDim.x + threadIdx.x
      kungpu.block_stock_count → blockDim.x
  }];
  let constructor = "::kungpu::createConvertKunGpuToLLVMPass()";
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

#endif // KUNGPU_PASSES_TD
