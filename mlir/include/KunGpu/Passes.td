#ifndef KUNGPU_PASSES_TD
#define KUNGPU_PASSES_TD

include "mlir/Pass/PassBase.td"

def WindowedTempMemoryPlanning
    : Pass<"kungpu-memory-planning", "::mlir::func::FuncOp"> {
  let summary = "Assign shared/local memory to kungpu.windowed_temp ops";
  let description = [{
    Greedy memory planning pass for kungpu.windowed_temp allocations.

    Sorts windowed_temp ops by ascending window size and assigns shared memory
    to as many as fit within the per-block budget:

      budget_per_block = total_smem_size / target_occupancy
      bytes_per_buffer = N * num_threads_per_block * vector_size * elem_bytes

    The result is written as a discardable boolean attribute "kungpu.smem" on
    each windowed_temp op.  The pass does not change IR structure; address-space
    selection is deferred to the to-LLVM lowering.

    Parameters (passed at construction, not CLI options):
      total_smem_size      - device shared memory in bytes
      target_occupancy     - concurrent blocks per SM
      num_threads_per_block
      vector_size          - scalar elements per thread per time step
  }];
  let constructor = "::kungpu::createWindowedTempMemoryPlanningPass()";
  let dependentDialects = ["::mlir::func::FuncDialect"];
}

#endif // KUNGPU_PASSES_TD
