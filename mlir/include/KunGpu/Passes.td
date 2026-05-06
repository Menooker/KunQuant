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

#endif // KUNGPU_PASSES_TD
