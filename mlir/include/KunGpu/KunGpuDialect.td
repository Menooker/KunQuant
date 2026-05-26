#ifndef KUNGPU_DIALECT_TD
#define KUNGPU_DIALECT_TD

include "mlir/IR/OpBase.td"

def KunGpu_Dialect : Dialect {
  let name = "kungpu";
  let summary = "KunQuant GPU dialect for NVGPU-targeted computation";
  let description = [{
    The kungpu dialect lowers kunir dataflow operations to explicit GPU
    control flow with an NVIDIA GPU thread model. It retains !kunir.ts
    types and introduces explicit time loops, stock-thread mapping, and
    shared memory operations. Raw pointers appear only when lowering to
    llvm+nvvm dialects.
  }];
  let cppNamespace = "::kungpu";
}

#endif // KUNGPU_DIALECT_TD
