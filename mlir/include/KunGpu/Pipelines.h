//===- Pipelines.h - Reusable kunir → LLVM lowering pipeline -------------===//
//
// Defines the canonical lowering pipeline that converts a `kunir.func`-based
// module all the way down to the LLVM dialect.  Phase ordering:
//
//   1. kunir-to-kungpu                    (kunir.func nested)
//   2. kungpu-memory-planning             (kunir.func nested)
//   3. convert-kungpu-to-llvm             (module — also lowers kunir.func
//                                          to func.func)
//   4. loop-invariant-code-motion         (per func)
//   5. canonicalize
//   6. cse
//   7. convert-scf-to-cf
//   8. convert-control-flow-to-llvm
//   9. convert-arith-to-llvm
//  10. convert-index-to-llvm
//  11. convert-func-to-llvm
//  12. reconcile-unrealized-casts
//
// `kunir_to_ptx` will reuse `buildKunIrToLLVMPipeline` and append the
// gpu→nvvm/llvm-translation passes after it.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;
} // namespace mlir

namespace kungpu {

/// Append the kunir → LLVM dialect lowering passes to `pm`.  This is the
/// shared entry point used by both the test wrapper pass below and by any
/// downstream pipeline that needs to lower further (e.g. kunir_to_ptx).
void buildKunIrToLLVMPipeline(::mlir::OpPassManager &pm);

/// Single-pass wrapper that runs `buildKunIrToLLVMPipeline` on the current
/// module.  Mainly for lit-testing the pipeline as a whole.
std::unique_ptr<::mlir::Pass> createKunIrToLLVMPass();

void registerKunIrToLLVMPass();

} // namespace kungpu
