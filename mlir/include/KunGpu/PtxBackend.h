//===- PtxBackend.h - Compile a kunir module all the way to PTX ---------===//
//
// Companion to `Pipelines.h` — runs the kunir-to-llvm dialect pipeline,
// translates the resulting MLIR `gpu.module` to an `llvm::Module`, applies
// the standard LLVM optimization pipeline (PassBuilder default
// per-module pipeline, the same one mlir::makeOptimizingTransformer uses,
// which is what upstream `gpu-module-to-binary` invokes via
// `ModuleToObject::optimizeModule`), and finally emits PTX through
// `NVPTXTargetMachine::addPassesToEmitFile(AssemblyFile)`.
//
// This is the single C++ entry point downstream `kunir_to_ptx` callers
// (host runtime, JIT) should plumb to.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace kungpu {

struct PtxCompileOptions {
  /// LLVM optimization level (0..3, mapped to OptimizationLevel::O0..O3).
  unsigned optLevel = 3;

  /// LLVM size level (0..2). 0 disables size opts; rarely needed for GPU.
  unsigned sizeLevel = 0;

  /// SM target, e.g. "sm_80".  Defaults to a widely-supported value; the
  /// caller should set it to whatever GPU it actually targets.
  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string targetCpu    = "sm_80";
  std::string targetFeatures = "+ptx80";
};

/// End-to-end compile a `builtin.module` containing `gpu.module` kernels.
///
/// 1. Runs the kunir → LLVM dialect pipeline (see Pipelines.h).
/// 2. Translates the LLVM-dialect module to llvm::Module via upstream
///    `mlir::translateModuleToLLVMIR`.
/// 3. Runs LLVM optimizations: `PassBuilder::buildPerModuleDefaultPipeline`
///    at the chosen OptimizationLevel — this includes DCE, InstCombine,
///    CSE, LICM, vectorization, etc.  The TargetMachine is the
///    NVPTXTargetMachine for the requested SM, so target-specific
///    pipeline tweaks fire too.
/// 4. Emits PTX assembly via `TargetMachine::addPassesToEmitFile` with
///    `CodeGenFileType::AssemblyFile`.
///
/// On success, `ptxOut` contains the PTX text.  On failure, returns
/// `failure()` after reporting diagnostics through MLIR's standard
/// channels.
::mlir::LogicalResult compileKunIrToPtx(::mlir::ModuleOp module,
                                          const PtxCompileOptions &options,
                                          std::string &ptxOut);

} // namespace kungpu
