//===- PtxBackend.h - Compile a kunir module to a CUDA cubin -----------===//
//
// Pipeline (single source of truth):
//
//   kunir → llvm dialect (our buildKunIrToLLVMPipeline)
//        → upstream `gpu-module-to-binary{format=bin}`
//        → cubin bytes pulled off the resulting `gpu.binary` op
//
// `gpu-module-to-binary` (via NVVMTargetAttrImpl) takes care of:
//   * MLIR → LLVM IR translation,
//   * libdevice.10.bc location + linking + AlwaysInline + DCE,
//   * the LLVM optimization pipeline,
//   * PTX emission via NVPTXTargetMachine,
//   * ptxas invocation.
//
// We just attach an `#nvvm.target<chip = ..., O = ...>` to the gpu.module
// and run the pass.  No more manual ptxas plumbing on the main path.
//
// `compileKunIrToPtx` is kept for **debug / inspection**: same pipeline,
// but `format=isa` so we can read the PTX text instead of the cubin.
// The main `compileKunIrToExecutable` does NOT route through it.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "KunCuda/Runtime.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace kungpu {

/// Knobs forwarded to the upstream `#nvvm.target` attribute (which the
/// `gpu-module-to-binary` pass reads via NVVMTargetAttrImpl).
///
/// `targetCpu` is the LLVM/NVPTX term — it carries the SM string
/// ("sm_80", "sm_120", …) that ptxas / NVPTXTargetMachine consume.  The
/// user-facing Python kwarg is `gpu_arch`.
struct PtxCompileOptions {
  unsigned    optLevel       = 3;       ///< maps to #nvvm.target<O = N>
  std::string targetTriple   = "nvptx64-nvidia-cuda";
  std::string targetCpu      = "sm_80"; ///< chip, e.g. "sm_120"
  std::string targetFeatures;           ///< empty → derived from chip

  /// Forwarded to gpu-module-to-binary's `toolkit` option.  Empty → the
  /// pass searches CUDA_HOME / CUDA_PATH / standard paths.  Useful when
  /// the right CUDA toolkit (the one with libdevice.10.bc + a matching
  /// ptxas) isn't on PATH.
  std::string toolkitPath;
};

/// Lower kunir → llvm dialect → emit PTX text.  **Debug / inspection
/// only** — the main compile path goes straight to cubin.
///
/// On success `ptxOut` holds the PTX assembly produced by the upstream
/// `gpu-module-to-binary{format=isa}` pass.  Module is mutated in place
/// (the gpu.module gets replaced with a gpu.binary op).
::mlir::LogicalResult compileKunIrToPtx(::mlir::ModuleOp module,
                                          const PtxCompileOptions &options,
                                          std::string &ptxOut);

/// Main entry point: lower kunir, run gpu-module-to-binary{format=bin},
/// and pull the cubin + per-kernel name metadata into an
/// `ExecutableData` ready for `kun_cuda::Executable`.
///
/// Walks the lowered module for kernel metadata (name, target spec,
/// I/O names) BEFORE the pass runs, since `gpu-module-to-binary`
/// replaces the gpu.module with a gpu.binary op.  graphInputs /
/// graphOutputs are NOT set here — the caller fills them on `out`
/// after this returns (see KunCuda/Runtime.h).
///
/// The module is mutated in-place by the pipeline.
::mlir::LogicalResult
compileKunIrToExecutable(::mlir::ModuleOp module,
                          const PtxCompileOptions &options,
                          ::kun_cuda::ExecutableData &out);

} // namespace kungpu
