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

#include "KunCuda/Runtime.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

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
  /// Empty by default — let LLVM pick a PTX version compatible with the
  /// chosen `targetCpu` (sm_80 → ptx70 etc., sm_120 → ptx87 etc.).
  std::string targetFeatures;
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

struct PtxToCubinOptions {
  /// SM architecture to assemble for, e.g. "sm_80".
  std::string gpuArch = "sm_80";
  /// PTX ISA version for ptxas (passed via --gpu-name and -V).  Empty =
  /// let ptxas choose its default.
  std::string ptxasVersion;
  /// Extra arguments forwarded verbatim to ptxas (e.g. {"-O3"}).
  std::vector<std::string> extraArgs;
  /// Optional override for the ptxas binary path.  When empty we search
  /// PATH and the CUDA_HOME / CUDA_PATH env vars (same logic upstream
  /// NVPTXSerializer uses).
  std::string ptxasPath;
};

/// Assemble PTX text into a CUBIN binary.  This is the same operation
/// upstream `NVPTXSerializer::compileToBinary` performs internally —
/// shell out to `ptxas` — exposed as a standalone helper because the
/// upstream class isn't part of the public C++ API.
///
/// On success, `cubinOut` contains the raw CUBIN bytes.
::mlir::LogicalResult compilePtxToCubin(::llvm::StringRef ptx,
                                          const PtxToCubinOptions &options,
                                          std::vector<char> &cubinOut,
                                          std::string &errorMsg);

/// Compile-only: run the kunir → LLVM dialect pipeline, translate to
/// LLVM IR, optimize, emit PTX, assemble to CUBIN, then walk the
/// lowered module to populate the per-kernel name metadata (one
/// `KernelMeta` per `llvm.func` carrying `kungpu.target_spec`).  The
/// caller is expected to fill in `out.graphInputs` / `out.graphOutputs`
/// before constructing a `kun_cuda::Executable` from the result —
/// graph topology is a runtime concern, not a compile-time one.
///
/// On success `out` is populated with: cubin, warpsPerCta, vectorSize
/// (validated to be uniform across kernels), and the unordered list of
/// kernels (each with its name and the input/output names from
/// `kungpu.input_names` / `kungpu.output_names`).  Topology validation,
/// topo sort, buffer indexing and slot planning all happen later, in
/// the `Executable` ctor.
///
/// The module is mutated in-place by the pipeline (same as
/// `compileKunIrToPtx`).
::mlir::LogicalResult
compileKunIrToExecutable(::mlir::ModuleOp module,
                          const PtxCompileOptions &ptxOpts,
                          const PtxToCubinOptions &cubinOpts,
                          ::kun_cuda::ExecutableData &out);

} // namespace kungpu
