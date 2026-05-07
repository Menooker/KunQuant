//===- PtxBackend.cpp - Compile a kunir module all the way to PTX ------===//

#include "KunGpu/PtxBackend.h"
#include "KunGpu/Pipelines.h"

#ifndef KUN_HAS_NVPTX

// LLVM was built without the NVPTX target.  Provide a stub so callers
// still link, but compiling actual PTX is unavailable.

#include "mlir/IR/BuiltinOps.h"

namespace kungpu {
::mlir::LogicalResult compileKunIrToPtx(::mlir::ModuleOp module,
                                          const PtxCompileOptions &,
                                          std::string &) {
  return module.emitError(
      "compileKunIrToPtx: NVPTX target was not enabled in this LLVM build "
      "(missing 'NVPTX' in LLVM_TARGETS_TO_BUILD).");
}
} // namespace kungpu

#else  // KUN_HAS_NVPTX


#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

using namespace mlir;

namespace kungpu {

namespace {

/// Look up the LLVM target for the given triple, lazily initializing the
/// NVPTX target & asmprinter once per process.
static llvm::Expected<const llvm::Target *>
lookupNvptxTarget(llvm::StringRef triple) {
  static const bool kInit = [] {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    return true;
  }();
  (void)kInit;

  std::string err;
  const llvm::Target *t = llvm::TargetRegistry::lookupTarget(triple.str(), err);
  if (!t)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), err);
  return t;
}

} // namespace

LogicalResult compileKunIrToPtx(ModuleOp module,
                                 const PtxCompileOptions &options,
                                 std::string &ptxOut) {
  MLIRContext *ctx = module.getContext();

  // ─── Step 1.  Run the kunir → LLVM dialect pipeline ────────────────
  PassManager pm(ctx);
  buildKunIrToLLVMPipeline(pm);
  if (failed(pm.run(module)))
    return module.emitError(
        "compileKunIrToPtx: kunir-to-llvm pipeline failed");

  // ─── Step 2.  Translate MLIR LLVM dialect → llvm::Module ──────────
  // Make sure NVVM (and friends) know how to emit themselves to LLVM IR.
  DialectRegistry registry;
  registerAllToLLVMIRTranslations(registry);
  ctx->appendDialectRegistry(registry);

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmCtx);
  if (!llvmModule)
    return module.emitError(
        "compileKunIrToPtx: translation to LLVM IR failed");

  // ─── Step 3.  Build NVPTXTargetMachine ────────────────────────────
  auto targetOrErr = lookupNvptxTarget(options.targetTriple);
  if (!targetOrErr) {
    llvm::handleAllErrors(targetOrErr.takeError(),
                          [&](const llvm::ErrorInfoBase &eib) {
                            module.emitError(
                                "compileKunIrToPtx: NVPTX target lookup: ")
                                << eib.message();
                          });
    return failure();
  }
  llvm::TargetOptions opts;
  std::unique_ptr<llvm::TargetMachine> targetMachine{
      (*targetOrErr)
          ->createTargetMachine(llvm::Triple(options.targetTriple),
                                options.targetCpu, options.targetFeatures,
                                opts, /*RelocModel=*/std::nullopt,
                                /*CodeModel=*/std::nullopt,
                                llvm::CodeGenOptLevel::Aggressive)};
  if (!targetMachine)
    return module.emitError(
        "compileKunIrToPtx: failed to create NVPTXTargetMachine");

  llvmModule->setTargetTriple(llvm::Triple(options.targetTriple));
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  // ─── Step 4.  Run LLVM PassBuilder default pipeline ───────────────
  // This is the same entry point upstream `gpu-module-to-binary` uses
  // (see ModuleToObject::optimizeModule → makeOptimizingTransformer).
  // It builds the full new-PM per-module pipeline at the requested O level,
  // which includes mem2reg, SROA, GVN, LICM, instcombine, DCE, vectorise,
  // unroll, etc., plus NVPTX-specific tweaks (the TargetMachine is passed
  // to PassBuilder so its pipeline-tuning hooks fire).
  if (auto err = makeOptimizingTransformer(options.optLevel,
                                            options.sizeLevel,
                                            targetMachine.get())(
          llvmModule.get())) {
    llvm::handleAllErrors(std::move(err),
                          [&](const llvm::ErrorInfoBase &eib) {
                            module.emitError(
                                "compileKunIrToPtx: LLVM opt pipeline: ")
                                << eib.message();
                          });
    return failure();
  }

  // ─── Step 5.  Emit PTX (AssemblyFile) via legacy codegen pipeline ─
  // This is the standard path used by `mlir::ModuleToObject`: the legacy
  // PassManager is required because `addPassesToEmitFile` is a legacy
  // codegen API.  The new PM ran in step 4 — codegen still uses legacy.
  llvm::SmallString<0> ptxBuf;
  {
    llvm::raw_svector_ostream stream(ptxBuf);
    llvm::buffer_ostream bufStream(stream);
    llvm::legacy::PassManager codegenPM;
    if (targetMachine->addPassesToEmitFile(
            codegenPM, bufStream, /*DwoOut=*/nullptr,
            llvm::CodeGenFileType::AssemblyFile)) {
      return module.emitError(
          "compileKunIrToPtx: NVPTXTargetMachine cannot emit assembly");
    }
    codegenPM.run(*llvmModule);
  }
  ptxOut.assign(ptxBuf.begin(), ptxBuf.end());
  return success();
}

} // namespace kungpu

#endif // KUN_HAS_NVPTX
