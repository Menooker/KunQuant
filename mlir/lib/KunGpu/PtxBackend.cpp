//===- PtxBackend.cpp - Compile a kunir module all the way to PTX ------===//

#include "KunGpu/PtxBackend.h"
#include "KunGpu/KunGpuUtils.h"
#include "KunGpu/Pipelines.h"
#include "KunIr/KunIrAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace kungpu {

namespace {

/// Search for `ptxas` in the user-provided override, then PATH, then
/// CUDA_HOME / CUDA_PATH / standard CUDA install locations.  Mirrors the
/// search the upstream NVPTXSerializer does.
static llvm::ErrorOr<std::string> findPtxas(::llvm::StringRef override) {
  using namespace llvm::sys;
  if (!override.empty() && fs::exists(override))
    return std::string(override);
  if (auto p = findProgramByName("ptxas"))
    return p;
  for (const char *envName : {"CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"}) {
    if (const char *envVal = std::getenv(envName)) {
      llvm::SmallString<256> p(envVal);
      path::append(p, "bin", "ptxas");
      if (fs::exists(p))
        return std::string(p);
    }
  }
  if (fs::exists("/usr/local/cuda/bin/ptxas"))
    return std::string("/usr/local/cuda/bin/ptxas");
  return std::make_error_code(std::errc::no_such_file_or_directory);
}

} // namespace

::mlir::LogicalResult compilePtxToCubin(::llvm::StringRef ptx,
                                          const PtxToCubinOptions &opts,
                                          std::vector<char> &cubinOut,
                                          std::string &errorMsg) {
  using namespace llvm;

  auto ptxasOrErr = findPtxas(opts.ptxasPath);
  if (!ptxasOrErr) {
    errorMsg = "compilePtxToCubin: ptxas not found "
                "(looked in CUDA_HOME / CUDA_PATH / PATH / "
                "/usr/local/cuda/bin); set ptxas_path or CUDA_HOME.";
    return ::mlir::failure();
  }

  // Write PTX to a temp file.
  SmallString<128> ptxPath, cubinPath, logPath;
  if (auto ec = sys::fs::createTemporaryFile("kun-ptx", "ptx", ptxPath)) {
    errorMsg = "compilePtxToCubin: createTemporaryFile(ptx): " + ec.message();
    return ::mlir::failure();
  }
  if (auto ec = sys::fs::createTemporaryFile("kun-cubin", "cubin", cubinPath)) {
    sys::fs::remove(ptxPath);
    errorMsg = "compilePtxToCubin: createTemporaryFile(cubin): " + ec.message();
    return ::mlir::failure();
  }
  if (auto ec = sys::fs::createTemporaryFile("kun-ptxlog", "log", logPath)) {
    sys::fs::remove(ptxPath); sys::fs::remove(cubinPath);
    errorMsg = "compilePtxToCubin: createTemporaryFile(log): " + ec.message();
    return ::mlir::failure();
  }

  // Auto-cleanup.
  struct CleanupOnExit {
    SmallVectorImpl<char> &p; ~CleanupOnExit() { sys::fs::remove(p); }
  };
  CleanupOnExit c1{ptxPath}, c2{cubinPath}, c3{logPath};

  {
    std::error_code ec;
    raw_fd_ostream os(ptxPath, ec, sys::fs::OF_None);
    if (ec) {
      errorMsg = "compilePtxToCubin: writing PTX: " + ec.message();
      return ::mlir::failure();
    }
    os << ptx;
  }

  // Build argv:
  //   ptxas --gpu-name=<sm_xx> -o <cubin> <ptx> [extra...]
  std::string gpuArg = "--gpu-name=" + opts.gpuArch;
  std::string outArg = "-o";
  SmallVector<StringRef> argv = {*ptxasOrErr, gpuArg, outArg, cubinPath, ptxPath};
  for (const auto &a : opts.extraArgs) argv.push_back(a);

  std::string errBuf;
  std::optional<StringRef> redirects[] = {std::nullopt,        // stdin
                                            StringRef(logPath),  // stdout
                                            StringRef(logPath)}; // stderr
  int rc = sys::ExecuteAndWait(*ptxasOrErr, argv, /*Env=*/std::nullopt,
                                 redirects, /*SecondsToWait=*/0,
                                 /*MemoryLimit=*/0, &errBuf);
  if (rc != 0) {
    auto logBuf = MemoryBuffer::getFile(logPath);
    errorMsg = "compilePtxToCubin: ptxas failed (exit " + std::to_string(rc) + ")";
    if (!errBuf.empty()) errorMsg += ": " + errBuf;
    if (logBuf && (*logBuf)->getBufferSize() > 0) {
      errorMsg += "\n--- ptxas log ---\n";
      errorMsg += (*logBuf)->getBuffer().str();
    }
    return ::mlir::failure();
  }

  auto cubinBuf = MemoryBuffer::getFile(cubinPath);
  if (!cubinBuf) {
    errorMsg = "compilePtxToCubin: cannot read cubin: " +
                  cubinBuf.getError().message();
    return ::mlir::failure();
  }
  StringRef bytes = (*cubinBuf)->getBuffer();
  cubinOut.assign(bytes.begin(), bytes.end());
  return ::mlir::success();
}

} // namespace kungpu

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
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
  // Register only the translations we actually need (builtin + LLVM +
  // NVVM + GPU); the upstream `registerAllToLLVMIRTranslations` would
  // pull in ArmSVE / SPIR-V / etc. and force us to link them all.
  DialectRegistry registry;
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  ctx->appendDialectRegistry(registry);

  // Mirror upstream `gpu-module-to-binary` / NVPTXSerializer: translate
  // the gpu.module (the kernel container) rather than the outer
  // builtin.module — only the gpu.module's body is meant to become LLVM
  // IR.  We just take the first gpu.module; multi-module support can
  // come later.
  gpu::GPUModuleOp gpuMod;
  module.walk([&](gpu::GPUModuleOp m) { gpuMod = m; return WalkResult::interrupt(); });
  if (!gpuMod)
    return module.emitError(
        "compileKunIrToPtx: no gpu.module found after lowering");

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(gpuMod, llvmCtx);
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

//===----------------------------------------------------------------------===//
// All-in-one: kunir → cubin + per-kernel name metadata
//
// Compile-time concerns only.  Topology / topo sort / buffer indices /
// slot planning all happen later, in `kun_cuda::Executable`'s ctor —
// see KunCuda/Runtime.h.
//===----------------------------------------------------------------------===//

LogicalResult compileKunIrToExecutable(ModuleOp module,
                                        const PtxCompileOptions &ptxOpts,
                                        const PtxToCubinOptions &cubinOpts,
                                        ::kun_cuda::ExecutableData &out) {
  // 1.  Run the kunir → LLVM dialect pipeline + emit PTX.  This mutates
  //     `module` in place so the discardable kunir metadata ends up on
  //     each lowered llvm.func.
  std::string ptx;
  if (failed(compileKunIrToPtx(module, ptxOpts, ptx)))
    return failure();

  // 2.  Walk every kernel function (carries kungpu.target_spec) and
  //     emit a KernelMeta with names and target spec.
  std::vector<::kun_cuda::KernelMeta> kernels;
  std::vector<std::pair<int64_t, int64_t>> targetSpecs;  // (warps, vector)
  std::vector<std::string> targetSpecOwners;             // for diagnostics

  module.walk([&](LLVM::LLVMFuncOp f) {
    if (!f->hasAttr(kFuncTargetSpecAttr))
      return WalkResult::advance();

    ::kun_cuda::KernelMeta km;
    km.kernelName = f.getSymName().str();
    if (auto inNames = getFuncInputNames(f))
      for (auto a : inNames)
        km.inputNames.push_back(llvm::cast<StringAttr>(a).str());
    if (auto outNames = getFuncOutputNames(f))
      for (auto a : outNames)
        km.outputNames.push_back(llvm::cast<StringAttr>(a).str());

    int64_t w = 1, v = 1;
    if (auto ts = getFuncTargetSpec(f)) {
      w = ts.getWarpsPerCta();
      v = ts.getVectorSize();
    }
    targetSpecs.emplace_back(w, v);
    targetSpecOwners.push_back(km.kernelName);
    kernels.push_back(std::move(km));
    return WalkResult::advance();
  });
  if (kernels.empty())
    return module.emitError(
        "compileKunIrToExecutable: no llvm.func with kungpu metadata "
        "found in the lowered module");

  // 3.  Target spec must be uniform across kernels (block / grid config
  //     is graph-wide in v0).
  auto [warpsPerCta, vectorSize] = targetSpecs.front();
  for (size_t i = 1; i < targetSpecs.size(); ++i) {
    auto [w, v] = targetSpecs[i];
    if (w != warpsPerCta || v != vectorSize)
      return module.emitError(
          "compileKunIrToExecutable: kernels disagree on warps_per_cta / "
          "vector_size — graph-wide target spec required (")
          << "kernel '" << targetSpecOwners[i] << "': warps_per_cta="
          << w << " vector_size=" << v
          << "; expected warps_per_cta=" << warpsPerCta
          << " vector_size=" << vectorSize << ")";
  }

  // 4.  Assemble PTX → CUBIN.
  std::vector<char> cubin;
  std::string err;
  if (failed(compilePtxToCubin(ptx, cubinOpts, cubin, err)))
    return module.emitError("compileKunIrToExecutable: ") << err;

  // 5.  Populate `out`.  graphInputs / graphOutputs are caller-supplied
  //     after this returns — leave them empty.
  out = ::kun_cuda::ExecutableData{};
  out.cubin       = std::move(cubin);
  out.warpsPerCta = warpsPerCta;
  out.vectorSize  = vectorSize;
  out.kernels     = std::move(kernels);
  return success();
}

} // namespace kungpu
