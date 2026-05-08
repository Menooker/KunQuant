//===- PtxBackend.cpp - kunir → cubin (single upstream-pass pipeline) -===//

#include "KunGpu/PtxBackend.h"
#include "KunGpu/KunGpuUtils.h"
#include "KunGpu/Pipelines.h"
#include "KunIr/KunIrAttrs.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace kungpu {

namespace {

//===----------------------------------------------------------------------===//
// Step 1: kunir → llvm dialect.  Same pipeline both compileKunIrToPtx and
// compileKunIrToExecutable need before they hand off to upstream
// gpu-module-to-binary.
//===----------------------------------------------------------------------===//

LogicalResult lowerKunIrToLLVMDialect(ModuleOp module) {
  PassManager pm(module.getContext());
  buildKunIrToLLVMPipeline(pm);
  if (failed(pm.run(module)))
    return module.emitError(
        "compileKunIr*: kunir-to-llvm pipeline failed");
  return success();
}

//===----------------------------------------------------------------------===//
// Step 2: attach #nvvm.target to the (single) gpu.module so the upstream
// pass knows what chip/O/etc. to compile for.  We do this by hand instead
// of running `nvvm-attach-target` to keep the chip / O knobs typed and
// avoid re-parsing the pass options string.
//===----------------------------------------------------------------------===//

LogicalResult attachNvvmTarget(ModuleOp module,
                                 const PtxCompileOptions &opts) {
  gpu::GPUModuleOp gpuMod;
  module.walk([&](gpu::GPUModuleOp m) {
    gpuMod = m;
    return WalkResult::interrupt();
  });
  if (!gpuMod)
    return module.emitError(
        "compileKunIr*: no gpu.module found after the kunir-to-llvm "
        "pipeline");

  MLIRContext *ctx = module.getContext();
  auto targetAttr = NVVM::NVVMTargetAttr::get(
      ctx, /*optLevel=*/static_cast<int>(opts.optLevel),
      /*triple=*/opts.targetTriple,
      /*chip=*/opts.targetCpu,
      /*features=*/opts.targetFeatures);
  gpuMod.setTargetsAttr(ArrayAttr::get(ctx, {targetAttr}));
  return success();
}

//===----------------------------------------------------------------------===//
// Step 3: run gpu-module-to-binary, then dig out the resulting object's
// payload (PTX text or cubin bytes).
//===----------------------------------------------------------------------===//

LogicalResult runGpuModuleToBinary(ModuleOp module,
                                     const std::string &compilationTarget,
                                     const std::string &toolkitPath,
                                     std::string &outBytes) {
  // The Python wrapper (`KunQuant.jit.cuda.find_cuda_toolkit`) is
  // responsible for resolving an empty toolkit path.  If it's still
  // empty here, the caller is using the C++ API directly without
  // hand-resolving — pass it on and let the upstream pass try its own
  // (limited) defaults.
  GpuModuleToBinaryPassOptions passOpts;
  passOpts.compilationTarget = compilationTarget;   // "isa" (PTX) | "bin" (cubin)
  passOpts.toolkitPath       = toolkitPath;

  PassManager pm(module.getContext());
  pm.addPass(createGpuModuleToBinaryPass(passOpts));
  if (failed(pm.run(module)))
    return module.emitError(
        "compileKunIr*: gpu-module-to-binary{format=")
        << compilationTarget << "} failed";

  // The pass replaces every gpu.module with a gpu.binary holding one
  // gpu.object per target attribute.  We attached exactly one target,
  // so we expect one binary with one object — pull its bytes out.
  gpu::BinaryOp binary;
  module.walk([&](gpu::BinaryOp op) {
    binary = op;
    return WalkResult::interrupt();
  });
  if (!binary)
    return module.emitError(
        "compileKunIr*: gpu-module-to-binary produced no gpu.binary "
        "(target attr missing on gpu.module?)");

  ArrayAttr objects = binary.getObjectsAttr();
  if (!objects || objects.empty())
    return module.emitError(
        "compileKunIr*: gpu.binary has no objects");
  auto obj = llvm::dyn_cast<gpu::ObjectAttr>(objects[0]);
  if (!obj)
    return module.emitError(
        "compileKunIr*: gpu.binary's first object is not a #gpu.object");

  StringAttr payload = obj.getObject();
  outBytes.assign(payload.getValue().begin(), payload.getValue().end());
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Public: PTX (debug / inspection)
//===----------------------------------------------------------------------===//

LogicalResult compileKunIrToPtx(ModuleOp module,
                                  const PtxCompileOptions &options,
                                  std::string &ptxOut) {
  if (failed(lowerKunIrToLLVMDialect(module))) return failure();
  if (failed(attachNvvmTarget(module, options))) return failure();
  return runGpuModuleToBinary(module, /*compilationTarget=*/"isa",
                                options.toolkitPath, ptxOut);
}

//===----------------------------------------------------------------------===//
// Public: kunir → cubin + per-kernel name metadata
//===----------------------------------------------------------------------===//

LogicalResult compileKunIrToExecutable(ModuleOp module,
                                        const PtxCompileOptions &options,
                                        ::kun_cuda::ExecutableData &out) {
  // 1.  kunir → llvm dialect.  After this the gpu.module body is fully
  //     lowered and our discardable kungpu.* attrs sit on llvm.func ops.
  if (failed(lowerKunIrToLLVMDialect(module))) return failure();

  // 2.  Walk every kernel function (carries kungpu.target_spec) and
  //     gather its name + I/O lists.  Must happen BEFORE the next pass
  //     since gpu-module-to-binary replaces the gpu.module with a
  //     gpu.binary that has no llvm.func to walk.
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

  // 3.  Validate target spec is graph-wide.
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

  // 4.  Attach #nvvm.target + run gpu-module-to-binary{format=bin}.
  if (failed(attachNvvmTarget(module, options))) return failure();
  std::string cubin;
  if (failed(runGpuModuleToBinary(module, /*compilationTarget=*/"bin",
                                    options.toolkitPath, cubin)))
    return failure();

  // 5.  Populate `out`.  graphInputs / graphOutputs are caller-supplied
  //     after this returns — leave them empty.
  out = ::kun_cuda::ExecutableData{};
  out.cubin.assign(cubin.begin(), cubin.end());
  out.warpsPerCta = warpsPerCta;
  out.vectorSize  = vectorSize;
  out.kernels     = std::move(kernels);
  return success();
}

} // namespace kungpu
