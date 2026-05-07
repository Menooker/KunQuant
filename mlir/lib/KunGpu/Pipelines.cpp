//===- Pipelines.cpp - kunir → LLVM lowering pipeline --------------------===//

#include "KunGpu/KunGpuDialect.h"
#include "KunGpu/Passes.h"
#include "KunGpu/Pipelines.h"
#include "KunIr/KunIrOps.h"
#include "KunIr/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace kungpu {

void buildKunIrToLLVMPipeline(OpPassManager &pm) {
  // pm's anchor is builtin.module; the kernels live one level down inside
  // a gpu.module, so kunir.func / gpu.func passes must nest through it.

  // ── 1–2.  Per-kunir.func passes (nested: gpu.module → kunir.func) ────
  {
    OpPassManager &gpuModPM = pm.nest<gpu::GPUModuleOp>();
    gpuModPM.addNestedPass<::kunir::FuncOp>(::kunir::createKunIrToKunGpuPass());
    gpuModPM.addNestedPass<::kunir::FuncOp>(
        ::kungpu::createWindowedTempMemoryPlanningPass());
  }

  // ── 3.  kunir.func → gpu.func + kungpu ops → LLVM (module-level) ─────
  pm.addPass(::kungpu::createConvertKunGpuToLLVMPass());

  // ── 4.  LICM per gpu.func (nested: gpu.module → gpu.func) ────────────
  {
    OpPassManager &gpuModPM = pm.nest<gpu::GPUModuleOp>();
    gpuModPM.addNestedPass<gpu::GPUFuncOp>(createLoopInvariantCodeMotionPass());
  }

  // ── 5–6.  Generic cleanup ─────────────────────────────────────────────
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // ── 7.  scf → cf (control flow) ───────────────────────────────────────
  pm.addPass(createSCFToControlFlowPass());

  // ── 8.  index / arith / cf → LLVM, in order.  These lower the device-
  //       side body of gpu.func before gpu-to-nvvm, so the latter only
  //       has to deal with gpu ops + the gpu.func wrapper.
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());

  // ── 9.  gpu.thread_id / block_id / block_dim → nvvm intrinsics, plus
  //       gpu.func → llvm.func (with `nvvm.kernel`).
  // indexBitwidth = 32 matches our function-signature i32 (no spurious
  // sext/trunc around the i32 NVVM intrinsics).
  {
    ConvertGpuOpsToNVVMOpsOptions gpuOpts;
    gpuOpts.indexBitwidth = 32;
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(gpuOpts));
  }

  // ── 10.  func.func → llvm.func (host-side helpers, if any).
  pm.addPass(createConvertFuncToLLVMPass());

  // ── 11.  Resolve any leftover unrealized_conversion_casts ──────────
  pm.addPass(createReconcileUnrealizedCastsPass());
}

namespace {

// Lit-test wrapper: runs the whole pipeline as a single -kunir-to-llvm pass.
struct KunIrToLLVMPass
    : PassWrapper<KunIrToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KunIrToLLVMPass)
  StringRef getArgument()    const override { return "kunir-to-llvm"; }
  StringRef getDescription() const override {
    return "Lower kunir.func down to the LLVM dialect (test wrapper)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    // Pulls in everything the nested pipeline will create / load.
    registry.insert<::kungpu::KunGpuDialect, scf::SCFDialect,
                    arith::ArithDialect, math::MathDialect,
                    func::FuncDialect, gpu::GPUDialect,
                    LLVM::LLVMDialect, NVVM::NVVMDialect,
                    cf::ControlFlowDialect, index::IndexDialect>();
  }

  void runOnOperation() override {
    OpPassManager pm("builtin.module");
    buildKunIrToLLVMPipeline(pm);
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createKunIrToLLVMPass() {
  return std::make_unique<KunIrToLLVMPass>();
}

void registerKunIrToLLVMPass() {
  PassRegistration<KunIrToLLVMPass>();
}

} // namespace kungpu
