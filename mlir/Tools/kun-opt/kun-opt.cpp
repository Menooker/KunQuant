#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "KunGpu/KunGpuDialect.h"
#include "KunGpu/Passes.h"
#include "KunGpu/Pipelines.h"
#include "KunIr/KunIrDialect.h"
#include "KunIr/KunIrOps.h"
#include "KunIr/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Core dialects used by kunir/kungpu
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::index::IndexDialect>();

  // KunQuant dialects
  registry.insert<kunir::KunIrDialect>();
  registry.insert<kungpu::KunGpuDialect>();

  // KunQuant passes & pipelines
  kunir::registerKunIrToKunGpuPass();
  kungpu::registerKunGpuPasses();
  kungpu::registerKunIrToLLVMPass();

  // Upstream passes used by the kunir-to-llvm pipeline (also lets users
  // build the pipeline manually via --pass-pipeline=… for debugging).
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertGpuOpsToNVVMOpsPass();
  mlir::registerReconcileUnrealizedCastsPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "KunQuant MLIR optimizer\n", registry));
}
