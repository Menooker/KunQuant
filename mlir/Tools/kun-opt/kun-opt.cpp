#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "KunGpu/KunGpuDialect.h"
#include "KunGpu/KunGpuOps.h"
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

  // KunQuant dialects
  registry.insert<kunir::KunIrDialect>();
  registry.insert<kungpu::KunGpuDialect>();

  // KunQuant passes
  kunir::registerKunIrToKunGpuPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "KunQuant MLIR optimizer\n", registry));
}
