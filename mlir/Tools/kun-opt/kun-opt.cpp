#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "KunGpu/KunGpuDialect.h"
#include "KunGpu/KunGpuOps.h"
#include "KunIr/KunIrDialect.h"
#include "KunIr/KunIrOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Core dialects used by kunir/kungpu
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();

  // KunQuant dialects
  registry.insert<kunir::KunIrDialect>();
  registry.insert<kungpu::KunGpuDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "KunQuant MLIR optimizer\n", registry));
}
