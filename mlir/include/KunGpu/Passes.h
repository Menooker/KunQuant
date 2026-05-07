#pragma once

// KunIrOps.h must be included before Passes.h.inc so that ::kunir::FuncOp
// is fully declared when the OperationPass<::kunir::FuncOp> template is used.
#include "KunIr/KunIrOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace kungpu {

#define GEN_PASS_DECL
#include "KunGpu/Passes.h.inc"

std::unique_ptr<mlir::Pass> createWindowedTempMemoryPlanningPass();
std::unique_ptr<mlir::Pass> createConvertKunGpuToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "KunGpu/Passes.h.inc"

} // namespace kungpu
