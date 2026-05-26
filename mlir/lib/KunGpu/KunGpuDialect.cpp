#include "KunGpu/KunGpuDialect.h"
#include "KunGpu/KunGpuOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace kungpu;

//===----------------------------------------------------------------------===//
// KunGpu dialect
//===----------------------------------------------------------------------===//

#include "KunGpu/KunGpuOpsDialect.cpp.inc"

void KunGpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KunGpu/KunGpuOps.cpp.inc"
  >();
}
