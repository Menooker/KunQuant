#include "KunIr/KunIrAttrs.h"
#include "KunIr/KunIrDialect.h"
#include "KunIr/KunIrOps.h"
#include "KunIr/KunIrTypes.h"

using namespace mlir;
using namespace kunir;

//===----------------------------------------------------------------------===//
// KunIr dialect
//===----------------------------------------------------------------------===//

#include "KunIr/KunIrOpsDialect.cpp.inc"

void KunIrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KunIr/KunIrOps.cpp.inc"
  >();
  registerTypes();
  registerAttrs();
}
