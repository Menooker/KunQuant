#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

// Emit class declarations (storage struct is only forward-declared here;
// the full definition lives in KunIrTypes.cpp via KunIrOpsTypes.cpp.inc).
#define GET_TYPEDEF_CLASSES
#include "KunIr/KunIrOpsTypes.h.inc"
