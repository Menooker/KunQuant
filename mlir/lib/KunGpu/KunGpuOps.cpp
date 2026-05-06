#include "KunGpu/KunGpuOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace kungpu;

// Emits op class method implementations (verifyInvariantsImpl, print, parse, etc.)
#define GET_OP_CLASSES
#include "KunGpu/KunGpuOps.cpp.inc"
