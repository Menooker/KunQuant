#include "KunGpu/KunGpuOps.h"
#include "KunIr/KunIrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace kungpu;

// Emits op class method implementations (verifyInvariantsImpl, print, parse, etc.)
#define GET_OP_CLASSES
#include "KunGpu/KunGpuOps.cpp.inc"

// The `ts` operand of ts.get and ts.put must be a function argument (block
// argument of an entry block), the result of a windowed_temp op, or the
// result of an accumulator op.
static bool isValidTsSource(Value v) {
  if (isa<BlockArgument>(v))
    return true;
  if (auto *def = v.getDefiningOp())
    return isa<WindowedTempOp, AccumulatorOp>(def);
  return false;
}

//===----------------------------------------------------------------------===//
// TsGetOp
//===----------------------------------------------------------------------===//

LogicalResult TsGetOp::verify() {
  auto tsTy = llvm::cast<kunir::TsType>(getTs().getType());
  if (tsTy.getElementType() != getResult().getType())
    return emitOpError("result type '")
           << getResult().getType()
           << "' must match ts element type '" << tsTy.getElementType() << "'";
  if (!isValidTsSource(getTs()))
    return emitOpError("ts operand must be a function argument or "
                       "the result of 'kungpu.windowed_temp' / "
                       "'kungpu.accumulator'");
  return success();
}

//===----------------------------------------------------------------------===//
// TsPutOp
//===----------------------------------------------------------------------===//

LogicalResult TsPutOp::verify() {
  auto tsTy = llvm::cast<kunir::TsType>(getTs().getType());
  if (tsTy.getElementType() != getValue().getType())
    return emitOpError("value type '")
           << getValue().getType()
           << "' must match ts element type '" << tsTy.getElementType() << "'";
  if (!isValidTsSource(getTs()))
    return emitOpError("ts operand must be a function argument or "
                       "the result of 'kungpu.windowed_temp' / "
                       "'kungpu.accumulator'");
  return success();
}
