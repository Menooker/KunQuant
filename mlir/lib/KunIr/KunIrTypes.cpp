#include "KunIr/KunIrTypes.h"
#include "KunIr/KunIrDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>

using namespace mlir;
using namespace kunir;

// Emits full TsTypeStorage definition + TypeBase method implementations.
#define GET_TYPEDEF_CLASSES
#include "KunIr/KunIrOpsTypes.cpp.inc"

static constexpr uint64_t kInfLookback = std::numeric_limits<uint64_t>::max();

// Custom assembly format: !kunir.ts<elemType, N>  or  !kunir.ts<elemType, inf>
mlir::Type TsType::parse(mlir::AsmParser &parser) {
  mlir::Type elemType;
  uint64_t maxLookback;

  if (parser.parseLess() || parser.parseType(elemType) || parser.parseComma())
    return {};

  if (parser.parseOptionalKeyword("inf").succeeded()) {
    maxLookback = kInfLookback;
  } else {
    if (parser.parseInteger(maxLookback))
      return {};
  }

  if (parser.parseGreater())
    return {};

  return TsType::get(parser.getContext(), elemType, maxLookback);
}

void TsType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getElementType() << ", ";
  if (getMaxLookback() == kInfLookback)
    printer << "inf";
  else
    printer << getMaxLookback();
  printer << ">";
}

void KunIrDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KunIr/KunIrOpsTypes.cpp.inc"
  >();
}
