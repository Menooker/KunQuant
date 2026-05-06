#include "KunIr/KunIrAttrs.h"
#include "KunIr/KunIrDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace kunir;

#define GET_ATTRDEF_CLASSES
#include "KunIr/KunIrOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TargetSpecAttr — custom assembly format
//
// Inline format (used inside kunir.func):
//   {occupancy = V, warps_per_cta = V, smem_size = V}
//
// Canonical MLIR attribute form (used stand-alone):
//   #kunir.target_spec<{occupancy = V, warps_per_cta = V, smem_size = V}>
//===----------------------------------------------------------------------===//

Attribute TargetSpecAttr::parse(AsmParser &parser, Type) {
  int64_t occupancy = 0, warpsPerCta = 0, smemSize = 0, vectorSize = 1;
  if (parser.parseLBrace() ||
      parser.parseKeyword("occupancy") || parser.parseEqual() ||
      parser.parseInteger(occupancy) || parser.parseComma() ||
      parser.parseKeyword("warps_per_cta") || parser.parseEqual() ||
      parser.parseInteger(warpsPerCta) || parser.parseComma() ||
      parser.parseKeyword("smem_size") || parser.parseEqual() ||
      parser.parseInteger(smemSize) || parser.parseComma() ||
      parser.parseKeyword("vector_size") || parser.parseEqual() ||
      parser.parseInteger(vectorSize) || parser.parseRBrace())
    return {};
  return TargetSpecAttr::get(parser.getContext(), occupancy, warpsPerCta,
                              smemSize, vectorSize);
}

void TargetSpecAttr::print(AsmPrinter &printer) const {
  printer << "{occupancy = " << getOccupancy()
          << ", warps_per_cta = " << getWarpsPerCta()
          << ", smem_size = " << getSmemSize()
          << ", vector_size = " << getVectorSize() << "}";
}

//===----------------------------------------------------------------------===//
// Dialect attr registration
//===----------------------------------------------------------------------===//

void KunIrDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KunIr/KunIrOpsAttrDefs.cpp.inc"
  >();
}
