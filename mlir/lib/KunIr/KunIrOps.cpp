#include "KunIr/KunIrOps.h"
#include "KunIr/KunIrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include <limits>

using namespace mlir;
using namespace kunir;

static constexpr uint64_t kInfLookback = std::numeric_limits<uint64_t>::max();

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KunIr/KunIrOps.cpp.inc"

//===----------------------------------------------------------------------===//
// YieldOp — manual zero-arg build (declared by OpBuilder<(ins), [{}]>)
//===----------------------------------------------------------------------===//

void kunir::YieldOp::build(mlir::OpBuilder &, mlir::OperationState &) {
  // Empty build: produces a zero-operand yield for ensureTerminator.
}

//===----------------------------------------------------------------------===//
// Binary elemwise ops — verify only (inferReturnTypes is in ElemwiseTsResultType)
//===----------------------------------------------------------------------===//

// Shared verifier: both inputs must share the same element type.
static LogicalResult verifyBinaryElemwise(Operation *op,
                                          Value lhs, Value rhs) {
  auto lhsTy = llvm::cast<TsType>(lhs.getType());
  auto rhsTy = llvm::cast<TsType>(rhs.getType());
  if (lhsTy.getElementType() != rhsTy.getElementType())
    return op->emitOpError("lhs element type '")
           << lhsTy.getElementType() << "' must match rhs element type '"
           << rhsTy.getElementType() << "'";
  return success();
}

LogicalResult AddOp::verify() { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult SubOp::verify() { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult MulOp::verify() { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult DivOp::verify() { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult MaxOp::verify() { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult MinOp::verify() { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }

//===----------------------------------------------------------------------===//
// Unary elemwise ops + CsRankOp — verify only
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::verify()    { return success(); }
LogicalResult LogOp::verify()    { return success(); }
LogicalResult SignOp::verify()   { return success(); }
LogicalResult CsRankOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// WindowedOutputOp
//===----------------------------------------------------------------------===//

LogicalResult WindowedOutputOp::verify() {
  auto inputTy  = llvm::cast<TsType>(getInput().getType());
  auto resultTy = llvm::cast<TsType>(getResult().getType());

  if (inputTy.getElementType() != resultTy.getElementType())
    return emitOpError("result element type '")
           << resultTy.getElementType()
           << "' must match input element type '"
           << inputTy.getElementType() << "'";

  int64_t len = getLength();
  if (len <= 0)
    return emitOpError("length must be positive, got ") << len;

  if (resultTy.getMaxLookback() != static_cast<uint64_t>(len))
    return emitOpError("result maxLookback (")
           << resultTy.getMaxLookback()
           << ") must equal length attribute (" << len << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// Reduce ops — verify they are inside a ForEachBackWindow body
//
// Uses Operation* directly (no template); SameOperandsAndResultType already
// enforces input == result type, so only the parent check is needed.
//===----------------------------------------------------------------------===//

static LogicalResult verifyInsideForEachBackWindow(Operation *op) {
  if (!llvm::isa_and_nonnull<ForEachBackWindowOp>(op->getParentOp()))
    return op->emitOpError(
        "must be directly inside a 'kunir.for_each_back_window' region");
  return success();
}

LogicalResult ReduceAddOp::verify() { return verifyInsideForEachBackWindow(*this); }
LogicalResult ReduceMulOp::verify() { return verifyInsideForEachBackWindow(*this); }
LogicalResult ReduceMaxOp::verify() { return verifyInsideForEachBackWindow(*this); }
LogicalResult ReduceMinOp::verify() { return verifyInsideForEachBackWindow(*this); }

//===----------------------------------------------------------------------===//
// ForEachBackWindowOp — verifier + custom assembly format
//
// Format:
//   %r = kunir.for_each_back_window
//       (%in0 : !kunir.ts<f32, 10>, %in1 : !kunir.ts<f32, 10>)
//       [window = 5]
//       (%cur0 : !kunir.ts<f32, 1>, %cur1 : !kunir.ts<f32, 1>)
//       -> (!kunir.ts<f32, 1>) {
//     %s = kunir.reduce_add %cur0 : !kunir.ts<f32, 1>
//     kunir.yield %s : !kunir.ts<f32, 1>
//   }
//===----------------------------------------------------------------------===//

LogicalResult ForEachBackWindowOp::verify() {
  int64_t win = getWindow();
  if (win <= 0)
    return emitOpError("window must be positive, got ") << win;

  auto inputs = getInputs();
  Block &bodyBlock = getBody().front();

  // Each input's maxLookback must be >= window.
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    auto inputTy = llvm::cast<TsType>(input.getType());
    uint64_t lookback = inputTy.getMaxLookback();
    if (lookback != kInfLookback && lookback < static_cast<uint64_t>(win))
      return emitOpError("input #")
             << idx << " maxLookback (" << lookback
             << ") must be >= window (" << win << ")";
  }

  // Block must have exactly one arg per input, typed ts<elemType_i, 1>.
  if (bodyBlock.getNumArguments() != inputs.size())
    return emitOpError("body block has ")
           << bodyBlock.getNumArguments()
           << " argument(s) but op has " << inputs.size() << " input(s)";

  for (auto [idx, input] : llvm::enumerate(inputs)) {
    auto inputTy = llvm::cast<TsType>(input.getType());
    Type expectedArgTy =
        TsType::get(getContext(), inputTy.getElementType(), 1);
    Type actualArgTy = bodyBlock.getArgument(idx).getType();
    if (actualArgTy != expectedArgTy)
      return emitOpError("body block argument #")
             << idx << " must have type '" << expectedArgTy
             << "', got '" << actualArgTy << "'";
  }

  // Body must terminate with YieldOp.
  auto yieldOp = llvm::dyn_cast<YieldOp>(bodyBlock.getTerminator());
  if (!yieldOp)
    return emitOpError("body must terminate with 'kunir.yield'");

  // results count == yield operands count.
  unsigned numResults = getNumResults();
  if (yieldOp.getValues().size() != numResults)
    return emitOpError("yield operands count (")
           << yieldOp.getValues().size()
           << ") must match op results count (" << numResults << ")";

  // Every result and yield operand must be ts<elemType, 1>.
  for (auto [idx, res] : llvm::enumerate(getResults())) {
    auto resTy = llvm::dyn_cast<TsType>(res.getType());
    if (!resTy)
      return emitOpError("result #") << idx << " must be a kunir ts type";
    if (resTy.getMaxLookback() != 1)
      return emitOpError("result #") << idx << " maxLookback must be 1, got "
             << resTy.getMaxLookback();
  }

  for (auto [idx, val] : llvm::enumerate(yieldOp.getValues())) {
    auto valTy = llvm::dyn_cast<TsType>(val.getType());
    if (!valTy)
      return emitOpError("yield operand #") << idx << " must be a kunir ts type";
    if (valTy.getMaxLookback() != 1)
      return emitOpError("yield operand #") << idx
             << " maxLookback must be 1, got " << valTy.getMaxLookback();
    if (val.getType() != getResult(idx).getType())
      return emitOpError("yield operand #")
             << idx << " type '" << val.getType()
             << "' must match result type '" << getResult(idx).getType() << "'";
  }

  return success();
}

ParseResult ForEachBackWindowOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  Builder &builder = parser.getBuilder();

  // (%in0 : type0, %in1 : type1, ...)
  SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  SmallVector<Type> inputTypes;
  if (parser.parseLParen())
    return failure();
  if (parser.parseOptionalRParen().failed()) {
    do {
      OpAsmParser::UnresolvedOperand operand;
      Type type;
      if (parser.parseOperand(operand) || parser.parseColonType(type))
        return failure();
      inputOperands.push_back(operand);
      inputTypes.push_back(type);
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRParen())
      return failure();
  }
  if (parser.resolveOperands(inputOperands, inputTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  // [window = <integer>]
  int64_t window;
  if (parser.parseLSquare() || parser.parseKeyword("window") ||
      parser.parseEqual() || parser.parseInteger(window) ||
      parser.parseRSquare())
    return failure();
  result.addAttribute("window", builder.getI64IntegerAttr(window));

  // (%cur0 : ts0, %cur1 : ts1, ...)
  SmallVector<OpAsmParser::Argument> blockArgs;
  if (parser.parseArgumentList(blockArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  // -> (types) or -> type
  SmallVector<Type> resultTypes;
  if (parser.parseArrow())
    return failure();
  if (parser.parseOptionalLParen().succeeded()) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  } else {
    Type singleTy;
    if (parser.parseType(singleTy))
      return failure();
    resultTypes.push_back(singleTy);
  }
  for (Type t : resultTypes)
    result.addTypes(t);

  // { body }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs))
    return failure();
  ForEachBackWindowOp::ensureTerminator(*body, builder, result.location);
  return success();
}

void ForEachBackWindowOp::print(OpAsmPrinter &printer) {
  Block &bodyBlock = getBody().front();

  // (%in0 : type0, %in1 : type1, ...)
  printer << " (";
  llvm::interleaveComma(getInputs(), printer, [&](Value input) {
    printer << input << " : " << input.getType();
  });
  printer << ")";

  printer << " [window = " << getWindow() << "]";

  // (%cur0 : ts0, %cur1 : ts1, ...)
  printer << " (";
  llvm::interleaveComma(bodyBlock.getArguments(), printer,
                        [&](BlockArgument arg) {
                          printer.printRegionArgument(arg);
                        });
  printer << ")";

  // -> (types) or -> type
  auto resultTypes = getResultTypes();
  if (resultTypes.size() == 1) {
    printer << " -> " << resultTypes[0];
  } else {
    printer << " -> (";
    llvm::interleaveComma(resultTypes, printer);
    printer << ")";
  }

  // Body (block args already printed above).
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}
