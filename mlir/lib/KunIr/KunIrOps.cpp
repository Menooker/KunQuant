#include "KunIr/KunIrOps.h"
#include "KunIr/KunIrAttrs.h"
#include "KunIr/KunIrInterfaces.h"
#include "KunIr/KunIrTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include <limits>

using namespace mlir;
using namespace kunir;

static constexpr uint64_t kInfLookback = std::numeric_limits<uint64_t>::max();

//===----------------------------------------------------------------------===//
// Interface table (generated)
//===----------------------------------------------------------------------===//

#include "KunIr/KunIrInterfaces.cpp.inc"

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

void kunir::ReturnOp::build(mlir::OpBuilder &, mlir::OperationState &) {
  // Empty build: produces a zero-operand return for ensureTerminator.
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
LogicalResult EqualOp::verify()        { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult GreaterOp::verify()      { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult GreaterEqualOp::verify() { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult LessOp::verify()         { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }
LogicalResult LessEqualOp::verify()    { return verifyBinaryElemwise(*this, getLhs(), getRhs()); }

// Logical ops also require both operands to be i1 ts.
static LogicalResult verifyLogicalBinary(Operation *op, Value lhs, Value rhs) {
  if (failed(verifyBinaryElemwise(op, lhs, rhs)))
    return failure();
  auto elemTy = llvm::cast<TsType>(lhs.getType()).getElementType();
  if (!elemTy.isInteger(1))
    return op->emitOpError("operand element type must be i1, got '")
           << elemTy << "'";
  return success();
}
LogicalResult AndOp::verify() { return verifyLogicalBinary(*this, getLhs(), getRhs()); }
LogicalResult OrOp::verify()  { return verifyLogicalBinary(*this, getLhs(), getRhs()); }

//===----------------------------------------------------------------------===//
// Unary elemwise ops — verify only
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::verify()  { return success(); }
LogicalResult LogOp::verify()  { return success(); }
LogicalResult ExpOp::verify()  { return success(); }
LogicalResult SqrtOp::verify() { return success(); }
LogicalResult SignOp::verify() { return success(); }

LogicalResult NotOp::verify() {
  auto elemTy = llvm::cast<TsType>(getInput().getType()).getElementType();
  if (!elemTy.isInteger(1))
    return emitOpError("operand element type must be i1, got '")
           << elemTy << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp — cond must be ts<i1, *>; true/false must share elem type.
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  auto condTy  = llvm::cast<TsType>(getCond().getType());
  auto trueTy  = llvm::cast<TsType>(getTrueValue().getType());
  auto falseTy = llvm::cast<TsType>(getFalseValue().getType());
  if (!condTy.getElementType().isInteger(1))
    return emitOpError("cond element type must be i1, got '")
           << condTy.getElementType() << "'";
  if (trueTy.getElementType() != falseTy.getElementType())
    return emitOpError("true_value element type '")
           << trueTy.getElementType()
           << "' must match false_value element type '"
           << falseTy.getElementType() << "'";
  return success();
}

// Result type: ts<true_value.elem, 1>.
LogicalResult SelectOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange,
    SmallVectorImpl<Type> &inferred) {
  auto trueTy = llvm::cast<TsType>(operands[1].getType());
  inferred.push_back(TsType::get(ctx, trueTy.getElementType(), 1));
  return success();
}

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
LogicalResult ReduceArgMinOp::verify() { return verifyInsideForEachBackWindow(*this); }
LogicalResult ReduceArgMaxOp::verify() { return verifyInsideForEachBackWindow(*this); }
LogicalResult ReduceRankOp::verify()   { return verifyInsideForEachBackWindow(*this); }
LogicalResult WindowLoopIndexOp::verify() {
  return verifyInsideForEachBackWindow(*this);
}

//===----------------------------------------------------------------------===//
// BackRef + FastWindowedSum — share a verifier (same shape / constraints)
//===----------------------------------------------------------------------===//

static LogicalResult
verifyWindowedScalarOrTsResultOp(Operation *op, Value input, int64_t window,
                                  Type resultTy) {
  auto inputTy = llvm::cast<TsType>(input.getType());
  if (window <= 0)
    return op->emitOpError("window must be positive, got ") << window;

  // Need both the current value and the value `window` steps back, so the
  // input must retain at least `window + 1` time steps.
  uint64_t need = static_cast<uint64_t>(window) + 1;
  uint64_t have = inputTy.getMaxLookback();
  if (have != kInfLookback && have < need)
    return op->emitOpError("input.maxLookback (")
           << have << ") must be >= window+1 (" << need << ")";

  // Result type: either ts<inputElemType, 1> (source form) or the input's
  // element type itself (lowered form, after kunir-to-kungpu).
  Type elemTy = inputTy.getElementType();
  if (auto resTs = llvm::dyn_cast<TsType>(resultTy)) {
    if (resTs.getElementType() != elemTy)
      return op->emitOpError("result element type '")
             << resTs.getElementType()
             << "' must match input element type '" << elemTy << "'";
    if (resTs.getMaxLookback() != 1)
      return op->emitOpError("result maxLookback must be 1, got ")
             << resTs.getMaxLookback();
    return success();
  }
  if (resultTy != elemTy)
    return op->emitOpError(
               "scalar result type must equal input element type '")
           << elemTy << "', got '" << resultTy << "'";
  return success();
}

LogicalResult BackRefOp::verify() {
  return verifyWindowedScalarOrTsResultOp(*this, getInput(), getWindow(),
                                            getResult().getType());
}
LogicalResult FastWindowedSumOp::verify() {
  return verifyWindowedScalarOrTsResultOp(*this, getInput(), getWindow(),
                                            getResult().getType());
}

//===----------------------------------------------------------------------===//
// ConstantOp — result must be ts<T, 1>.  The value attr is f64; we don't
// pre-check finiteness so that quiet-NaN (0x7FF8...) can flow through.
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto resultTy = llvm::cast<TsType>(getResult().getType());
  if (resultTy.getMaxLookback() != 1)
    return emitOpError("result maxLookback must be 1, got ")
           << resultTy.getMaxLookback();
  return success();
}

//===----------------------------------------------------------------------===//
// AccumulatorOp / SetAccumulatorOp
//===----------------------------------------------------------------------===//

LogicalResult AccumulatorOp::verify() {
  auto resultTy = llvm::cast<TsType>(getResult().getType());
  if (resultTy.getMaxLookback() != 1)
    return emitOpError("accumulator result maxLookback must be 1, got ")
           << resultTy.getMaxLookback();
  if (getName().empty())
    return emitOpError("accumulator name must be non-empty");
  return success();
}

LogicalResult SetAccumulatorOp::verify() {
  auto *accOp = getAcc().getDefiningOp();
  if (!accOp || !llvm::isa<AccumulatorOp>(accOp))
    return emitOpError(
        "first operand must be the result of a 'kunir.accumulator'");
  auto accTy   = llvm::cast<TsType>(getAcc().getType());
  auto maskTy  = llvm::cast<TsType>(getMask().getType());
  auto valueTy = llvm::cast<TsType>(getValue().getType());
  if (accTy.getElementType() != valueTy.getElementType())
    return emitOpError("value element type '")
           << valueTy.getElementType()
           << "' must match accumulator element type '"
           << accTy.getElementType() << "'";
  if (!llvm::isa<IntegerType>(maskTy.getElementType()) ||
      llvm::cast<IntegerType>(maskTy.getElementType()).getWidth() != 1)
    return emitOpError("mask element type must be i1, got '")
           << maskTy.getElementType() << "'";
  return success();
}

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

//===----------------------------------------------------------------------===//
// BinaryArithInterface implementations
//===----------------------------------------------------------------------===//

Value AddOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::AddFOp::create(b, loc, lhs, rhs);
}
Value SubOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::SubFOp::create(b, loc, lhs, rhs);
}
Value MulOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::MulFOp::create(b, loc, lhs, rhs);
}
Value DivOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::DivFOp::create(b, loc, lhs, rhs);
}
Value MaxOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::MaximumFOp::create(b, loc, lhs, rhs);
}
Value MinOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::MinimumFOp::create(b, loc, lhs, rhs);
}

// Comparison ops: dispatch arith.cmpf for FloatType operands and
// arith.cmpi for IntegerType operands.  Verifier guarantees lhs.type == rhs.type.
static Value buildCmpScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                              arith::CmpFPredicate fp,
                              arith::CmpIPredicate ip) {
  if (llvm::isa<FloatType>(lhs.getType()))
    return arith::CmpFOp::create(b, loc, fp, lhs, rhs);
  return arith::CmpIOp::create(b, loc, ip, lhs, rhs);
}
Value GreaterOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return buildCmpScalarOp(b, loc, lhs, rhs,
                          arith::CmpFPredicate::OGT, arith::CmpIPredicate::sgt);
}
Value GreaterEqualOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return buildCmpScalarOp(b, loc, lhs, rhs,
                          arith::CmpFPredicate::OGE, arith::CmpIPredicate::sge);
}
Value LessOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return buildCmpScalarOp(b, loc, lhs, rhs,
                          arith::CmpFPredicate::OLT, arith::CmpIPredicate::slt);
}
Value LessEqualOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return buildCmpScalarOp(b, loc, lhs, rhs,
                          arith::CmpFPredicate::OLE, arith::CmpIPredicate::sle);
}
Value EqualOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return buildCmpScalarOp(b, loc, lhs, rhs,
                          arith::CmpFPredicate::OEQ, arith::CmpIPredicate::eq);
}

// Logical binary ops on i1.
Value AndOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::AndIOp::create(b, loc, lhs, rhs);
}
Value OrOp::buildScalarOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::OrIOp::create(b, loc, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// UnaryArithInterface implementations
//===----------------------------------------------------------------------===//

Value AbsOp::buildScalarOp(OpBuilder &b, Location loc, Value operand) {
  return math::AbsFOp::create(b, loc, operand);
}
Value LogOp::buildScalarOp(OpBuilder &b, Location loc, Value operand) {
  return math::LogOp::create(b, loc, operand);
}
Value ExpOp::buildScalarOp(OpBuilder &b, Location loc, Value operand) {
  return math::ExpOp::create(b, loc, operand);
}
Value SqrtOp::buildScalarOp(OpBuilder &b, Location loc, Value operand) {
  return math::SqrtOp::create(b, loc, operand);
}
Value SignOp::buildScalarOp(OpBuilder &b, Location loc, Value operand) {
  // sign(x) ≈ copysign(1.0, x)
  Value one = arith::ConstantOp::create(
      b, loc, operand.getType(), b.getFloatAttr(operand.getType(), 1.0));
  return math::CopySignOp::create(b, loc, one, operand);
}
Value NotOp::buildScalarOp(OpBuilder &b, Location loc, Value operand) {
  // not(x) = x ^ 1 on i1
  Value one = arith::ConstantOp::create(b, loc, b.getI1Type(),
                                            b.getIntegerAttr(b.getI1Type(), 1));
  return arith::XOrIOp::create(b, loc, operand, one);
}

//===----------------------------------------------------------------------===//
// ReduceArithInterface implementations
//===----------------------------------------------------------------------===//

TypedAttr ReduceAddOp::getInitValue(FloatType elemType) {
  return FloatAttr::get(elemType, 0.0);
}
Value ReduceAddOp::buildAccumOp(OpBuilder &b, Location loc, Value acc, Value elem) {
  return arith::AddFOp::create(b, loc, acc, elem);
}

TypedAttr ReduceMulOp::getInitValue(FloatType elemType) {
  return FloatAttr::get(elemType, 1.0);
}
Value ReduceMulOp::buildAccumOp(OpBuilder &b, Location loc, Value acc, Value elem) {
  return arith::MulFOp::create(b, loc, acc, elem);
}

TypedAttr ReduceMaxOp::getInitValue(FloatType elemType) {
  return FloatAttr::get(elemType, -std::numeric_limits<double>::infinity());
}
Value ReduceMaxOp::buildAccumOp(OpBuilder &b, Location loc, Value acc, Value elem) {
  return arith::MaximumFOp::create(b, loc, acc, elem);
}

TypedAttr ReduceMinOp::getInitValue(FloatType elemType) {
  return FloatAttr::get(elemType, std::numeric_limits<double>::infinity());
}
Value ReduceMinOp::buildAccumOp(OpBuilder &b, Location loc, Value acc, Value elem) {
  return arith::MinimumFOp::create(b, loc, acc, elem);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &b, OperationState &result,
                   StringRef name, FunctionType type,
                   ArrayAttr inputNames, ArrayAttr outputNames,
                   TargetSpecAttr targetSpec, int64_t unreliableCount) {
  result.addAttribute(getSymNameAttrName(result.name), b.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name), TypeAttr::get(type));
  result.addAttribute(getInputNamesAttrName(result.name), inputNames);
  result.addAttribute(getOutputNamesAttrName(result.name), outputNames);
  result.addAttribute(getTargetSpecAttrName(result.name), targetSpec);
  result.addAttribute(getUnreliableCountAttrName(result.name),
                        b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/true),
                                          unreliableCount));
  Region *body = result.addRegion();
  Block *block = new Block;
  for (Type inputType : type.getInputs())
    block->addArgument(inputType, result.location);
  body->push_back(block);
}

LogicalResult FuncOp::verify() {
  FunctionType ft = getFunctionTypeTyped();
  Block &block = getBodyBlock();

  // Block args must match function input types
  if (block.getNumArguments() != ft.getNumInputs())
    return emitOpError("body block has ") << block.getNumArguments()
           << " args but function type has " << ft.getNumInputs() << " inputs";
  for (auto [i, argType] : llvm::enumerate(ft.getInputs())) {
    if (block.getArgument(i).getType() != argType)
      return emitOpError("block arg #") << i << " type mismatch";
  }

  // Validate input_names / output_names counts
  auto inputNames  = getInputNames();
  auto outputNames = getOutputNames();
  unsigned numResults = ft.getNumResults();

  if (numResults > 0) {
    // Non-void: inputs == num_args, outputs == num_results
    if (inputNames.size() != ft.getNumInputs())
      return emitOpError("non-void func: input_names count (")
             << inputNames.size() << ") != num args ("
             << ft.getNumInputs() << ")";
    if (outputNames.size() != numResults)
      return emitOpError("non-void func: output_names count (")
             << outputNames.size() << ") != num results (" << numResults << ")";
  } else {
    // Void: inputs + outputs == num_args
    if (inputNames.size() + outputNames.size() != ft.getNumInputs())
      return emitOpError("void func: input_names + output_names count (")
             << (inputNames.size() + outputNames.size())
             << ") != num args (" << ft.getNumInputs() << ")";
  }

  // Validate all names are StringAttr
  for (auto [i, a] : llvm::enumerate(inputNames))
    if (!llvm::isa<StringAttr>(a))
      return emitOpError("input_names[") << i << "] is not a StringAttr";
  for (auto [i, a] : llvm::enumerate(outputNames))
    if (!llvm::isa<StringAttr>(a))
      return emitOpError("output_names[") << i << "] is not a StringAttr";

  // Validate target_spec
  auto ts = getTargetSpec();
  if (ts.getOccupancy() <= 0)
    return emitOpError("target occupancy must be positive, got ")
           << ts.getOccupancy();
  if (ts.getWarpsPerCta() <= 0)
    return emitOpError("target warps_per_cta must be positive, got ")
           << ts.getWarpsPerCta();
  if (ts.getSmemSize() < 0)
    return emitOpError("target smem_size must be non-negative, got ")
           << ts.getSmemSize();

  // Validate unreliable_count.  `-1` is a sentinel meaning "whole time
  // history required" — the runtime collapses such functions to a
  // single chunk.  Any other negative value is rejected.
  if (getUnreliableCount() < -1)
    return emitOpError("unreliable_count must be -1 (whole-time) or "
                       "non-negative, got ")
           << getUnreliableCount();

  return success();
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &b = parser.getBuilder();

  // @sym_name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  // (%arg0 : type0, ...)
  SmallVector<OpAsmParser::Argument> blockArgs;
  if (parser.parseArgumentList(blockArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  // inputs { %name = "str", ... }
  SmallVector<Attribute> inputNameAttrs;
  if (parser.parseKeyword("inputs") || parser.parseLBrace())
    return failure();
  if (parser.parseOptionalRBrace().failed()) {
    do {
      OpAsmParser::UnresolvedOperand argRef;
      StringAttr nameStr;
      if (parser.parseOperand(argRef) || parser.parseEqual() ||
          parser.parseAttribute(nameStr))
        return failure();
      inputNameAttrs.push_back(nameStr);
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRBrace()) return failure();
  }

  // outputs { ["str", ...] | [%name = "str", ...] }
  SmallVector<Attribute> outputNameAttrs;
  if (parser.parseKeyword("outputs") || parser.parseLBrace())
    return failure();
  if (parser.parseOptionalRBrace().failed()) {
    do {
      // Try %name = "str" form; if no %, fall through to "str" form
      OpAsmParser::UnresolvedOperand argRef;
      auto optArg = parser.parseOptionalOperand(argRef);
      if (optArg.has_value()) {
        if (failed(*optArg) || parser.parseEqual()) return failure();
      }
      StringAttr nameStr;
      if (parser.parseAttribute(nameStr)) return failure();
      outputNameAttrs.push_back(nameStr);
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRBrace()) return failure();
  }

  // target { occupancy = V, warps_per_cta = V, smem_size = V }
  if (parser.parseKeyword("target")) return failure();
  auto targetSpec = TargetSpecAttr::parse(parser, Type{});
  if (!targetSpec) return failure();
  result.addAttribute(getTargetSpecAttrName(result.name), targetSpec);

  // unreliable_count = N
  if (parser.parseKeyword("unreliable_count") || parser.parseEqual())
    return failure();
  int64_t unrelVal = 0;
  if (parser.parseInteger(unrelVal)) return failure();
  result.addAttribute(getUnreliableCountAttrName(result.name),
                       b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/true),
                                         unrelVal));

  // -> (result_type, ...) or -> result_type  [optional]
  SmallVector<Type> resultTypes;
  if (parser.parseOptionalArrow().succeeded()) {
    if (parser.parseOptionalLParen().succeeded()) {
      if (!parser.parseOptionalRParen().succeeded()) {
        if (parser.parseTypeList(resultTypes) || parser.parseRParen())
          return failure();
      }
    } else {
      Type singleTy;
      if (parser.parseType(singleTy)) return failure();
      resultTypes.push_back(singleTy);
    }
  }

  // Build function type from block arg types + result types
  SmallVector<Type> inputTypes;
  for (auto &arg : blockArgs) inputTypes.push_back(arg.type);
  auto funcType = FunctionType::get(result.getContext(), inputTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(funcType));
  result.addAttribute(getInputNamesAttrName(result.name),
                      b.getArrayAttr(inputNameAttrs));
  result.addAttribute(getOutputNamesAttrName(result.name),
                      b.getArrayAttr(outputNameAttrs));

  // Body region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs)) return failure();
  FuncOp::ensureTerminator(*body, b, result.location);
  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  Block &block = getBodyBlock();
  FunctionType ft = getFunctionTypeTyped();

  // @name
  p << " @" << getSymName();

  // (%arg0 : type0, ...)
  p << "(";
  llvm::interleaveComma(block.getArguments(), p, [&](BlockArgument arg) {
    p << arg << ": " << arg.getType();
  });
  p << ")";

  // inputs {%arg0 = "name0", ...}
  auto inputNames = getInputNames();
  unsigned numInputs = inputNames.size();
  p << " inputs {";
  for (unsigned i = 0; i < numInputs; ++i) {
    if (i) p << ", ";
    p << block.getArgument(i) << " = "
      << llvm::cast<StringAttr>(inputNames[i]);
  }
  p << "}";

  // outputs {...}
  auto outputNames = getOutputNames();
  p << " outputs {";
  if (ft.getNumResults() == 0) {
    // void: %argN = "name" form
    for (unsigned i = 0; i < outputNames.size(); ++i) {
      if (i) p << ", ";
      p << block.getArgument(numInputs + i) << " = "
        << llvm::cast<StringAttr>(outputNames[i]);
    }
  } else {
    // non-void: just "name" strings
    llvm::interleaveComma(outputNames, p,
                          [&](Attribute a) { p << llvm::cast<StringAttr>(a); });
  }
  p << "}";

  // target {occupancy = ..., ...}
  p << " target ";
  getTargetSpec().print(p);

  // unreliable_count = N
  p << " unreliable_count = " << getUnreliableCount();

  // -> result types (non-void)
  auto resultTypes = ft.getResults();
  if (!resultTypes.empty()) {
    p << " -> ";
    if (resultTypes.size() == 1) {
      p << resultTypes[0];
    } else {
      p << "(";
      llvm::interleaveComma(resultTypes, p);
      p << ")";
    }
  }

  // body
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto funcOp = llvm::cast<FuncOp>((*this)->getParentOp());
  FunctionType ft = funcOp.getFunctionTypeTyped();
  auto resultTypes = ft.getResults();

  if (getOperands().size() != resultTypes.size())
    return emitOpError("returns ") << getOperands().size()
           << " value(s) but function has " << resultTypes.size()
           << " result type(s)";

  for (auto [i, opType, resType] :
       llvm::enumerate(getOperandTypes(), resultTypes)) {
    if (opType != resType)
      return emitOpError("operand #") << i << " type '" << opType
             << "' does not match function result type '" << resType << "'";
  }
  return success();
}
