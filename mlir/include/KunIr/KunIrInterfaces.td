#ifndef KUNIR_INTERFACES_TD
#define KUNIR_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BinaryArithInterface
// Implemented by binary elemwise ops (add, sub, mul, div, max, min).
//===----------------------------------------------------------------------===//

def KunIr_BinaryArithInterface : OpInterface<"BinaryArithInterface"> {
  let cppNamespace = "::kunir";
  let description = [{
    Factory interface for binary kunir ops.  Each implementing op knows how to
    lower itself to its corresponding scalar arith/math op.
  }];
  let methods = [
    InterfaceMethod<
      "Build the scalar arith/math op corresponding to this binary kunir op.",
      "::mlir::Value", "buildScalarOp",
      (ins "::mlir::OpBuilder &":$b, "::mlir::Location":$loc,
           "::mlir::Value":$lhs, "::mlir::Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// UnaryArithInterface
// Implemented by unary elemwise ops (abs, log, sign).
//===----------------------------------------------------------------------===//

def KunIr_UnaryArithInterface : OpInterface<"UnaryArithInterface"> {
  let cppNamespace = "::kunir";
  let description = [{
    Factory interface for unary kunir ops.  Each implementing op knows how to
    lower itself to its corresponding scalar arith/math op.
  }];
  let methods = [
    InterfaceMethod<
      "Build the scalar arith/math op corresponding to this unary kunir op.",
      "::mlir::Value", "buildScalarOp",
      (ins "::mlir::OpBuilder &":$b, "::mlir::Location":$loc,
           "::mlir::Value":$operand)>
  ];
}

//===----------------------------------------------------------------------===//
// ReduceArithInterface
// Implemented by reduce_* ops (reduce_add, reduce_mul, reduce_max, reduce_min).
//===----------------------------------------------------------------------===//

def KunIr_ReduceArithInterface : OpInterface<"ReduceArithInterface"> {
  let cppNamespace = "::kunir";
  let description = [{
    Factory interface for reduction kunir ops.  Provides the identity element
    for iter_args initialisation and a factory that emits one accumulation step.
  }];
  let methods = [
    InterfaceMethod<
      "Return the identity (init) value for this reduction as a typed float attribute.",
      "::mlir::TypedAttr", "getInitValue",
      (ins "::mlir::FloatType":$elemType)>,
    InterfaceMethod<
      "Build one accumulation step: newAcc = reduce_op(oldAcc, element).",
      "::mlir::Value", "buildAccumOp",
      (ins "::mlir::OpBuilder &":$b, "::mlir::Location":$loc,
           "::mlir::Value":$acc, "::mlir::Value":$elem)>
  ];
}

#endif // KUNIR_INTERFACES_TD
