//===- KunGpuUtils.h - Lookup helpers for kungpu metadata on func ops ----===//
//
// After convert-kungpu-to-llvm phase 1 lowers `kunir.func` to `func.func`,
// the original kunir.func metadata (target spec, input/output names) is
// preserved as discardable attributes on the new func.func.  Use these
// accessors instead of reading attributes by name in callers — they are
// the func.func equivalents of `kunir::FuncOp::getTargetSpec` etc.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "KunIr/KunIrAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

namespace kungpu {

/// Discardable-attribute names used to attach kunir.func metadata to a
/// func.func after phase 1 of convert-kungpu-to-llvm.
constexpr llvm::StringLiteral kFuncTargetSpecAttr  = "kungpu.target_spec";
constexpr llvm::StringLiteral kFuncInputNamesAttr  = "kungpu.input_names";
constexpr llvm::StringLiteral kFuncOutputNamesAttr = "kungpu.output_names";

/// Read the target_spec attribute from a func.func (lowered from a
/// kunir.func).  Returns null if the attribute is missing.
inline ::kunir::TargetSpecAttr
getFuncTargetSpec(::mlir::func::FuncOp fn) {
  return fn->getAttrOfType<::kunir::TargetSpecAttr>(kFuncTargetSpecAttr);
}
inline void setFuncTargetSpec(::mlir::func::FuncOp fn,
                                ::kunir::TargetSpecAttr spec) {
  fn->setAttr(kFuncTargetSpecAttr, spec);
}

/// Read the input_names array attribute from a func.func.  The array
/// contains one StringAttr per ts input parameter; null if missing.
inline ::mlir::ArrayAttr
getFuncInputNames(::mlir::func::FuncOp fn) {
  return fn->getAttrOfType<::mlir::ArrayAttr>(kFuncInputNamesAttr);
}
inline void setFuncInputNames(::mlir::func::FuncOp fn,
                                ::mlir::ArrayAttr names) {
  fn->setAttr(kFuncInputNamesAttr, names);
}

/// Read the output_names array attribute from a func.func.  The array
/// contains one StringAttr per ts output (function arg in void form, or
/// result in non-void form); null if missing.
inline ::mlir::ArrayAttr
getFuncOutputNames(::mlir::func::FuncOp fn) {
  return fn->getAttrOfType<::mlir::ArrayAttr>(kFuncOutputNamesAttr);
}
inline void setFuncOutputNames(::mlir::func::FuncOp fn,
                                 ::mlir::ArrayAttr names) {
  fn->setAttr(kFuncOutputNamesAttr, names);
}

} // namespace kungpu
