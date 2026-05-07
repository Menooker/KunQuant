//===- KunGpuUtils.h - Lookup helpers for kungpu metadata on func ops ----===//
//
// After convert-kungpu-to-llvm lowers `kunir.func` to `gpu.func`, the
// original kunir.func metadata (target spec, input/output names) is
// preserved as discardable attributes on the new gpu.func.  Accessors take
// `Operation*` so they also work on whatever the gpu.func is later
// rewritten to (e.g. `llvm.func` after convert-gpu-to-nvvm).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "KunIr/KunIrAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace kungpu {

/// Discardable-attribute names used to attach kunir.func metadata to the
/// kernel function after phase 1 of convert-kungpu-to-llvm.
constexpr llvm::StringLiteral kFuncTargetSpecAttr  = "kungpu.target_spec";
constexpr llvm::StringLiteral kFuncInputNamesAttr  = "kungpu.input_names";
constexpr llvm::StringLiteral kFuncOutputNamesAttr = "kungpu.output_names";

inline ::kunir::TargetSpecAttr getFuncTargetSpec(::mlir::Operation *fn) {
  return fn->getAttrOfType<::kunir::TargetSpecAttr>(kFuncTargetSpecAttr);
}
inline void setFuncTargetSpec(::mlir::Operation *fn,
                                ::kunir::TargetSpecAttr spec) {
  fn->setAttr(kFuncTargetSpecAttr, spec);
}

inline ::mlir::ArrayAttr getFuncInputNames(::mlir::Operation *fn) {
  return fn->getAttrOfType<::mlir::ArrayAttr>(kFuncInputNamesAttr);
}
inline void setFuncInputNames(::mlir::Operation *fn,
                                ::mlir::ArrayAttr names) {
  fn->setAttr(kFuncInputNamesAttr, names);
}

inline ::mlir::ArrayAttr getFuncOutputNames(::mlir::Operation *fn) {
  return fn->getAttrOfType<::mlir::ArrayAttr>(kFuncOutputNamesAttr);
}
inline void setFuncOutputNames(::mlir::Operation *fn,
                                 ::mlir::ArrayAttr names) {
  fn->setAttr(kFuncOutputNamesAttr, names);
}

} // namespace kungpu
