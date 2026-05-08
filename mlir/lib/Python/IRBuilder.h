//===- IRBuilder.h - Programmatic kunir module construction from Python ---===//
//
// Exposes a stateful builder to Python so a translator (e.g. KunQuant's
// codegen pass) can emit kunir ops without going through textual MLIR.
//
// Lifecycle:
//   ir = kun_mlir.IRBuilder()
//   ir.begin_func(name, in_types, in_names, out_names, target_spec, result_types)
//   args = ir.func_args
//   v = ir.add(args[0], args[1])
//   ir.end_func([v])
//   ...
//   mod = ir.finish()                # → kun_mlir.ModuleOp
//
// `Value` and `Type` are opaque wrappers around mlir::Value / mlir::Type.
// They are valid only while the IRBuilder (and the resulting PyModule)
// are alive.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <pybind11/pybind11.h>

namespace kun_mlir_py {
/// Register the IRBuilder + Value + Type pybind classes on `m`.
void registerIRBuilder(::pybind11::module &m);
} // namespace kun_mlir_py
