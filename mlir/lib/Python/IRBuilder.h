//===- IRBuilder.h - Programmatic kunir module construction from Python ---===//
//
// Exposes a stateful builder to Python so a translator (e.g. KunQuant's
// codegen pass) can emit kunir ops without going through textual MLIR.
//
// Lifecycle:
//   ir = KunMLIR.IRBuilder()
//   ir.begin_func(name, in_types, in_names, out_names, target_spec, result_types)
//   args = ir.func_args
//   v = ir.add(args[0], args[1])
//   ir.end_func([v])
//   ...
//   mod = ir.finish()                # → KunMLIR.ModuleOp
//
// `Value` and `Type` are opaque wrappers around mlir::Value / mlir::Type.
// They are valid only while the IRBuilder (and the resulting PyModule)
// are alive.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

namespace kun_mlir_py {
/// Register the IRBuilder + Value + Type nanobind classes on `m`.
void registerIRBuilder(::nanobind::module_ &m);
} // namespace kun_mlir_py
