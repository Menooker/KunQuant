//===- PyModule.h - PyModule (MLIR ctx + ModuleOp) shared by bindings --===//
//
// Used by both MlirBinding.cpp (parse / compile entry points) and
// IRBuilder.cpp (programmatic construction of a kunir module from
// Python).  The header is deliberately thin: dialect / translation /
// target registrations all live in PyModule.cpp so nobody pays for them
// transitively.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"   // mlir::ModuleOp
#include "mlir/IR/OwningOpRef.h"  // mlir::OwningOpRef

#include <memory>
#include <string>

namespace mlir { class MLIRContext; }

namespace kun_mlir_py {

class PyModule {
public:
  PyModule();                        // sets up ctx + dialects + registrations
  ~PyModule();                       // out-of-line so MLIRContext can stay
                                      // forward-declared in this header
  PyModule(const PyModule &)            = delete;
  PyModule &operator=(const PyModule &) = delete;

  /// Parse an MLIR text fragment into a fresh PyModule.  Throws on
  /// parse failure.
  static std::unique_ptr<PyModule> parse(const std::string &text);

  /// Pretty-print the held module.
  std::string toString() const;

  std::unique_ptr<mlir::MLIRContext> ctx;
  mlir::OwningOpRef<mlir::ModuleOp>  module;
};

} // namespace kun_mlir_py
