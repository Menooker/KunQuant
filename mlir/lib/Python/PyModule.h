//===- PyModule.h - PyModule (MLIR ctx + ModuleOp) shared by bindings --===//
//
// Used by both MlirBinding.cpp (parse / compile entry points) and
// IRBuilder.cpp (programmatic construction of a kunir module from Python).
//
//===----------------------------------------------------------------------===//

#pragma once

#include <pybind11/pybind11.h>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "KunGpu/KunGpuDialect.h"
#include "KunIr/KunIrDialect.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace kun_mlir_py {

class PyModule {
public:
  PyModule()
      : ctx(std::make_unique<mlir::MLIRContext>(
            makeRegistry(), mlir::MLIRContext::Threading::DISABLED)) {
    ctx->loadAllAvailableDialects();
  }

  static mlir::DialectRegistry makeRegistry() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::index::IndexDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::NVVM::NVVMDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<kunir::KunIrDialect>();
    registry.insert<kungpu::KunGpuDialect>();
    return registry;
  }

  static std::unique_ptr<PyModule> parse(const std::string &text) {
    auto pm = std::make_unique<PyModule>();
    pm->module = mlir::parseSourceString<mlir::ModuleOp>(text, pm->ctx.get());
    if (!pm->module)
      throw std::runtime_error("kun_mlir.parse: failed to parse MLIR text");
    return pm;
  }

  std::string toString() const {
    std::string out;
    llvm::raw_string_ostream os(out);
    module.get().print(os);
    os.flush();
    return out;
  }

  std::unique_ptr<mlir::MLIRContext> ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace kun_mlir_py
