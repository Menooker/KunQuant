//===- PyModule.cpp - dialect / translation / target registration -----===//
//
// Everything that touches a specific dialect or translation lives here,
// not in PyModule.h, so consumers of `class PyModule` only pay for the
// MLIRContext + ModuleOp typedefs.
//
//===----------------------------------------------------------------------===//

#include "PyModule.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// Required for `gpu-module-to-binary` to dispatch to the NVVM target
// implementation (libdevice link + LLVM opt + ptxas).
#include "mlir/Target/LLVM/NVVM/Target.h"

// MLIR → LLVM IR translation registrations consumed by the NVVM target
// serializer.  Keep the list minimal — `registerAllToLLVMIRTranslations`
// would force linking ArmSVE / SPIR-V / etc.
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "KunGpu/KunGpuDialect.h"
#include "KunIr/KunIrDialect.h"

#include "llvm/Support/raw_ostream.h"

#include <stdexcept>

namespace kun_mlir_py {

namespace {

mlir::DialectRegistry makeRegistry() {
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

  // Wire up `#nvvm.target`'s serializeToObject impl so
  // `gpu-module-to-binary` can lower gpu.module → cubin / PTX.
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  // ...and the dialect → LLVM IR translation hooks the NVVM target
  // calls once it has its hands on the gpu.module body.
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  return registry;
}

} // namespace

PyModule::PyModule()
    : ctx(std::make_unique<mlir::MLIRContext>(
          makeRegistry(), mlir::MLIRContext::Threading::DISABLED)) {
  ctx->loadAllAvailableDialects();
}

PyModule::~PyModule() = default;

std::unique_ptr<PyModule> PyModule::parse(const std::string &text) {
  auto pm = std::make_unique<PyModule>();
  pm->module = mlir::parseSourceString<mlir::ModuleOp>(text, pm->ctx.get());
  if (!pm->module)
    throw std::runtime_error("kun_mlir.parse: failed to parse MLIR text");
  return pm;
}

std::string PyModule::toString() const {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.get().print(os);
  os.flush();
  return out;
}

} // namespace kun_mlir_py
