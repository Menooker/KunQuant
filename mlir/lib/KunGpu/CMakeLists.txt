add_mlir_dialect_library(MLIRKunGpuDialect
  KunGpuDialect.cpp
  KunGpuOps.cpp
  KunGpuMemoryPlanning.cpp
  KunGpuToLLVM.cpp
  Pipelines.cpp
  PtxBackend.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include

  DEPENDS
  MLIRKunGpuOpsIncGen
  MLIRKunGpuPassIncGen
  MLIRKunIrOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRArithDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRSideEffectInterfaces
  MLIRKunIrDialect
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRGPUToNVVMTransforms
  MLIRNVVMDialect

  # Main path: kunir → llvm dialect → upstream gpu-module-to-binary.
  # NVVMTarget supplies the serializeToObject impl that handles libdevice
  # linking + LLVM optimization + PTX emission + ptxas invocation.
  MLIRGPUTransforms
  MLIRNVVMTarget
  MLIRTargetLLVM

  # LLVM IR translation registrations consumed transitively by
  # NVVMTargetAttrImpl::serializeToObject.
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
)
