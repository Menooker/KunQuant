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

  # LLVM IR translation + PTX emission
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRExecutionEngineUtils
)
