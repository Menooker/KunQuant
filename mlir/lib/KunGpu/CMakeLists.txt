# PtxBackend.cpp pulls in the NVPTX LLVM backend; it compiles to an
# empty TU and we don't add the NVPTX-only deps when LLVM was configured
# without NVPTX in LLVM_TARGETS_TO_BUILD.
set(_KUN_HAS_NVPTX OFF)
if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(_KUN_HAS_NVPTX ON)
endif()

set(_KUN_KUNGPU_EXTRA_LIBS )
if(_KUN_HAS_NVPTX)
  list(APPEND _KUN_KUNGPU_EXTRA_LIBS
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMToLLVMIRTranslation
    MLIRExecutionEngineUtils
  )
else()
  message(STATUS "[KunGpu] NVPTX target not in LLVM_TARGETS_TO_BUILD; "
                  "kungpu::compileKunIrToPtx will be a stub.")
endif()

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
  ${_KUN_KUNGPU_EXTRA_LIBS}
)

if(_KUN_HAS_NVPTX)
  target_compile_definitions(obj.MLIRKunGpuDialect PRIVATE KUN_HAS_NVPTX=1)
endif()
