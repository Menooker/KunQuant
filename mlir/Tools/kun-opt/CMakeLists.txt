set(LLVM_LINK_COMPONENTS Support)
if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND LLVM_LINK_COMPONENTS
    NVPTXCodeGen NVPTXAsmParser NVPTXDesc NVPTXInfo
    Passes Target TargetParser Core IRReader CodeGen MC AsmPrinter)
endif()

add_llvm_executable(kun-opt kun-opt.cpp)

llvm_update_compile_flags(kun-opt)

target_link_libraries(kun-opt PRIVATE
  # KunQuant dialects + passes
  MLIRKunIrDialect
  MLIRKunGpuDialect
  MLIRKunIrToKunGpu

  # MLIR opt infrastructure
  MLIROptLib

  # Standard dialects used inside kunir/kungpu IR
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRIndexDialect

  # Conversion passes used by the kunir-to-llvm pipeline
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRGPUToNVVMTransforms
  MLIRNVVMDialect
  MLIRConvertToLLVMPass
  MLIRReconcileUnrealizedCasts

  # Core MLIR libraries
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRSideEffectInterfaces
)

# Verify no unexpected MLIR dialects are pulled in transitively
mlir_check_all_link_libraries(kun-opt)
