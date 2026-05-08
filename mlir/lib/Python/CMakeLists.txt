# Python binding module for the kunir → PTX → CUBIN → launch flow.
#
# pybind11 is added at the top-level KunQuant CMakeLists, so the
# `pybind11_add_module` macro is already in scope.

# pybind11 modules have undefined symbols (PyObject_*, PyExc_*, …) that
# the Python interpreter resolves at module-load time.  MLIR's
# HandleLLVMOptions adds `-Wl,-z,defs` globally, which is incompatible
# with that policy.  Strip it locally for this subdirectory.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
string(REPLACE "-Wl,-z,defs" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")

pybind11_add_module(kun_mlir SHARED
  MlirBinding.cpp
  IRBuilder.cpp
)

# Co-locate the binding with libKunCudaRuntime.so so $ORIGIN
# (CMAKE_INSTALL_RPATH set at top level) resolves dependencies as
# siblings — same pattern as KunRunner ↔ KunRuntime.
set_target_properties(kun_mlir PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/mlir/python")

target_link_libraries(kun_mlir PRIVATE
  # Compiler side
  MLIRKunIrDialect
  MLIRKunGpuDialect
  MLIRKunIrToKunGpu

  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRNVVMDialect
  MLIRSCFDialect

  # Pulled in by the kunir-to-llvm pipeline that runs inside lower_to_ptx.
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRGPUToNVVMTransforms
  MLIRReconcileUnrealizedCasts

  # Runtime side — owns cuda.h + libcuda; we just hand it ExecutableData
  # and call launch().
  KunCudaRuntime
)
