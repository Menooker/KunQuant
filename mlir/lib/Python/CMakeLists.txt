# Python binding module for the kunir → PTX → CUBIN → launch flow.
#
# nanobind is added at the top-level KunQuant CMakeLists, so the
# `nanobind_add_module` macro is already in scope.

# nanobind modules have undefined symbols (PyObject_*, PyExc_*, …) that
# the Python interpreter resolves at module-load time.  MLIR's
# HandleLLVMOptions adds `-Wl,-z,defs` globally, which is incompatible
# with that policy.  Strip it locally for this subdirectory.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
string(REPLACE "-Wl,-z,defs" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")

# STABLE_ABI: single .abi3.so on CPython ≥ 3.12; falls back to per-version
# on older Pythons.  Matches the runner binding (cpp/Python).
nanobind_add_module(KunMLIR STABLE_ABI
  MlirBinding.cpp
  IRBuilder.cpp
  PyModule.cpp
)

# Drop the .so directly into the KunQuant-MLIR Python package.  The
# libKunCudaRuntime.so runtime is co-located there too (see
# mlir/lib/KunCuda/CMakeLists.txt) so the $ORIGIN rpath resolves the sibling
# at load time.
set_target_properties(KunMLIR PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${KUN_MLIR_PYTHON_PACKAGE_DIR}")

target_link_libraries(KunMLIR PRIVATE
  # cuda.h + libcuda stub — the binding's runGraph wrapper calls
  # cuMemAlloc / cuMemFree directly to back caller-omitted output
  # buffers.  The CUDA::cuda_driver target is provided by the root
  # KUN_BUILD_MLIR CUDAToolkit discovery.
  CUDA::cuda_driver

  # Compiler side
  MLIRKunIrDialect
  MLIRKunGpuDialect
  MLIRKunIrToKunGpu

  # Runtime side — owns cuda.h + libcuda; we just hand it ExecutableData
  # and call launch().
  KunCudaRuntime
)

mlir_target_link_libraries(KunMLIR PRIVATE
  # Compiler side
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

  # Required for `gpu-module-to-binary` (the upstream pass we drive in
  # PtxBackend.cpp) + the NVVM target serializer it dispatches to.
  MLIRGPUTransforms
  MLIRNVVMTarget
)
