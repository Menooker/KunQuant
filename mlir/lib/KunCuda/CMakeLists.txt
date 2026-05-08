# libKunCudaRuntime.so — pure CUDA runtime, decoupled from the MLIR
# compiler library and the Python binding.  Holds the ExecutableData
# struct, the Executable class (cuModuleLoadData / cuLaunchKernel) and
# nothing else.

add_library(KunCudaRuntime SHARED Runtime.cpp)

# Project-wide compile flags set -fvisibility=hidden + inlines-hidden to
# minimise the size of MLIR static libs.  This shared runtime needs to
# export its public class methods so downstream .so's (KunMLIR, host
# runners, …) can resolve them at load time.
#
# We also put the .so next to the KunMLIR Python module, mirroring the
# existing project pattern (INSTALL_RPATH=$ORIGIN at the top level): all
# co-distributed shared libs live in one directory and find each other
# as siblings.
set_target_properties(KunCudaRuntime PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/KunQuant/jit")

target_include_directories(KunCudaRuntime PUBLIC
    "${PROJECT_SOURCE_DIR}/mlir/include")

# CUDA Driver API (cuda.h + libcuda stub).  Resolved via CMake's
# standard FindCUDAToolkit — pass -DCUDAToolkit_ROOT=<path> or set
# $CUDA_PATH if the toolkit is not in a default search location.
# The stub is only used at link time; the runtime loader resolves
# libcuda.so.1 from the NVIDIA driver.  This dep is PRIVATE:
# downstream consumers see only Runtime.h, which never includes
# <cuda.h>.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(KunCudaRuntime PRIVATE CUDA::cuda_driver)

# Like the Python module, this shared library has no CPython symbols to
# satisfy the global -Wl,-z,defs check; but we *do* link the CUDA stub
# so all libcuda symbols *are* resolved.  Nothing to strip here.
