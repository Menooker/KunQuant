# libKunCudaRuntime.so — CUDA runtime, decoupled from the MLIR compiler
# library and the Python binding.

# ── CUDA language setup ──────────────────────────────────────────────
# The root CMakeLists discovers CUDAToolkit once when KUN_BUILD_MLIR is
# enabled.  Reuse its CMAKE_CUDA_COMPILER / CUDA::cuda_driver target here.
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# sm_75 (Turing) baseline.  cs_rank — and any other bundled kernel
# we add to kernels/ — uses no arch-specific instructions, so sm_75
# covers every NVIDIA GPU CUDA 13 still supports.  Later GPUs run
# the PTX via driver-JIT (cached system-wide in ~/.nv/ComputeCache).
set(CMAKE_CUDA_ARCHITECTURES 75)

# ── Recipe: compile `.cu` → `.ptx` → embed as a C array ──────────────
# Drop a new `kernels/<name>.cu` and the foreach below picks it up —
# the kernel becomes available to Runtime.cpp via
# `#include "<name>_ptx.inc"` exposing `kun_<name>_ptx[]`.
#
# nvcc stamps a `.version` directive on every emitted PTX matching
# its own toolkit release, which the deployed CUDA driver may not
# yet understand.  Our kernels here all use baseline features
# available since PTX 7.x, so EmbedFile.cmake's PTX_VERSION knob
# rewrites the `.version` directive to 7.8 before embedding —
# accepted by every driver from R510 (CUDA 11.6) onwards.  Same
# header-rewrite trick Triton uses when its LLVM emits new ISA.
set(_kun_ptx_embed_includes "")
set(_kun_embed_cmake "${CMAKE_CURRENT_SOURCE_DIR}/EmbedFile.cmake")

function(kun_add_bundled_ptx_kernel cu_path)
  get_filename_component(_stem "${cu_path}" NAME_WE)
  set(_obj_tgt "KunPtxObj_${_stem}")
  set(_inc     "${CMAKE_CURRENT_BINARY_DIR}/${_stem}_ptx.inc")
  set(_symbol  "kun_${_stem}_ptx")

  # Native CUDA OBJECT lib with CUDA_PTX_COMPILATION → CMake invokes
  # nvcc with `-ptx`; output is one .ptx per source.
  add_library(${_obj_tgt} OBJECT "${cu_path}")
  set_target_properties(${_obj_tgt} PROPERTIES
      CUDA_PTX_COMPILATION ON)
  target_compile_options(${_obj_tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

  # Embed step — `$<TARGET_OBJECTS:tgt>` is a generator expression that
  # resolves to the .ptx output path(s) at build time.
  add_custom_command(
      OUTPUT  "${_inc}"
      COMMAND "${CMAKE_COMMAND}"
              -DINPUT=$<TARGET_OBJECTS:${_obj_tgt}>
              -DOUTPUT=${_inc}
              -DSYMBOL=${_symbol}
              -DPTX_VERSION=7.8
              -DSM_VERSION=61
              -P "${_kun_embed_cmake}"
      DEPENDS ${_obj_tgt} $<TARGET_OBJECTS:${_obj_tgt}> "${_kun_embed_cmake}"
      COMMENT "Embedding ${_stem}.ptx as ${_symbol}[] (downgrading to ISA 7.8)"
      VERBATIM
      COMMAND_EXPAND_LISTS)

  # Hand the generated header back to the parent scope (proper list
  # append: read the parent's value via the auto-captured local,
  # APPEND in-function, write back).
  set(_local "${_kun_ptx_embed_includes}")
  list(APPEND _local "${_inc}")
  set(_kun_ptx_embed_includes "${_local}" PARENT_SCOPE)
endfunction()

# All bundled kernels live in kernels/.  CONFIGURE_DEPENDS makes
# ninja/make re-run cmake when a new .cu is added — no manual re-config.
file(GLOB _kun_cu_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")
foreach(_cu IN LISTS _kun_cu_sources)
  kun_add_bundled_ptx_kernel("${_cu}")
endforeach()

add_library(KunCudaRuntime SHARED
    ExecutableData.cpp
    Runtime.cpp
    RuntimeCudaGraph.cpp
    ${_kun_ptx_embed_includes})

# Project-wide compile flags set -fvisibility=hidden + inlines-hidden to
# minimise the size of MLIR static libs.  This shared runtime needs to
# export its public class methods so downstream .so's (KunMLIR, host
# runners, …) can resolve them at load time.
#
# We also put the .so next to the KunMLIR Python module
# (INSTALL_RPATH=$ORIGIN at the top level): all co-distributed shared libs
# live in one directory and find each other as siblings.
set_target_properties(KunCudaRuntime PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
    LIBRARY_OUTPUT_DIRECTORY "${KUN_MLIR_PYTHON_PACKAGE_DIR}")

target_include_directories(KunCudaRuntime PUBLIC
    "${PROJECT_SOURCE_DIR}/mlir/include")

# Runtime.cpp #include's the generated <kernel>_ptx.inc files.
target_include_directories(KunCudaRuntime PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}")

# ExecutableData.cpp uses nlohmann/json for artifact metadata.  This is a
# header-only third-party dependency kept private to the runtime library.
target_include_directories(KunCudaRuntime SYSTEM PRIVATE
    "${PROJECT_SOURCE_DIR}/3rdparty/nlohmann_json/include")

# CUDA Driver API.  `CUDA::cuda_driver` is FindCUDAToolkit's imported
# target wrapping `lib64/stubs/libcuda.so` with the right INCLUDE
# INTERFACE — gets cuda.h + the link-time stub in one go.  This dep
# is PRIVATE: downstream consumers see only Runtime.h, which never
# includes <cuda.h>.
target_link_libraries(KunCudaRuntime PRIVATE CUDA::cuda_driver)

# Like the Python module, this shared library has no CPython symbols to
# satisfy the global -Wl,-z,defs check; but we *do* link the CUDA stub
# so all libcuda symbols *are* resolved.  Nothing to strip here.
