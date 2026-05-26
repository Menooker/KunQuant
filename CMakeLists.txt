cmake_policy(SET CMP0048 NEW)
# nanobind requires CMake ≥ 3.15 (we use ≥ 3.18 so FindPython's
# `Development.SABIModule` component is reliably available, which is
# what STABLE_ABI mode below needs to link against).
cmake_minimum_required(VERSION 3.18)
project(KunRunner VERSION 0.0.1)

# nanobind requires C++17.  Set this before any add_subdirectory so
# subprojects (nanobind itself + our own targets) all pick it up.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect the matching Python with the SABIModule component so
# nanobind_add_module(... STABLE_ABI) can produce a single .abi3.so
# that works on every CPython ≥ 3.12.  Falls back to the regular
# Development.Module ABI on older interpreters.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)

add_subdirectory(3rdparty/nanobind)
include_directories(${PROJECT_SOURCE_DIR}/cpp)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

option(KUN_AVX512 "Enable AVX512 instruction set" OFF)
option(KUN_AVX512DQ "Enable AVX512DQ instruction set" OFF)
option(KUN_AVX512VL "Enable AVX512VL instruction set" OFF)
option(KUN_NO_AVX2 "Disable AVX2 and FMA instruction set" OFF)
option(KUN_SANITIZER "Enable sanitizer" OFF)
option(KUN_BUILD_CPU_RUNNER "Build the CPU KunRunner extension/runtime" ON)


if (CMAKE_CXX_COMPILER_ID MATCHES "(Clang|GNU|AppleClang)")
    # NOTE: C++ standard is set via CMAKE_CXX_STANDARD above; don't
    # re-add -std=c++NN here or it'll clobber subprojects' choice.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -pthread")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(amd64|x86_64|AMD64)")
        message(STATUS "Building for x86_64 architecture")
        if (NOT KUN_NO_AVX2)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        else()
            message(STATUS "AVX2 and FMA instruction sets are disabled")
        endif()
    endif()
    if(KUN_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
    endif()
    if(KUN_AVX512DQ)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512dq")
    endif()
    if(KUN_AVX512VL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl")
    endif()
    if(KUN_SANITIZER)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -static-libasan")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4200 /wd4305")
    if (NOT KUN_NO_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    else()
        message(STATUS "AVX2 and FMA instruction sets are disabled")
    endif()
    if(KUN_AVX512 OR KUN_AVX512DQ OR KUN_AVX512VL)
        # MSVC has only one flag for AVX512
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
    endif()
endif()

if(NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif()

message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

if(KUN_BUILD_CPU_RUNNER)
    file(GLOB_RECURSE KunRuntimeSrc ${PROJECT_SOURCE_DIR}/cpp/Kun/*.cpp
        ${PROJECT_SOURCE_DIR}/cpp/KunSIMD/*.cpp)
    add_library(KunRuntime SHARED ${KunRuntimeSrc})
    target_compile_definitions(KunRuntime PRIVATE KUN_CORE_LIB=1)
    if (NOT WIN32)
        target_link_libraries(KunRuntime PRIVATE dl)
    endif()

    file(GLOB_RECURSE KunPythonSrc ${PROJECT_SOURCE_DIR}/cpp/Python/*.cpp)
    # STABLE_ABI: build a single Python-version-independent .abi3.so on
    # CPython ≥ 3.12; nanobind silently disables this and falls back to
    # regular ABI on older Pythons.  Saves us one rebuild per Python
    # minor we want to ship.
    nanobind_add_module(KunRunner STABLE_ABI ${KunPythonSrc})

    target_link_libraries(KunRunner PUBLIC KunRuntime)



    file(GLOB_RECURSE KunTestSrc ${PROJECT_SOURCE_DIR}/tests/cpp/*.cpp)
    add_library(KunTest SHARED EXCLUDE_FROM_ALL ${KunTestSrc})
    target_link_libraries(KunTest KunRuntime)


    file(GLOB_RECURSE KunCApiTestSrc ${PROJECT_SOURCE_DIR}/tests/capi/*.cpp)
    add_executable(KunCApiTest EXCLUDE_FROM_ALL ${KunCApiTestSrc})
    target_link_libraries(KunCApiTest KunRuntime)
    add_dependencies(KunCApiTest KunTest)

    add_custom_target(TestingTargets DEPENDS KunCApiTest KunTest KunRunner)
endif()

#===------------------------------------------------------------------------===#
# Optional MLIR backend (kun-opt + kunir/kungpu dialects)
#===------------------------------------------------------------------------===#
option(KUN_BUILD_MLIR "Build MLIR backend with kunir/kungpu dialects" OFF)
if(KUN_BUILD_MLIR)
  set(KUN_MLIR_PYTHON_PACKAGE_DIR "${PROJECT_SOURCE_DIR}/KunQuantMLIR"
      CACHE PATH "Output directory for the KunQuant-MLIR Python package binaries")

  find_package(MLIR REQUIRED CONFIG)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH
        "nvcc used by CMake's CUDA-language support.")
  endif()
  get_filename_component(KUN_CUDA_TOOLKIT_ROOT
                         "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
  get_filename_component(KUN_CUDA_TOOLKIT_ROOT
                         "${KUN_CUDA_TOOLKIT_ROOT}" DIRECTORY)
  message(STATUS
      "KunQuant MLIR CUDA toolkit = ${KUN_CUDA_TOOLKIT_ROOT} "
      "(version ${CUDAToolkit_VERSION})")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  add_subdirectory(mlir)
endif()
