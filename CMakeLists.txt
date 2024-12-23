cmake_policy(SET CMP0048 NEW)
project(KunRunner VERSION 0.0.1)
cmake_minimum_required(VERSION 3.5)
add_subdirectory(3rdparty/pybind11)
include_directories(${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp)

set(CMAKE_INSTALL_RPATH "\$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

option(KUN_AVX512 "Enable AVX512 instruction set" OFF)
option(KUN_AVX512DQ "Enable AVX512DQ instruction set" OFF)
option(KUN_AVX512VL "Enable AVX512VL instruction set" OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden -mavx2 -mfma -pthread")
    if(KUN_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
    endif()
    if(KUN_AVX512DQ)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512dq")
    endif()
    if(KUN_AVX512VL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4200 /wd4305")
    if(KUN_AVX512 OR KUN_AVX512DQ OR KUN_AVX512VL)
        # MSVC has only one flag for AVX512
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
    endif()
endif()


file(GLOB_RECURSE KunRuntimeSrc ${PROJECT_SOURCE_DIR}/cpp/Kun/*.cpp
    ${PROJECT_SOURCE_DIR}/cpp/KunSIMD/*.cpp)
add_library(KunRuntime SHARED ${KunRuntimeSrc})
target_compile_definitions(KunRuntime PRIVATE KUN_CORE_LIB=1)
if (NOT WIN32)
    target_link_libraries(KunRuntime PRIVATE dl)
endif()

file(GLOB_RECURSE KunPythonSrc ${PROJECT_SOURCE_DIR}/cpp/Python/*.cpp)
pybind11_add_module(KunRunner SHARED ${KunPythonSrc})

target_link_libraries(KunRunner PUBLIC KunRuntime)



file(GLOB_RECURSE KunTestSrc ${PROJECT_SOURCE_DIR}/tests/cpp/*.cpp)
add_library(KunTest SHARED EXCLUDE_FROM_ALL ${KunTestSrc})
target_link_libraries(KunTest KunRunner)


file(GLOB_RECURSE KunCApiTestSrc ${PROJECT_SOURCE_DIR}/tests/capi/*.cpp)
add_executable(KunCApiTest EXCLUDE_FROM_ALL ${KunCApiTestSrc})
target_link_libraries(KunCApiTest KunRuntime)
add_dependencies(KunCApiTest KunTest)

if(NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif()

message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

add_custom_target(TestingTargets DEPENDS KunCApiTest KunTest KunRunner)