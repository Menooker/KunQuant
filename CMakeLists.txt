project(KunRunner)
cmake_minimum_required(VERSION 3.5)
add_subdirectory(3rdparty/pybind11)
include_directories(${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp)
message(STATUS ${pybind11_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden -mavx2")
file(GLOB_RECURSE KunRuntimeSrc ${PROJECT_SOURCE_DIR}/cpp/Kun/*.cpp)
pybind11_add_module(KunRunner SHARED ${KunRuntimeSrc})
target_link_libraries(KunRunner PRIVATE dl)


file(GLOB_RECURSE KunTestSrc ${PROJECT_SOURCE_DIR}/tests/cpp/*.cpp)
add_library(KunTest SHARED EXCLUDE_FROM_ALL ${KunTestSrc})
target_link_libraries(KunTest KunRunner)