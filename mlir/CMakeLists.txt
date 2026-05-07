set(KUN_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KUN_MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Include paths: MLIR/LLVM installed headers, our source headers,
# and the build dir (for generated .h.inc files from tablegen)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${KUN_MLIR_SOURCE_DIR}/include)
include_directories(${KUN_MLIR_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

# The MLIR backend requires LLVM to have been built with the NVPTX target
# — we emit PTX and load cubins.  Fail loudly here so the diagnostic is
# clear rather than getting cryptic linker errors later.
if(NOT "NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(FATAL_ERROR
    "KunQuant MLIR backend requires LLVM with NVPTX target.  Reconfigure "
    "your LLVM build with -DLLVM_TARGETS_TO_BUILD=\"X86;NVPTX\" "
    "(currently: '${LLVM_TARGETS_TO_BUILD}').")
endif()

# Enable gc-sections so kun-opt pulls in only the code it actually uses
# from MLIR static libraries, keeping the binary small.
if(NOT MSVC)
  add_compile_options(-ffunction-sections -fdata-sections)
  if(APPLE)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-dead_strip")
  else()
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,--gc-sections")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--gc-sections")
  endif()
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(Tools/kun-opt)
add_subdirectory(test)
