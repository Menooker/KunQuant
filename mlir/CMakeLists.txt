set(KUN_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KUN_MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Include paths: MLIR/LLVM installed headers, our source headers,
# and the build dir (for generated .h.inc files from tablegen)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${KUN_MLIR_SOURCE_DIR}/include)
include_directories(${KUN_MLIR_BINARY_DIR}/include)

# LLVMConfig.cmake exports LLVM_DEFINITIONS as a single space-separated
# *string* (e.g. "-D_GNU_SOURCE -D_GLIBCXX_USE_CXX11_ABI=1 ...").
# Plain `add_definitions(${LLVM_DEFINITIONS})` works most of the time
# (CMake splits on whitespace during unquoted expansion), but some
# CMake versions cache the string as a single COMPILE_DEFINITIONS
# entry, which then re-emits as one malformed `-D_GLIBCXX_USE_CXX11_ABI="1
# -D__STDC_CONSTANT_MACROS ..."` flag — duplicating the macro and
# producing the "_GLIBCXX_USE_CXX11_ABI redefined" warning.
#
# Tokenise the string explicitly first to keep CMake honest, then add.
separate_arguments(_KUN_LLVM_DEFS UNIX_COMMAND "${LLVM_DEFINITIONS}")
add_compile_options(${_KUN_LLVM_DEFS})
unset(_KUN_LLVM_DEFS)

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
