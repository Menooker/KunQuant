# kun-opt is placed in LLVM_RUNTIME_OUTPUT_INTDIR (= ${CMAKE_BINARY_DIR}/bin)
set(KUN_OPT_BINARY "${LLVM_RUNTIME_OUTPUT_INTDIR}/kun-opt")

# MLIR_TOOLS_DIR may not be set when building out-of-tree; fall back to
# the LLVM tools directory which contains FileCheck, mlir-opt, etc.
if(NOT MLIR_TOOLS_DIR)
  set(MLIR_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(KUN_MLIR_TEST_DEPENDS
  kun-opt
  KunMLIR
  FileCheck
)

add_lit_testsuite(check-kun-mlir "Running KunQuant MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${KUN_MLIR_TEST_DEPENDS}
)
set_target_properties(check-kun-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(KUN_MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${KUN_MLIR_TEST_DEPENDS}
)
