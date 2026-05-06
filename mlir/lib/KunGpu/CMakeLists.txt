add_mlir_dialect_library(MLIRKunGpuDialect
  KunGpuDialect.cpp
  KunGpuOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include

  DEPENDS
  MLIRKunGpuOpsIncGen
  MLIRKunIrOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRKunIrDialect
)
