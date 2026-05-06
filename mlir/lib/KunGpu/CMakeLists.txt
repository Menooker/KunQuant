add_mlir_dialect_library(MLIRKunGpuDialect
  KunGpuDialect.cpp
  KunGpuOps.cpp
  KunGpuMemoryPlanning.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include

  DEPENDS
  MLIRKunGpuOpsIncGen
  MLIRKunGpuPassIncGen
  MLIRKunIrOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRSideEffectInterfaces
  MLIRKunIrDialect
)
