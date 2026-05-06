add_mlir_dialect_library(MLIRKunIrDialect
  KunIrDialect.cpp
  KunIrTypes.cpp
  KunIrOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include

  DEPENDS
  MLIRKunIrOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
