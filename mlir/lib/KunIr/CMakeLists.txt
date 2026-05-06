add_mlir_dialect_library(MLIRKunIrDialect
  KunIrDialect.cpp
  KunIrTypes.cpp
  KunIrInterfaces.cpp
  KunIrOps.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include

  DEPENDS
  MLIRKunIrOpsIncGen
  MLIRKunIrInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

add_mlir_library(MLIRKunIrToKunGpu
  KunIrToKunGpu.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include

  DEPENDS
  MLIRKunIrOpsIncGen
  MLIRKunIrInterfacesIncGen
  MLIRKunGpuOpsIncGen

  LINK_LIBS PUBLIC
  MLIRKunIrDialect
  MLIRKunGpuDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
)
