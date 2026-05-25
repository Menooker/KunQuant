add_mlir_dialect_library(MLIRKunIrDialect
  KunIrDialect.cpp
  KunIrTypes.cpp
  KunIrInterfaces.cpp
  KunIrOps.cpp
  KunIrAttrs.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include

  DEPENDS
  MLIRKunIrOpsIncGen
  MLIRKunIrInterfacesIncGen
  MLIRKunIrAttrsIncGen
)

mlir_target_link_libraries(MLIRKunIrDialect PUBLIC
  MLIRIR
  MLIRFuncDialect
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
)

mlir_target_link_libraries(MLIRKunIrToKunGpu PUBLIC
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
)
