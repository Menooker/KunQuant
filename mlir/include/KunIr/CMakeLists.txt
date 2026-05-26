add_mlir_dialect(KunIrOps kunir)
add_mlir_interface(KunIrInterfaces)

# AttrDef generation (add_mlir_dialect does not cover attrdefs)
set(LLVM_TARGET_DEFINITIONS KunIrOps.td)
mlir_tablegen(KunIrOpsAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=kunir)
mlir_tablegen(KunIrOpsAttrDefs.cpp.inc -gen-attrdef-defs  -attrdefs-dialect=kunir)
add_mlir_dialect_tablegen_target(MLIRKunIrAttrsIncGen)
