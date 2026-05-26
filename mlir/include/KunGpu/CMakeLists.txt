add_mlir_dialect(KunGpuOps kungpu)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name KunGpu)
add_mlir_dialect_tablegen_target(MLIRKunGpuPassIncGen)
