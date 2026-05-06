#ifndef KUNIR_DIALECT_TD
#define KUNIR_DIALECT_TD

include "mlir/IR/OpBase.td"

def KunIr_Dialect : Dialect {
  let name = "kunir";
  let summary = "KunQuant IR dialect for dataflow financial factor computation";
  let description = [{
    The kunir dialect represents financial factor computations as a pure
    dataflow graph. Operations consume and produce !kunir.ts values; there
    is no explicit time-dimension iteration or memory access at this level.
  }];
  let cppNamespace = "::kunir";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // KUNIR_DIALECT_TD
