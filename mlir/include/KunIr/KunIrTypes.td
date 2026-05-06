#ifndef KUNIR_TYPES_TD
#define KUNIR_TYPES_TD

include "KunIr/KunIrDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class KunIr_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<KunIr_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def KunIr_TsType : KunIr_Type<"Ts", "ts"> {
  let summary = "Time series type for financial data";
  let description = [{
    Represents a time series of financial data across all stocks.
    Logically a [num_stocks x num_time] array; both dimensions are
    determined at runtime. The ts type is a pure value flowing between
    kunir ops — it carries no load/store semantics at this level.

    The maxLookback parameter (uint64) controls how many past time steps
    this stream retains:
      1        — only the current value (no history)
      N        — up to N past values available for windowed ops
      UINT64_MAX — unlimited history (printed/parsed as "inf"), used for
                   function input parameters
  }];
  let parameters = (ins "::mlir::Type":$elementType, "uint64_t":$maxLookback);
  let hasCustomAssemblyFormat = 1;
}

// Type constraint for use in op definitions
def KunIr_AnyTs : Type<CPred<"::llvm::isa<::kunir::TsType>($_self)">,
                        "kunir time series type", "::kunir::TsType">;

#endif // KUNIR_TYPES_TD
