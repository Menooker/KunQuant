#ifndef KUNIR_ATTRS_TD
#define KUNIR_ATTRS_TD

include "KunIr/KunIrDialect.td"

// TargetSpec attribute: hardware parameters for a kunir.func op.
def KunIr_TargetSpecAttr : AttrDef<KunIr_Dialect, "TargetSpec"> {
  let mnemonic = "target_spec";
  let summary = "Hardware target parameters for a kunir function";
  let description = [{
    Describes the GPU launch configuration for a kunir function:
      occupancy      — target concurrent blocks per SM
      warps_per_cta  — warps per thread block
      smem_size      — total shared memory bytes available on one SM
                       (per-block budget = smem_size / occupancy)
      vector_size    — scalar elements per thread per time step

    Printed inline inside kunir.func as:
      target {occupancy = N, warps_per_cta = N, smem_size = N, vector_size = N}
  }];
  let parameters = (ins
    "int64_t":$occupancy,
    "int64_t":$warps_per_cta,
    "int64_t":$smem_size,
    "int64_t":$vector_size
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // KUNIR_ATTRS_TD
