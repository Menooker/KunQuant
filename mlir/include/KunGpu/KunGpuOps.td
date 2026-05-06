#ifndef KUNGPU_OPS_TD
#define KUNGPU_OPS_TD

include "KunGpu/KunGpuDialect.td"
include "KunIr/KunIrTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class KunGpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<KunGpu_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Thread/block indexing
//===----------------------------------------------------------------------===//

def KunGpu_StockIdOp : KunGpu_Op<"stock_id", [Pure]> {
  let summary = "Get the stock index assigned to this GPU thread";
  let description = [{
    Returns the logical stock index for the current GPU thread.
    Lowers to: blockIdx.x * blockDim.x + threadIdx.x
  }];
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

def KunGpu_BlockStockCountOp : KunGpu_Op<"block_stock_count", [Pure]> {
  let summary = "Number of stocks handled per GPU block (blockDim.x)";
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

def KunGpu_TimeLengthOp : KunGpu_Op<"time_length", [Pure]> {
  let summary = "Number of time steps this GPU kernel must process";
  let description = [{
    Returns the length of the time dimension the current kernel invocation
    is responsible for. Each thread iterates over [0, time_length).
  }];
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Time-series memory ops
//
// kungpu.windowed_temp allocates a per-thread circular (ring) buffer.
// kungpu.ts.get / kungpu.ts.put provide the bridge between the abstract
// !kunir.ts<elemType, N> handle and actual scalar element loads/stores.
// The `ts` operand of get/put must be a function argument or a windowed_temp.
// The `time` operand is an index into the time dimension; the `result`
// (or `value` for put) must match the ts element type.
//===----------------------------------------------------------------------===//

def KunGpu_WindowedTempOp : KunGpu_Op<"windowed_temp", [Pure]> {
  let summary = "Allocate a per-thread windowed (circular) temporary buffer";
  let description = [{
    Allocates a thread-local circular buffer whose element type and window
    length are encoded in the result type `!kunir.ts<elemType, N>`.
    The buffer is used as the backing store for a windowed reduction;
    it must be the `ts` operand of `ts.get` or `ts.put`.
  }];
  let results = (outs KunIr_AnyTs:$result);
  let assemblyFormat = "`:` type($result) attr-dict";
}

def KunGpu_TsGetOp : KunGpu_Op<"ts.get", [Pure]> {
  let summary = "Load a scalar element from a time series at a given time index";
  let description = [{
    Loads the per-stock element from time series `ts` at time index `time`.
    The result type must equal the element type of `ts`.

    Example:
      %v = kungpu.ts.get %close[%t] : !kunir.ts<f32, inf> -> f32
  }];
  let arguments = (ins KunIr_AnyTs:$ts, Index:$time);
  let results = (outs AnyFloat:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$ts `[` $time `]` `:` type($ts) `->` type($result) attr-dict";
}

def KunGpu_TsPutOp : KunGpu_Op<"ts.put"> {
  let summary = "Store a scalar value into a time series at a given time index";
  let description = [{
    Stores scalar `value` into time series `ts` at time index `time`.
    `value` must have the same type as the element type of `ts`.

    Example:
      kungpu.ts.put %out[%t], %v : !kunir.ts<f32, 1>, f32
  }];
  let arguments = (ins KunIr_AnyTs:$ts, Index:$time, AnyFloat:$value);
  let hasVerifier = 1;
  let assemblyFormat =
    "$ts `[` $time `]` `,` $value `:` type($ts) `,` type($value) attr-dict";
}

#endif // KUNGPU_OPS_TD
