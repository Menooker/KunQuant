#ifndef KUNGPU_OPS_TD
#define KUNGPU_OPS_TD

include "KunGpu/KunGpuDialect.td"
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

#endif // KUNGPU_OPS_TD
