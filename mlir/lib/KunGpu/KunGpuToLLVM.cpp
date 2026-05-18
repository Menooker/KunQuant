//===- KunGpuToLLVM.cpp - Lower kungpu + kunir.func → gpu.func + LLVM ---===//
//
// Assumes the input module is a `gpu.module` (or that the kunir.func lives
// inside one).  Two-phase pass.
//
// Phase 1 (convertFuncSignature, simple imperative helper):
//   kunir.func @f(%a: !kunir.ts<…>, …)
//     → gpu.func @f(%t: i32, %n: i32, %a: !kunir.ts<…>, …) kernel
//   inserted into the same gpu.module that contained the kunir.func.
//   The two prepended i32 arguments are time_length and num_stocks
//   (i32 because 64-bit ops are slow on GPUs; the linear gmem address
//   is still computed in i64).  ts arg types are preserved here — phase 2
//   converts them to !llvm.ptr via the standard signature-conversion pat.
//   target_spec, input_names and output_names are moved to discardable
//   attributes (see KunGpuUtils.h accessors).
//   kunir.return → gpu.return.
//
// Phase 2 (applyPartialConversion, one OpConversionPattern per op):
//   TypeConverter:  !kunir.ts<T,N> → !llvm.ptr
//
// Op semantics (post-redesign):
//   ts.put %ts, %v          : append %v at the tail of %ts.
//   ts.get %ts[%offset_i32] : read %ts at tail-relative offset (0 = latest).
//
// Lowering of windowed_temp head state — single i32 alloca holding the
// next-writable position (modeled on cpp/Kun/Ops.hpp::OutputWindow):
//
//   on put(v):  buf[pos] = v;
//               pos = (pos + 1 >= N) ? 0 : pos + 1;     // no modulo
//
//   on get(off):
//               adj = off + 1;                          // off=0 → most-recent put
//               idx = (pos >= adj) ? pos - adj : pos + N - adj;
//               return buf[idx];
//
// Lowering for global ts (function-arg pointer, TxS layout):
//   the "tail" is the current time step, given by the enclosing scf.for iv.
//   put :   gmem[iv * num_stocks + sid]            = v
//   get :   load gmem[(iv - off) * num_stocks + sid]
//
//===----------------------------------------------------------------------===//

#include "KunGpu/KunGpuOps.h"
#include "KunGpu/KunGpuUtils.h"
#include "KunGpu/Passes.h"
#include "KunIr/KunIrAttrs.h"
#include "KunIr/KunIrOps.h"
#include "KunIr/KunIrTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_CONVERTKUNGPUTOLLVM
#include "KunGpu/Passes.h.inc"

using namespace mlir;
using namespace kunir;
using namespace kungpu;

namespace {

// Per-windowed_temp side state.
//   posPtr — i32 alloca holding the next-writable circular position.
//            NULL means the entry is an accumulator (single slot, no
//            circular wrap; ts.get / ts.put always touch slot 0).
//   stride — slot stride in bytes-of-T units:
//              1 for local (alloca buffer is per-thread)
//              K for shared (slot-major across the K threads in a block);
//                K = warps_per_cta * 32, captured as an i32 SSA value.
// Keyed on the original windowed_temp / accumulator result Value so the
// ts.get / ts.put patterns can find it.
struct WTDesc {
  Value posPtr;     // null → accumulator (no position counter)
  int64_t stride;   // 1 → no multiply at access time
};
using WTDescMap = llvm::DenseMap<Value, WTDesc>;

// Per-function cache for the write-start SSA value shared across
// multiple output-store rewrites.  Each gpu.func builds (at most) one
// write_start; subsequent ts.put rewrites against an output arg reuse
// it, so we don't lean on a downstream CSE pass.
//
// `writeStart` is index-typed (not i32) because it's compared against
// the scf.for IV which is index-typed.  The runtime scalar args (mask,
// chunk_size, warmup) come in as i32; the helper below inserts the
// i32 → index cast once at function entry.
//
//   writeStart   : (block_id y == 0) ? mask : block_id y * chunk_size.
//                  Output stores below this time-index are suppressed —
//                  they fall in the warmup-overlap region.
//
// Emitted at the very top of the function entry block so it dominates
// every store site, regardless of how deeply nested.
struct ChunkContext {
  Value writeStart;
};
using ChunkCtxMap = llvm::DenseMap<Operation *, ChunkContext>;

//===----------------------------------------------------------------------===//
// Helper: stock_id = blockIdx.x * blockDim.x + threadIdx.x  (index-typed)
// Defined here so phase 1 (`convertFuncSignature` below) can reuse it
// for the active-thread guard, in addition to the conversion patterns.
//===----------------------------------------------------------------------===//

static Value emitStockId(OpBuilder &b, Location loc, Type idxTy) {
  Value tid  = gpu::ThreadIdOp::create(b, loc, idxTy, gpu::Dimension::x);
  Value bid  = gpu::BlockIdOp::create(b, loc, idxTy, gpu::Dimension::x);
  Value bdim = gpu::BlockDimOp::create(b, loc, idxTy, gpu::Dimension::x);
  return arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, bid, bdim), tid);
}

//===----------------------------------------------------------------------===//
// Phase 1: kunir.func → func.func (signature only)
//===----------------------------------------------------------------------===//

static LogicalResult convertFuncSignature(kunir::FuncOp fn) {
  auto *ctx  = fn.getContext();
  Location loc = fn.getLoc();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto idxTy = IndexType::get(ctx);

  // We only support vector_size = 1 right now.  When vector_size > 1 a
  // single thread handles `vector_size` consecutive stocks; if those
  // straddle the num_stocks boundary, the kernel either has to:
  //   - clamp the lane index to min(base + k, num_stocks - 1) on
  //     every gmem load (safe re-read), and per-lane predicate the
  //     gmem stores to skip the out-of-range cells;
  //   - or refuse non-aligned num_stocks at launch time.
  // TODO(vector_size>1): implement the clamp scheme above and remove
  // this check.  See discussion in KunGpuToLLVM history for why
  // PTX vector loads can't mask individual lanes.
  auto tsAttr = fn.getTargetSpecAttr();
  int64_t vectorSize = tsAttr ? tsAttr.getVectorSize() : 1;
  if (vectorSize != 1) {
    return fn.emitError("convert-kungpu-to-llvm: vector_size = ")
            << vectorSize << " not yet supported (only vector_size = 1). "
            << "TODO: implement clamp + per-lane store predicate for the "
            << "tail block.";
  }

  FunctionType oldFT = fn.getFunctionTypeTyped();
  // Prepend (time_length, num_stocks, mask, chunk_size, warmup) — all
  // i32.  time_length / num_stocks shape the linear gmem indexing;
  // mask / chunk_size / warmup feed the multi-chunk time-axis path
  // (kungpu.time_lb / time_ub / output-store gating).  64-bit math is
  // slow on GPUs, so we keep them as i32 and cast to index only at the
  // few places that need it.
  SmallVector<Type> newArgTypes = {i32Ty, i32Ty, i32Ty, i32Ty, i32Ty};
  for (Type t : oldFT.getInputs())
    newArgTypes.push_back(t);

  // Build gpu.func right before the kunir.func — both live inside the
  // enclosing gpu.module.
  OpBuilder b(fn);
  auto newFunc = gpu::GPUFuncOp::create(
      b, loc, fn.getSymName(), FunctionType::get(ctx, newArgTypes, {}));
  // Mark as a kernel (sets the op-level `kernel` attribute) so that
  // convert-gpu-to-nvvm tags the resulting llvm.func with `nvvm.kernel`.
  newFunc.setKernelAttr(UnitAttr::get(ctx));
  setFuncTargetSpec (newFunc, fn.getTargetSpecAttr());
  setFuncInputNames (newFunc, fn.getInputNames());
  setFuncOutputNames(newFunc, fn.getOutputNames());
  setFuncUnreliableCount(newFunc, fn.getUnreliableCount());

  // gpu.func's auto-created entry block is replaced with the kunir.func
  // body.  Block-arg types initially still match the kunir.func signature;
  // phase 2's signature-conversion pattern reconciles them with the new
  // gpu.func type (ts → !llvm.ptr).
  newFunc.getBody().takeBody(fn.getBody());
  Block &entry = newFunc.getBody().front();
  entry.insertArgument(0u, i32Ty, loc); // time_length
  entry.insertArgument(1u, i32Ty, loc); // num_stocks
  entry.insertArgument(2u, i32Ty, loc); // mask
  entry.insertArgument(3u, i32Ty, loc); // chunk_size
  entry.insertArgument(4u, i32Ty, loc); // warmup

  SmallVector<kunir::ReturnOp> returns;
  newFunc.walk([&](kunir::ReturnOp r) { returns.push_back(r); });
  for (kunir::ReturnOp r : returns) {
    OpBuilder rb(r);
    gpu::ReturnOp::create(rb, r.getLoc());
    r.erase();
  }
  fn.erase();

  // ── Tail-block guard ────────────────────────────────────────────────
  // grid_x is sized as ceil(num_stocks / block_x), so the last block
  // contains threads with stock_id ≥ num_stocks.  Without a guard those
  // threads do gmem GEPs at out-of-bounds addresses (UB).  Compute
  // stock_id at the top of the kernel and wrap the original body in
  // `scf.if (stock_id < num_stocks)`.  Inactive threads fall through to
  // gpu.return without touching gmem; their smem column is sized to the
  // block (not num_stocks), so leaving it uninitialised is safe.
  //
  // For vector_size = 1 this is the entire fix; vector_size > 1 is
  // gated above.
  Operation *gpuRet = entry.getTerminator();
  Operation *origFirst = entry.empty() ? nullptr : &entry.front();
  if (!origFirst || origFirst == gpuRet) {
    // Empty body — nothing to guard.
    return success();
  }

  OpBuilder pb(ctx);
  pb.setInsertionPointToStart(&entry);
  Value sidIdx = emitStockId(pb, loc, idxTy);
  Value sidI32 = arith::IndexCastOp::create(pb, loc, i32Ty, sidIdx);
  Value numStocks = entry.getArgument(1); // i32
  Value active = arith::CmpIOp::create(pb, loc, arith::CmpIPredicate::slt,
                                            sidI32, numStocks);
  auto ifOp = scf::IfOp::create(pb, loc, /*resultTypes=*/TypeRange{},
                                     active, /*withElseRegion=*/false);

  // Move all original ops (everything between the prologue we just
  // inserted and the gpu.return) into the scf.if's then-region, before
  // its implicit scf.yield.
  Block &thenBlk = ifOp.getThenRegion().front();
  Operation *thenYield = thenBlk.getTerminator();
  thenBlk.getOperations().splice(thenYield->getIterator(),
                                   entry.getOperations(),
                                   origFirst->getIterator(),
                                   gpuRet->getIterator());

  return success();
}

//===----------------------------------------------------------------------===//
// Helpers used inside conversion patterns
//===----------------------------------------------------------------------===//

// Read num_stocks (i32 func arg[1]) sign-extended to i64 for the linear gmem
// address computation.  The bare i32 value is in arg[1]; we extend at every
// use site (cheap, and lets the caller decide).
static Value getNumStocksI64(OpBuilder &b, Operation *op, Location loc) {
  Value ns32 = op->getParentOfType<gpu::GPUFuncOp>()
                   .getBody().front().getArgument(1);
  return arith::ExtSIOp::create(b, loc, b.getI64Type(), ns32);
}
static Value getCurrentTimeIdx(Operation *op) {
  // The enclosing function may contain nested scf.for's — outermost is
  // the per-thread time loop, inner ones come from for_each_back_window
  // bodies.  Reads/writes against a global ts (function-arg or graph
  // intermediate) must use the OUTER time loop's IV regardless of how
  // deep they sit; `op->getParentOfType<scf::ForOp>()` would otherwise
  // grab the FBW's window-step IV and produce gmem addresses indexed
  // by `w ∈ [0, window)` instead of the actual time `t`.
  scf::ForOp outermost;
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto f = dyn_cast<scf::ForOp>(p))
      outermost = f;
  }
  return outermost ? outermost.getInductionVar() : Value();
}

// linear gmem address = base + (timeIdx - offsetIdx) * num_stocks + stock_id
static Value gmemGEPWithOffset(OpBuilder &b, Location loc, Type elemTy,
                                LLVM::LLVMPointerType ptrTy, Value basePt,
                                Value timeIdx, Value offsetIdx,
                                Value numStocksI64, Type idxTy, Type i64Ty) {
  Value effIdx = offsetIdx ? arith::SubIOp::create(b, loc, timeIdx, offsetIdx).getResult()
                            : timeIdx;
  Value tI64   = arith::IndexCastOp::create(b, loc, i64Ty, effIdx);
  Value sid    = emitStockId(b, loc, idxTy);
  Value sidI64 = arith::IndexCastOp::create(b, loc, i64Ty, sid);
  Value lin    = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, tI64, numStocksI64), sidI64);
  return LLVM::GEPOp::create(b, loc, ptrTy, elemTy, basePt, ValueRange{lin});
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct TimeLengthPattern : OpConversionPattern<TimeLengthOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TimeLengthOp op, OpAdaptor /*a*/,
                  ConversionPatternRewriter &rewriter) const override {
    Value tl32 = op->getParentOfType<gpu::GPUFuncOp>()
                     .getBody().front().getArgument(0);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        op, rewriter.getIndexType(), tl32);
    return success();
  }
};

// time_lb = (block_id y == 0) ? 0 : block_id y * chunk_size - warmup
// All arithmetic happens in i32 (64-bit ops are slow on GPU); a single
// index_cast at the end produces the index-typed scf.for bound.
// chunk_size / warmup come from gpu.func args[3] / args[4]; the op has
// no operands at the kungpu level.
struct TimeLbPattern : OpConversionPattern<TimeLbOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TimeLbOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();
    auto idxTy = rewriter.getIndexType();
    auto fn = op->getParentOfType<gpu::GPUFuncOp>();
    Value chunkSize = fn.getBody().front().getArgument(3);
    Value warmup    = fn.getBody().front().getArgument(4);
    Value cyIdx = gpu::BlockIdOp::create(rewriter, loc, idxTy, gpu::Dimension::y);
    Value cy = arith::IndexCastOp::create(rewriter, loc, i32Ty, cyIdx);
    Value c0 = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
    Value isFirst = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, cy, c0);
    Value off = arith::MulIOp::create(rewriter, loc, cy, chunkSize);
    Value offMinusW = arith::SubIOp::create(rewriter, loc, off, warmup);
    Value lbI32 = arith::SelectOp::create(rewriter, loc, isFirst, c0, offMinusW);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, idxTy, lbI32);
    return success();
  }
};

// time_ub = min((block_id y + 1) * chunk_size, time_length)
// chunk_size / time_length come from gpu.func args[3] / args[0]; both
// are i32 so the math stays in i32 with one final cast to index.
struct TimeUbPattern : OpConversionPattern<TimeUbOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TimeUbOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();
    auto idxTy = rewriter.getIndexType();
    auto fn = op->getParentOfType<gpu::GPUFuncOp>();
    Value timeLen   = fn.getBody().front().getArgument(0);
    Value chunkSize = fn.getBody().front().getArgument(3);
    Value cyIdx = gpu::BlockIdOp::create(rewriter, loc, idxTy, gpu::Dimension::y);
    Value cy = arith::IndexCastOp::create(rewriter, loc, i32Ty, cyIdx);
    Value c1 = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value next = arith::AddIOp::create(rewriter, loc, cy, c1);
    Value end = arith::MulIOp::create(rewriter, loc, next, chunkSize);
    Value ubI32 = arith::MinUIOp::create(rewriter, loc, end, timeLen);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, idxTy, ubI32);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Chunk-context lazy helper.  See ChunkContext above.
//
// mask / chunk_size / warmup come in as i32 func args (positions 2 / 3 /
// 4 after time_length / num_stocks).  We build writeStart from the i32
// mask + chunk_size args, then cast to index once and cache.  Emitted
// at the very top of the function entry block so the resulting SSA
// value dominates every store-site inside the kernel.
//===----------------------------------------------------------------------===//

static Value getOrCreateWriteStart(Operation *op, ChunkCtxMap &map,
                                     ConversionPatternRewriter &rewriter) {
  auto fn = op->getParentOfType<gpu::GPUFuncOp>();
  ChunkContext &ctx = map[fn.getOperation()];
  if (ctx.writeStart) return ctx.writeStart;

  // Compute in i32 (cheap on GPU) then cast once to index, since the
  // result is compared against the scf.for IV (index-typed).
  Block &entry = fn.getBody().front();
  Value maskI32      = entry.getArgument(2);
  Value chunkSizeI32 = entry.getArgument(3);
  Location loc = fn.getLoc();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&entry);
  auto i32Ty = rewriter.getI32Type();
  auto idxTy = rewriter.getIndexType();
  Value cyIdx = gpu::BlockIdOp::create(rewriter, loc, idxTy, gpu::Dimension::y);
  Value cy = arith::IndexCastOp::create(rewriter, loc, i32Ty, cyIdx);
  Value c0 = arith::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
  Value isFirst = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, cy, c0);
  Value off = arith::MulIOp::create(rewriter, loc, cy, chunkSizeI32);
  Value wsI32 = arith::SelectOp::create(rewriter, loc, isFirst, maskI32, off);
  ctx.writeStart = arith::IndexCastOp::create(rewriter, loc, idxTy, wsI32);
  return ctx.writeStart;
}

struct StockIdPattern : OpConversionPattern<StockIdOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StockIdOp op, OpAdaptor /*a*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op,
        emitStockId(rewriter, op.getLoc(), rewriter.getIndexType()));
    return success();
  }
};

struct BlockStockCountPattern : OpConversionPattern<BlockStockCountOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BlockStockCountOp op, OpAdaptor /*a*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<gpu::BlockDimOp>(
        op, rewriter.getIndexType(), gpu::Dimension::x);
    return success();
  }
};

// Each windowed_temp lowers to:
//   %buf = llvm.alloca N x elemTy   (or smem GEP slice)
//   %pos = llvm.alloca 1 x i32
//   llvm.store 0 : i32, %pos        (next writable position starts at 0)
struct WindowedTempPattern : OpConversionPattern<WindowedTempOp> {
  WTDescMap &descMap;
  int &smemCounter;

  WindowedTempPattern(TypeConverter &tc, MLIRContext *ctx, WTDescMap &m, int &sc)
      : OpConversionPattern(tc, ctx), descMap(m), smemCounter(sc) {}

  LogicalResult
  matchAndRewrite(WindowedTempOp op, OpAdaptor /*a*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx     = op.getContext();
    Location loc  = op.getLoc();
    auto i32Ty    = rewriter.getI32Type();
    auto idxTy    = rewriter.getIndexType();
    auto ptrTy    = LLVM::LLVMPointerType::get(ctx);

    auto tsTy   = llvm::cast<TsType>(op.getType());
    int64_t N   = static_cast<int64_t>(tsTy.getMaxLookback());
    Type elemTy = tsTy.getElementType();

    // Buffer (alloca or smem slice).  All counters/offsets are i32.
    //
    // Local memory:
    //   bufPtr = alloca [N x T]  (per-thread, contiguous)  — stride = 1
    //
    // Shared memory (slot-major, bank-conflict-free):
    //   global [N * K x T] (addr_space=3) where K = threads_per_block
    //   Slot j of thread t lives at index   j * K + t.
    //   bufPtr = smem + tid                                 — stride = K
    //   ts.put/get use bufPtr[idx * K], landing on
    //     smem + tid + idx*K = slot_idx*K + tid  (correct).
    Value bufPtr;
    int64_t stride;

    if (op.isSmem()) {
      auto fn = op->getParentOfType<gpu::GPUFuncOp>();
      auto gpuModule = op->getParentOfType<gpu::GPUModuleOp>();
      auto tsAttr = getFuncTargetSpec(fn);
      int64_t blockSize = tsAttr ? (tsAttr.getWarpsPerCta() * 32) : 32;
      stride = blockSize;

      std::string name =
          ("__smem_" + fn.getName() + "_" +
           llvm::Twine(smemCounter++)).str();
      {
        OpBuilder::InsertionGuard g(rewriter);
        Block *modBody = &gpuModule.getBodyRegion().front();
        rewriter.setInsertionPoint(modBody, modBody->begin());
        LLVM::GlobalOp::create(
            rewriter, loc, LLVM::LLVMArrayType::get(elemTy, N * blockSize), false,
            LLVM::Linkage::Internal, name, Attribute{}, 0, 3);
      }
      Value raw = LLVM::AddressOfOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(ctx, 3), name);
      Value gen    = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, raw);
      Value tid    = gpu::ThreadIdOp::create(rewriter, loc, idxTy, gpu::Dimension::x);
      Value tidI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, tid);
      // bufPtr = smem + tid  (slot-major: slot j thread t lives at j*K + t)
      bufPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, gen,
                                             ValueRange{tidI32});
    } else {
      stride = 1;
      Value nCst = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(N));
      bufPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemTy, nCst);
    }

    // Single i32 cell tracking next-writable position; init to 0.
    Value c1_i32 = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value posPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i32Ty, c1_i32);
    Value zeroI32 = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
    LLVM::StoreOp::create(rewriter, loc, zeroI32, posPtr);

    // Side state, keyed on the original (pre-replacement) ts Value.
    descMap[op.getResult()] = {posPtr, stride};

    rewriter.replaceOp(op, bufPtr);
    return success();
  }
};

// kungpu.accumulator → single-slot alloca, zero-initialised.  Modeled in
// descMap with a null posPtr so the ts.get / ts.put dispatch can recognise
// it and emit a plain load/store at slot 0 (no circular wrap, no position
// counter).  The op MUST be lowered for offset = 0 only — verified at the
// ts.get / ts.put pattern level.
struct AccumulatorPattern : OpConversionPattern<kungpu::AccumulatorOp> {
  WTDescMap &descMap;
  AccumulatorPattern(TypeConverter &tc, MLIRContext *ctx, WTDescMap &m)
      : OpConversionPattern(tc, ctx), descMap(m) {}

  LogicalResult
  matchAndRewrite(kungpu::AccumulatorOp op, OpAdaptor /*a*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx    = op.getContext();
    Location loc = op.getLoc();
    auto i32Ty   = rewriter.getI32Type();
    auto ptrTy   = LLVM::LLVMPointerType::get(ctx);

    auto tsTy   = llvm::cast<TsType>(op.getType());
    Type elemTy = tsTy.getElementType();

    auto fn = op->getParentOfType<gpu::GPUFuncOp>();
    if (!fn)
      return rewriter.notifyMatchFailure(
          op, "kungpu.accumulator must be inside a gpu.func");

    // Alloca + zero-init at function entry so the slot is well-defined
    // before the time loop begins.
    Value bufPtr;
    {
      OpBuilder::InsertionGuard g(rewriter);
      Block &entry = fn.getBody().front();
      rewriter.setInsertionPointToStart(&entry);
      Value c1_i32 = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
      bufPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, elemTy, c1_i32);
      Value zero = LLVM::ConstantOp::create(
          rewriter, loc, elemTy, rewriter.getZeroAttr(elemTy));
      LLVM::StoreOp::create(rewriter, loc, zero, bufPtr);
    }

    // posPtr = null → ts.get / ts.put treat as accumulator (slot 0 only).
    descMap[op.getResult()] = {Value(), 1};
    rewriter.replaceOp(op, bufPtr);
    return success();
  }
};

// Multiply an i32 index by a compile-time stride.  stride==1 is a no-op.
static Value applyStride(OpBuilder &b, Location loc, Value idx, int64_t stride,
                          Type i32Ty) {
  if (stride == 1)
    return idx;
  Value k = LLVM::ConstantOp::create(b, loc, i32Ty,
                                        b.getI32IntegerAttr(stride));
  return LLVM::MulOp::create(b, loc, idx, k);
}

struct TsGetPattern : OpConversionPattern<TsGetOp> {
  WTDescMap &descMap;

  TsGetPattern(TypeConverter &tc, MLIRContext *ctx, WTDescMap &m)
      : OpConversionPattern(tc, ctx), descMap(m) {}

  LogicalResult
  matchAndRewrite(TsGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx    = op.getContext();
    Location loc = op.getLoc();
    auto i32Ty   = rewriter.getI32Type();
    auto i64Ty   = rewriter.getI64Type();
    auto idxTy   = rewriter.getIndexType();
    auto ptrTy   = LLVM::LLVMPointerType::get(ctx);
    Type elemTy  = op.getType();

    Value tsPtr     = adaptor.getTs();      // !llvm.ptr
    Value offsetI32 = adaptor.getOffset();  // i32

    auto it = descMap.find(op.getTs());
    if (it != descMap.end()) {
      const WTDesc &desc = it->second;
      // ── accumulator: single-slot load.  offset must be 0. ─────────
      if (!desc.posPtr) {
        int64_t offsetVal = -1;
        if (auto a = offsetI32.getDefiningOp<arith::ConstantOp>())
          offsetVal = llvm::cast<IntegerAttr>(a.getValue()).getInt();
        else if (auto l = offsetI32.getDefiningOp<LLVM::ConstantOp>())
          offsetVal = llvm::cast<IntegerAttr>(l.getValue()).getInt();
        if (offsetVal != 0)
          return rewriter.notifyMatchFailure(
              op, "ts.get on accumulator must use offset = 0");
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemTy, tsPtr);
        return success();
      }
      // ── windowed_temp: circular get without modulo ────────────────
      //   adj = offset + 1                  (offset=0 → most-recent put)
      //   idx = pos >= adj ? pos - adj : pos + N - adj
      //   return buf[idx * stride]
      int64_t N = static_cast<int64_t>(
          llvm::cast<TsType>(op.getTs().getType()).getMaxLookback());
      Value pos    = LLVM::LoadOp::create(rewriter, loc, i32Ty, desc.posPtr);
      Value c1     = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
      Value nCst   = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(N));
      Value adj    = LLVM::AddOp::create(rewriter, loc, offsetI32, c1);
      Value cmp    = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge,
                                                    pos, adj);
      Value posMinusAdj = LLVM::SubOp::create(rewriter, loc, pos, adj);
      Value posPlusN    = LLVM::AddOp::create(rewriter, loc, pos, nCst);
      Value wrapped     = LLVM::SubOp::create(rewriter, loc, posPlusN, adj);
      Value idx32       = LLVM::SelectOp::create(
          rewriter, loc, cmp, posMinusAdj, wrapped);
      // LLVM GEP accepts any integer index type — keep it i32 to avoid the
      // 64-bit ops that are slow on GPUs.
      Value gepIdx = applyStride(rewriter, loc, idx32, desc.stride, i32Ty);
      Value gep = LLVM::GEPOp::create(
          rewriter, loc, ptrTy, elemTy, tsPtr, ValueRange{gepIdx});
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemTy, gep);
    } else {
      // ── global ts (function arg, TxS layout) ──────────────────────
      //   effective time = (enclosing scf.for iv) − offset
      //   load gmem[effTime * num_stocks + stock_id]
      Value timeIdx = getCurrentTimeIdx(op);
      Value offsetIdx = arith::IndexCastOp::create(
          rewriter, loc, idxTy, offsetI32);
      Value gep = gmemGEPWithOffset(rewriter, loc, elemTy, ptrTy, tsPtr,
                                     timeIdx, offsetIdx,
                                     getNumStocksI64(rewriter, op, loc),
                                     idxTy, i64Ty);
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemTy, gep);
    }
    return success();
  }
};

struct TsPutPattern : OpConversionPattern<TsPutOp> {
  WTDescMap &descMap;
  ChunkCtxMap &chunkCtx;

  TsPutPattern(TypeConverter &tc, MLIRContext *ctx, WTDescMap &m,
                ChunkCtxMap &c)
      : OpConversionPattern(tc, ctx), descMap(m), chunkCtx(c) {}

  LogicalResult
  matchAndRewrite(TsPutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx    = op.getContext();
    Location loc = op.getLoc();
    auto i32Ty   = rewriter.getI32Type();
    auto i64Ty   = rewriter.getI64Type();
    auto idxTy   = rewriter.getIndexType();
    auto ptrTy   = LLVM::LLVMPointerType::get(ctx);

    Value tsPtr = adaptor.getTs();
    Value v     = adaptor.getValue();
    Type elemTy = v.getType();

    auto it = descMap.find(op.getTs());
    if (it != descMap.end()) {
      const WTDesc &desc = it->second;
      // ── accumulator: single-slot store, no pos counter to advance. ─
      if (!desc.posPtr) {
        LLVM::StoreOp::create(rewriter, loc, v, tsPtr);
        rewriter.eraseOp(op);
        return success();
      }
      // ── windowed_temp: store at buf[pos*stride], then advance pos ─
      //   buf[pos * stride] = v
      //   pos = (pos + 1 >= N) ? 0 : pos + 1
      int64_t N = static_cast<int64_t>(
          llvm::cast<TsType>(op.getTs().getType()).getMaxLookback());
      Value pos = LLVM::LoadOp::create(rewriter, loc, i32Ty, desc.posPtr);

      // Keep GEP index in i32 (cheap on GPU); LLVM accepts any int type.
      Value gepIdx = applyStride(rewriter, loc, pos, desc.stride, i32Ty);
      Value gep = LLVM::GEPOp::create(
          rewriter, loc, ptrTy, elemTy, tsPtr, ValueRange{gepIdx});
      LLVM::StoreOp::create(rewriter, loc, v, gep);

      Value c1     = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
      Value nCst   = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(N));
      Value zero32 = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
      Value posP1  = LLVM::AddOp::create(rewriter, loc, pos, c1);
      Value cmp    = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge,
                                                    posP1, nCst);
      Value newPos = LLVM::SelectOp::create(rewriter, loc, cmp, zero32, posP1);
      LLVM::StoreOp::create(rewriter, loc, newPos, desc.posPtr);
      rewriter.eraseOp(op);
    } else {
      // ── global ts: write at current time, gated by per-chunk write_start.
      //    Output time dim == time_length (== input time dim); the warmup
      //    region [0, mask) is just left unwritten by the kernel.
      //
      //   if (t >= write_start)
      //     out[t, sid] = v
      //
      // The `t >= write_start` comparison is uniform across the CTA (all
      // threads share the same scf.for IV), so the lowered branch is a
      // single uniform predicate — no warp divergence at chunk boundaries.
      Value timeIdx    = getCurrentTimeIdx(op);
      Value writeStart = getOrCreateWriteStart(op, chunkCtx, rewriter);

      Value doWrite = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sge, timeIdx, writeStart);
      auto ifOp = scf::IfOp::create(
          rewriter, loc, /*resultTypes=*/TypeRange{}, doWrite,
          /*withElseRegion=*/false);

      OpBuilder ib = OpBuilder::atBlockBegin(&ifOp.getThenRegion().front());
      Value gep = gmemGEPWithOffset(ib, loc, elemTy, ptrTy, tsPtr,
                                     timeIdx, /*offsetIdx=*/Value(),
                                     getNumStocksI64(ib, op, loc),
                                     idxTy, i64Ty);
      LLVM::StoreOp::create(ib, loc, v, gep);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FastWindowedSum — running sum with Kahan compensation + NaN tracking.
//
// Per-thread state (4 cells, alloca'd at function entry, promoted to
// registers by mem2reg):
//   v             — running sum                                    (T)
//   compAdd       — Kahan compensation for the +cur step           (T)
//   compSub       — Kahan compensation for the -old step           (T)
//   numNans       — count of NaNs currently inside the trailing-N window (i32)
//
// Algorithm — direct port of cpp/Kun/Ops.hpp::FastWindowedSum::step:
//
//   cur = input[t]                                                 ts.get  off=0
//   old = (t - loop_lb >= window) ? input[t - window] : NaN        ts.get  off=window  (guarded)
//   old_is_nan = isnan(old)
//   new_is_nan = isnan(cur)
//   v = old_is_nan ? v : kahanAdd(v, -old, &compSub)               // subtract old
//   v = new_is_nan ? v : kahanAdd(v, +cur, &compAdd)               // add cur
//   numNans += (new_is_nan ? 1 : 0) - (old_is_nan ? 1 : 0)
//   out = (numNans == 0) ? v : NaN
//
// Guard uses `t - loop_lb`, not bare `t`: state is per-CTA alloca
// (zero-init) so each chunk needs its own N-step warmup with old=NaN
// to build v up.  Chunk 0 has loop_lb = 0 so the guard collapses to
// CPU's `t >= window`.
//===----------------------------------------------------------------------===//

struct FastWindowedSumPattern : OpConversionPattern<FastWindowedSumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FastWindowedSumOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx    = op.getContext();
    Location loc = op.getLoc();
    auto i32Ty   = rewriter.getI32Type();
    auto ptrTy   = LLVM::LLVMPointerType::get(ctx);

    auto resultTy = op.getResult().getType();
    auto floatTy  = llvm::dyn_cast<FloatType>(resultTy);
    if (!floatTy)
      return rewriter.notifyMatchFailure(
          op, "fast_windowed_sum result must be a scalar float "
              "(post kunir-to-kungpu lowering)");

    int64_t window = op.getWindow();
    Value origInput = op.getInput();

    // ── 1. Allocate state at function entry + initialise. ──────────
    auto fn = op->getParentOfType<gpu::GPUFuncOp>();
    if (!fn)
      return rewriter.notifyMatchFailure(
          op, "fast_windowed_sum must be inside a gpu.func");

    Value vPtr, addPtr, subPtr, nansPtr;
    {
      OpBuilder::InsertionGuard g(rewriter);
      Block &entry = fn.getBody().front();
      rewriter.setInsertionPointToStart(&entry);
      Value c1_i32 = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
      Value zeroF = LLVM::ConstantOp::create(
          rewriter, loc, floatTy, rewriter.getFloatAttr(floatTy, 0.0));
      Value windowI32 = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(window));

      vPtr    = LLVM::AllocaOp::create(rewriter, loc, ptrTy, floatTy, c1_i32);
      addPtr  = LLVM::AllocaOp::create(rewriter, loc, ptrTy, floatTy, c1_i32);
      subPtr  = LLVM::AllocaOp::create(rewriter, loc, ptrTy, floatTy, c1_i32);
      nansPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i32Ty,   c1_i32);

      LLVM::StoreOp::create(rewriter, loc, zeroF,     vPtr);
      LLVM::StoreOp::create(rewriter, loc, zeroF,     addPtr);
      LLVM::StoreOp::create(rewriter, loc, zeroF,     subPtr);
      LLVM::StoreOp::create(rewriter, loc, windowI32, nansPtr);
    }

    // ── 2. Read cur (off=0) and old (off=window, guarded). ─────────
    Value zeroOff   = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
    Value windowOff = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(window));
    Value cur = TsGetOp::create(rewriter, loc, floatTy, origInput, zeroOff);

    auto forOp = op->getParentOfType<scf::ForOp>();
    if (!forOp)
      return rewriter.notifyMatchFailure(
          op, "fast_windowed_sum must be inside a scf.for time loop");
    Value timeIdx   = forOp.getInductionVar();
    Value loopLb    = forOp.getLowerBound();
    Value localT    = arith::SubIOp::create(rewriter, loc, timeIdx, loopLb);
    Value windowIdx = arith::ConstantIndexOp::create(rewriter, loc, window);
    Value tGeWindow = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sge, localT, windowIdx);

    auto ifOp = scf::IfOp::create(
        rewriter, loc, TypeRange{floatTy}, tGeWindow, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value loaded =
          TsGetOp::create(rewriter, loc, floatTy, origInput, windowOff);
      scf::YieldOp::create(rewriter, loc, loaded);
    }
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value nanV = LLVM::ConstantOp::create(
          rewriter, loc, floatTy,
          rewriter.getFloatAttr(
              floatTy, std::numeric_limits<double>::quiet_NaN()));
      scf::YieldOp::create(rewriter, loc, nanV);
    }
    Value old = ifOp.getResult(0);

    // ── 3. Algorithm step.  All arith is via LLVM ops at this phase. ──
    auto fcmp_isnan = [&](Value x) {
      // isnan(x) ⇔ x != x  (UNE catches NaN, == NaN is false)
      return LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::une, x, x);
    };
    Value oldIsNan = fcmp_isnan(old);
    Value newIsNan = fcmp_isnan(cur);

    // Loaded state.
    Value v       = LLVM::LoadOp::create(rewriter, loc, floatTy, vPtr);
    Value compAdd = LLVM::LoadOp::create(rewriter, loc, floatTy, addPtr);
    Value compSub = LLVM::LoadOp::create(rewriter, loc, floatTy, subPtr);
    Value numNans = LLVM::LoadOp::create(rewriter, loc, i32Ty,   nansPtr);

    Value zeroF = LLVM::ConstantOp::create(
        rewriter, loc, floatTy, rewriter.getFloatAttr(floatTy, 0.0));

    // kahanAdd(isnan_small, sum, small, &comp):
    //   y = small - comp;  t = sum + y;
    //   newComp = (t - sum) - y;
    //   comp = isnan_small ? comp : newComp;
    //   return t
    auto kahanAdd = [&](Value isnan_small, Value sum, Value small, Value &comp) {
      Value y     = LLVM::FSubOp::create(rewriter, loc, small, comp);
      Value t     = LLVM::FAddOp::create(rewriter, loc, sum, y);
      Value tMs   = LLVM::FSubOp::create(rewriter, loc, t, sum);
      Value newC  = LLVM::FSubOp::create(rewriter, loc, tMs, y);
      comp = LLVM::SelectOp::create(rewriter, loc, isnan_small, comp, newC);
      return t;
    };

    // v -= old  (skip when old is NaN)
    Value negOld = LLVM::FSubOp::create(rewriter, loc, zeroF, old);
    Value tSub   = kahanAdd(oldIsNan, v, negOld, compSub);
    v = LLVM::SelectOp::create(rewriter, loc, oldIsNan, v, tSub);

    // v += cur  (skip when cur is NaN)
    Value tAdd   = kahanAdd(newIsNan, v, cur, compAdd);
    v = LLVM::SelectOp::create(rewriter, loc, newIsNan, v, tAdd);

    // numNans += (new_is_nan ? 1 : 0) - (old_is_nan ? 1 : 0)
    Value oneI32  = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value zeroI32 = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
    Value oldDelta = LLVM::SelectOp::create(
        rewriter, loc, oldIsNan, oneI32, zeroI32);
    Value newDelta = LLVM::SelectOp::create(
        rewriter, loc, newIsNan, oneI32, zeroI32);
    numNans = LLVM::SubOp::create(rewriter, loc, numNans, oldDelta);
    numNans = LLVM::AddOp::create(rewriter, loc, numNans, newDelta);

    // result = (numNans == 0) ? v : NaN
    Value isFull = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, numNans, zeroI32);
    Value nanV = LLVM::ConstantOp::create(
        rewriter, loc, floatTy,
        rewriter.getFloatAttr(floatTy,
                                std::numeric_limits<double>::quiet_NaN()));
    Value out = LLVM::SelectOp::create(rewriter, loc, isFull, v, nanV);

    // ── 4. Store back state. ────────────────────────────────────────
    LLVM::StoreOp::create(rewriter, loc, v,       vPtr);
    LLVM::StoreOp::create(rewriter, loc, compAdd, addPtr);
    LLVM::StoreOp::create(rewriter, loc, compSub, subPtr);
    LLVM::StoreOp::create(rewriter, loc, numNans, nansPtr);

    rewriter.replaceOp(op, out);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertKunGpuToLLVMPass
    : ::impl::ConvertKunGpuToLLVMBase<ConvertKunGpuToLLVMPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx       = &getContext();

    // ── Phase 1 ────────────────────────────────────────────────────────
    {
      SmallVector<kunir::FuncOp> kfns;
      module.walk([&](kunir::FuncOp fn) { kfns.push_back(fn); });
      for (kunir::FuncOp fn : kfns)
        if (failed(convertFuncSignature(fn)))
          return signalPassFailure();
    }

    // ── Phase 2 ────────────────────────────────────────────────────────
    TypeConverter typeConv;
    typeConv.addConversion([](Type t) { return t; });
    typeConv.addConversion([](TsType t) -> Type {
      return LLVM::LLVMPointerType::get(t.getContext());
    });
    auto materialize = [](OpBuilder &b, Type t, ValueRange vs, Location l) -> Value {
      if (vs.size() != 1) return Value();
      return UnrealizedConversionCastOp::create(b, l, t, vs).getResult(0);
    };
    typeConv.addSourceMaterialization(materialize);
    typeConv.addTargetMaterialization(materialize);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                           LLVM::LLVMDialect, gpu::GPUDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    target.addIllegalOp<WindowedTempOp, kungpu::AccumulatorOp,
                        TsGetOp, TsPutOp,
                        TimeLengthOp, TimeLbOp, TimeUbOp,
                        StockIdOp, BlockStockCountOp>();
    target.addIllegalOp<kunir::FastWindowedSumOp>();
    // gpu.func is legal only after its signature has been converted from
    // (...kunir.ts) to (...!llvm.ptr) by the FunctionOpInterface pattern
    // we register below.
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) {
      return typeConv.isSignatureLegal(op.getFunctionType()) &&
             typeConv.isLegal(&op.getBody());
    });
    // gpu.return is void in our IR — always legal.

    WTDescMap descMap;
    ChunkCtxMap chunkCtx;
    int smemCounter = 0;

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(
        patterns, typeConv);
    patterns.add<TimeLengthPattern, TimeLbPattern, TimeUbPattern,
                  StockIdPattern, BlockStockCountPattern>(typeConv, ctx);
    patterns.add<WindowedTempPattern>(typeConv, ctx, descMap, smemCounter);
    patterns.add<AccumulatorPattern>(typeConv, ctx, descMap);
    patterns.add<TsGetPattern>(typeConv, ctx, descMap);
    patterns.add<TsPutPattern>(typeConv, ctx, descMap, chunkCtx);
    patterns.add<FastWindowedSumPattern>(typeConv, ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace kungpu {
std::unique_ptr<mlir::Pass> createConvertKunGpuToLLVMPass() {
  return std::make_unique<ConvertKunGpuToLLVMPass>();
}
} // namespace kungpu
