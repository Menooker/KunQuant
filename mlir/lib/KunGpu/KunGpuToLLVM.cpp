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
//   stride — slot stride in bytes-of-T units:
//              1 for local (alloca buffer is per-thread)
//              K for shared (slot-major across the K threads in a block);
//                K = warps_per_cta * 32, captured as an i32 SSA value.
// Keyed on the original windowed_temp result Value so the ts.get / ts.put
// patterns can find it.
struct WTDesc {
  Value posPtr;
  int64_t stride; // 1 → no multiply at access time
};
using WTDescMap = llvm::DenseMap<Value, WTDesc>;

//===----------------------------------------------------------------------===//
// Phase 1: kunir.func → func.func (signature only)
//===----------------------------------------------------------------------===//

static void convertFuncSignature(kunir::FuncOp fn) {
  auto *ctx = fn.getContext();
  Location loc = fn.getLoc();
  auto i32Ty = IntegerType::get(ctx, 32);

  FunctionType oldFT = fn.getFunctionTypeTyped();
  SmallVector<Type> newArgTypes = {i32Ty, i32Ty};
  for (Type t : oldFT.getInputs())
    newArgTypes.push_back(t);

  // Build gpu.func right before the kunir.func — both live inside the
  // enclosing gpu.module.
  OpBuilder b(fn);
  auto newFunc = b.create<gpu::GPUFuncOp>(
      loc, fn.getSymName(), FunctionType::get(ctx, newArgTypes, {}));
  // Mark as a kernel (sets the op-level `kernel` attribute) so that
  // convert-gpu-to-nvvm tags the resulting llvm.func with `nvvm.kernel`.
  newFunc.setKernelAttr(UnitAttr::get(ctx));
  setFuncTargetSpec (newFunc, fn.getTargetSpecAttr());
  setFuncInputNames (newFunc, fn.getInputNames());
  setFuncOutputNames(newFunc, fn.getOutputNames());

  // gpu.func's auto-created entry block is replaced with the kunir.func
  // body.  Block-arg types initially still match the kunir.func signature;
  // phase 2's signature-conversion pattern reconciles them with the new
  // gpu.func type (ts → !llvm.ptr).
  newFunc.getBody().takeBody(fn.getBody());
  Block &entry = newFunc.getBody().front();
  entry.insertArgument(0u, i32Ty, loc);
  entry.insertArgument(1u, i32Ty, loc);

  SmallVector<kunir::ReturnOp> returns;
  newFunc.walk([&](kunir::ReturnOp r) { returns.push_back(r); });
  for (kunir::ReturnOp r : returns) {
    OpBuilder rb(r);
    rb.create<gpu::ReturnOp>(r.getLoc());
    r.erase();
  }
  fn.erase();
}

//===----------------------------------------------------------------------===//
// Helpers used inside conversion patterns
//===----------------------------------------------------------------------===//

static Value emitStockId(OpBuilder &b, Location loc, Type idxTy) {
  Value tid  = b.create<gpu::ThreadIdOp>(loc, idxTy, gpu::Dimension::x);
  Value bid  = b.create<gpu::BlockIdOp>(loc, idxTy, gpu::Dimension::x);
  Value bdim = b.create<gpu::BlockDimOp>(loc, idxTy, gpu::Dimension::x);
  return b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, bdim), tid);
}

// Read num_stocks (i32 func arg[1]) sign-extended to i64 for the linear gmem
// address computation.  The bare i32 value is in arg[1]; we extend at every
// use site (cheap, and lets the caller decide).
static Value getNumStocksI64(OpBuilder &b, Operation *op, Location loc) {
  Value ns32 = op->getParentOfType<gpu::GPUFuncOp>()
                   .getBody().front().getArgument(1);
  return b.create<arith::ExtSIOp>(loc, b.getI64Type(), ns32);
}
static Value getCurrentTimeIdx(Operation *op) {
  auto fOp = op->getParentOfType<scf::ForOp>();
  return fOp ? fOp.getInductionVar() : Value();
}

// linear gmem address = base + (timeIdx - offsetIdx) * num_stocks + stock_id
static Value gmemGEPWithOffset(OpBuilder &b, Location loc, Type elemTy,
                                LLVM::LLVMPointerType ptrTy, Value basePt,
                                Value timeIdx, Value offsetIdx,
                                Value numStocksI64, Type idxTy, Type i64Ty) {
  Value effIdx = offsetIdx ? b.create<arith::SubIOp>(loc, timeIdx, offsetIdx).getResult()
                            : timeIdx;
  Value tI64   = b.create<arith::IndexCastOp>(loc, i64Ty, effIdx);
  Value sid    = emitStockId(b, loc, idxTy);
  Value sidI64 = b.create<arith::IndexCastOp>(loc, i64Ty, sid);
  Value lin    = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, tI64, numStocksI64), sidI64);
  return b.create<LLVM::GEPOp>(loc, ptrTy, elemTy, basePt, ValueRange{lin});
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
        rewriter.create<LLVM::GlobalOp>(
            loc, LLVM::LLVMArrayType::get(elemTy, N * blockSize), false,
            LLVM::Linkage::Internal, name, Attribute{}, 0, 3);
      }
      Value raw = rewriter.create<LLVM::AddressOfOp>(
          loc, LLVM::LLVMPointerType::get(ctx, 3), name);
      Value gen    = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrTy, raw);
      Value tid    = rewriter.create<gpu::ThreadIdOp>(loc, idxTy, gpu::Dimension::x);
      Value tidI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, tid);
      // bufPtr = smem + tid  (slot-major: slot j thread t lives at j*K + t)
      bufPtr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, elemTy, gen,
                                             ValueRange{tidI32});
    } else {
      stride = 1;
      Value nCst = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(N));
      bufPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrTy, elemTy, nCst);
    }

    // Single i32 cell tracking next-writable position; init to 0.
    Value c1_i32 = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value posPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrTy, i32Ty, c1_i32);
    Value zeroI32 = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(0));
    rewriter.create<LLVM::StoreOp>(loc, zeroI32, posPtr);

    // Side state, keyed on the original (pre-replacement) ts Value.
    descMap[op.getResult()] = {posPtr, stride};

    rewriter.replaceOp(op, bufPtr);
    return success();
  }
};

// Multiply an i32 index by a compile-time stride.  stride==1 is a no-op.
static Value applyStride(OpBuilder &b, Location loc, Value idx, int64_t stride,
                          Type i32Ty) {
  if (stride == 1)
    return idx;
  Value k = b.create<LLVM::ConstantOp>(loc, i32Ty,
                                        b.getI32IntegerAttr(stride));
  return b.create<LLVM::MulOp>(loc, idx, k);
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
      // ── windowed_temp: circular get without modulo ────────────────
      //   adj = offset + 1                  (offset=0 → most-recent put)
      //   idx = pos >= adj ? pos - adj : pos + N - adj
      //   return buf[idx * stride]
      const WTDesc &desc = it->second;
      int64_t N = static_cast<int64_t>(
          llvm::cast<TsType>(op.getTs().getType()).getMaxLookback());
      Value pos    = rewriter.create<LLVM::LoadOp>(loc, i32Ty, desc.posPtr);
      Value c1     = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(1));
      Value nCst   = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(N));
      Value adj    = rewriter.create<LLVM::AddOp>(loc, offsetI32, c1);
      Value cmp    = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge,
                                                    pos, adj);
      Value posMinusAdj = rewriter.create<LLVM::SubOp>(loc, pos, adj);
      Value posPlusN    = rewriter.create<LLVM::AddOp>(loc, pos, nCst);
      Value wrapped     = rewriter.create<LLVM::SubOp>(loc, posPlusN, adj);
      Value idx32       = rewriter.create<LLVM::SelectOp>(
          loc, cmp, posMinusAdj, wrapped);
      // LLVM GEP accepts any integer index type — keep it i32 to avoid the
      // 64-bit ops that are slow on GPUs.
      Value gepIdx = applyStride(rewriter, loc, idx32, desc.stride, i32Ty);
      Value gep = rewriter.create<LLVM::GEPOp>(
          loc, ptrTy, elemTy, tsPtr, ValueRange{gepIdx});
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemTy, gep);
    } else {
      // ── global ts (function arg, TxS layout) ──────────────────────
      //   effective time = (enclosing scf.for iv) − offset
      //   load gmem[effTime * num_stocks + stock_id]
      Value timeIdx = getCurrentTimeIdx(op);
      Value offsetIdx = rewriter.create<arith::IndexCastOp>(
          loc, idxTy, offsetI32);
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

  TsPutPattern(TypeConverter &tc, MLIRContext *ctx, WTDescMap &m)
      : OpConversionPattern(tc, ctx), descMap(m) {}

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
      // ── windowed_temp: store at buf[pos*stride], then advance pos ─
      //   buf[pos * stride] = v
      //   pos = (pos + 1 >= N) ? 0 : pos + 1
      const WTDesc &desc = it->second;
      int64_t N = static_cast<int64_t>(
          llvm::cast<TsType>(op.getTs().getType()).getMaxLookback());
      Value pos = rewriter.create<LLVM::LoadOp>(loc, i32Ty, desc.posPtr);

      // Keep GEP index in i32 (cheap on GPU); LLVM accepts any int type.
      Value gepIdx = applyStride(rewriter, loc, pos, desc.stride, i32Ty);
      Value gep = rewriter.create<LLVM::GEPOp>(
          loc, ptrTy, elemTy, tsPtr, ValueRange{gepIdx});
      rewriter.create<LLVM::StoreOp>(loc, v, gep);

      Value c1     = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(1));
      Value nCst   = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(N));
      Value zero32 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(0));
      Value posP1  = rewriter.create<LLVM::AddOp>(loc, pos, c1);
      Value cmp    = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge,
                                                    posP1, nCst);
      Value newPos = rewriter.create<LLVM::SelectOp>(loc, cmp, zero32, posP1);
      rewriter.create<LLVM::StoreOp>(loc, newPos, desc.posPtr);
      rewriter.eraseOp(op);
    } else {
      // ── global ts: write at current time ──────────────────────────
      Value timeIdx = getCurrentTimeIdx(op);
      Value gep = gmemGEPWithOffset(rewriter, loc, elemTy, ptrTy, tsPtr,
                                     timeIdx, /*offsetIdx=*/Value(),
                                     getNumStocksI64(rewriter, op, loc),
                                     idxTy, i64Ty);
      rewriter.create<LLVM::StoreOp>(loc, v, gep);
      rewriter.eraseOp(op);
    }
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
        convertFuncSignature(fn);
    }

    // ── Phase 2 ────────────────────────────────────────────────────────
    TypeConverter typeConv;
    typeConv.addConversion([](Type t) { return t; });
    typeConv.addConversion([](TsType t) -> Type {
      return LLVM::LLVMPointerType::get(t.getContext());
    });
    auto materialize = [](OpBuilder &b, Type t, ValueRange vs, Location l) -> Value {
      if (vs.size() != 1) return Value();
      return b.create<UnrealizedConversionCastOp>(l, t, vs).getResult(0);
    };
    typeConv.addSourceMaterialization(materialize);
    typeConv.addTargetMaterialization(materialize);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                           LLVM::LLVMDialect, gpu::GPUDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    target.addIllegalOp<WindowedTempOp, TsGetOp, TsPutOp,
                        TimeLengthOp, StockIdOp, BlockStockCountOp>();
    // gpu.func is legal only after its signature has been converted from
    // (...kunir.ts) to (...!llvm.ptr) by the FunctionOpInterface pattern
    // we register below.
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) {
      return typeConv.isSignatureLegal(op.getFunctionType()) &&
             typeConv.isLegal(&op.getBody());
    });
    // gpu.return is void in our IR — always legal.

    WTDescMap descMap;
    int smemCounter = 0;

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(
        patterns, typeConv);
    patterns.add<TimeLengthPattern, StockIdPattern, BlockStockCountPattern>(
        typeConv, ctx);
    patterns.add<WindowedTempPattern>(typeConv, ctx, descMap, smemCounter);
    patterns.add<TsGetPattern, TsPutPattern>(typeConv, ctx, descMap);

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
