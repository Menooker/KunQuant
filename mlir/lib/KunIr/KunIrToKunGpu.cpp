//===- KunIrToKunGpu.cpp - Lower kunir ops to kungpu + scf + arith --------===//
//
// Lowers a kunir.func whose body contains kunir ops into a form that uses:
//   - kungpu.time_length / kungpu.ts.get / kungpu.ts.put  for ts I/O
//   - scf.for for the outer time loop and inner back-window loops
//   - arith.* / math.* for scalar arithmetic
//
// Assumptions / limitations:
//   - The function body is a single block.
//   - ts-typed return values are converted to output parameters (void return).
//   - All inputs to kunir.for_each_back_window must be ts handles (function
//     arguments or kunir.windowed_output results).
//   - Each yield operand of for_each_back_window must come from a reduce_* op.
//   - Cross-sectional kernels (cs_rank) never enter kunir — the Python
//     frontend (CodegenMLIR._maybe_external_partition) routes them
//     directly to a pre-compiled CUmodule bundled with the runtime.
//
//===----------------------------------------------------------------------===//

#include "KunGpu/KunGpuOps.h"
#include "KunIr/KunIrInterfaces.h"
#include "KunIr/KunIrOps.h"
#include "KunIr/KunIrTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace kunir;
using namespace kungpu;

// In a function returning LogicalResult / FailureOr<U>:
//   KUN_ASSIGN_OR_FAIL(T x, callReturningFailureOrT(...));
// On failure → `return failure();`.  On success → declare `x = *result`.
// Multi-statement expansion; do not use without braces in if/while/for bodies.
#define KUN_DETAIL_CAT_(a, b) a##b
#define KUN_DETAIL_CAT(a, b)  KUN_DETAIL_CAT_(a, b)
#define KUN_ASSIGN_OR_FAIL_IMPL(decl, expr, tmp)        \
    auto tmp = (expr);                                  \
    if (::mlir::failed(tmp)) return ::mlir::failure();  \
    decl = *std::move(tmp)
#define KUN_ASSIGN_OR_FAIL(decl, expr)                  \
    KUN_ASSIGN_OR_FAIL_IMPL(decl, expr,                 \
        KUN_DETAIL_CAT(_kunOrFail_, __COUNTER__))

namespace {

//===----------------------------------------------------------------------===//
// Value tracking
//
// Two disjoint maps, both keyed by kunir SSA Values:
//   tsMap     : ts SSA value  -> the ts handle SSA value to load from.
//               Populated by: function-arg seeding, WindowedOutputOp
//               (handle = windowed_temp), inner LowerHelper copy of outer.tsMap.
//   scalarMap : SSA value     -> the scalar SSA value already materialised
//               for it.  Populated by: arith op results, reduce accumulator
//               pre-seed + update, BackRefOp result, for_each_back_window
//               result, FastWindowedSumOp result, getScalar's offset-0 cache.
//
// A given SSA value lives in at most one map at construction time, but a ts
// handle can become "additionally" represented in scalarMap once it has been
// read at offset 0 — that's the getScalar cache.
//===----------------------------------------------------------------------===//

using HandleMap = llvm::DenseMap<Value, Value>;

// One LowerHelper per scope (outer function body / for_each_back_window body).
// `zeroOffsetI32` is the function-scope i32 zero constant created once before
// the outer scf.for; all LowerHelper instances share it.
struct LowerHelper {
  HandleMap tsMap;
  HandleMap scalarMap;
  Value zeroOffsetI32;

  // Shared util: look up `v` (a ts SSA value) in tsMap, emit
  // ts.get(handle, offsetI32), return the loaded scalar.  Does NOT touch
  // scalarMap — callers decide whether/where to cache the result.  Returns
  // failure (with an in-flight diagnostic at `loc`) if `v` is not a
  // registered ts handle.
  //
  // Used by:
  //   - getScalar (offset = zeroOffsetI32)
  //   - BackRefOp branch (offset = constant(window))
  //   - for_each_back_window block-arg pre-load (offset = window-1-w)
  FailureOr<Value> getScalarUncached(Value v, Value offsetI32,
                                      OpBuilder &b, Location loc) {
    auto it = tsMap.find(v);
    if (it == tsMap.end())
      return emitError(loc,
          "kunir-to-kungpu: value is not a registered ts handle in tsMap");
    auto tsTy = llvm::cast<TsType>(v.getType());
    return b.create<TsGetOp>(loc, tsTy.getElementType(),
                              it->second, offsetI32).getResult();
  }

  // Offset-0 read with scalarMap caching.  Looks up scalarMap first; on miss
  // loads at offset 0 via getScalarUncached and caches the result.  This is
  // the standard "current time step" read used by all in-body operand
  // lookups inside lowerBlock.
  FailureOr<Value> getScalar(Value v, OpBuilder &b, Location loc) {
    auto sit = scalarMap.find(v);
    if (sit != scalarMap.end()) return sit->second;
    KUN_ASSIGN_OR_FAIL(Value scalar,
                       getScalarUncached(v, zeroOffsetI32, b, loc));
    scalarMap[v] = scalar;
    return scalar;
  }

  // Lower non-terminator ops in `ops` in definition order.
  //
  // For each op:
  //   - Anything else: call handleUnknown if provided, else return failure.
  LogicalResult lowerBlock(
      llvm::ArrayRef<Operation *> ops, OpBuilder &b,
      llvm::function_ref<LogicalResult(Operation &)> handleUnknown = nullptr) {
    for (Operation *op : ops) {
      Location ol = op->getLoc();
      if (auto iface = dyn_cast<BinaryArithInterface>(op)) {
        KUN_ASSIGN_OR_FAIL(Value lhs, getScalar(op->getOperand(0), b, ol));
        KUN_ASSIGN_OR_FAIL(Value rhs, getScalar(op->getOperand(1), b, ol));
        scalarMap[op->getResult(0)] = iface.buildScalarOp(b, ol, lhs, rhs);
      } else if (auto iface = dyn_cast<UnaryArithInterface>(op)) {
        KUN_ASSIGN_OR_FAIL(Value operand, getScalar(op->getOperand(0), b, ol));
        scalarMap[op->getResult(0)] = iface.buildScalarOp(b, ol, operand);
      } else if (auto ri = dyn_cast<ReduceArithInterface>(op)) {
        KUN_ASSIGN_OR_FAIL(Value elem, getScalar(op->getOperand(0), b, ol));
        auto it = scalarMap.find(op->getResult(0));
        assert(it != scalarMap.end() &&
               "reduce result must be pre-seeded in scalarMap with current acc");
        it->second = ri.buildAccumOp(b, ol, it->second, elem);
      } else if (auto sel = dyn_cast<SelectOp>(op)) {
        KUN_ASSIGN_OR_FAIL(Value cond, getScalar(sel.getCond(),      b, ol));
        KUN_ASSIGN_OR_FAIL(Value tv,   getScalar(sel.getTrueValue(), b, ol));
        KUN_ASSIGN_OR_FAIL(Value fv,   getScalar(sel.getFalseValue(),b, ol));
        scalarMap[sel.getResult()] =
            b.create<arith::SelectOp>(ol, cond, tv, fv).getResult();
      } else if (auto br = dyn_cast<BackRefOp>(op)) {
        Value offset = b.create<arith::ConstantOp>(
            ol, b.getI32Type(), b.getI32IntegerAttr(br.getWindow()));
        KUN_ASSIGN_OR_FAIL(Value scalar,
            getScalarUncached(br.getInput(), offset, b, ol));
        scalarMap[br.getResult()] = scalar;
      } else if (handleUnknown) {
        if (failed(handleUnknown(*op))) return failure();
      } else {
        return op->emitError("kunir-to-kungpu: cannot lower op in block");
      }
    }
    return success();
  }

  LogicalResult lowerBlock(
      Block &block, OpBuilder &b,
      llvm::function_ref<LogicalResult(Operation &)> handleUnknown = nullptr) {
    SmallVector<Operation *> ops;
    for (Operation &op : block.without_terminator())
      ops.push_back(&op);
    return lowerBlock(ops, b, handleUnknown);
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct LowerKunIrToKunGpuPass
    : PassWrapper<LowerKunIrToKunGpuPass, OperationPass<kunir::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKunIrToKunGpuPass)
  StringRef getArgument()    const override { return "kunir-to-kungpu"; }
  StringRef getDescription() const override {
    return "Lower kunir ops to kungpu + scf + arith/math"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<kungpu::KunGpuDialect, arith::ArithDialect,
                    math::MathDialect, scf::SCFDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void LowerKunIrToKunGpuPass::runOnOperation() {
  kunir::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  Location loc = funcOp.getLoc();

  Block &entry = funcOp.getBody().front();

  // ------------------------------------------------------------------
  // 1. Extend function signature: ts return types → extra output params.
  //    Runtime-scalar args (time_length, num_stocks, mask, chunk_size,
  //    warmup) are added later by convert-kungpu-to-llvm's
  //    convertFuncSignature, not here.
  // ------------------------------------------------------------------
  FunctionType oldFT = funcOp.getFunctionTypeTyped();
  SmallVector<Type> newArgTys(oldFT.getInputs());
  SmallVector<unsigned> tsRetIdx;
  for (auto [i, ty] : llvm::enumerate(oldFT.getResults()))
    if (isa<TsType>(ty)) tsRetIdx.push_back(i);

  SmallVector<Value> outParams;
  for (unsigned i : tsRetIdx) {
    outParams.push_back(entry.addArgument(oldFT.getResult(i), loc));
    newArgTys.push_back(oldFT.getResult(i));
  }
  SmallVector<Type> newRetTys;
  for (auto [i, ty] : llvm::enumerate(oldFT.getResults()))
    if (!isa<TsType>(ty)) newRetTys.push_back(ty);
  funcOp.setFunctionTypeAttr(
      TypeAttr::get(FunctionType::get(ctx, newArgTys, newRetTys)));


  // ------------------------------------------------------------------
  // 2. Snapshot original ops and find the original return.
  // ------------------------------------------------------------------
  SmallVector<Operation *> origOps;
  kunir::ReturnOp retOp;
  for (Operation &op : entry) origOps.push_back(&op);
  for (Operation *op : origOps)
    if (auto r = dyn_cast<kunir::ReturnOp>(op)) { retOp = r; break; }

  // Collect ts return values from the original return.
  SmallVector<Value> tsRetVals;
  if (retOp)
    for (Value v : retOp.getOperands())
      if (isa<TsType>(v.getType())) tsRetVals.push_back(v);
  assert(tsRetVals.size() == outParams.size());

  // ------------------------------------------------------------------
  // 3. Insert outer scf.for loop before the first original op.
  //    windowed_temp ops are inserted before this loop (via `b`).
  // ------------------------------------------------------------------
  OpBuilder b(ctx);
  b.setInsertionPoint(origOps.front());

  // Per-chunk bounds.  Both ops are operandless — chunk_size / warmup /
  // time_length all live as kernel scalar args added by
  // convert-kungpu-to-llvm and are read at lowering time.  When the
  // caller's launcher uses num_chunks = 1 it sets chunk_size =
  // time_length so chunk 0 covers the full range.
  Value lb = b.create<TimeLbOp>(loc, b.getIndexType());
  Value ub = b.create<TimeUbOp>(loc, b.getIndexType());
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  // Outer-loop ts.get/put always reference the current time step, i.e.
  // tail-relative offset = 0 (i32).  Created before outerFor so it dominates
  // every use inside the loop body.
  Value zeroOffsetI32 = b.create<arith::ConstantOp>(
      loc, b.getI32Type(), b.getI32IntegerAttr(0));
  auto outerFor = b.create<scf::ForOp>(loc, lb, ub, c1);

  // Erase the implicit empty scf.yield (no iter_args → zero-operand yield).
  outerFor.getBody()->back().erase();
  OpBuilder fb = OpBuilder::atBlockEnd(outerFor.getBody());

  // Point b before outerFor so windowed_temp ops land outside the time loop.
  b.setInsertionPoint(outerFor);

  // ------------------------------------------------------------------
  // 4. Build outer LowerHelper; seed each ts-typed function argument into tsMap.
  // ------------------------------------------------------------------
  LowerHelper outer;
  outer.zeroOffsetI32 = zeroOffsetI32;
  unsigned numOrigArgs = oldFT.getNumInputs();
  for (unsigned i = 0; i < numOrigArgs; ++i) {
    Value arg = entry.getArgument(i);
    if (isa<TsType>(arg.getType()))
      outer.tsMap[arg] = arg;
  }

  // ------------------------------------------------------------------
  // 5. Lower original ops in definition order.
  //
  //    LowerHelper::lowerBlock handles binary/unary/reduce/select/back_ref ops.
  //    windowed_output, for_each_back_window, fast_windowed_sum, and
  //    func.return are outer-scope only and are handled by the callback
  //    below.
  // ------------------------------------------------------------------
  auto outerHandler = [&](Operation &op) -> LogicalResult {
    if (isa<kunir::ReturnOp>(op)) return success(); // handled in step 7

    Location ol = op.getLoc();

    // windowed_output → allocate windowed_temp outside the loop,
    //                   fill circular buffer at each time step inside.
    if (auto woOp = dyn_cast<WindowedOutputOp>(op)) {
      auto wt = b.create<WindowedTempOp>(ol, woOp.getResult().getType());
      outer.tsMap[woOp.getResult()] = wt.getResult();
      KUN_ASSIGN_OR_FAIL(Value inputScalar,
                         outer.getScalar(woOp.getInput(), fb, ol));
      fb.create<TsPutOp>(ol, wt.getResult(), inputScalar);
      return success();
    }

    // for_each_back_window → nested scf.for with iter_args.
    if (auto fwOp = dyn_cast<ForEachBackWindowOp>(op)) {
      int64_t window = fwOp.getWindow();
      Block &body = fwOp.getBody().front();
      auto yieldOp = llvm::cast<YieldOp>(body.getTerminator());

      // Verify all inputs are ts handles (already in outer.tsMap).
      for (Value inp : fwOp.getInputs()) {
        if (!outer.tsMap.count(inp))
          return op.emitError("kunir-to-kungpu: for_each_back_window input "
                              "must be a ts handle");
      }

      // Each yield operand must come from a reduce_* op — collect init values.
      SmallVector<Value> initVals;
      for (Value yv : yieldOp.getValues()) {
        auto *defOp = yv.getDefiningOp();
        auto ri = defOp ? dyn_cast<ReduceArithInterface>(defOp)
                        : ReduceArithInterface{};
        if (!ri) {
          return op.emitError("kunir-to-kungpu: for_each_back_window yield "
                              "operand must come from a reduce_* op");
        }
        auto elemTy = llvm::cast<FloatType>(
            llvm::cast<TsType>(defOp->getOperand(0).getType()).getElementType());
        initVals.push_back(fb.create<arith::ConstantOp>(ol, ri.getInitValue(elemTy)));
      }

      // Create inner scf.for %w = 0 to window step 1 iter_args(acc_i = init_i).
      // The lambda form lets us emit a proper scf.yield as the body terminator
      // without fighting the implicit yield created by ensureTerminator.
      Value wBound  = fb.create<arith::ConstantIndexOp>(ol, window);
      Value wM1_i32 = fb.create<arith::ConstantOp>(
          ol, fb.getI32Type(), fb.getI32IntegerAttr(window - 1));

      // Capture lowerBlock result since the lambda can't return LogicalResult.
      bool innerOk = true;
      auto innerFor = fb.create<scf::ForOp>(
          ol, c0, wBound, c1, initVals,
          [&](OpBuilder &ib, Location il, Value w, ValueRange iterArgs) {
            // Tail-relative offset for this window step.  Iterating w from 0
            // to window-1 reads oldest-to-newest, i.e. offset = window-1-w.
            Value w_i32 =
                ib.create<arith::IndexCastOp>(il, ib.getI32Type(), w);
            Value windowedOffset =
                ib.create<arith::SubIOp>(il, wM1_i32, w_i32);

            // Inner LowerHelper inherits the outer tsMap/scalarMap so reads
            // inside the body can still reach outer-scope handles (e.g. a
            // back_ref placed in the body) and outer-scope scalars, and
            // shares the function-scope zero-offset constant.
            //
            // Pre-loads, written directly into inner.scalarMap (bypassing
            // the offset-0 cache since these reads are at non-zero offsets):
            //   - Each block arg = its corresponding ts input loaded at the
            //     windowed offset (via getScalarUncached).
            //   - Each reduce result = the matching iter_arg accumulator.
            //
            // After this setup the body has no non-zero-offset reads left;
            // lowerBlock just uses offset 0 + scalarMap for everything.
            LowerHelper inner{outer.tsMap, outer.scalarMap, outer.zeroOffsetI32};
            for (auto [i, arg] : llvm::enumerate(body.getArguments())) {
              auto r = inner.getScalarUncached(fwOp.getInputs()[i],
                                                windowedOffset, ib, il);
              if (failed(r)) {
                innerOk = false;
                ib.create<scf::YieldOp>(il, initVals);
                return;
              }
              inner.scalarMap[arg] = *r;
            }
            for (auto [i, yv] : llvm::enumerate(yieldOp.getValues()))
              inner.scalarMap[yv] = iterArgs[i];

            if (failed(inner.lowerBlock(body, ib))) {
              innerOk = false;
              ib.create<scf::YieldOp>(il, initVals); // keep IR structurally valid
              return;
            }

            SmallVector<Value> newAccs;
            for (Value yv : yieldOp.getValues())
              newAccs.push_back(inner.scalarMap.find(yv)->second);
            ib.create<scf::YieldOp>(il, newAccs);
          });
      if (!innerOk) return failure();

      // Map for_each_back_window results (scalar reduce accs) to the inner
      // for's results.
      for (auto [i, res] : llvm::enumerate(fwOp.getResults()))
        outer.scalarMap[res] = innerFor.getResult(i);
      return success();
    }

    // fast_windowed_sum → preserved as a kunir op with scalar result and
    // ts-handle input.  The kungpu-to-llvm pass owns the actual lowering
    // (per-thread state allocas + the Kahan-corrected step).
    if (auto fws = dyn_cast<FastWindowedSumOp>(op)) {
      auto inputTs = llvm::cast<TsType>(fws.getInput().getType());
      auto inputIt = outer.tsMap.find(fws.getInput());
      if (inputIt == outer.tsMap.end())
        return op.emitError(
            "kunir-to-kungpu: fast_windowed_sum input must be a ts handle");
      auto newOp = fb.create<FastWindowedSumOp>(
          ol, /*resultType=*/inputTs.getElementType(),
          /*input=*/inputIt->second, fws.getWindowAttr());
      outer.scalarMap[fws.getResult()] = newOp.getResult();
      return success();
    }

    return op.emitError("kunir-to-kungpu: unhandled op in outer block");
  };

  if (failed(outer.lowerBlock(origOps, fb, outerHandler)))
    return signalPassFailure();

  // ------------------------------------------------------------------
  // 6. Emit ts.put for each ts return value, then close the outer for.
  // ------------------------------------------------------------------
  for (auto [outParam, rv] : llvm::zip(outParams, tsRetVals)) {
    auto it = outer.scalarMap.find(rv);
    assert(it != outer.scalarMap.end() &&
           "ts return value not materialised as a scalar");
    fb.create<TsPutOp>(loc, outParam, it->second);
  }
  fb.create<scf::YieldOp>(loc);

  // ------------------------------------------------------------------
  // 7. Insert a replacement return before the original return op.
  // ------------------------------------------------------------------
  if (retOp) {
    b.setInsertionPoint(retOp);
    SmallVector<Value> nonTsRets;
    for (Value v : retOp.getOperands())
      if (!isa<TsType>(v.getType())) nonTsRets.push_back(v);
    b.create<kunir::ReturnOp>(loc, mlir::ValueRange(nonTsRets));
  }

  // ------------------------------------------------------------------
  // 8. Erase original ops in reverse order.
  // ------------------------------------------------------------------
  for (Operation *op : llvm::reverse(origOps))
    op->erase();
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace kunir {
void registerKunIrToKunGpuPass() {
  PassRegistration<LowerKunIrToKunGpuPass>();
}
std::unique_ptr<mlir::Pass> createKunIrToKunGpuPass() {
  return std::make_unique<LowerKunIrToKunGpuPass>();
}
} // namespace kunir
