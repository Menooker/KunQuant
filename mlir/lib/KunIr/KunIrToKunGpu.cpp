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
//
// `outerTimeIdx` / `outerLoopLb` are the outer scf.for time loop's induction
// variable and lower bound (index type).  BackRef's warmup guard
// (`t - loop_lb < window` → NaN) needs both; threading them through
// LowerHelper keeps the guard available inside for_each_back_window bodies
// as well, since `t` there is still the OUTER time index.
struct LowerHelper {
  HandleMap tsMap;
  HandleMap scalarMap;
  Value zeroOffsetI32;
  Value outerTimeIdx;   // outer scf.for induction var (index)
  Value outerLoopLb;    // outer scf.for lower bound (index)
  // Inside a for_each_back_window body: the current window step offset
  // (window-1-w).  Used by argmin/argmax to record the position index.
  Value windowedOffsetI32;
  // Running accumulators for each reduce op in the enclosing FBW body.
  // Single-state reduce: 1 entry; argmin/max: {best_val, best_idx};
  // rank: {less_count, eq_count}.  Seeded by FBW pre-loop, updated by
  // each reduce step, read by scf.yield.
  llvm::DenseMap<Value, SmallVector<Value, 2>> multiAccs;

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
    return TsGetOp::create(b, loc, tsTy.getElementType(),
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

  // One step of a multi-state reduce (argmin/argmax/rank).  Mirrors
  // cpp/Kun/Ops.hpp's step() exactly so CPU and GPU match bit-for-bit
  // (modulo reduction-order changes).
  LogicalResult lowerMultiReduce(Operation *op, OpBuilder &b, Location ol) {
    auto isArgMin = isa<kunir::ReduceArgMinOp>(op);
    auto isArgMax = isa<kunir::ReduceArgMaxOp>(op);
    auto isRank   = isa<kunir::ReduceRankOp>(op);
    assert(isArgMin || isArgMax || isRank);

    KUN_ASSIGN_OR_FAIL(Value elem, getScalar(op->getOperand(0), b, ol));
    FloatType elemTy = llvm::cast<FloatType>(elem.getType());
    auto &accs = multiAccs[op->getResult(0)];
    assert(accs.size() == 2 &&
           "multi-state reduce must be pre-seeded with 2 iter_args");

    auto fconst = [&](double v) {
      return arith::ConstantOp::create(b, ol, elemTy,
                                          b.getFloatAttr(elemTy, v))
          .getResult();
    };
    auto fIsNan = [&](Value v) {
      return arith::CmpFOp::create(b, ol, arith::CmpFPredicate::UNE, v, v)
          .getResult();
    };
    Value nanF = fconst(std::numeric_limits<double>::quiet_NaN());
    Value one  = fconst(1.0);

    if (isArgMin || isArgMax) {
      // accs = {best_val, best_idx}.  Ordered compare so NaN doesn't
      // trigger the update; NaN is propagated by the final selects.
      Value bestVal = accs[0];
      Value bestIdx = accs[1];
      Value bestIsNan = fIsNan(bestVal);
      Value elemIsNan = fIsNan(elem);
      auto pred = isArgMin ? arith::CmpFPredicate::OGT
                            : arith::CmpFPredicate::OLT;
      Value cmp = arith::CmpFOp::create(b, ol, pred, bestVal, elem)
                      .getResult();
      Value newVal = arith::SelectOp::create(b, ol, cmp, elem, bestVal)
                          .getResult();
      // Record the window-relative position (window-1-w) so
      // TsArgMin = window - ReduceArgMin gives pandas's
      // np.argmin()+1 convention (1=oldest, window=newest).
      Value wIdxF = arith::SIToFPOp::create(b, ol, elemTy,
                                                windowedOffsetI32)
                        .getResult();
      Value newIdx = arith::SelectOp::create(b, ol, cmp, wIdxF, bestIdx)
                          .getResult();
      Value anyNan = arith::OrIOp::create(b, ol, bestIsNan, elemIsNan)
                          .getResult();
      newVal = arith::SelectOp::create(b, ol, anyNan, nanF, newVal)
                  .getResult();
      newIdx = arith::SelectOp::create(b, ol, anyNan, nanF, newIdx)
                  .getResult();
      accs[0] = newVal;
      accs[1] = newIdx;
      return success();
    }

    // ReduceRank: accs = {less_count, eq_count}; `current` is an
    // outer-scope ts<f, 1> already in scalarMap.
    KUN_ASSIGN_OR_FAIL(Value cur, getScalar(op->getOperand(1), b, ol));
    Value lessCnt = accs[0];
    Value eqCnt   = accs[1];
    Value curIsNan  = fIsNan(cur);
    Value elemIsNan = fIsNan(elem);
    Value anyNan    = arith::OrIOp::create(b, ol, curIsNan, elemIsNan)
                          .getResult();
    Value cmpLess = arith::CmpFOp::create(
                        b, ol, arith::CmpFPredicate::OLT, elem, cur)
                        .getResult();
    Value cmpEq   = arith::CmpFOp::create(
                        b, ol, arith::CmpFPredicate::OEQ, elem, cur)
                        .getResult();
    Value lessP1 = arith::AddFOp::create(b, ol, lessCnt, one).getResult();
    Value newLess = arith::SelectOp::create(b, ol, cmpLess, lessP1, lessCnt)
                        .getResult();
    // NaN routed only into less_count — the final rank extract
    // (`less + (eq + 1) / 2`, computed after the scf.for) then
    // propagates NaN out.
    newLess = arith::SelectOp::create(b, ol, anyNan, nanF, newLess)
                  .getResult();
    Value eqP1   = arith::AddFOp::create(b, ol, eqCnt, one).getResult();
    Value newEq  = arith::SelectOp::create(b, ol, cmpEq, eqP1, eqCnt)
                        .getResult();
    accs[0] = newLess;
    accs[1] = newEq;
    return success();
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
        // Running acc lives in multiAccs[result][0] (see FBW lowering
        // for the pre-seed); single- and multi-state reduces share
        // the same storage so scf.yield reads them uniformly.
        KUN_ASSIGN_OR_FAIL(Value elem, getScalar(op->getOperand(0), b, ol));
        auto mit = multiAccs.find(op->getResult(0));
        assert(mit != multiAccs.end() && mit->second.size() == 1 &&
               "reduce result must be pre-seeded in multiAccs with current acc");
        mit->second[0] = ri.buildAccumOp(b, ol, mit->second[0], elem);
      } else if (isa<kunir::ReduceArgMinOp, kunir::ReduceArgMaxOp,
                       kunir::ReduceRankOp>(op)) {
        if (failed(lowerMultiReduce(op, b, ol)))
          return failure();
      } else if (auto sel = dyn_cast<SelectOp>(op)) {
        KUN_ASSIGN_OR_FAIL(Value cond, getScalar(sel.getCond(),      b, ol));
        KUN_ASSIGN_OR_FAIL(Value tv,   getScalar(sel.getTrueValue(), b, ol));
        KUN_ASSIGN_OR_FAIL(Value fv,   getScalar(sel.getFalseValue(),b, ol));
        scalarMap[sel.getResult()] =
            arith::SelectOp::create(b, ol, cond, tv, fv).getResult();
      } else if (auto br = dyn_cast<BackRefOp>(op)) {
        // Warmup guard:  if   t - outer_loop_lb < window  →  NaN
        //                else                            →  ts.get(window)
        //
        // Chunk 0 has loop_lb = 0 so the guard collapses to CPU's
        // "first window-1 outputs are NaN".  Chunk k>=1's per-CTA state
        // (e.g. fast-stat accumulators) is zero-initialised at function
        // entry, so each chunk needs `window` add-only steps to rebuild
        // the trailing-window state — gating the "remove" value with
        // NaN here propagates through NaN-aware remove patterns
        // (Equals(oldx, oldx) === false on NaN) and auto-suppresses the
        // subtract step during warmup.
        //
        // The scf.if uses the manual-OpBuilder (not body-builder-lambda)
        // form, so the enclosing function context is still `lowerBlock`
        // — KUN_ASSIGN_OR_FAIL can return failure from here without
        // tripping any lambda return-type mismatch.
        int64_t window = br.getWindow();
        auto inputTs = llvm::cast<TsType>(br.getInput().getType());
        auto floatTy = llvm::dyn_cast<FloatType>(inputTs.getElementType());
        if (!floatTy)
          return br.emitError("kunir-to-kungpu: back_ref input must have a "
                              "float element type (NaN required for the "
                              "warmup guard)");

        Value delta =
            arith::SubIOp::create(b, ol, outerTimeIdx, outerLoopLb);
        Value windowIdx =
            arith::ConstantIndexOp::create(b, ol, window);
        Value inSteady = arith::CmpIOp::create(
            b, ol, arith::CmpIPredicate::sge, delta, windowIdx);
        auto ifOp = scf::IfOp::create(b, ol, TypeRange{floatTy}, inSteady,
                                          /*withElseRegion=*/true);
        {
          OpBuilder ib =
              OpBuilder::atBlockBegin(&ifOp.getThenRegion().front());
          Value offset = arith::ConstantOp::create(
              ib, ol, ib.getI32Type(), ib.getI32IntegerAttr(window));
          KUN_ASSIGN_OR_FAIL(Value loaded,
              getScalarUncached(br.getInput(), offset, ib, ol));
          scf::YieldOp::create(ib, ol, loaded);
        }
        {
          OpBuilder ib =
              OpBuilder::atBlockBegin(&ifOp.getElseRegion().front());
          llvm::APFloat qnan =
              llvm::APFloat::getQNaN(floatTy.getFloatSemantics());
          Value nanV = arith::ConstantOp::create(
              ib, ol, floatTy, FloatAttr::get(floatTy, qnan));
          scf::YieldOp::create(ib, ol, nanV);
        }
        scalarMap[br.getResult()] = ifOp.getResult(0);
      } else if (auto co = dyn_cast<ConstantOp>(op)) {
        auto resTs = llvm::cast<TsType>(co.getResult().getType());
        Type elemTy = resTs.getElementType();
        // The op carries an f64 attribute; convert to the element type
        // (f32 / f64) so arith.constant gets a type-matching attribute.
        llvm::APFloat apv(co.getValue());
        if (auto ft = llvm::dyn_cast<FloatType>(elemTy)) {
          bool losesInfo = false;
          apv.convert(ft.getFloatSemantics(),
                      llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        }
        scalarMap[co.getResult()] = arith::ConstantOp::create(
            b, ol, elemTy, b.getFloatAttr(elemTy, apv));
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
  Value lb = TimeLbOp::create(b, loc, b.getIndexType());
  Value ub = TimeUbOp::create(b, loc, b.getIndexType());
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  // Outer-loop ts.get/put always reference the current time step, i.e.
  // tail-relative offset = 0 (i32).  Created before outerFor so it dominates
  // every use inside the loop body.
  Value zeroOffsetI32 = arith::ConstantOp::create(
      b, loc, b.getI32Type(), b.getI32IntegerAttr(0));
  auto outerFor = scf::ForOp::create(b, loc, lb, ub, c1);

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
  outer.outerTimeIdx  = outerFor.getInductionVar();
  outer.outerLoopLb   = outerFor.getLowerBound();
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
      auto wt = WindowedTempOp::create(b, ol, woOp.getResult().getType());
      outer.tsMap[woOp.getResult()] = wt.getResult();
      KUN_ASSIGN_OR_FAIL(Value inputScalar,
                         outer.getScalar(woOp.getInput(), fb, ol));
      TsPutOp::create(fb, ol, wt.getResult(), inputScalar);
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

      // Build the iter_args layout: single-state reduce = 1 init,
      // argmin/max = (best_val, best_idx), rank = (less, eq).
      struct ReduceSlot {
        int numAccs;
        int startIdx;
      };
      SmallVector<ReduceSlot> slots; // parallel to yieldOp.getValues()
      SmallVector<Value> initVals;
      auto elemTyOf = [](Operation *defOp) -> FloatType {
        return llvm::cast<FloatType>(
            llvm::cast<TsType>(defOp->getOperand(0).getType()).getElementType());
      };
      auto pushConst = [&](FloatType elemTy, double v) {
        initVals.push_back(arith::ConstantOp::create(
            fb, ol, elemTy, fb.getFloatAttr(elemTy, v)));
      };
      for (Value yv : yieldOp.getValues()) {
        auto *defOp = yv.getDefiningOp();
        if (!defOp) {
          return op.emitError("kunir-to-kungpu: for_each_back_window yield "
                              "operand has no defining op");
        }
        ReduceSlot slot{0, (int)initVals.size()};
        if (auto ri = dyn_cast<ReduceArithInterface>(defOp)) {
          FloatType elemTy = elemTyOf(defOp);
          initVals.push_back(arith::ConstantOp::create(
              fb, ol, ri.getInitValue(elemTy)));
          slot.numAccs = 1;
        } else if (isa<kunir::ReduceArgMinOp>(defOp) ||
                     isa<kunir::ReduceArgMaxOp>(defOp)) {
          FloatType elemTy = elemTyOf(defOp);
          double inf = std::numeric_limits<double>::infinity();
          pushConst(elemTy, isa<kunir::ReduceArgMinOp>(defOp) ? inf : -inf);
          pushConst(elemTy, 0.0);
          slot.numAccs = 2;
        } else if (isa<kunir::ReduceRankOp>(defOp)) {
          FloatType elemTy = elemTyOf(defOp);
          pushConst(elemTy, 0.0);
          pushConst(elemTy, 0.0);
          slot.numAccs = 2;
        } else {
          return op.emitError("kunir-to-kungpu: for_each_back_window yield "
                              "operand must come from a reduce_* op");
        }
        slots.push_back(slot);
      }

      // Create inner scf.for %w = 0 to window step 1 iter_args(acc_i = init_i).
      // The lambda form lets us emit a proper scf.yield as the body terminator
      // without fighting the implicit yield created by ensureTerminator.
      Value wBound  = arith::ConstantIndexOp::create(fb, ol, window);
      Value wM1_i32 = arith::ConstantOp::create(
          fb, ol, fb.getI32Type(), fb.getI32IntegerAttr(window - 1));

      // Capture lowerBlock result since the lambda can't return LogicalResult.
      bool innerOk = true;
      auto innerFor = scf::ForOp::create(
          fb, ol, c0, wBound, c1, initVals,
          [&](OpBuilder &ib, Location il, Value w, ValueRange iterArgs) {
            // Tail-relative offset for this window step.  Iterating w from 0
            // to window-1 reads oldest-to-newest, i.e. offset = window-1-w.
            Value w_i32 =
                arith::IndexCastOp::create(ib, il, ib.getI32Type(), w);
            Value windowedOffset =
                arith::SubIOp::create(ib, il, wM1_i32, w_i32);

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
            LowerHelper inner{outer.tsMap, outer.scalarMap,
                                outer.zeroOffsetI32,
                                outer.outerTimeIdx, outer.outerLoopLb};
            // Hand the inner helper the current window-step offset
            // (window-1-w) so multi-state reductions (argmin/argmax)
            // can use it as the recorded `index`.
            inner.windowedOffsetI32 = windowedOffset;
            for (auto [i, arg] : llvm::enumerate(body.getArguments())) {
              auto r = inner.getScalarUncached(fwOp.getInputs()[i],
                                                windowedOffset, ib, il);
              if (failed(r)) {
                innerOk = false;
                scf::YieldOp::create(ib, il, initVals);
                return;
              }
              inner.scalarMap[arg] = *r;
            }
            // Pre-seed accumulators from iter_args.
            for (auto [i, yv] : llvm::enumerate(yieldOp.getValues())) {
              const auto &slot = slots[i];
              SmallVector<Value, 2> accs;
              accs.reserve(slot.numAccs);
              for (int j = 0; j < slot.numAccs; ++j)
                accs.push_back(iterArgs[slot.startIdx + j]);
              inner.multiAccs[yv] = std::move(accs);
            }

            if (failed(inner.lowerBlock(body, ib))) {
              innerOk = false;
              scf::YieldOp::create(ib, il, initVals); // keep IR structurally valid
              return;
            }

            // Yield the updated accumulators back into the iter_args.
            SmallVector<Value> newAccs(initVals.size());
            for (auto [i, yv] : llvm::enumerate(yieldOp.getValues())) {
              const auto &slot = slots[i];
              const auto &accs = inner.multiAccs[yv];
              for (int j = 0; j < slot.numAccs; ++j)
                newAccs[slot.startIdx + j] = accs[j];
            }
            scf::YieldOp::create(ib, il, newAccs);
          });
      if (!innerOk) return failure();

      // Project each fwOp result from the inner-for's iter_arg slice:
      // single-state passes through, argmin/max returns best_idx, rank
      // computes less + (eq + 1) / 2.
      OpBuilder::InsertionGuard guardPost(b);
      b.setInsertionPointAfter(innerFor);
      for (auto [i, res] : llvm::enumerate(fwOp.getResults())) {
        const auto &slot = slots[i];
        Value yv = yieldOp.getValues()[i];
        auto *defOp = yv.getDefiningOp();
        Value finalVal;
        if (slot.numAccs == 1) {
          finalVal = innerFor.getResult(slot.startIdx);
        } else if (isa<kunir::ReduceArgMinOp>(defOp) ||
                     isa<kunir::ReduceArgMaxOp>(defOp)) {
          finalVal = innerFor.getResult(slot.startIdx + 1);
        } else {
          // ReduceRankOp:  less + (eq + 1) / 2
          Value less = innerFor.getResult(slot.startIdx);
          Value eq   = innerFor.getResult(slot.startIdx + 1);
          auto elemTy = llvm::cast<FloatType>(less.getType());
          Value one = arith::ConstantOp::create(
              b, ol, elemTy, b.getFloatAttr(elemTy, 1.0));
          Value two = arith::ConstantOp::create(
              b, ol, elemTy, b.getFloatAttr(elemTy, 2.0));
          Value eqp1 = arith::AddFOp::create(b, ol, eq, one);
          Value half = arith::DivFOp::create(b, ol, eqp1, two);
          finalVal = arith::AddFOp::create(b, ol, less, half);
        }
        outer.scalarMap[res] = finalVal;
      }
      return success();
    }

    // kunir.accumulator → kungpu.accumulator (allocated outside the time
    // loop, like windowed_temp).  Stored in tsMap so that downstream reads
    // (via getScalar → kungpu.ts.get @ offset 0) resolve to the slot.
    if (auto acc = dyn_cast<kunir::AccumulatorOp>(op)) {
      auto ka = kungpu::AccumulatorOp::create(
          b, ol, acc.getResult().getType(), acc.getNameAttr());
      outer.tsMap[acc.getResult()] = ka.getResult();
      return success();
    }

    // kunir.set_accumulator → scf.if (mask) { kungpu.ts.put %acc, %value }
    // inside the outer time loop.  mask and value are loaded at offset 0
    // (current time step) via the standard scalarMap-cached getScalar.
    if (auto sa = dyn_cast<kunir::SetAccumulatorOp>(op)) {
      auto accIt = outer.tsMap.find(sa.getAcc());
      if (accIt == outer.tsMap.end())
        return op.emitError("kunir-to-kungpu: set_accumulator acc must come "
                            "from a kunir.accumulator");
      KUN_ASSIGN_OR_FAIL(Value maskScalar,
                         outer.getScalar(sa.getMask(),  fb, ol));
      KUN_ASSIGN_OR_FAIL(Value valueScalar,
                         outer.getScalar(sa.getValue(), fb, ol));
      auto ifOp = scf::IfOp::create(fb, ol, /*resultTypes=*/TypeRange{},
                                         maskScalar, /*withElseRegion=*/false);
      OpBuilder ib = OpBuilder::atBlockBegin(&ifOp.getThenRegion().front());
      TsPutOp::create(ib, ol, accIt->second, valueScalar);
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
      auto newOp = FastWindowedSumOp::create(
          fb, ol, /*resultType=*/inputTs.getElementType(),
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
    TsPutOp::create(fb, loc, outParam, it->second);
  }
  scf::YieldOp::create(fb, loc);

  // ------------------------------------------------------------------
  // 7. Insert a replacement return before the original return op.
  // ------------------------------------------------------------------
  if (retOp) {
    b.setInsertionPoint(retOp);
    SmallVector<Value> nonTsRets;
    for (Value v : retOp.getOperands())
      if (!isa<TsType>(v.getType())) nonTsRets.push_back(v);
    kunir::ReturnOp::create(b, loc, mlir::ValueRange(nonTsRets));
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
