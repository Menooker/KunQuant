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
//   - kunir.cs_rank is not yet supported.
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

namespace {

//===----------------------------------------------------------------------===//
// Value tracking: is a kunir ts value a ts handle or a scalar?
//
// HANDLE — the mapped Value is a ts memory object (!kunir.ts<*>); loading it
//          via ts.get at a given time index yields the element scalar.
// SCALAR — the mapped Value is an already-computed float scalar.
//===----------------------------------------------------------------------===//

enum class TsKind { Handle, Scalar };
struct TsEntry { TsKind kind; Value value; };
using TsMap = llvm::DenseMap<Value, TsEntry>;

// If `v` is mapped as a Handle in tsMap, emit ts.get(handle, timeIdx) and
// promote the entry to Scalar.  Returns the scalar value.
static Value getScalar(Value v, TsMap &tsMap, Value timeIdx,
                       OpBuilder &b, Location loc) {
  auto it = tsMap.find(v);
  assert(it != tsMap.end() && "value not found in tsMap");
  if (it->second.kind == TsKind::Scalar)
    return it->second.value;
  auto tsTy = llvm::cast<TsType>(v.getType());
  Value scalar = b.create<TsGetOp>(loc, tsTy.getElementType(),
                                    it->second.value, timeIdx);
  it->second = {TsKind::Scalar, scalar};
  return scalar;
}

// Lower non-terminator ops in `ops` in sequential (definition) order.
//
// For each op:
//   - BinaryArithInterface: emit scalar binary op, record result as Scalar.
//   - UnaryArithInterface:  emit scalar unary op,  record result as Scalar.
//   - ReduceArithInterface: caller must pre-seed the op's result in tsMap with
//     the current accumulator (iterArg).  This function emits the accumulation
//     step and updates the tsMap entry to the new accumulator.
//   - Anything else: call handleUnknown if provided, else return failure.
//
// Handle-typed operands are loaded via ts.get (getScalar) on first use.
static LogicalResult lowerBlock(
    llvm::ArrayRef<Operation *> ops,
    TsMap &tsMap, Value timeIdx, OpBuilder &b, Location loc,
    llvm::function_ref<LogicalResult(Operation &)> handleUnknown = nullptr) {
  for (Operation *op : ops) {
    Location ol = op->getLoc();
    if (auto iface = dyn_cast<BinaryArithInterface>(op)) {
      Value lhs = getScalar(op->getOperand(0), tsMap, timeIdx, b, ol);
      Value rhs = getScalar(op->getOperand(1), tsMap, timeIdx, b, ol);
      tsMap[op->getResult(0)] = {TsKind::Scalar,
          iface.buildScalarOp(b, ol, lhs, rhs)};
    } else if (auto iface = dyn_cast<UnaryArithInterface>(op)) {
      Value operand = getScalar(op->getOperand(0), tsMap, timeIdx, b, ol);
      tsMap[op->getResult(0)] = {TsKind::Scalar,
          iface.buildScalarOp(b, ol, operand)};
    } else if (auto ri = dyn_cast<ReduceArithInterface>(op)) {
      Value elem = getScalar(op->getOperand(0), tsMap, timeIdx, b, ol);
      auto it = tsMap.find(op->getResult(0));
      assert(it != tsMap.end() && it->second.kind == TsKind::Scalar
             && "reduce result must be pre-seeded in tsMap with current acc");
      it->second = {TsKind::Scalar,
          ri.buildAccumOp(b, ol, it->second.value, elem)};
    } else if (handleUnknown) {
      if (failed(handleUnknown(*op))) return failure();
    } else {
      return op->emitError("kunir-to-kungpu: cannot lower op in block");
    }
  }
  return success();
}

// Overload that collects non-terminator ops from `block` and delegates.
static LogicalResult lowerBlock(
    Block &block,
    TsMap &tsMap, Value timeIdx, OpBuilder &b, Location loc,
    llvm::function_ref<LogicalResult(Operation &)> handleUnknown = nullptr) {
  SmallVector<Operation *> ops;
  for (Operation &op : block.without_terminator())
    ops.push_back(&op);
  return lowerBlock(ops, tsMap, timeIdx, b, loc, handleUnknown);
}

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

  Value timeLen = b.create<TimeLengthOp>(loc, b.getIndexType());
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  auto outerFor = b.create<scf::ForOp>(loc, c0, timeLen, c1);
  Value t = outerFor.getInductionVar();

  // Erase the implicit empty scf.yield (no iter_args → zero-operand yield).
  outerFor.getBody()->back().erase();
  OpBuilder fb = OpBuilder::atBlockEnd(outerFor.getBody());

  // Point b before outerFor so windowed_temp ops land outside the time loop.
  b.setInsertionPoint(outerFor);

  // ------------------------------------------------------------------
  // 4. Seed tsMap: each ts-typed function argument is a Handle.
  // ------------------------------------------------------------------
  TsMap tsMap;
  unsigned numOrigArgs = oldFT.getNumInputs();
  for (unsigned i = 0; i < numOrigArgs; ++i) {
    Value arg = entry.getArgument(i);
    if (isa<TsType>(arg.getType()))
      tsMap[arg] = {TsKind::Handle, arg};
  }

  // ------------------------------------------------------------------
  // 5. Lower original ops in definition order.
  //
  //    lowerBlock handles binary/unary/reduce ops.  windowed_output,
  //    for_each_back_window, and func.return are handled by the callback.
  // ------------------------------------------------------------------
  auto outerHandler = [&](Operation &op) -> LogicalResult {
    if (isa<kunir::ReturnOp>(op)) return success(); // handled in step 7

    Location ol = op.getLoc();

    // windowed_output → allocate windowed_temp outside the loop,
    //                   fill circular buffer at each time step inside.
    if (auto woOp = dyn_cast<WindowedOutputOp>(op)) {
      auto wt = b.create<WindowedTempOp>(ol, woOp.getResult().getType());
      tsMap[woOp.getResult()] = {TsKind::Handle, wt.getResult()};
      Value inputScalar = getScalar(woOp.getInput(), tsMap, t, fb, ol);
      fb.create<TsPutOp>(ol, wt.getResult(), t, inputScalar);
      return success();
    }

    // for_each_back_window → nested scf.for with iter_args.
    if (auto fwOp = dyn_cast<ForEachBackWindowOp>(op)) {
      int64_t window = fwOp.getWindow();
      Block &body = fwOp.getBody().front();
      auto yieldOp = llvm::cast<YieldOp>(body.getTerminator());

      // Resolve inputs to ts handles.
      SmallVector<Value> inputHandles(fwOp.getInputs().size());
      for (auto [i, inp] : llvm::enumerate(fwOp.getInputs())) {
        auto it = tsMap.find(inp);
        if (it == tsMap.end() || it->second.kind != TsKind::Handle) {
          return op.emitError("kunir-to-kungpu: for_each_back_window input "
                              "must be a ts handle");
        }
        inputHandles[i] = it->second.value;
      }

      // Each yield operand must come from a reduce_* op — collect init values.
      SmallVector<ReduceArithInterface> reduces;
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
        reduces.push_back(ri);
      }

      // Create inner scf.for %w = 0 to window step 1 iter_args(acc_i = init_i).
      // The lambda form lets us emit a proper scf.yield as the body terminator
      // without fighting the implicit yield created by ensureTerminator.
      Value wBound = fb.create<arith::ConstantIndexOp>(ol, window);
      Value wM1    = fb.create<arith::ConstantIndexOp>(ol, window - 1);

      // Capture lowerBlock result since the lambda can't return LogicalResult.
      bool innerOk = true;
      auto innerFor = fb.create<scf::ForOp>(
          ol, c0, wBound, c1, initVals,
          [&](OpBuilder &ib, Location il, Value w, ValueRange iterArgs) {
            // elemIdx = t - (window - 1) + w
            Value base    = ib.create<arith::SubIOp>(il, t, wM1);
            Value elemIdx = ib.create<arith::AddIOp>(il, base, w);

            // Seed innerTsMap: block args as handles; reduce results as acc.
            TsMap innerTsMap;
            for (auto [i, arg] : llvm::enumerate(body.getArguments()))
              innerTsMap[arg] = {TsKind::Handle, inputHandles[i]};
            for (auto [i, yv] : llvm::enumerate(yieldOp.getValues()))
              innerTsMap[yv.getDefiningOp()->getResult(0)] = {TsKind::Scalar,
                                                              iterArgs[i]};

            if (failed(lowerBlock(body, innerTsMap, elemIdx, ib, il))) {
              innerOk = false;
              ib.create<scf::YieldOp>(il, initVals); // keep IR structurally valid
              return;
            }

            SmallVector<Value> newAccs;
            for (Value yv : yieldOp.getValues())
              newAccs.push_back(innerTsMap.find(yv)->second.value);
            ib.create<scf::YieldOp>(il, newAccs);
          });
      if (!innerOk) return failure();

      // Map for_each_back_window results to the inner for's results.
      for (auto [i, res] : llvm::enumerate(fwOp.getResults()))
        tsMap[res] = {TsKind::Scalar, innerFor.getResult(i)};
      return success();
    }

    if (isa<CsRankOp>(op)) {
      return op.emitError("kunir-to-kungpu: cs_rank lowering not yet implemented");
    }
    return op.emitError("kunir-to-kungpu: unhandled op in outer block");
  };

  if (failed(lowerBlock(origOps, tsMap, t, fb, loc, outerHandler)))
    return signalPassFailure();

  // ------------------------------------------------------------------
  // 6. Emit ts.put for each ts return value, then close the outer for.
  // ------------------------------------------------------------------
  for (auto [outParam, rv] : llvm::zip(outParams, tsRetVals)) {
    auto it = tsMap.find(rv);
    assert(it != tsMap.end() && it->second.kind == TsKind::Scalar);
    fb.create<TsPutOp>(loc, outParam, t, it->second.value);
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
} // namespace kunir
