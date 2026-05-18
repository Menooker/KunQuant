//===- IRBuilder.cpp - Programmatic kunir module construction from Python ===//

#include "IRBuilder.h"
#include "PyModule.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "KunIr/KunIrAttrs.h"
#include "KunIr/KunIrOps.h"
#include "KunIr/KunIrTypes.h"

#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace mlir;

namespace kun_mlir_py {

namespace {

/// Stateful kunir builder.  Holds an MLIRContext + ModuleOp (via PyModule),
/// a current OpBuilder insertion point, and a stack used by
/// for_each_back_window's region nesting.
class IRBuilder {
public:
  IRBuilder()
      : pm_(std::make_unique<PyModule>()), b_(pm_->ctx.get()) {
    Location loc = b_.getUnknownLoc();
    pm_->module = OwningOpRef<ModuleOp>(ModuleOp::create(loc));
    b_.setInsertionPointToEnd(pm_->module.get().getBody());
    // One gpu.module per IRBuilder — KunMLIR's pipeline expects exactly
    // one container for all kunir.func ops.
    gpuMod_ = gpu::GPUModuleOp::create(b_, loc, "kungpu_kernels");
    b_.setInsertionPointToStart(&gpuMod_.getBodyRegion().front());
  }

  // ── Type construction ─────────────────────────────────────────────
  Type tsType(const std::string &elemDtype, int64_t lookback) {
    Type elem;
    if (elemDtype == "f32" || elemDtype == "float")
      elem = b_.getF32Type();
    else if (elemDtype == "f64" || elemDtype == "double")
      elem = b_.getF64Type();
    else if (elemDtype == "i1" || elemDtype == "bool")
      elem = b_.getI1Type();
    else
      throw std::runtime_error("IRBuilder.ts_type: unsupported elem dtype '" +
                                 elemDtype + "' (expected f32/f64/i1)");
    uint64_t lb = lookback == 0 ? std::numeric_limits<uint64_t>::max()
                                  : static_cast<uint64_t>(lookback);
    return kunir::TsType::get(pm_->ctx.get(), elem, lb);
  }

  // ── Function ──────────────────────────────────────────────────────
  std::vector<Value>
  beginFunc(const std::string &name,
              std::vector<Type> inputTypes,
              std::vector<std::string> inputNames,
              std::vector<std::string> outputNames,
              int64_t occupancy, int64_t warpsPerCta,
              int64_t smemSize, int64_t vectorSize,
              int64_t unreliableCount,
              std::vector<Type> resultTypes) {
    if (curFunc_)
      throw std::runtime_error(
          "IRBuilder.begin_func: a function is already open — call "
          "end_func() first");
    if (inputTypes.size() != inputNames.size())
      throw std::runtime_error(
          "IRBuilder.begin_func: input_types and input_names must have "
          "the same length");
    if (resultTypes.size() != outputNames.size())
      throw std::runtime_error(
          "IRBuilder.begin_func: result_types and output_names must have "
          "the same length (non-void form: outputs become result types)");
    // `-1` is the whole-time sentinel.  Anything more negative is bogus.
    if (unreliableCount < -1)
      throw std::runtime_error(
          "IRBuilder.begin_func: unreliable_count must be -1 (whole-time) "
          "or non-negative, got "
          + std::to_string(unreliableCount));

    // Restore insertion point to the gpu.module body before starting a
    // new function (in case end_func left us at module scope already).
    b_.setInsertionPointToEnd(&gpuMod_.getBodyRegion().front());

    MLIRContext *ctx = pm_->ctx.get();
    Location loc = b_.getUnknownLoc();

    auto funcType = b_.getFunctionType(inputTypes, resultTypes);
    auto inNamesAttr = b_.getArrayAttr(llvm::map_to_vector(
        inputNames,
        [&](const std::string &s) -> Attribute { return b_.getStringAttr(s); }));
    auto outNamesAttr = b_.getArrayAttr(llvm::map_to_vector(
        outputNames,
        [&](const std::string &s) -> Attribute { return b_.getStringAttr(s); }));
    auto target = kunir::TargetSpecAttr::get(ctx, occupancy, warpsPerCta,
                                                smemSize, vectorSize);

    curFunc_ = kunir::FuncOp::create(b_, loc, name, funcType, inNamesAttr,
                                       outNamesAttr, target,
                                       unreliableCount);
    Block &entry = curFunc_.getBodyBlock();
    b_.setInsertionPointToStart(&entry);

    std::vector<Value> args(entry.args_begin(), entry.args_end());
    return args;
  }

  void endFunc(std::vector<Value> returnValues) {
    if (!curFunc_)
      throw std::runtime_error(
          "IRBuilder.end_func: no open function — call begin_func() first");
    if (!loopStack_.empty())
      throw std::runtime_error(
          "IRBuilder.end_func: " + std::to_string(loopStack_.size()) +
          " for_each_back_window region(s) still open — close them first");

    Location loc = b_.getUnknownLoc();
    kunir::ReturnOp::create(b_, loc, ValueRange(returnValues));

    // Restore insertion point to gpu.module so the next begin_func
    // appends a sibling.
    b_.setInsertionPointToEnd(&gpuMod_.getBodyRegion().front());
    curFunc_ = nullptr;
  }

  // ── Elemwise ops (InferTypeOpInterface — no result type needed) ──
  Value addOp(Value a, Value b) { return makeBin<kunir::AddOp>(a, b); }
  Value subOp(Value a, Value b) { return makeBin<kunir::SubOp>(a, b); }
  Value mulOp(Value a, Value b) { return makeBin<kunir::MulOp>(a, b); }
  Value divOp(Value a, Value b) { return makeBin<kunir::DivOp>(a, b); }
  Value maxOp(Value a, Value b) { return makeBin<kunir::MaxOp>(a, b); }
  Value minOp(Value a, Value b) { return makeBin<kunir::MinOp>(a, b); }

  Value absOp(Value x)  { return makeUn<kunir::AbsOp>(x); }
  Value logOp(Value x)  { return makeUn<kunir::LogOp>(x); }
  Value expOp(Value x)  { return makeUn<kunir::ExpOp>(x); }
  Value sqrtOp(Value x) { return makeUn<kunir::SqrtOp>(x); }
  Value signOp(Value x) { return makeUn<kunir::SignOp>(x); }

  // ── Comparison + logical (binary, return ts<i1, 1>) ─────────────
  Value gtOp(Value a, Value b) { return makeBin<kunir::GreaterOp>(a, b); }
  Value geOp(Value a, Value b) { return makeBin<kunir::GreaterEqualOp>(a, b); }
  Value ltOp(Value a, Value b) { return makeBin<kunir::LessOp>(a, b); }
  Value leOp(Value a, Value b) { return makeBin<kunir::LessEqualOp>(a, b); }
  Value eqOp(Value a, Value b) { return makeBin<kunir::EqualOp>(a, b); }
  Value andOp(Value a, Value b) { return makeBin<kunir::AndOp>(a, b); }
  Value orOp(Value a, Value b)  { return makeBin<kunir::OrOp>(a, b); }
  Value notOp(Value x) { return makeUn<kunir::NotOp>(x); }

  // ── Select (cond, true_value, false_value) ──────────────────────
  Value selectOp(Value cond, Value tv, Value fv) {
    return kunir::SelectOp::create(b_, b_.getUnknownLoc(), cond, tv, fv);
  }

  // ── Scalar constant lifted to ts<T, 1> ─────────────────────────
  Value constantOp(double value, Type tsTy) {
    auto attr = b_.getF64FloatAttr(value);
    return kunir::ConstantOp::create(b_, b_.getUnknownLoc(), tsTy, attr);
  }

  // ── Accumulator / SetAccumulator ───────────────────────────────
  Value accumulatorOp(std::string name, Type tsTy) {
    return kunir::AccumulatorOp::create(b_, b_.getUnknownLoc(), tsTy,
                                            b_.getStringAttr(name));
  }
  void setAccumulatorOp(Value acc, Value mask, Value value) {
    kunir::SetAccumulatorOp::create(b_, b_.getUnknownLoc(), acc, mask, value);
  }

  // ── Windowed buffer materialization ───────────────────────────────
  Value windowedOutputOp(Value x, int64_t length) {
    auto inTs = llvm::cast<kunir::TsType>(x.getType());
    auto resultTy = kunir::TsType::get(pm_->ctx.get(), inTs.getElementType(),
                                          static_cast<uint64_t>(length));
    return kunir::WindowedOutputOp::create(b_, b_.getUnknownLoc(), resultTy, x,
                                                length);
  }

  // ── Back-reference + Fast windowed sum (high-level: ts → ts<T,1>) ─
  Value backRefOp(Value x, int64_t window) {
    auto inTs = llvm::cast<kunir::TsType>(x.getType());
    auto resultTy = kunir::TsType::get(pm_->ctx.get(), inTs.getElementType(), 1);
    return kunir::BackRefOp::create(b_, b_.getUnknownLoc(), resultTy, x, window);
  }
  Value fastWindowedSumOp(Value x, int64_t window) {
    auto inTs = llvm::cast<kunir::TsType>(x.getType());
    auto resultTy = kunir::TsType::get(pm_->ctx.get(), inTs.getElementType(), 1);
    return kunir::FastWindowedSumOp::create(b_, b_.getUnknownLoc(), resultTy, x,
                                                 window);
  }

  // ── For-each-back-window region ───────────────────────────────────
  std::vector<Value>
  beginForEachBackWindow(std::vector<Value> inputs, int64_t window,
                            std::vector<Type> resultTypes) {
    Location loc = b_.getUnknownLoc();
    auto loopOp = kunir::ForEachBackWindowOp::create(b_, loc, resultTypes,
                                                          inputs, window);
    // Populate body block: one block arg per input, each ts<elemType, 1>.
    Block *body = new Block;
    for (Value in : inputs) {
      auto ts = llvm::cast<kunir::TsType>(in.getType());
      body->addArgument(
          kunir::TsType::get(pm_->ctx.get(), ts.getElementType(), 1), loc);
    }
    loopOp.getBody().push_back(body);

    // Descend into the body; remember where to resume.
    ipStack_.push_back(b_.saveInsertionPoint());
    loopStack_.push_back(loopOp);
    b_.setInsertionPointToStart(body);

    return std::vector<Value>(body->args_begin(), body->args_end());
  }

  std::vector<Value>
  endForEachBackWindow(std::vector<Value> yieldValues) {
    if (loopStack_.empty())
      throw std::runtime_error(
          "IRBuilder.end_for_each_back_window: no open loop");
    Location loc = b_.getUnknownLoc();
    kunir::YieldOp::create(b_, loc, ValueRange(yieldValues));

    auto loopOp = loopStack_.back();
    loopStack_.pop_back();
    b_.restoreInsertionPoint(ipStack_.back());
    ipStack_.pop_back();

    return std::vector<Value>(loopOp.getResults().begin(),
                                loopOp.getResults().end());
  }

  // ── Reductions (must be inside a loop body) ───────────────────────
  Value reduceAddOp(Value x) { return makeReduce<kunir::ReduceAddOp>(x); }
  Value reduceMulOp(Value x) { return makeReduce<kunir::ReduceMulOp>(x); }
  Value reduceMaxOp(Value x) { return makeReduce<kunir::ReduceMaxOp>(x); }
  Value reduceMinOp(Value x) { return makeReduce<kunir::ReduceMinOp>(x); }
  Value reduceArgMinOp(Value x) { return makeReduce<kunir::ReduceArgMinOp>(x); }
  Value reduceArgMaxOp(Value x) { return makeReduce<kunir::ReduceArgMaxOp>(x); }
  Value reduceRankOp(Value x, Value cur) {
    // SameOperandsAndResultType — pass x's type as the result type.
    return kunir::ReduceRankOp::create(b_, b_.getUnknownLoc(), x.getType(), x, cur);
  }
  Value windowLoopIndexOp(Type ts_ty) {
    return kunir::WindowLoopIndexOp::create(b_, b_.getUnknownLoc(), ts_ty);
  }

  // ── Finalize ──────────────────────────────────────────────────────
  std::unique_ptr<PyModule> finish() {
    if (curFunc_)
      throw std::runtime_error(
          "IRBuilder.finish: a function is still open — call end_func() "
          "first");
    if (!loopStack_.empty())
      throw std::runtime_error(
          "IRBuilder.finish: for_each_back_window region(s) still open");
    return std::move(pm_);
  }

  std::string toString() const {
    if (!pm_)
      throw std::runtime_error(
          "IRBuilder.to_string: builder has been consumed by finish()");
    return pm_->toString();
  }

private:
  template <typename OpTy> Value makeBin(Value a, Value b) {
    return OpTy::create(b_, b_.getUnknownLoc(), a, b);
  }
  template <typename OpTy> Value makeUn(Value x) {
    return OpTy::create(b_, b_.getUnknownLoc(), x);
  }
  template <typename OpTy> Value makeReduce(Value x) {
    // SameOperandsAndResultType — pass x's type as the result type.
    return OpTy::create(b_, b_.getUnknownLoc(), x.getType(), x);
  }

  std::unique_ptr<PyModule> pm_;
  OpBuilder b_;
  gpu::GPUModuleOp gpuMod_;
  kunir::FuncOp curFunc_;
  std::vector<OpBuilder::InsertPoint> ipStack_;
  std::vector<kunir::ForEachBackWindowOp> loopStack_;
};

std::string valueRepr(Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (v) v.print(os);
  else   os << "<null Value>";
  return s;
}

std::string typeRepr(Type t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (t) t.print(os);
  else   os << "<null Type>";
  return s;
}

} // namespace

void registerIRBuilder(nb::module_ &m) {
  // Opaque MLIR Value / Type wrappers.  No mutating methods — just an
  // identity / repr.  They live as long as the IRBuilder + its resulting
  // PyModule.
  nb::class_<Value>(m, "Value")
      .def("__repr__", [](Value v) { return "<KunMLIR.Value " + valueRepr(v) + ">"; })
      .def("__str__",  [](Value v) { return valueRepr(v); });

  nb::class_<Type>(m, "Type")
      .def("__repr__", [](Type t) { return "<KunMLIR.Type " + typeRepr(t) + ">"; })
      .def("__str__",  [](Type t) { return typeRepr(t); });

  nb::class_<IRBuilder>(m, "IRBuilder",
        "Stateful builder that constructs a kunir module programmatically.\n"
        "Wrap your translator around this — it's the canonical alternative "
        "to round-tripping through MLIR text via parse().")
      .def(nb::init<>())

      // Type
      .def("ts_type", &IRBuilder::tsType,
            nb::arg("elem_dtype"), nb::arg("lookback"),
            "Build a !kunir.ts<elem_dtype, lookback>.  lookback==0 → 'inf'.")

      // Function
      .def("begin_func", &IRBuilder::beginFunc,
            nb::arg("name"),
            nb::arg("input_types"), nb::arg("input_names"),
            nb::arg("output_names"),
            nb::arg("occupancy"), nb::arg("warps_per_cta"),
            nb::arg("smem_size"), nb::arg("vector_size"),
            nb::arg("unreliable_count"),
            nb::arg("result_types"),
            "Open a new kunir.func.  Returns its argument Values.  "
            "`unreliable_count` is the partition-local warmup depth "
            "(max windowed-chain depth from any input to any output).")
      .def("end_func", &IRBuilder::endFunc, nb::arg("return_values"),
            "Close the current kunir.func with a kunir.return.")

      // Elemwise
      .def("add",    &IRBuilder::addOp,    nb::arg("lhs"), nb::arg("rhs"))
      .def("sub",    &IRBuilder::subOp,    nb::arg("lhs"), nb::arg("rhs"))
      .def("mul",    &IRBuilder::mulOp,    nb::arg("lhs"), nb::arg("rhs"))
      .def("div",    &IRBuilder::divOp,    nb::arg("lhs"), nb::arg("rhs"))
      .def("max",    &IRBuilder::maxOp,    nb::arg("lhs"), nb::arg("rhs"))
      .def("min",    &IRBuilder::minOp,    nb::arg("lhs"), nb::arg("rhs"))
      .def("abs",    &IRBuilder::absOp,    nb::arg("x"))
      .def("log",    &IRBuilder::logOp,    nb::arg("x"))
      .def("exp",    &IRBuilder::expOp,    nb::arg("x"))
      .def("sqrt",   &IRBuilder::sqrtOp,   nb::arg("x"))
      .def("sign",   &IRBuilder::signOp,   nb::arg("x"))

      // Comparison + logical (binary). Cmp ops return ts<i1, 1>;
      // and/or expect ts<i1, *> operands and also return ts<i1, 1>.
      .def("gt",     &IRBuilder::gtOp,     nb::arg("lhs"), nb::arg("rhs"))
      .def("ge",     &IRBuilder::geOp,     nb::arg("lhs"), nb::arg("rhs"))
      .def("lt",     &IRBuilder::ltOp,     nb::arg("lhs"), nb::arg("rhs"))
      .def("le",     &IRBuilder::leOp,     nb::arg("lhs"), nb::arg("rhs"))
      .def("eq",     &IRBuilder::eqOp,     nb::arg("lhs"), nb::arg("rhs"))
      .def("and_",   &IRBuilder::andOp,    nb::arg("lhs"), nb::arg("rhs"))
      .def("or_",    &IRBuilder::orOp,     nb::arg("lhs"), nb::arg("rhs"))
      .def("not_",   &IRBuilder::notOp,    nb::arg("x"))

      // Select: cond ? true_value : false_value
      .def("constant", &IRBuilder::constantOp,
            nb::arg("value"), nb::arg("type"),
            "Build a kunir.constant of element-type matching `type` (a "
            "ts<T, 1>).  Pass float('nan') for NaN.")

      .def("accumulator", &IRBuilder::accumulatorOp,
            nb::arg("name"), nb::arg("type"),
            "Build a kunir.accumulator with the given name and ts<T, 1> "
            "result type.  Same-name accumulators CSE to a single slot.")
      .def("set_accumulator", &IRBuilder::setAccumulatorOp,
            nb::arg("acc"), nb::arg("mask"), nb::arg("value"),
            "Conditionally store `value` into `acc` when `mask` is true. "
            "Side-effecting; returns no SSA value.")

      .def("select", &IRBuilder::selectOp,
            nb::arg("cond"), nb::arg("true_value"), nb::arg("false_value"))

      // Windowed materialization
      .def("windowed_output", &IRBuilder::windowedOutputOp,
            nb::arg("x"), nb::arg("length"))

      // Back-reference + Fast windowed sum
      .def("back_ref",          &IRBuilder::backRefOp,
            nb::arg("x"), nb::arg("window"))
      .def("fast_windowed_sum", &IRBuilder::fastWindowedSumOp,
            nb::arg("x"), nb::arg("window"))

      // Loop
      .def("begin_for_each_back_window", &IRBuilder::beginForEachBackWindow,
            nb::arg("inputs"), nb::arg("window"), nb::arg("result_types"),
            "Open a for_each_back_window region.  Returns block args (one "
            "per loop input, type ts<elem,1>).")
      .def("end_for_each_back_window", &IRBuilder::endForEachBackWindow,
            nb::arg("yield_values"),
            "Close the current for_each_back_window with a kunir.yield, "
            "returning the loop op's results.")

      // Reductions
      .def("reduce_add", &IRBuilder::reduceAddOp, nb::arg("x"))
      .def("reduce_mul", &IRBuilder::reduceMulOp, nb::arg("x"))
      .def("reduce_max", &IRBuilder::reduceMaxOp, nb::arg("x"))
      .def("reduce_min", &IRBuilder::reduceMinOp, nb::arg("x"))
      .def("reduce_argmin", &IRBuilder::reduceArgMinOp, nb::arg("x"))
      .def("reduce_argmax", &IRBuilder::reduceArgMaxOp, nb::arg("x"))
      .def("reduce_rank",   &IRBuilder::reduceRankOp,
            nb::arg("x"), nb::arg("current"))
      .def("window_loop_index", &IRBuilder::windowLoopIndexOp,
            nb::arg("ts_ty"))

      // Finalize / debug
      .def("to_string", &IRBuilder::toString,
            "Print the module under construction (for debugging — does "
            "not consume the builder).")
      .def("finish", &IRBuilder::finish,
            "Hand off the module to a KunMLIR.ModuleOp.  Builder is "
            "consumed.");
}

} // namespace kun_mlir_py
