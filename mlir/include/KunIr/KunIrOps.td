#ifndef KUNIR_OPS_TD
#define KUNIR_OPS_TD

include "KunIr/KunIrDialect.td"
include "KunIr/KunIrTypes.td"
include "KunIr/KunIrInterfaces.td"
include "KunIr/KunIrAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class KunIr_Op<string mnemonic, list<Trait> traits = []>
    : Op<KunIr_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Element-wise binary ops
//
// Inputs may have different maxLookback values but must share the same element
// type. The result always has maxLookback = 1 (only the current value).
//===----------------------------------------------------------------------===//

class KunIr_BinaryElemwiseOp<string mnemonic, list<Trait> traits = []>
    : KunIr_Op<mnemonic, !listconcat(traits, [
        Pure,
        InferTypeOpInterface,
        NativeOpTrait<"KunIrElemwiseTsResultType">,
        DeclareOpInterfaceMethods<KunIr_BinaryArithInterface>
    ])> {
  let arguments = (ins KunIr_AnyTs:$lhs, KunIr_AnyTs:$rhs);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict";
}

def KunIr_AddOp : KunIr_BinaryElemwiseOp<"add"> {
  let summary = "Element-wise addition of two time series";
}
def KunIr_SubOp : KunIr_BinaryElemwiseOp<"sub"> {
  let summary = "Element-wise subtraction of two time series";
}
def KunIr_MulOp : KunIr_BinaryElemwiseOp<"mul"> {
  let summary = "Element-wise multiplication of two time series";
}
def KunIr_DivOp : KunIr_BinaryElemwiseOp<"div"> {
  let summary = "Element-wise division of two time series";
}
def KunIr_MaxOp : KunIr_BinaryElemwiseOp<"max"> {
  let summary = "Element-wise maximum of two time series";
}
def KunIr_MinOp : KunIr_BinaryElemwiseOp<"min"> {
  let summary = "Element-wise minimum of two time series";
}

//===----------------------------------------------------------------------===//
// Element-wise unary ops
//
// The input may have any maxLookback. The result always has maxLookback = 1.
//===----------------------------------------------------------------------===//

class KunIr_UnaryElemwiseOp<string mnemonic, list<Trait> traits = []>
    : KunIr_Op<mnemonic, !listconcat(traits, [
        Pure,
        InferTypeOpInterface,
        NativeOpTrait<"KunIrElemwiseTsResultType">,
        DeclareOpInterfaceMethods<KunIr_UnaryArithInterface>
    ])> {
  let arguments = (ins KunIr_AnyTs:$input);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$input `:` type($input) attr-dict";
}

def KunIr_AbsOp  : KunIr_UnaryElemwiseOp<"abs">  {
  let summary = "Element-wise absolute value";
}
def KunIr_LogOp  : KunIr_UnaryElemwiseOp<"log">  {
  let summary = "Element-wise natural logarithm";
}
def KunIr_SignOp : KunIr_UnaryElemwiseOp<"sign"> {
  let summary = "Element-wise sign (-1, 0, 1)";
}

//===----------------------------------------------------------------------===//
// Logical ops on i1 ts
//
// And/Or are binary, Not is unary.  Operand element types must be i1 and the
// result is always ts<i1, 1>.  Verifier enforces the i1 constraint; the
// elemwise traits above already give us the right result-type inference
// (result = ts<input.elem, 1> = ts<i1, 1> when input is i1).
//===----------------------------------------------------------------------===//

def KunIr_AndOp : KunIr_BinaryElemwiseOp<"and"> {
  let summary = "Element-wise logical AND on ts<i1>";
}
def KunIr_OrOp  : KunIr_BinaryElemwiseOp<"or"> {
  let summary = "Element-wise logical OR on ts<i1>";
}
def KunIr_NotOp : KunIr_UnaryElemwiseOp<"not"> {
  let summary = "Element-wise logical NOT on ts<i1>";
}

//===----------------------------------------------------------------------===//
// Comparison ops
//
// Operands are two ts values with matching element types; result is always
// !kunir.ts<i1, 1>.  Lowering dispatches arith.cmpf for float operands and
// arith.cmpi for integer operands.
//===----------------------------------------------------------------------===//

class KunIr_BinaryCmpOp<string mnemonic, list<Trait> traits = []>
    : KunIr_Op<mnemonic, !listconcat(traits, [
        Pure,
        InferTypeOpInterface,
        NativeOpTrait<"KunIrCmpTsResultType">,
        DeclareOpInterfaceMethods<KunIr_BinaryArithInterface>
    ])> {
  let arguments = (ins KunIr_AnyTs:$lhs, KunIr_AnyTs:$rhs);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict";
}

def KunIr_GreaterOp      : KunIr_BinaryCmpOp<"gt"> {
  let summary = "Element-wise greater-than (lhs > rhs)";
}
def KunIr_GreaterEqualOp : KunIr_BinaryCmpOp<"ge"> {
  let summary = "Element-wise greater-or-equal (lhs >= rhs)";
}
def KunIr_LessOp         : KunIr_BinaryCmpOp<"lt"> {
  let summary = "Element-wise less-than (lhs < rhs)";
}
def KunIr_LessEqualOp    : KunIr_BinaryCmpOp<"le"> {
  let summary = "Element-wise less-or-equal (lhs <= rhs)";
}
def KunIr_EqualOp        : KunIr_BinaryCmpOp<"eq"> {
  let summary = "Element-wise equality (lhs == rhs)";
}

//===----------------------------------------------------------------------===//
// Select op — cond ? true_value : false_value
//
// cond is ts<i1, *>; true_value / false_value share the same element type T.
// Result is ts<T, 1>.
//===----------------------------------------------------------------------===//

def KunIr_SelectOp : KunIr_Op<"select", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Element-wise select on three ts values";
  let arguments = (ins KunIr_AnyTs:$cond,
                       KunIr_AnyTs:$true_value,
                       KunIr_AnyTs:$false_value);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$cond `,` $true_value `,` $false_value `:` type($cond) `,` "
    "type($true_value) `,` type($false_value) attr-dict";
}

//===----------------------------------------------------------------------===//
// Constant op
//
// A scalar value lifted into a ts<T, 1> at every time step.  The value
// attribute is f64; kunir-to-kungpu converts it to an arith.constant of
// the result's element type (f32, f64, i1, ...).  Pass NaN by storing
// `0x7FF8000000000000` (the quiet-NaN bit pattern) into the f64 attr.
//===----------------------------------------------------------------------===//

def KunIr_ConstantOp : KunIr_Op<"constant", [Pure]> {
  let summary = "Scalar constant broadcast to every time step";
  let arguments = (ins F64Attr:$value);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$value `:` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// Accumulator / SetAccumulator
//
// A scalar "register" that survives across time steps within one kernel
// invocation.  Modeled as a ts<T, 1> handle so existing ts.get / ts.put
// load and store its current value.  The handle storage is owned by a
// single LLVM alloca after kungpu-to-llvm lowering.
//
// `kunir.accumulator` is Pure with a `name` StrAttr — two accumulators
// sharing the same name CSE to a single handle (single backing slot).
//
// `kunir.set_accumulator` is NOT Pure: writing to the slot is a side
// effect and must not be CSE'd or hoisted past dependent reads.  When the
// scalar `mask` is true at the current time step the accumulator is
// overwritten with `value`; otherwise it retains the previous value.
//===----------------------------------------------------------------------===//

def KunIr_AccumulatorOp : KunIr_Op<"accumulator", [Pure]> {
  let summary = "Stateful single-slot scalar register (read via ts.get @0)";
  let arguments = (ins StrAttr:$name);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

def KunIr_SetAccumulatorOp : KunIr_Op<"set_accumulator"> {
  let summary = "Conditionally overwrite an accumulator slot";
  let description = [{
    `acc` must be the result of a `kunir.accumulator`.  When the scalar
    `mask` is true at the current time step, stores `value` into the
    accumulator slot; otherwise the slot is unchanged.  Side-effecting
    (NOT Pure): never CSE / dedup.
  }];
  let arguments = (ins KunIr_AnyTs:$acc,
                       KunIr_AnyTs:$mask,
                       KunIr_AnyTs:$value);
  let hasVerifier = 1;
  let assemblyFormat =
    "$acc `,` $mask `,` $value `:` type($acc) `,` type($mask) `,` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// Cross-sectional ops
//
// kunir has no first-class cross-sectional op.  Cross-sectional kernels
// (currently only cs_rank) are routed by the Python frontend
// (KunQuant.passes.CodegenMLIR._maybe_external_partition) directly to a
// pre-compiled CUmodule bundled with the runtime — they never appear
// in kunir IR or the MLIR pipeline.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WindowedOutput op
//
// Materialises a lookback window: takes any-length ts and produces a ts
// with a specific maxLookback.  The element types must match.
//===----------------------------------------------------------------------===//

def KunIr_WindowedOutputOp : KunIr_Op<"windowed_output", [Pure]> {
  let summary = "Store a segment of a time-series stream with fixed lookback";
  let description = [{
    Takes an input ts with any maxLookback and produces a ts whose
    maxLookback equals the `length` attribute.  This op is used to
    explicitly bound how far back subsequent windowed ops may look.

    Constraints:
      - result element type == input element type
      - result.maxLookback == length
  }];
  let arguments = (ins KunIr_AnyTs:$input, I64Attr:$length);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$input `[` `length` `=` $length `]` `:` type($input) `->` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// Region terminator
//===----------------------------------------------------------------------===//

def KunIr_YieldOp : KunIr_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "Yield zero or more scalar values from a kunir region body";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  // Zero-argument builder for ensureTerminator (empty yield).
  let builders = [OpBuilder<(ins), [{}]>];
}

//===----------------------------------------------------------------------===//
// ForEachBackWindow op
//
// Accepts N input ts values, iterates over their shared back window, and
// produces M ts<elemType, 1> results via kunir.reduce_* ops and kunir.yield.
//
// Body block:
//   - Has exactly N arguments, one per input.
//   - Block arg i has type !kunir.ts<input_i.elemType, 1>  (the current slice).
//   - May contain kunir.reduce_* ops that accumulate the block arguments.
//   - Must terminate with kunir.yield returning M ts<elemType, 1> values,
//     one per result of this op.
//
// Constraints:
//   - window > 0
//   - For each input i: input_i.maxLookback >= window (verified)
//   - body block arg count == inputs count
//   - results count == yield operands count
//   - Each result has maxLookback == 1
//===----------------------------------------------------------------------===//

def KunIr_ForEachBackWindowOp : KunIr_Op<"for_each_back_window", [
    Pure,
    SingleBlockImplicitTerminator<"::kunir::YieldOp">
]> {
  let summary = "Apply reductions over a sliding back window of N time series";
  let description = [{
    For each time step t, iterates over the window [t-window+1 .. t] of every
    input ts and applies the reductions described in the region body.
    Each input ts contributes one block argument of type ts<elemType_i, 1>
    representing the current window slice.
    kunir.reduce_* ops accumulate those values; kunir.yield collects the
    results which become the M outputs of this op, each of type ts<elemType, 1>.

    Every input's maxLookback must be >= window (or inf).
  }];
  let arguments = (ins Variadic<KunIr_AnyTs>:$inputs, I64Attr:$window);
  let results = (outs Variadic<KunIr_AnyTs>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Reduction ops (must appear inside a ForEachBackWindow body)
//
// Each op takes a !kunir.ts<elemType, 1> value (the current window slice
// from the enclosing ForEachBackWindow block argument) and returns the
// accumulated !kunir.ts<elemType, 1> result after the window is traversed.
// SameOperandsAndResultType enforces that input and result types match.
//===----------------------------------------------------------------------===//

class KunIr_ReduceOp<string mnemonic, list<Trait> traits = []>
    : KunIr_Op<mnemonic, !listconcat(traits, [
        Pure,
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<KunIr_ReduceArithInterface>
    ])> {
  let arguments = (ins KunIr_AnyTs:$value);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

def KunIr_ReduceAddOp : KunIr_ReduceOp<"reduce_add"> {
  let summary = "Sum reduction over the back window";
}
def KunIr_ReduceMulOp : KunIr_ReduceOp<"reduce_mul"> {
  let summary = "Product reduction over the back window";
}
def KunIr_ReduceMaxOp : KunIr_ReduceOp<"reduce_max"> {
  let summary = "Maximum reduction over the back window";
}
def KunIr_ReduceMinOp : KunIr_ReduceOp<"reduce_min"> {
  let summary = "Minimum reduction over the back window";
}

//===----------------------------------------------------------------------===//
// BackRef — read input value at t - window
//
// At time step t, returns input[t - window]; semantically NaN when t < window
// (the upstream pipeline writes the high time-domain region first, so cells
// before t = window-1 should be ignored by the user).
//
// The op carries through `kunir-to-kungpu` in two forms:
//   * Source kunir:  result = !kunir.ts<T, 1>
//   * Lowered form:  result = T (scalar) — a single ts.get on the ts handle
//
// Constraints:
//   * input.maxLookback >= window + 1   (or inf — function args)
//   * window > 0
//   * if result is ts: must be ts<input.elemType, 1>
//   * if result is scalar: must equal input.elemType
//===----------------------------------------------------------------------===//

def KunIr_BackRefOp : KunIr_Op<"back_ref", [Pure]> {
  let summary = "Reference the input value `window` time steps in the past";
  let arguments = (ins KunIr_AnyTs:$input, I64Attr:$window);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$input `[` `window` `=` $window `]` `:` type($input) `->` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// FastWindowedSum — running sum over the trailing `window` time steps
//
// Equivalent to KunQuant's CPU `FastWindowedSum<T, stride, window>` (see
// cpp/Kun/Ops.hpp:325).  Per time step:
//
//     v_t = v_{t-1}                              (init 0, Kahan-corrected)
//         - input[t-window]   if input[t-window] not NaN
//         + input[t]          if input[t]        not NaN
//     out_t = (num_nans == 0) ? v_t : NaN
//
// The op is preserved past `kunir-to-kungpu` (with ts→scalar result rewrite)
// because its 4-element per-thread state (v, compAdd, compSub, num_nans) is
// most naturally materialised as LLVM allocas in the kungpu-to-llvm pass.
// See FastWindowedSumPattern there.
//
// Constraints (same as BackRef): input.maxLookback >= window + 1, window > 0.
//===----------------------------------------------------------------------===//

def KunIr_FastWindowedSumOp : KunIr_Op<"fast_windowed_sum", [Pure]> {
  let summary = "Stateful rolling-window sum with Kahan compensation + NaN tracking";
  let arguments = (ins KunIr_AnyTs:$input, I64Attr:$window);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$input `[` `window` `=` $window `]` `:` type($input) `->` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// FuncOp — function with named I/O and hardware target spec
//
// Custom assembly format:
//
//   kunir.func @name(%arg0: type0, ...)
//       inputs {%arg0 = "name0", ...}
//       outputs {"out0", ...}            // non-void: one string per result
//       target {occupancy = V, warps_per_cta = V, smem_size = V}
//       unreliable_count = N
//       -> (result_type0, ...) {
//     body
//   }
//
//   kunir.func @name(%arg0: type0, %arg1: type1)
//       inputs {%arg0 = "input0"}
//       outputs {%arg1 = "output0"}      // void: %argN = "name" form
//       target {...}
//       unreliable_count = N {
//     body
//   }
//
// Constraints (void case):   len(inputs) + len(outputs) == len(block_args)
// Constraints (non-void):    len(inputs) == len(block_args),
//                            len(outputs) == num_results
//
// `unreliable_count` is the per-partition warmup depth: max over all
// outputs of (sum of windowed op windows from any input to that output).
// Required because callers (runtime) need it to size the time-axis
// chunk grid and to thread the kernel's warmup arg.
//===----------------------------------------------------------------------===//

def KunIr_FuncOp : KunIr_Op<"func", [
    IsolatedFromAbove,
    Symbol,
    SingleBlockImplicitTerminator<"::kunir::ReturnOp">
]> {
  let summary = "KunQuant function with named I/O and hardware target spec";
  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$function_type,
    ArrayAttr:$input_names,
    ArrayAttr:$output_names,
    KunIr_TargetSpecAttr:$target_spec,
    I64Attr:$unreliable_count
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                   "::mlir::FunctionType":$type,
                   "::mlir::ArrayAttr":$inputNames,
                   "::mlir::ArrayAttr":$outputNames,
                   "::kunir::TargetSpecAttr":$targetSpec,
                   "int64_t":$unreliableCount)>
  ];
  let extraClassDeclaration = [{
    // getFunctionType() is generated by tablegen and returns mlir::Type.
    // This typed helper casts it for callers that need mlir::FunctionType.
    ::mlir::FunctionType getFunctionTypeTyped() {
      return llvm::cast<::mlir::FunctionType>(getFunctionType());
    }
    ::mlir::Block &getBodyBlock() { return getBody().front(); }
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp — terminator for kunir.func body
//===----------------------------------------------------------------------===//

def KunIr_ReturnOp : KunIr_Op<"return", [
    Pure,
    Terminator,
    ReturnLike,
    HasParent<"::kunir::FuncOp">
]> {
  let summary = "Return from a kunir.func body";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let hasVerifier = 1;
  // Zero-arg builder used by SingleBlockImplicitTerminator (ensureTerminator).
  let builders = [OpBuilder<(ins), [{}]>];
}

#endif // KUNIR_OPS_TD
