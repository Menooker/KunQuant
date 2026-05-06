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
// Cross-sectional ops
//
// Operates time-step-by-time-step across all stocks. Result has maxLookback=1.
//===----------------------------------------------------------------------===//

def KunIr_CsRankOp : KunIr_Op<"cs_rank", [
    Pure,
    InferTypeOpInterface,
    NativeOpTrait<"KunIrElemwiseTsResultType">
]> {
  let summary = "Cross-sectional rank of a time series";
  let description = [{
    For each time step t, computes the rank of each stock's value among
    all stocks at time t. Output values are in the range [0, 1].
    The input may have any maxLookback; the result always has maxLookback = 1.
  }];
  let arguments = (ins KunIr_AnyTs:$input);
  let results = (outs KunIr_AnyTs:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$input `:` type($input) attr-dict";
}

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
// FuncOp — function with named I/O and hardware target spec
//
// Custom assembly format:
//
//   kunir.func @name(%arg0: type0, ...)
//       inputs {%arg0 = "name0", ...}
//       outputs {"out0", ...}            // non-void: one string per result
//       target {occupancy = V, warps_per_cta = V, smem_size = V}
//       -> (result_type0, ...) {
//     body
//   }
//
//   kunir.func @name(%arg0: type0, %arg1: type1)
//       inputs {%arg0 = "input0"}
//       outputs {%arg1 = "output0"}      // void: %argN = "name" form
//       target {...} {
//     body
//   }
//
// Constraints (void case):   len(inputs) + len(outputs) == len(block_args)
// Constraints (non-void):    len(inputs) == len(block_args),
//                            len(outputs) == num_results
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
    KunIr_TargetSpecAttr:$target_spec
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
                   "::kunir::TargetSpecAttr":$targetSpec)>
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
