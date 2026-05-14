"""Translate a (post-optimize) KunQuant Function into a KunMLIR module
holding a single kunir.func inside a gpu.module.

This is the GPU-side counterpart to passes.CodegenCpp.codegen_cpp; it
runs after the same Driver.optimize() pipeline the CPU path uses, then
walks the lowered IR and emits kunir ops via the KunMLIR.IRBuilder
pybind class.

Scope (v0): only the ops kunir currently supports.
  - Elemwise binary: Add, Sub, Mul, Div, Max, Min
  - Elemwise unary:  Abs, Log, Sign
  - Cross-sectional: Rank
  - Windowed:        WindowedTempOutput, ForeachBackWindow + IterValue,
                      ReduceAdd / ReduceMul / ReduceMax / ReduceMin
  - Boundaries:      Input, Output

Anything else raises NotImplementedError with the offending op printed.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # KunMLIR is a compiled extension built alongside the MLIR support,
    # only imported here for type checking — no runtime dependency added
    # to the codegen path itself.
    from KunQuant.jit import KunMLIR

from KunQuant.Op import (
    OpBase, Input, Output, ForeachBackWindow, IterValue, WindowedTempOutput,
    ReductionOp, SimpleCrossSectionalOp, ConstantOp,
)
from KunQuant.ops.ElewiseOp import (
    Add, Sub, Mul, Div, Max, Min, Abs, Log, Sign,
    GreaterThan, GreaterEqual, LessThan, LessEqual, Equals,
    And, Or, Not, Select,
)
from KunQuant.ops.ReduceOp import (
    ReduceAdd, ReduceMul, ReduceMax, ReduceMin,
)
from KunQuant.ops.MiscOp import (
    BackRef, FastWindowedSum,
    Accumulator, SetAccumulator, ReturnFirstValue,
)
from KunQuant.Stage import Function


# ── Op-class → IRBuilder method dispatch ────────────────────────────

_BINARY = {
    Add: "add", Sub: "sub", Mul: "mul", Div: "div",
    Max: "max", Min: "min",
    GreaterThan:  "gt", GreaterEqual: "ge",
    LessThan:     "lt", LessEqual:    "le",
    Equals:       "eq",
    And:          "and_", Or:         "or_",
}
_UNARY = {
    Abs: "abs", Log: "log", Sign: "sign",
    Not: "not_",
    # NOTE: `Rank` is intentionally absent.  Cross-sectional rank
    # partitions are routed to a pre-compiled CUmodule by
    # `_maybe_external_partition` below; they never become kunir ops.
}
_REDUCE = {
    ReduceAdd: "reduce_add", ReduceMul: "reduce_mul",
    ReduceMax: "reduce_max", ReduceMin: "reduce_min",
}


# ── Target spec carrier ─────────────────────────────────────────────

class TargetSpec:
    """GPU launch parameters mirrored from kunir.target_spec."""
    def __init__(self, *, occupancy: int = 1, warps_per_cta: int = 4,
                 smem_size: int = 49152, vector_size: int = 1):
        self.occupancy     = occupancy
        self.warps_per_cta = warps_per_cta
        self.smem_size     = smem_size
        self.vector_size   = vector_size


# ── Helpers ─────────────────────────────────────────────────────────

def _kunir_symbol(name: str) -> str:
    """Coerce a partition name into a valid kunir / PTX symbol.

    The partitioner derives a partition's name from the names of its
    Output ops; when a partition is "intermediate-only" (every output
    is consumed by a downstream partition, none is a user-facing
    Output), those names come from `OpBase.hash_hex` which starts with
    a digit half the time.  Digits are fine for buffer-table keys
    (CPU runtime indexes by name) but ptxas rejects them as
    `.entry` symbols.

    Prefix any such name with a single `_` so the kunir.func symbol
    is always a valid identifier, while leaving `input_names` /
    `output_names` (the public buffer-table keys) untouched.
    """
    if name and name[0].isdigit():
        return "_" + name
    return name

def _index_loop_members(f: Function) -> Tuple[
        Dict[ForeachBackWindow, List[OpBase]],
        Dict[ForeachBackWindow, List[ReductionOp]]]:
    """For each ForeachBackWindow in `f`, collect the body ops (those
    whose `_parent_loop` is the loop) and the reduction ops (whose
    `get_loop()` is the loop).  Both lists keep f.ops topo order."""
    body_ops: Dict[ForeachBackWindow, List[OpBase]] = {}
    reductions: Dict[ForeachBackWindow, List[ReductionOp]] = {}
    for op in f.ops:
        if isinstance(op, ReductionOp):
            loop = op.get_loop()
            reductions.setdefault(loop, []).append(op)
        elif op.get_parent() is not None:
            body_ops.setdefault(op.get_parent(), []).append(op)
    return body_ops, reductions


def _emit_simple(op: OpBase,
                  ir: KunMLIR.IRBuilder,
                  val_map: Dict[OpBase, KunMLIR.Value],
                  ts_1: KunMLIR.Type) -> KunMLIR.Value:
    """Emit a non-control-flow op via IRBuilder dispatch.  `ts_1` is the
    kunir ts type with maxLookback=1, used by ops whose result has no
    input to infer the element type from (currently only ConstantOp)."""
    cls = type(op)
    if cls in _BINARY:
        getattr(ir, _BINARY[cls])
        return getattr(ir, _BINARY[cls])(val_map[op.inputs[0]],
                                           val_map[op.inputs[1]])
    if cls in _UNARY:
        return getattr(ir, _UNARY[cls])(val_map[op.inputs[0]])
    if isinstance(op, WindowedTempOutput):
        return ir.windowed_output(val_map[op.inputs[0]],
                                    int(op.attrs["window"]))
    if isinstance(op, BackRef):
        return ir.back_ref(val_map[op.inputs[0]], int(op.attrs["window"]))
    if isinstance(op, FastWindowedSum):
        return ir.fast_windowed_sum(val_map[op.inputs[0]],
                                      int(op.attrs["window"]))
    if isinstance(op, Select):
        return ir.select(val_map[op.inputs[0]],
                          val_map[op.inputs[1]],
                          val_map[op.inputs[2]])
    if isinstance(op, ConstantOp):
        v = op.attrs["value"]
        fv = float("nan") if v == "nan" else float(v)
        return ir.constant(fv, ts_1)
    if isinstance(op, Accumulator):
        # The Python op's `inputs[0]` is a keep-alive in the graph IR;
        # it does NOT feed the slot.  Only the `name` attr matters at
        # the MLIR level — same-name accumulators CSE to one slot.
        return ir.accumulator(op.attrs["name"], ts_1)
    if isinstance(op, SetAccumulator):
        # Side-effect: returns no SSA value.  ReturnFirstValue is what
        # keeps this op alive in the Python graph (see MiscOp.py).
        ir.set_accumulator(val_map[op.inputs[0]],
                            val_map[op.inputs[1]],
                            val_map[op.inputs[2]])
        return None
    if isinstance(op, ReturnFirstValue):
        # In the Python graph IR, ReturnFirstValue's only job is to keep
        # side-effecting siblings (SetAccumulator etc.) reachable from a
        # graph output so the GC does not drop them.  In SSA-MLIR the
        # side-effect ops are preserved by their own MemWrite semantics;
        # ReturnFirstValue carries no new MLIR-level meaning, so we just
        # forward the first input's Value.  Other inputs were already
        # emitted in topo order before we got here.
        return val_map[op.inputs[0]]
    raise NotImplementedError(
        f"CodegenMLIR: op type {cls.__name__} is not supported by the "
        f"GPU backend yet (op = {op})")


def _emit_reduction(op: ReductionOp,
                     ir: KunMLIR.IRBuilder,
                     val_map: Dict[OpBase, KunMLIR.Value]) -> KunMLIR.Value:
    cls = type(op)
    if cls not in _REDUCE:
        raise NotImplementedError(
            f"CodegenMLIR: reduction {cls.__name__} not supported yet "
            f"(op = {op})")
    if len(op.inputs) != 1:
        raise NotImplementedError(
            f"CodegenMLIR: reductions with init_val are not supported "
            f"yet (op = {op})")
    return getattr(ir, _REDUCE[cls])(val_map[op.inputs[0]])


# ── Main entry point ────────────────────────────────────────────────

def _maybe_external_partition(f: Function, dtype: str) -> Optional[dict]:
    """If `f` is a partition the GPU runtime handles as a pre-compiled
    external kernel (bundled PTX loaded as a separate CUmodule), return
    a descriptor dict that KunMLIR.compile() should append to the
    executable's kernel list.  Otherwise return None.

    The descriptor matches what KunMLIR.compile's `external_kernels=`
    parameter expects:
        {"name": <str>, "kind": <str>,
         "inputs": [<str>...], "outputs": [<str>...]}

    Detection mirrors CodegenCpp's "simple cross-sectional fast path"
    (CodegenCpp.codegen_cpp's `len(f.ops) == 3` check): a partition
    whose only compute op is a `SimpleCrossSectionalOp` (Rank, Scale,
    …).  The partitioner places every CrossSectionalOp into its own
    partition without other compute, so this shape is what we get.

    The `kind` string is `cs_<lowercased class name>_f{32,64}`, e.g.
    `cs_rank_f32`, `cs_scale_f64`.  The C++ binding maps it to a
    `KernelKind` enum; unknown kinds raise there with a clear error,
    so adding a new SimpleCrossSectionalOp on the Python side does
    not silently succeed without a matching bundled PTX kernel.
    """
    compute = [op for op in f.ops
                if not isinstance(op, (Input, Output))]
    if len(compute) != 1 or not isinstance(compute[0], SimpleCrossSectionalOp):
        return None
    inputs  = [op for op in f.ops if isinstance(op, Input)]
    outputs = [op for op in f.ops if isinstance(op, Output)]
    if len(inputs) != 1 or len(outputs) != 1:
        return None  # surprising shape, let the regular path emit an error
    if dtype not in ("f32", "f64"):
        return None
    op_kind = compute[0].__class__.__name__.lower()
    return {
        "name":    f.name or f"cs_{op_kind}",
        "kind":    f"cs_{op_kind}_{dtype}",
        "inputs":  [op.attrs["name"] for op in inputs],
        "outputs": [op.attrs["name"] for op in outputs],
    }


def translate_function(f: Function, target: TargetSpec,
                        ir: KunMLIR.IRBuilder,
                        dtype: str = "f32",
                        unreliable_count: int = 0) -> Optional[dict]:
    """Emit `f` as a single kunir.func into the open `ir` (KunMLIR.IRBuilder).

    If `f` is an externally-dispatched partition (e.g. a single cs_rank
    op handled by the bundled cs_rank.ptx CUmodule), emit nothing into
    the IRBuilder and return its descriptor dict so the caller can pass
    it to KunMLIR.compile()'s `external_kernels=` list.  Otherwise
    return `None` after emitting a kunir.func.

    `unreliable_count` is the partition-local warmup depth — the caller
    (`KunQuant.jit.cuda`) computes it via `infer_window(f)` on this
    post-partition Function and feeds it in.
    """
    ext = _maybe_external_partition(f, dtype)
    if ext is not None:
        return ext

    # 1.  Boundary ops in topo order — the kunir.func's I/O.
    inputs:  List[Input]  = [op for op in f.ops if isinstance(op, Input)]
    outputs: List[Output] = [op for op in f.ops if isinstance(op, Output)]
    if not inputs:
        raise ValueError("CodegenMLIR: function has no Input ops")
    if not outputs:
        raise ValueError("CodegenMLIR: function has no Output ops")

    in_names  = [op.attrs["name"] for op in inputs]
    out_names = [op.attrs["name"] for op in outputs]

    # 2.  Pre-index loop members so we can emit each loop's body +
    #     reductions contiguously (regardless of topo interleaving with
    #     other loops).
    body_ops_by_loop, reductions_by_loop = _index_loop_members(f)

    # 3.  Open the kunir.func.  All inputs are ts<dtype, inf>; all
    #     graph results are ts<dtype, 1>.
    ts_inf = ir.ts_type(dtype, 0)
    ts_1   = ir.ts_type(dtype, 1)

    func_args = ir.begin_func(
        name=_kunir_symbol(f.name or "kernel"),
        input_types=[ts_inf] * len(inputs),
        input_names=in_names,
        output_names=out_names,
        occupancy=target.occupancy, warps_per_cta=target.warps_per_cta,
        smem_size=target.smem_size, vector_size=target.vector_size,
        unreliable_count=unreliable_count,
        result_types=[ts_1] * len(outputs),
    )

    val_map: Dict[OpBase, KunMLIR.Value] = {}
    emitted = set()
    for inp, val in zip(inputs, func_args):
        val_map[inp] = val
        emitted.add(inp)

    # 4.  Walk f.ops in topo order, emitting one op (or one whole loop)
    #     at a time.
    for op in f.ops:
        if op in emitted:
            continue
        if isinstance(op, Input):
            continue                      # already mapped from func_args
        if isinstance(op, Output):
            continue                      # handled at the end via Return
        if isinstance(op, ForeachBackWindow):
            _emit_loop(op, ir, val_map, ts_1,
                        body_ops_by_loop.get(op, []),
                        reductions_by_loop.get(op, []),
                        emitted)
            continue
        if isinstance(op, ReductionOp) or op.get_parent() is not None:
            # Should have been emitted as part of its enclosing loop;
            # if we hit it here, the loop never appeared first — that's
            # a bug in topo sort or in this translator's iteration.
            raise RuntimeError(
                f"CodegenMLIR: reduction/body op visited before its "
                f"enclosing loop ({op})")
        val_map[op] = _emit_simple(op, ir, val_map, ts_1)
        emitted.add(op)

    # 5.  Close the function with Outputs in declared order.
    return_values = [val_map[o.inputs[0]] for o in outputs]
    ir.end_func(return_values)
    return None


def _emit_loop(loop: ForeachBackWindow,
                ir: KunMLIR.IRBuilder,
                val_map: Dict[OpBase, KunMLIR.Value],
                ts_1: KunMLIR.Type,
                body_ops: List[OpBase],
                reductions: List[ReductionOp],
                emitted: set) -> None:
    loop_input_vals = [val_map[i] for i in loop.inputs]
    n_results = len(reductions)
    if n_results == 0:
        raise NotImplementedError(
            f"CodegenMLIR: ForeachBackWindow with no reductions "
            f"(loop = {loop})")

    block_args = ir.begin_for_each_back_window(
        inputs=loop_input_vals,
        window=int(loop.attrs["window"]),
        result_types=[ts_1] * n_results,
    )
    # Block args mirror loop.inputs positionally.  Map the source-op
    # → block-arg so IterValue can be resolved to the right one.
    block_arg_by_src = {src: block_args[i]
                          for i, src in enumerate(loop.inputs)}

    # Body ops: IterValue → block arg; everything else uses _emit_simple.
    for body_op in body_ops:
        if isinstance(body_op, IterValue):
            val_map[body_op] = block_arg_by_src[body_op.inputs[1]]
        else:
            val_map[body_op] = _emit_simple(body_op, ir, val_map, ts_1)
        emitted.add(body_op)

    # Reductions accumulate yield values, in topo order.
    yield_vals = [_emit_reduction(r, ir, val_map) for r in reductions]
    loop_results = ir.end_for_each_back_window(yield_vals)
    for r, lr in zip(reductions, loop_results):
        val_map[r] = lr
        emitted.add(r)

    emitted.add(loop)
