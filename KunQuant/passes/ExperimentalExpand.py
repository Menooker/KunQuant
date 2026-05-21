"""Experimental stateful-op expansion pass (currently GPU-only).

Gated on ``options["experimental_expand"]`` — when False (default), the
pass returns immediately so the CPU pipeline is untouched.  Runs after
the first ``decompose`` so user-facing composite ops (e.g.
``WindowedLinearRegressionSlope``) have already been broken into
``WindowedLinearRegression`` + per-extractor ``Impl`` ops.

Replaces ops that the kunir codegen doesn't lower directly with
``Accumulator + Select + SetAccumulator`` chains (and FBW reductions) it
does support:

* ``ExpMovingAvg(v, span)`` → an ``Accumulator(init_val="nan")`` carrying
  the running EMA.  The NaN init doubles as the "not yet seeded" sentinel
  — first non-NaN ``v`` is stored verbatim, subsequent non-NaN ``v`` uses
  the pandas ``ewm(adjust=False, ignore_na=True)`` update.  An ``__init``
  Input is not supported yet — the pass raises on encounter.

* ``WindowedLinearRegression(v, window)`` → ``FastWindowedSum`` for the
  running sum / sum-of-squares, plus a ``ForeachBackWindow`` +
  ``WindowLoopIndex`` + ``ReduceAdd`` for the position-weighted sum_xy.
  Intermediate ops are stashed in ``state[lin_op] : List[OpBase]`` so each
  consumer Impl (``Slope``, ``RSqaure``, ``Resi``) can pick the entries it
  needs and emit its final formula.

* ``SetInfOrNanToValue(a, value)`` → ``Select(isnan(a - a), value, a)``
  (mirrors the C++ implementation; ``a - a`` is NaN for both NaN and ±Inf).

* ``ReduceDecayLinear(v, window)`` → ``ReduceAdd(v * weight)`` where
  ``weight = (WindowLoopIndex + 1) / (window * (window + 1) / 2)``.
"""

from typing import Callable, Dict, List, Optional, Tuple, Type

from KunQuant.Op import (
    OpBase, Builder, ConstantOp, ForeachBackWindow, IterValue,
    WindowedTempOutput, WindowLoopIndex,
)
from KunQuant.ops.ElewiseOp import Select, Equals, Not, SetInfOrNanToValue
from KunQuant.ops.ReduceOp import ReduceAdd, ReduceDecayLinear
from KunQuant.ops.MiscOp import (
    FastWindowedSum, Accumulator, SetAccumulator,
    ExpMovingAvg, WindowedLinearRegression,
    WindowedLinearRegressionSlopeImpl,
    WindowedLinearRegressionRSqaureImpl,
    WindowedLinearRegressionResiImpl,
)
from KunQuant.Stage import Function
from .Util import kun_pass


# ── EMA expansion ───────────────────────────────────────────────────

def _expand_ema(op: ExpMovingAvg) -> OpBase:
    """Build the Accumulator-based chain inside the current Builder.

    The slot is initialised to NaN, which serves as the "not yet seeded"
    sentinel: a NaN ``prev`` means we still need to seed with the first
    non-NaN ``x``.  The SetAccumulator's mask is ``notnan_x``, so NaN
    inputs leave the slot unchanged (pandas ignore_na=True).
    """
    if len(op.inputs) >= 2:
        raise RuntimeError(
            "experimental_expand: ExpMovingAvg with an `__init` Input is "
            "not supported yet on the GPU backend")
    span  = op.attrs["window"]
    alpha = 2.0 / (span + 1)
    x     = op.inputs[0]

    # `is_whole_time_required=True` propagates the kernel's
    # unreliable_count to the sentinel so the runtime collapses to a
    # single chunk — EMA's per-stock state can't survive a chunk
    # boundary reset.
    prev    = Accumulator(x, f"ema_{span}", init_val="nan",
                          is_whole_time_required=True)
    notnan_x  = Equals(x, x)
    prev_nan  = Not(Equals(prev, prev))

    formula = x * alpha + prev * (1.0 - alpha)
    #   prev is NaN (still warmup):
    #     - x non-NaN → seed with x
    #     - x NaN     → keep NaN (Select returns `x`)
    #   prev is set:
    #     - x non-NaN → standard formula
    #     - x NaN     → carry prev unchanged
    new_ema = Select(prev_nan, x, Select(notnan_x, formula, prev))
    # mask = notnan_x: on NaN x we don't touch the slot (preserves
    # both the NaN-sentinel and the carried prev).  SetAccumulator
    # returns the slot's new value for this step (mask ? value : prev),
    # which matches `new_ema` here — use it directly as the EMA result.
    return SetAccumulator(prev, notnan_x, new_ema)


# ── WindowedLinearRegression intermediate state ────────────────────

# Field names in the per-op `state` list returned by `_expand_linreg`.
# Consumers index by these constants for clarity.
_LR_SUM_Y    = 0   # FastWindowedSum(v,    window)
_LR_SUM_YY   = 1   # FastWindowedSum(v*v,  window)
_LR_SUM_XY   = 2   # Σ_{i=0..window-1} i * v[t-window+1+i]
_LR_SLOPE    = 3
_LR_INTERCEPT = 4
_LR_V        = 5   # original v (for the Resi consumer)


def _expand_linreg(op: WindowedLinearRegression) -> List[OpBase]:
    """Emit running sums + the closed-form slope/intercept for v
    regressed on the integer position x = 0..window-1 within the window.

    The x positions are treated as constants (i.e. no NaN-aware
    re-indexing) — for an input with NaN entries the running sums become
    NaN via the FastWindowedSum / FBW NaN propagation and consumers
    return NaN through.
    """
    window = op.attrs["window"]
    v      = op.inputs[0]

    # sum_y = rolling sum of v over the window; NaN until window full.
    # FastWindowedSum requires a WindowedDataSourceOp input sized window+1.
    sum_y  = FastWindowedSum(WindowedTempOutput(v, window + 1), window)
    # sum_yy = rolling sum of v² — same pattern over a v*v intermediate.
    sum_yy = FastWindowedSum(WindowedTempOutput(v * v, window + 1), window)
    # sum_xy = Σ idx * v where idx is the window position (0=oldest,
    # window-1=newest).  Express via FBW + WindowLoopIndex + Mul +
    # ReduceAdd; OOB reads (warmup) return NaN, so sum_xy is NaN until
    # the window fills.
    wtemp = WindowedTempOutput(v, window)
    with ForeachBackWindow(wtemp, window) as each:
        idx     = WindowLoopIndex(each)
        val     = IterValue(each, wtemp)
        contrib = idx * val
    sum_xy = ReduceAdd(contrib)

    # Compile-time constants for x:
    #   sum_x  = Σ i  for i in [0, window)       = window*(window-1)/2
    #   sum_xx = Σ i² for i in [0, window)       = window*(window-1)*(2*window-1)/6
    # ⇒ denom = window*sum_xx - sum_x² = window²(window-1)(window+1)/12
    n      = float(window)
    sum_x  = n * (n - 1) / 2.0
    denom  = (n * n) * (n - 1.0) * (n + 1.0) / 12.0   # constant; assume window > 1
    slope     = (sum_xy * n - sum_y * sum_x) / denom
    intercept = (sum_y - slope * sum_x) / n

    state = [None] * 6
    state[_LR_SUM_Y]     = sum_y
    state[_LR_SUM_YY]    = sum_yy
    state[_LR_SUM_XY]    = sum_xy
    state[_LR_SLOPE]     = slope
    state[_LR_INTERCEPT] = intercept
    state[_LR_V]         = v
    return state


# ── Consumer formulas (one per Impl op) ─────────────────────────────

def _expand_lr_slope(op: OpBase,
                     state: Dict[OpBase, List[OpBase]]) -> OpBase:
    return state[op.inputs[0]][_LR_SLOPE]


def _expand_lr_rsquare(op: OpBase,
                       state: Dict[OpBase, List[OpBase]]) -> OpBase:
    lin_op = op.inputs[0]
    lr_state = state[lin_op]
    # SS_reg = slope² * (window*sum_xx - sum_x²) / window = slope² * denom / window
    # SS_tot = sum_yy - sum_y²/window
    # R²     = SS_reg / SS_tot
    n     = float(lin_op.attrs["window"])
    denom = (n * n) * (n - 1.0) * (n + 1.0) / 12.0
    slope = lr_state[_LR_SLOPE]
    ss_reg = (slope * slope) * (denom / n)
    ss_tot = (
        lr_state[_LR_SUM_YY] -
        (lr_state[_LR_SUM_Y] * lr_state[_LR_SUM_Y]) / n)
    return ss_reg / ss_tot


def _expand_lr_resi(op: OpBase,
                    state: Dict[OpBase, List[OpBase]]) -> OpBase:
    lin_op = op.inputs[0]
    lr_state = state[lin_op]
    # residual at the newest window position (x = window-1):
    #   v_t - (slope * (window-1) + intercept)
    pred = (
        lr_state[_LR_SLOPE] * float(lin_op.attrs["window"] - 1) +
        lr_state[_LR_INTERCEPT])
    return lr_state[_LR_V] - pred


# ── SetInfOrNanToValue expansion ────────────────────────────────────

def _expand_set_inf_or_nan(op: SetInfOrNanToValue) -> OpBase:
    # Mirrors the C++ implementation in cpp/Kun/Ops.hpp:
    # `mask = isnan(a - a); return select(mask, v, a)`.
    # `a - a` is 0 for finite `a` and NaN for NaN/±Inf (Inf-Inf == NaN),
    # so isnan-of-diff catches both NaN and Inf in one shot.
    a = op.inputs[0]
    diff = a - a
    mask = Not(Equals(diff, diff))
    return Select(mask, ConstantOp(op.attrs["value"]), a)


# ── DecayLinear reduction expansion ─────────────────────────────────

def _expand_decay_linear(op: ReduceDecayLinear) -> OpBase:
    if len(op.inputs) != 1:
        raise RuntimeError(
            f"experimental_expand: ReduceDecayLinear expects one input "
            f"(op = {op})")
    window = int(op.attrs["window"])
    denom = (1.0 + window) * window / 2.0
    loop = op.get_loop()
    with loop:
        idx = WindowLoopIndex(loop)
        weight = (idx + 1.0) * (1.0 / denom)
        contrib = op.inputs[0] * weight
    return ReduceAdd(contrib)


# ── Dispatch table helpers ──────────────────────────────────────────

ExpandFunc = Callable[[OpBase, Dict[OpBase, List[OpBase]]], OpBase]
ExpandRule = Tuple[Type[OpBase], ExpandFunc]


_EXPAND_RULES: List[ExpandRule] = [
    (ExpMovingAvg, lambda op, state: _expand_ema(op)),
    (WindowedLinearRegressionSlopeImpl, _expand_lr_slope),
    (WindowedLinearRegressionRSqaureImpl, _expand_lr_rsquare),
    (WindowedLinearRegressionResiImpl, _expand_lr_resi),
    (SetInfOrNanToValue, lambda op, state: _expand_set_inf_or_nan(op)),
    (ReduceDecayLinear, lambda op, state: _expand_decay_linear(op)),
]


def _find_expand_rule(op: OpBase) -> Optional[ExpandFunc]:
    for op_type, expand in _EXPAND_RULES:
        if isinstance(op, op_type):
            return expand
    return None


# ── Pass driver ─────────────────────────────────────────────────────

def _experimental_expand_impl(
    ops: List[OpBase], options: dict,
) -> List[OpBase]:
    # state[lin_op] = list of intermediate ops; consumers pick by index.
    state: Dict[OpBase, List[OpBase]] = {}
    replace_map: Dict[OpBase, OpBase] = {}
    out: List[OpBase] = []
    changed = False

    for op in ops:
        op.replace_inputs(replace_map)

        if isinstance(op, WindowedLinearRegression):
            b = Builder(op.get_parent())
            with b:
                lin_state = _expand_linreg(op)
            out.extend(b.ops)
            state[op] = lin_state
            # The LinearRegression op produces a "state handle" Value
            # consumed only by its Impl ops, which we lower below via
            # `state[]` lookup — so we don't keep `op` in `out` and we
            # don't enter it in `replace_map`.  Consumers find the same
            # original Python object by identity through `op.inputs[0]`.
            changed = True
            continue

        expand = _find_expand_rule(op)
        if expand is not None:
            b = Builder(op.get_parent())
            with b:
                new_val = expand(op, state)
            out.extend(b.ops)
            replace_map[op] = new_val
            changed = True
            continue

        out.append(op)

    return out if changed else None


@kun_pass
def experimental_expand(f: Function, options: dict = {}):
    if not options.get("experimental_expand", False):
        return
    newops = _experimental_expand_impl(f.ops, options)
    if newops is not None:
        f.set_ops(newops)
