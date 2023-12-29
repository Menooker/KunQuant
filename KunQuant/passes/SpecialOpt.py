from KunQuant.Op import OpBase, ForeachBackWindow, WindowedTempOutput, Output
from KunQuant.ops import ReduceAdd, FastWindowedSum
from KunQuant.Stage import Function
from typing import List, Dict, Tuple

def _is_ok_for_reduce_opt(op: OpBase) -> Tuple[OpBase, int]:
    if not isinstance(op, ReduceAdd):
        return None
    if op.get_parent() is not None:
        return None
    loop = op.inputs[0]
    if not isinstance(loop, ForeachBackWindow):
        return None
    window = loop.attrs["window"]
    window_data = loop.inputs[0]
    return window_data, window

def special_impl(ops: List[OpBase]) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    for op in ops:
        op.replace_inputs(replace_map)
        # if it is reduce-sum in non-loop context
        result = _is_ok_for_reduce_opt(op)
        if result is None:
            out.append(op)
            continue
        opt_in, window = result
        newop = FastWindowedSum(opt_in, window)
        out.append(newop)
        changed = True    
        replace_map[op] = newop
    if changed:
        return out
    return None

def special_optimize(f: Function):
    '''
    Optimize:
    y = ...
    x = WindowedTempOutput(y) or Output(y)
    x2 = ForeachBackWindow(x)
    x3 = ReduceAdd(x2)
    =======================
    Into
    x3 = FastWindowedSum(y)
    '''
    newops = special_impl(f.ops)
    if newops is not None:
        f.set_ops(newops)