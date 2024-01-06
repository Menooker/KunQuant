from KunQuant.passes.Util import kun_pass
from KunQuant.Op import OpBase, ForeachBackWindow, WindowedTempOutput, Output, IterValue
from KunQuant.ops import ReduceAdd, FastWindowedSum, SubConst, MulConst
from KunQuant.Stage import Function
from typing import List, Dict, Tuple

def _is_ok_for_reduce_opt(op: OpBase) -> Tuple[OpBase, int]:
    if not isinstance(op, ReduceAdd):
        return None
    if op.get_parent() is not None:
        return None
    itr = op.inputs[0]
    if not isinstance(itr, IterValue):
        return None
    loop = itr.inputs[0]
    window_data = itr.inputs[1]
    if not isinstance(loop, ForeachBackWindow):
        return None
    window = loop.attrs["window"]
    return window_data, window

def special_impl(ops: List[OpBase]) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    for op in ops:
        op.replace_inputs(replace_map)
        if isinstance(op, MulConst) and op.attrs["value"] == -1:
            newop = SubConst(op.inputs[0], 0, True)
            newop.set_parent(op.get_parent())
            out.append(newop)
            changed = True    
            replace_map[op] = newop
            continue
        # if it is reduce-sum in non-loop context
        result = _is_ok_for_reduce_opt(op)
        if result is None:
            out.append(op)
            continue
        opt_in, window = result
        newop = FastWindowedSum(opt_in, window)
        # FastWindowedSum needs an additional window size than original Sum
        if isinstance(opt_in, WindowedTempOutput):
            assert(opt_in.attrs["window"] >= window)
            if opt_in.attrs["window"] < window + 1:
                opt_in.attrs["window"] = window + 1
        out.append(newop)
        changed = True    
        replace_map[op] = newop
    if changed:
        return out
    return None

@kun_pass
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

    And Mul(-1) => Sub(0, X)
    '''
    newops = special_impl(f.ops)
    if newops is not None:
        f.set_ops(newops)