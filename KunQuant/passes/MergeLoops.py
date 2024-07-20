from KunQuant.passes.Util import kun_pass
from KunQuant.Op import OpBase, ForeachBackWindow, ReductionOp, IterValue
from KunQuant.Stage import Function
from typing import List, Dict

def _get_single_user(f: Function, op: OpBase) -> OpBase:
    loop_vars = f.op_to_id[op].uses
    if len(loop_vars) != 1:
        return None
    return next(iter(loop_vars))

def _should_merge_loops(f: Function, op: OpBase) -> List[List[OpBase]]:
    '''
    Check if op is used by multiple single-itervalue loops and itervalue is used by a single reductionop
    '''
    op_uses = f.op_to_id[op].uses
    user_loops: Dict[int, List[OpBase]] = {}
    for op_use in op_uses:
        if op_use.inputs.__len__() != 1 or not isinstance(op_use, ForeachBackWindow):
            continue
        iter_var = _get_single_user(f, op_use)
        if not iter_var:
            continue
        reducer = _get_single_user(f, iter_var)
        if not reducer or not isinstance(reducer, ReductionOp) or reducer.inputs.__len__() !=  1:
            continue
        window = op_use.attrs["window"]
        if window not in user_loops:
            user_loops[window] = [op_use, iter_var, reducer]
        else:
            user_loops[window].append(reducer)
    if len(user_loops) <= 1:
        return None
    user_loops = sorted(user_loops.values(), key=lambda x: x[0].attrs["window"], reverse=True)
    return user_loops

def _merge_impl(f: Function, ops: List[OpBase]) -> List[OpBase]:
    replace_map = set()
    out = []
    changed = False
    for op in ops:
        if op in replace_map:
            continue
        reduce_loop = _should_merge_loops(f, op)
        if reduce_loop:
            changed = True
            out.append(op)
            for idx, loop_ops in enumerate(reduce_loop):
                loop = loop_ops[0]
                loopvar = loop_ops[1]
                if idx != 0:
                    loop.attrs["copy_prev_body"] = True
                if idx != len(reduce_loop) - 1:
                    loop.attrs["segment_end"] = reduce_loop[idx+1][0].attrs["window"]
                # update the loopvar and parent_loop of reduce_op
                for i in range(3, len(loop_ops)):
                    loop_ops[i].inputs[0] = loopvar
                for o in loop_ops:
                    out.append(o)
                    replace_map.add(o)
            continue
        out.append(op)
    if changed:
        return out
    return None

@kun_pass
def merge_loops(f: Function, options: dict = {}):
    newops = _merge_impl(f, f.ops)
    if newops is not None:
        f.set_ops(newops)