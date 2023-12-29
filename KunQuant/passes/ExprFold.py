from KunQuant.Op import OpBase, ForeachBackWindow, Output
from KunQuant.Stage import Function
from typing import List, Dict

def fold_impl(ops: List[OpBase]) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    # need to consider the loop of the Op. We don't fold the Ops of different loops
    computed: Dict[(str, ForeachBackWindow), OpBase] = dict()
    for op in ops:
        op.replace_inputs(replace_map)
        if isinstance(op, ForeachBackWindow) or isinstance(op, Output):
            # don't fold for-each-backwindow
            out.append(op)
            continue
        thekey = (str(op), op.get_parent())
        found_op = computed.get(thekey, None)
        if not found_op:
            out.append(op)
            computed[thekey] = op
            continue
        changed = True    
        replace_map[op] = found_op
    if changed:
        return out
    return None

def expr_fold(f: Function):
    newops = fold_impl(f.ops)
    if newops is not None:
        f.set_ops(newops)