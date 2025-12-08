from KunQuant.passes.Util import kun_pass
from KunQuant.Op import OpBase, ForeachBackWindow, Output
from KunQuant.Stage import Function
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class _OpProxy:
    op: OpBase
    cache: Dict[OpBase, int]
    parent: ForeachBackWindow

    def __hash__(self):
        return self.op.fast_hash(self.cache)
    def __eq__(self, o: '_OpProxy') -> bool:
        if self.parent != o.parent:
            return False
        if self.op.__class__ != o.op.__class__:
            return False
        if self.op.attrs != o.op.attrs:
            return False
        if self.op.inputs == o.op.inputs:
            return True
        for x,y in zip(self.op.items(), o.op.items()):
            if x != y:
                return False
        return True
    
def fold_impl(ops: List[OpBase]) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    # need to consider the loop of the Op. We don't fold the Ops of different loops
    computed: Dict[_OpProxy, OpBase] = dict()
    cache: Dict[OpBase, int] = dict()
    for op in ops:
        op.replace_inputs(replace_map)
        if isinstance(op, ForeachBackWindow) or isinstance(op, Output):
            # don't fold for-each-backwindow
            out.append(op)
            continue
        thekey = _OpProxy(op, cache, op.get_parent())
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

@kun_pass
def expr_fold(f: Function, options: dict = {}):
    newops = fold_impl(f.ops)
    if newops is not None:
        f.set_ops(newops)