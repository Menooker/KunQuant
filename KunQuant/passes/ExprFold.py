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
        return _is_same(self.op, o.op)

def _is_same(x: OpBase, y: OpBase) -> bool:
    '''
    Returns whether x.items() and y.items() are equal. It walks the two graphs side by side and
    visits each pair of ops once, so that a value used by many ops is not walked once per path
    to it.
    '''
    stack = [(x, {}, y, {})]
    visited = set()
    while stack:
        a, akwargs, b, bkwargs = stack.pop()
        if a is b and akwargs.keys() == bkwargs.keys() and all(akwargs[k] is bkwargs[k] for k in akwargs):
            continue
        key = (id(a), id(b), *map(id, akwargs.values()), *map(id, bkwargs.values()))
        if key in visited:
            continue
        visited.add(key)
        if a.__class__ != b.__class__ or a.attrs != b.attrs:
            return False
        aargs = a.get_args(True, **akwargs)
        bargs = b.get_args(True, **bkwargs)
        if len(aargs) != len(bargs):
            return False
        for (aarg, aargkwargs), (barg, bargkwargs) in zip(aargs, bargs):
            stack.append((aarg, aargkwargs, barg, bargkwargs))
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