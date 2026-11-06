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
    replace_map: Dict[OpBase, OpBase]

    def __hash__(self):
        return self.op.fast_hash(self.cache)
    def __eq__(self, o: '_OpProxy', self_kwargs: dict = None, o_kwargs: dict = None, checked: set = None) -> bool:
        if self.parent != o.parent:
            return False
        if self.op.__class__ != o.op.__class__:
            return False
        if self.op.attrs != o.op.attrs:
            return False
        self_args = self.op.get_args(True, **(self_kwargs or {}))
        o_args = o.op.get_args(True, **(o_kwargs or {}))
        if self_args == o_args:
            return True
        if len(self_args) != len(o_args):
            return False
        # the pairs of args already found equal in this comparison, so that a value used by many ops is compared once
        if checked is None:
            checked = set()
        for self_arg, o_arg in zip(self_args, o_args):
            lhs_op, lhs_kwargs = self_arg
            rhs_op, rhs_kwargs = o_arg
            lhs_replaced = self.replace_map.get(lhs_op)
            if lhs_replaced is not None and lhs_replaced == self.replace_map.get(rhs_op) and lhs_kwargs == rhs_kwargs:
                continue   #fast path
            key = (
                lhs_op, rhs_op,
                frozenset(lhs_kwargs.items()),
                frozenset(rhs_kwargs.items()),
            )
            if key in checked:
                continue
            if not _OpProxy(lhs_op, self.cache, self.parent, self.replace_map).__eq__(_OpProxy(rhs_op, self.cache, self.parent, self.replace_map), lhs_kwargs, rhs_kwargs, checked):
                return False
            checked.add(key)
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
        thekey = _OpProxy(op, cache, op.get_parent(), replace_map)
        found_op = computed.get(thekey, None)
        if not found_op:
            out.append(op)
            computed[thekey] = op
            replace_map[op] = op
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