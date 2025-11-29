from KunQuant.Op import OpBase, Output, CompositiveOp, WindowedTrait, WindowedTempOutput, WindowedDataSourceOp, CrossSectionalOp, Input
from KunQuant.Stage import Function, OpInfo
from typing import List, Dict
from .Util import kun_pass


def decompose_impl(ops: List[OpBase], options) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    for op in ops:
        op.replace_inputs(replace_map)
        if isinstance(op, CompositiveOp):
            changed = True
            decomposed = op.decompose(options)
            # recursively call decompose
            inner = decompose_impl(decomposed, options)
            if inner is not None:
                decomposed = inner
            out.extend(decomposed)
            replace_map[op] = decomposed[-1]
        elif isinstance(op, WindowedTrait):
            window = op.required_input_window()
            for idx, inp in enumerate(op.inputs):
                if not isinstance(inp, WindowedDataSourceOp):
                    changed = True
                    newin = WindowedTempOutput(inp, window)
                    op.inputs[idx] = newin
                    out.append(newin)
            out.append(op)
        else:
            out.append(op)
    if changed:
        return out
    return None

def decompose_rank_impl(ops: List[OpBase], info: Dict[OpBase, OpInfo]) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    already_added_out = dict()
    visited = set()
    hash_cache: Dict['OpBase', int] = dict()
    for op in ops:
        op.replace_inputs(replace_map)
        if op in visited:
            continue
        if isinstance(op, CrossSectionalOp):
            for idx, inp in enumerate(op.inputs):
                if not isinstance(inp, Input) and not isinstance(inp, Output):
                    changed = True
                    newin = None
                    if inp in already_added_out:
                        newin = already_added_out[inp]
                    else:
                        for user in info[inp].uses:
                            if isinstance(user, Output):
                                newin = user
                                if user not in visited:
                                    visited.add(user)
                                    out.append(user)
                                break
                        if newin is None:
                            newin = Output(inp, inp.hash_hex(hash_cache))
                            out.append(newin)
                        already_added_out[inp] = newin
                    op.inputs[idx] = newin
            out.append(op)
            visited.add(op)
        else:
            out.append(op)
            visited.add(op)
    if changed:
        return out
    return None

@kun_pass
def decompose(f: Function, options: dict = {}):
    newops = decompose_impl(f.ops, options)
    f.strict_window = True
    if newops is not None:
        f.set_ops(newops)

@kun_pass
def decompose_rank(f: Function, options: dict = {}):
    newops = decompose_rank_impl(f.ops, f.op_to_id)
    if newops is not None:
        f.set_ops(newops)

def move_dup_rank_output_impl(ops: List[OpBase], info: Dict[OpBase, OpInfo]) -> List[OpBase]:
    '''
    before
    v1 = Rank(...)
    o1 = Output(v1)
    o2 = Output(v2)
    after this pass:
    v1 = Rank(...)
    o1 = Output(v1)
    o2 = Output(o1)
    '''
    replace_map = dict()
    out = []
    changed = False
    already_added_out = dict()
    for op in ops:
        op.replace_inputs(replace_map)
        if isinstance(op, Output):
            inp = op.inputs[0]
            if isinstance(inp, CrossSectionalOp):
                if inp in already_added_out:
                    op.inputs[0] = already_added_out[inp]
                else:
                    already_added_out[inp] = op
            out.append(op)
        else:
            out.append(op)
    if changed:
        return out
    return None


@kun_pass
def move_dup_rank_output(f: Function, options: dict = {}):
    newops = move_dup_rank_output_impl(f.ops, f.op_to_id)
    if newops is not None:
        f.set_ops(newops)