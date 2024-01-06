from KunQuant.Op import OpBase, Output, CompositiveOp, WindowedTrait, WindowedTempOutput, WindowedDataSourceOp, Rank, Input
from KunQuant.Stage import Function
from typing import List
from .Util import kun_pass


def decompose_impl(ops: List[OpBase]) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    for op in ops:
        op.replace_inputs(replace_map)
        if isinstance(op, CompositiveOp):
            changed = True
            decomposed = op.decompose()
            # recursively call decompose
            inner = decompose_impl(decomposed)
            if inner is not None:
                decomposed = inner
            out.extend(decomposed)
            replace_map[op] = decomposed[-1]
        elif isinstance(op, WindowedTrait):
            window = op.attrs["window"]
            for idx, inp in enumerate(op.inputs):
                if not isinstance(inp, WindowedDataSourceOp):
                    changed = True
                    newin = WindowedTempOutput(inp, window)
                    op.inputs[idx] = newin
                    out.append(newin)
            out.append(op)
        elif isinstance(op, Rank):
            inp = op.inputs[0]
            if not isinstance(inp, Input):
                changed = True
                newin = Output(inp, inp.hash_hex())
                op.inputs[0] = newin
                out.append(newin)
            out.append(op)
        else:
            out.append(op)
    if changed:
        return out
    return None

@kun_pass
def decompose(f: Function):
    newops = decompose_impl(f.ops)
    f.strict_window = True
    if newops is not None:
        f.set_ops(newops)
    
            