from KunQuant.Op import OpBase, Output, WindowedTrait, Input
from KunQuant.Stage import Function
from typing import Dict, List
from .Util import kun_pass

def _impl(op: OpBase, result: Dict[OpBase, int]) -> int:
    if op in result:
        return result[op]
    v = 0
    for inp in op.inputs:
        v = max(v, _impl(inp, result))
    if isinstance(op, WindowedTrait):
        v += op.attrs["window"]
    result[op] = v
    return v

def infer_window(f: Function, options: dict = {}) -> Dict[str, int]:
    result: Dict[OpBase, int] = dict()
    ret = dict()
    for op in f.ops:
        if isinstance(op, Output):
            ret[op.attrs["name"]] = _impl(op, result)
    return ret

def infer_input_window(f: Function, result: Dict[str, int]) -> None:
    for op in f.ops:
        if isinstance(op, Input) or isinstance(op, Output):
            opname = op.attrs["name"]
            result[opname] = result.get(opname, 1)
            for use in f.op_to_id[op].uses:
                if isinstance(use, WindowedTrait):
                    result[opname] = max(use.required_input_window(), result.get(opname, 0))

