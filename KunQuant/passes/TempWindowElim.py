from KunQuant.passes.Util import kun_pass
from KunQuant.Op import OpBase, WindowedTempOutput, Input, Output, traverse_replace_map
from KunQuant.Stage import Function
from typing import List, Dict, Tuple

def _get_temp_out_with_window(op: OpBase, window: int):
    if not isinstance(op, WindowedTempOutput):
        return False, 0
    w = op.attrs["window"]
    return w >= window, w

def for_each_op(op: OpBase, f: Function, replace_map: dict) -> Tuple[OpBase, OpBase]:
    if not isinstance(op, WindowedTempOutput):
        return (op, None)
    inp = op.inputs[0]
    # temp window on input, simply eliminate it
    if isinstance(inp, Input):
        return (None, inp)
    # check if the input of WindowedTempOutput is used in Output or other WindowedTempOutput
    inp_info = f.op_to_id[inp]
    window = op.attrs["window"]
    max_window = 0
    max_window_op = None
    for user, _ in inp_info.uses.items():
        if user == op:
            continue
        # if the user is used by Output, return the output
        if isinstance(user, Output):
            return (None, traverse_replace_map(user, replace_map))
        # select the max window op with the larger id
        checked, w = _get_temp_out_with_window(user, window)
        if checked:
            if w > max_window or (w == max_window and id(user) > id(max_window_op)):
                max_window = w
                max_window_op = user
    if max_window_op is not None:
        return (None, traverse_replace_map(max_window_op, replace_map))
    return (op, None)

def temp_window_elim_impl(ops: List[OpBase], f: Function) -> List[OpBase]:
    replace_map = dict()
    out = []
    changed = False
    for idx, op in enumerate(ops):
        if op in replace_map:
            continue
        op.replace_inputs(replace_map)
        normal, replacer = for_each_op(op, f, replace_map)
        if normal is not None:
            out.append(op)
        else:
            changed = True
            replace_map[op] = replacer
    if changed:
        return out
    return None

@kun_pass
def temp_window_elim(f: Function, options: dict = {}):
    newops = temp_window_elim_impl(f.ops, f)
    if newops is not None:
        newops = Function.topo_sort_ops(newops)
        f.set_ops(newops)