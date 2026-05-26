from KunQuant.passes.Util import kun_pass
from KunQuant.Op import (
    OpBase, WindowedTempOutput, Input, Output, WindowedTrait,
    traverse_replace_map,
)
from KunQuant.Stage import Function
from typing import List, Dict, Tuple

def _get_temp_out_with_window(op: OpBase, window: int):
    if not isinstance(op, WindowedTempOutput):
        return False, 0
    w = op.attrs["window"]
    return w >= window, w

def for_each_op(op: OpBase, f: Function, replace_map: dict, may_slice_time: bool) -> Tuple[OpBase, OpBase]:
    if not isinstance(op, WindowedTempOutput):
        return (op, None)
    inp = op.inputs[0]
    # temp window on input, simply eliminate it
    if isinstance(inp, Input):
        return (None, inp)
    # If nobody consumes this as a windowed source, the temp window is just
    # the current input value and can be folded away.
    if not any(isinstance(user, WindowedTrait)
               for user in f.op_to_id[op].uses):
        return (None, inp)
    # check if the input of WindowedTempOutput is used in Output or other WindowedTempOutput
    inp_info = f.op_to_id[inp]
    window = op.attrs["window"]
    max_window = 0
    max_window_op = None
    for user, _ in inp_info.uses.items():
        if user == op:
            continue
        # if the user is used by Output, return the output.  When the
        # runtime may slice time, reading history from an output buffer can
        # race across time chunks; keep a local temp window instead.
        if not may_slice_time and isinstance(user, Output):
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

def _unwrap_output_wto(ops: List[OpBase], f: Function) -> bool:
    """Rewrite Output(WindowedTempOutput(x)) → Output(x)."""
    changed = False
    for op in ops:
        if not isinstance(op, Output):
            continue
        src = op.inputs[0]
        if not isinstance(src, WindowedTempOutput):
            continue
        old_src = src
        while isinstance(src, WindowedTempOutput):
            src = src.inputs[0]
        if op in f.op_to_id[old_src].uses:
            del f.op_to_id[old_src].uses[op]
        f.op_to_id[src].uses[op] = 1
        op.inputs[0] = src
        changed = True
    return changed

def temp_window_elim_impl(ops: List[OpBase], f: Function, options: dict) -> List[OpBase]:
    _unwrap_output_wto(ops, f)
    may_slice_time = options.get("may_slice_time", False)
    replace_map = dict()
    out = []
    changed = False
    for idx, op in enumerate(ops):
        if op in replace_map:
            continue
        op.replace_inputs(replace_map)
        normal, replacer = for_each_op(op, f, replace_map, may_slice_time)
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
    newops = temp_window_elim_impl(f.ops, f, options)
    if newops is not None:
        newops = Function.topo_sort_ops(newops)
        f.set_ops(newops)
