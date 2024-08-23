from KunQuant.passes import *
from KunQuant.Stage import Function
from KunQuant.Op import Input, Output, OpBase, CrossSectionalOp
from typing import Dict, List, Union
import typing
from collections import OrderedDict
from dataclasses import dataclass
from KunQuant.passes import Util as PassUtil

required_version = "0x64100003"

def optimize(f: Function, options: dict)->Dict[str, int]:
    if PassUtil.debug_mode:
        print("Before optimize: ", f)
    ret = infer_window(f, options)
    # optimize before decompose to let value ranges work
    special_optimize(f, options)
    decompose(f, options)
    expr_fold(f, options)
    special_optimize(f, options)
    expr_fold(f, options)
    decompose_rank(f, options)
    temp_window_elim(f, options)
    return ret

def post_optimize(impl: List[Function], options: dict)->Dict[str, int]:
    window_result: Dict[str, int] = dict()
    if PassUtil.debug_mode:
        print("Post optimize:","=====================")
    for f in impl:
        temp_window_elim(f, options)
        infer_input_window(f, window_result)
        merge_loops(f, options)
    return window_result

@dataclass
class _Buffer:
    idx: int
    name: str
    kind: str
    num_users: int = 0

    def to_str(self, unreliables: Dict[str, int], stream_windows: Dict[str, int]) -> str:
        unrel = unreliables.get(self.name, 0)
        swindow = stream_windows.get(self.name, 1)
        return f'{{{self.idx}, "{self.name}", {self.num_users}, BufferKind::{self.kind}, {unrel}, {swindow}}}'

@dataclass
class _Partition:
    name: str
    idx: int
    in_buf: List[_Buffer]
    out_buf: List[_Buffer]
    outs: List['_Partition'] = None
    num_in_dep = 0
    is_cross_sectional = False

    def __post_init__(self):
        for buf in self.in_buf:
            buf.num_users += 1

def _deprecation_check(name: str, argname: str) -> str:
    if name == "ST8s":
        print(f"The layout name given in {argname} ST8s is depracated. Use STs and blocking_len=8 instead")
        return "STs"
    return name

def compileit(f: Function, module_name: str, partition_factor = 3, dtype = "float", blocking_len = None, input_layout = "STs", output_layout = "STs", allow_unaligned: Union[bool, None] = None, options = {}):
    input_layout = _deprecation_check(input_layout, "input_layout")
    output_layout = _deprecation_check(output_layout, "input_layout")
    if dtype not in ["float", "double"]:
        raise RuntimeError("Bad dtype " + dtype)
    if blocking_len is None:
        suggested_len = {"float": 8, "double": 4}
        blocking_len = suggested_len[dtype]
    if output_layout not in ["STs", "TS", "STREAM"]:
        raise RuntimeError("Bad output_layout name " + output_layout)
    if input_layout not in ["STs", "TS", "STREAM"]:
        raise RuntimeError("Bad input_layout name " + input_layout)
    stream_mode = output_layout == "STREAM"
    if stream_mode and input_layout != "STREAM":
        print("Ignoring input_layout because output_layout is stream mode")
        input_layout = "STREAM"

    if stream_mode and options.get("opt_reduce", False):
        raise RuntimeError("Currently opt_reduce in stream mode is not supported.")
    if stream_mode and allow_unaligned is None:
        allow_unaligned = False
    if allow_unaligned and stream_mode:
        raise RuntimeError("Currently allow_unaligned in stream mode is not supported.")

    input_name_to_idx: Dict[str, int] = dict()
    buffer_names: List[_Buffer] = []
    partitions: typing.OrderedDict[str, _Partition] = OrderedDict()
    num_temp_buffer = 0
    def insert_name_str(name: str, kind: str) -> _Buffer:
        nonlocal input_name_to_idx, num_temp_buffer
        if name not in input_name_to_idx:
            newidx = len(input_name_to_idx)
            newbuf =  _Buffer(newidx, name, kind)
            if kind == "TEMP":
                num_temp_buffer += 1
            input_name_to_idx[name] = newidx
            buffer_names.append(newbuf)
            return newbuf
        return buffer_names[input_name_to_idx[name]]
    def insert_name(op: OpBase, kind: str) -> _Buffer:
        return insert_name_str(op.attrs["name"], kind)
    def set_buffer_layout(op: OpBase, buf: _Buffer):
        if stream_mode:
            op.attrs["layout"] = "STREAM"
            return
        if buf.kind == "TEMP":
            op.attrs["layout"] = "STs"
        elif buf.kind == "INPUT":
            op.attrs["layout"] = input_layout
        elif buf.kind == "OUTPUT":
            op.attrs["layout"] = output_layout

    for op in f.ops:
        if isinstance(op, Input):
            insert_name(op, "INPUT")
        elif isinstance(op, Output):
            insert_name(op, "OUTPUT")

    required_windows = optimize(f, options)
    mainf, impl = do_partition(f, partition_factor, options)
    input_windows = post_optimize(impl, options)

    impl_src = ['''#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/Ops.hpp>
#include <Kun/Rank.hpp>
#include <Kun/Scale.hpp>
#include <Kun/Ops/Quantile.hpp>


using namespace kun;
using namespace kun::ops;
''']    
    for func in impl:
        pins = []
        pouts = []
        ins = []
        outs = []
        for op in func.ops:
            if isinstance(op, Input):
                buf = insert_name(op, "TEMP")
                pins.append(buf)
                ins.append((op, buf.kind))
                set_buffer_layout(op, buf)
            elif isinstance(op, Output):
                buf = insert_name(op, "TEMP")
                pouts.append(buf)
                outs.append((op, buf.kind))
                set_buffer_layout(op, buf)
        def query_temp_buf_id(tempname: str, window: int) -> int:
            input_windows[tempname] = window
            return insert_name_str(tempname, "TEMP").idx
        src = codegen_cpp(func, input_name_to_idx, ins, outs, options, stream_mode, query_temp_buf_id, input_windows, dtype, blocking_len, not allow_unaligned)
        impl_src.append(src)
        newparti = _Partition(func.name, len(partitions), pins, pouts)
        if len(func.ops) == 3 and isinstance(func.ops[1], CrossSectionalOp):
            newparti.is_cross_sectional = True
        partitions[func.name] = newparti
    for p in mainf.ops:
        cur = partitions[p.attrs["name"]]
        cur.num_in_dep = len(p.inputs)
        cur.outs = [partitions[use.attrs["name"]] for use in mainf.op_to_id[p].uses]

    if PassUtil.debug_mode:
        print("Num temp buffers: ", num_temp_buffer)

    buffer_src = ",\n".join(["    "+ v.to_str(required_windows, input_windows) for v in buffer_names])
    impl_src.append(f"static BufferInfo __buffers[]{{\n{buffer_src}\n}};")

    parti_buffer_src = []
    for name, parti in partitions.items():
        buffer_lines = ", ".join([f"&__buffers[{v.idx}]" for v in parti.in_buf])
        parti_buffer_src.append(f"static BufferInfo *stage_{name}_in_buf[] = {{{buffer_lines}}};")
        buffer_lines = ", ".join([f"&__buffers[{v.idx}]" for v in parti.out_buf])
        parti_buffer_src.append(f"static BufferInfo *stage_{name}_out_buf[] = {{{buffer_lines}}};")
    impl_src.append("\n".join(parti_buffer_src))

    parti_dep_src = "\n".join([f"extern Stage *stage_{name}_dep[{len(parti.outs)}];" if len(parti.outs) else f"Stage **stage_{name}_dep = nullptr;"
                                for name, parti in partitions.items()])
    impl_src.append(f'''namespace {{
{parti_dep_src}
}}
''')
    
    parti_info_src = ",\n".join([f'''    {{/*f*/ stage_{parti.name}, /*dependers*/ stage_{parti.name}_dep, /*num_dependers*/ {len(parti.outs)},
     /*in_buffers*/ stage_{parti.name}_in_buf, /*num_in_buffers*/ {len(parti.in_buf)},
     /*out_buffers*/ stage_{parti.name}_out_buf, /*num_out_buffers*/ {len(parti.out_buf)}, /*pending_out*/ {parti.num_in_dep},
     /*num_tasks*/ TaskExecKind::{"SLICE_BY_TIME" if parti.is_cross_sectional else "SLICE_BY_STOCK"}, /*id*/ {parti.idx}}}''' for parti in partitions.values()])
    impl_src.append(f'''static Stage __stages[] = {{
{parti_info_src}
}};''')
                    
    parti_dep_src = []
    for name, parti in partitions.items():
        if len(parti.outs):
            details = ", ".join([f"&__stages[{out.idx}]" for out in parti.outs])
            parti_dep_src.append(f"Stage *stage_{parti.name}_dep[] = {{{details}}};")
    parti_dep_src2 = "\n".join(parti_dep_src)
    impl_src.append(f'''namespace {{
{parti_dep_src2}
}}
''')
    dty = dtype[0].upper() + dtype[1:]
    impl_src.append(f'''KUN_EXPORT Module {module_name}{{
    {required_version},
    {len(partitions)},
    __stages,
    {len(buffer_names)},
    __buffers,
    MemoryLayout::{input_layout},
    MemoryLayout::{output_layout},
    {blocking_len},
    Datatype::{dty},
    {"0" if allow_unaligned else "1"}
}};''')
    return "\n\n".join(impl_src)
