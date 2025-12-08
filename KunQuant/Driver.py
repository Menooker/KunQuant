from KunQuant.passes import *
from KunQuant.Stage import Function
from KunQuant.Op import Input, Output, OpBase, CrossSectionalOp
from typing import Dict, List, Union
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from KunQuant.passes import Util as PassUtil

# get the cpu architecture of the machine
from KunQuant.jit.env import cpu_arch as _cpu_arch

required_version = "0x64100003"
@dataclass
class KunCompilerConfig:
    partition_factor : int = 3
    dtype:str = "float"
    blocking_len: int = None
    input_layout:str = "STs"
    output_layout:str = "STs"
    allow_unaligned: Union[bool, None] = None
    split_source: int = 0
    options: dict = field(default_factory=dict)

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
    move_dup_rank_output(f, options)
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

def _is_cross_sectional(f: Function) -> bool:
    for op in f.ops:
        if isinstance(op, Input) or isinstance(op, Output):
            continue
        if isinstance(op, CrossSectionalOp):
            return True
        return False
    return False

def _deprecation_check(name: str, argname: str) -> str:
    if name == "ST8s":
        print(f"The layout name given in {argname} ST8s is depracated. Use STs and blocking_len=8 instead")
        return "STs"
    return name

def compileit(f: Function, module_name: str, partition_factor = 3, dtype = "float", blocking_len = None, input_layout = "STs", output_layout = "STs", allow_unaligned: Union[bool, None] = None, split_source = 0, options = {}) -> List[str]:
    element_size = {"float": 32, "double": 64}
    if _cpu_arch == "x86_64":
        suggested_len = {"float": 8, "double": 4}
        simd_len = {256, 512}
    elif _cpu_arch == "aarch64":
        suggested_len = {"float": 4, "double": 2}
        simd_len = {128}
    else:
        raise RuntimeError(f"Unsupported CPU architecture: {_cpu_arch}")
    input_layout = _deprecation_check(input_layout, "input_layout")
    output_layout = _deprecation_check(output_layout, "input_layout")
    if dtype not in ["float", "double"]:
        raise RuntimeError("Bad dtype " + dtype)
    if blocking_len is None:
        blocking_len = suggested_len[dtype]
    if element_size[dtype] * blocking_len not in simd_len:
        raise RuntimeError(f"Blocking length {blocking_len} is not supported for {dtype} on {_cpu_arch}")
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
    stats_no_warn = False
    if options.get('no_fast_stat', False) == "no_warn":
        options['no_fast_stat'] = False
        stats_no_warn = True
    if stream_mode:
        if 'no_fast_stat' not in options:
            options['no_fast_stat'] = True
        else:
            if not options['no_fast_stat']:
                raise RuntimeError("no_fast_stat=False is not supported in stream mode.")
    else:
        if 'no_fast_stat' not in options:
            options['no_fast_stat'] = dtype == "float"
        if not options['no_fast_stat'] and not stats_no_warn:
            print(f"Warning: fast stat optimization is ON for {module_name}. This may result in lower precision and faster execution in windowed statistics functions. You can turn it off by setting options['no_fast_stat'] = True. If you are sure about the precision, you can set options['no_fast_stat'] = 'no_warn' to disable this warning.")

    if stream_mode and allow_unaligned is None:
        allow_unaligned = False
    elif allow_unaligned is None:
        allow_unaligned = _cpu_arch != "aarch64"
    if allow_unaligned and stream_mode:
        raise RuntimeError("Currently allow_unaligned in stream mode is not supported.")
    if allow_unaligned and _cpu_arch == "aarch64":
        raise RuntimeError("Currently allow_unaligned is not supported on aarch64")
    
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
        if op.attrs.get("single_value", False):
            op.attrs["layout"] = "TS"
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

    shared_header = '''#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/Ops.hpp>
#include <Kun/Rank.hpp>
#include <Kun/Scale.hpp>
#include <Kun/Ops/Quantile.hpp>


using namespace kun;
using namespace kun::ops;
'''
    shared_header_simple = '''#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
using namespace kun;
'''
    impl_src = [shared_header]
    def push_source(is_simple=False):
        nonlocal impl_src, cur_count
        out_src.append("\n\n".join(impl_src))
        cur_count = 0
        impl_src = [shared_header_simple if is_simple else shared_header]
    out_src = []
    decl_src = []
    cur_count = 0
    is_single_source = split_source == 0
    # the set of names of custom cross sectional functions
    generated_cross_sectional_func = set()
    for func in impl:
        if split_source > 0 and cur_count > split_source:
            push_source()
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
        src, decl = codegen_cpp(module_name, func, input_name_to_idx, ins, outs, options, stream_mode, query_temp_buf_id, input_windows, generated_cross_sectional_func, dtype, blocking_len, not allow_unaligned, is_single_source)
        impl_src.append(src)
        decl_src.append(decl)
        newparti = _Partition(func.name, len(partitions), pins, pouts)
        if _is_cross_sectional(func):
            newparti.is_cross_sectional = True
        partitions[func.name] = newparti
        # if not cross session
        if len(src) > 0:
            cur_count+=1
    for p in mainf.ops:
        cur = partitions[p.attrs["name"]]
        cur.num_in_dep = len(p.inputs)
        cur.outs = [partitions[use.attrs["name"]] for use in mainf.op_to_id[p].uses]

    if PassUtil.debug_mode:
        print("Num temp buffers: ", num_temp_buffer)
    if not is_single_source:
        push_source(True)
    static_or_none = "static" if is_single_source else ""
    buffer_src = ",\n".join(["    "+ v.to_str(required_windows, input_windows) for v in buffer_names])
    impl_src.append(f"{static_or_none} BufferInfo __buffers_{module_name}[]{{\n{buffer_src}\n}};")
    buffer_decl = f"extern BufferInfo __buffers_{module_name}[{len(buffer_names)}];"

    if not is_single_source:
        push_source(True)
        impl_src.append(buffer_decl)
    parti_buffer_src = []
    parti_buffer_decl = []
    for name, parti in partitions.items():
        buffer_lines = ", ".join([f"&__buffers_{module_name}[{v.idx}]" for v in parti.in_buf])
        parti_buffer_src.append(f"{static_or_none} BufferInfo *stage_{name}_in_buf[] = {{{buffer_lines}}};")
        parti_buffer_decl.append(f"extern BufferInfo *stage_{name}_in_buf[{len(parti.in_buf)}];")
        buffer_lines = ", ".join([f"&__buffers_{module_name}[{v.idx}]" for v in parti.out_buf])
        parti_buffer_src.append(f"{static_or_none} BufferInfo *stage_{name}_out_buf[] = {{{buffer_lines}}};")
        parti_buffer_decl.append(f"extern BufferInfo *stage_{name}_out_buf[{len(parti.in_buf)}];")
    impl_src.append("\n".join(parti_buffer_src))

    parti_dep_src = "\n".join([f"extern Stage *stage_{module_name}__{name}_dep[{len(parti.outs)}];" if len(parti.outs) else f"Stage **stage_{module_name}__{name}_dep = nullptr;"
                                for name, parti in partitions.items()])
    if not is_single_source:
        push_source(False)
        impl_src.append(parti_dep_src)
        impl_src.extend(parti_buffer_decl)
    else:
        impl_src.append(f'''namespace {{
{parti_dep_src}
}}
''')
    # push extern decls
    impl_src.append("\n".join(decl_src).rstrip())
    
    parti_info_src = ",\n".join([f'''    {{/*f*/ stage_{module_name}__{parti.name}, /*dependers*/ stage_{module_name}__{parti.name}_dep, /*num_dependers*/ {len(parti.outs)},
     /*in_buffers*/ stage_{parti.name}_in_buf, /*num_in_buffers*/ {len(parti.in_buf)},
     /*out_buffers*/ stage_{parti.name}_out_buf, /*num_out_buffers*/ {len(parti.out_buf)}, /*pending_out*/ {parti.num_in_dep},
     /*num_tasks*/ TaskExecKind::{"SLICE_BY_TIME" if parti.is_cross_sectional else "SLICE_BY_STOCK"}, /*id*/ {parti.idx}}}''' for parti in partitions.values()])
    impl_src.append(f'''{static_or_none} Stage __stages_{module_name}[] = {{
{parti_info_src}
}};''')

    if not is_single_source:
        push_source(True)
        impl_src.append(f'''extern Stage __stages_{module_name}[{len(partitions)}];''')
        impl_src.append(buffer_decl)
    parti_dep_src = []
    for name, parti in partitions.items():
        if len(parti.outs):
            details = ", ".join([f"&__stages_{module_name}[{out.idx}]" for out in parti.outs])
            parti_dep_src.append(f"Stage *stage_{module_name}__{parti.name}_dep[] = {{{details}}};")
    parti_dep_src2 = "\n".join(parti_dep_src)
    if not is_single_source:
        impl_src.append(parti_dep_src2)
    else:
        impl_src.append(f'''namespace {{
{parti_dep_src2}
}}
''')
    dty = dtype[0].upper() + dtype[1:]
    impl_src.append(f'''KUN_EXPORT Module {module_name}{{
    {required_version},
    {len(partitions)},
    __stages_{module_name},
    {len(buffer_names)},
    __buffers_{module_name},
    MemoryLayout::{input_layout},
    MemoryLayout::{output_layout},
    {blocking_len},
    Datatype::{dty},
    {"0" if allow_unaligned else "1"}
}};''')
    push_source()
    if not is_single_source:
        out_src[0], out_src[-2] = out_src[-2], out_src[0]
    return out_src
