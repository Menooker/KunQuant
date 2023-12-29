from KunQuant.passes import *
from KunQuant.Stage import Function
from KunQuant.Op import Input, Output, OpBase
from typing import Dict, List

def optimize(f: Function)->None:
    decompose(f)
    expr_fold(f)
    temp_window_elim(f)
    special_optimize(f)

def compileit(f: Function, input_stride: int, output_stride: int, buffer_names: List[str]):
    optimize(f)
    mainf, impl = do_partition(f, 4)
    input_name_to_idx = dict()
    for idx, name in enumerate(buffer_names):
        input_name_to_idx[name] = idx
    def insert_name(op: OpBase) -> None:
        nonlocal input_name_to_idx
        name = op.attrs["name"]
        if name not in input_name_to_idx:
            newidx = len(input_name_to_idx)
            input_name_to_idx[name] =  newidx
            
    for func in impl:
        ins = []
        outs = []
        for op in func.ops:
            if isinstance(op, Input):
                insert_name(op)
                ins.append(op)
            elif isinstance(op, Output):
                insert_name(op)
                outs.append(op)
        src = codegen_cpp(func, input_stride, output_stride, input_name_to_idx, ins, outs)
        print(src)
