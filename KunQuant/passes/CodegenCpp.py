from KunQuant.Op import *
from KunQuant.Stage import Function, OpInfo
from KunQuant.ops import *
from typing import List, Dict, Set, Tuple
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import OrderedDict

class _CppStmt(ABC):
    def __init__(self, parent: '_CppStmt') -> None:
        super().__init__()
        self.parent = parent
        if parent:
            self.indents = parent.indents + 1
        else:
            self.indents = 0
    
    @abstractmethod
    def __str__(self) -> str:
        pass

class _CppScope(_CppStmt):
    def __init__(self, parent: _CppStmt) -> None:
        super().__init__(parent)
        self.scope: List[_CppStmt] = []
        self.parent_for: '_CppFor' = None

    def __str__(self) -> str:
        body = "\n".join([str(a) for a in self.scope])
        ind = make_indents(self.indents, 4)
        return f"{{\n{body}\n{ind}}}" 

class _CppFor(_CppStmt):
    def __init__(self, parent: _CppStmt, header: str) -> None:
        super().__init__(parent)
        self.header = header
        self.body = _CppScope(parent)
        self.body.parent_for = self

    def __str__(self) -> str:
        ind = make_indents(self.indents, 4)
        return ind + self.header + str(self.body)

class _CppSingleLine(_CppStmt):
    def __init__(self, parent: _CppStmt, line: str) -> None:
        super().__init__(parent)
        self.line = line
    def __str__(self) -> str:
        return make_indents(self.indents, 4) + self.line

def _get_buffer_name(op: OpBase, idx: int) -> str:
    if isinstance(op, Input) or isinstance(op, Output):
        name = op.attrs["name"]
        return f"buf_{name}"
    elif isinstance(op, WindowedTempOutput):
        return f"temp_{idx}"
    raise RuntimeError("Bad buffer" + str(op))

def _float_value_to_float(v: float, dtype: str) -> str:
    ret = str(v)
    if '.' not in ret and 'e' not in ret:
        ret += '.'
    if dtype == "float":
        ret += 'f'
    return ret

def _value_to_float(op: OpBase, dtype: str) -> str:
    return _float_value_to_float(op.attrs["value"], dtype)

vector_len = 8

def codegen_cpp(f: Function, input_name_to_idx: Dict[str, int], inputs: List[Tuple[Input, bool]], outputs: List[Tuple[Output, bool]], options: dict, stream_mode: bool, query_temp_buffer_id, stream_window_size: Dict[str, int], elem_type: str, simd_lanes: int, aligned: bool) -> str:
    if len(f.ops) == 3 and isinstance(f.ops[1], CrossSectionalOp):
        return f'''static auto stage_{f.name} = {f.ops[1].__class__.__name__}Stocks<Mapper{f.ops[0].attrs["layout"]}<{elem_type}, {simd_lanes}>, Mapper{f.ops[2].attrs["layout"]}<{elem_type}, {simd_lanes}>>;'''
    header = f'''static void stage_{f.name}(Context* __ctx, size_t __stock_idx, size_t __total_time, size_t __start, size_t __length) '''
    toplevel = _CppScope(None)
    buffer_type: Dict[OpBase, str] = dict()
    ptrname = "" if elem_type == "float" else "D"
    for inp, buf_kind in inputs:
        name = inp.attrs["name"]
        layout = inp.attrs["layout"]
        idx_in_ctx = input_name_to_idx[name]
        not_user_input = buf_kind != "INPUT" # if is user input, the time should start from __start and length of time is __total_time
        start_str = "0" if not_user_input else "__start"
        total_str = "__length" if not_user_input else "__total_time"
        if stream_mode:
            window_size = stream_window_size.get(name, 1)
            buffer_type[inp] = f"StreamWindow<{elem_type}, {simd_lanes}, {window_size}>"
            code = f"StreamWindow<{elem_type}, {simd_lanes}, {window_size}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].stream_buf, __stock_idx, __ctx->stock_count}};"
        else:
            buffer_type[inp] = f"Input{layout}<{elem_type}, {simd_lanes}>"
            code = f"Input{layout}<{elem_type}, {simd_lanes}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].ptr{ptrname}, __stock_idx, __ctx->stock_count, {total_str}, {start_str}}};"
        toplevel.scope.append(_CppSingleLine(toplevel, code))
    if not aligned:
        toplevel.scope.append(_CppSingleLine(toplevel, f'''auto todo_count = __ctx->stock_count - __stock_idx  * {simd_lanes};'''))
        toplevel.scope.append(_CppSingleLine(toplevel, f'''auto mask = kun_simd::vec<{elem_type}, {simd_lanes}>::make_mask(todo_count > {simd_lanes} ? {simd_lanes} : todo_count);'''))
    for idx, (outp, is_tmp) in enumerate(outputs):
        name = outp.attrs["name"]
        layout = outp.attrs["layout"]
        idx_in_ctx = input_name_to_idx[name]
        buffer_type[outp] = f"Output{layout}<{elem_type}, {simd_lanes}>"
        if stream_mode:
            window_size = stream_window_size.get(name, 1)
            buffer_type[inp] = f"StreamWindow<{elem_type}, {simd_lanes}, {window_size}>"
            code = f"StreamWindow<{elem_type}, {simd_lanes}, {window_size}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].stream_buf, __stock_idx, __ctx->stock_count}};"
        else:
            buffer_type[inp] = f"Output{layout}<{elem_type}, {simd_lanes}>"
            code = f"Output{layout}<{elem_type}, {simd_lanes}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].ptr{ptrname}, __stock_idx, __ctx->stock_count, __length, 0}};"
        toplevel.scope.append(_CppSingleLine(toplevel, code))
    for op in f.ops:
        if op.get_parent() is None and isinstance(op, WindowedTempOutput):
            window = op.attrs["window"]
            idx = f.get_op_idx(op)
            if stream_mode:
                buffer_type[op] = f"StreamWindow<{elem_type}, {simd_lanes}, {window}>"
                bufname = f"{f.name}_{idx}"
                code = f"StreamWindow<{elem_type}, {simd_lanes}, {window}> temp_{idx}{{__ctx->buffers[{query_temp_buffer_id(bufname, window)}].stream_buf, __stock_idx, __ctx->stock_count}};"
            else:
                buffer_type[op] = f"OutputWindow<{elem_type}, {simd_lanes}, {window}>"
                code = f"OutputWindow<{elem_type}, {simd_lanes}, {window}> temp_{idx}{{}};"
            toplevel.scope.append(_CppSingleLine(toplevel, code))

    top_for = _CppFor(toplevel, "for(size_t i = 0;i < __length;i++) ")
    toplevel.scope.append(top_for)
    top_body = top_for.body
    cur_body = top_body
    loop_to_cpp_loop: Dict[ForeachBackWindow, _CppScope] = {None: top_body}
    prev_for = None
    for op in f.ops:
        idx = f.get_op_idx(op)
        inp = [f.get_op_idx(inpv) for inpv in op.inputs]
        scope = loop_to_cpp_loop[op.get_parent()]
        if isinstance(op, Input):
            name = op.attrs["name"]
            if not aligned:
                mask_str = ", mask"
            else:
                mask_str = ""
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = buf_{name}.step(i{mask_str});"))
        elif isinstance(op, Output):
            name = op.attrs["name"]
            if not aligned:
                mask_str = ", mask"
            else:
                mask_str = ""
            scope.scope.append(_CppSingleLine(scope, f"buf_{name}.store(i, v{inp[0]}{mask_str});"))
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = v{inp[0]};"))
        elif isinstance(op, WindowedTempOutput):
            scope.scope.append(_CppSingleLine(scope, f"temp_{idx}.store(i, v{inp[0]});"))
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = v{inp[0]};"))
        elif isinstance(op, ConstantOp):
            scope.scope.append(_CppSingleLine(scope, f'auto v{idx} = constVec<{simd_lanes}>({_value_to_float(op, elem_type)});'))
        elif isinstance(op, Log):
            funcname = "LogFast" if options.get("fast_log", True) else "Log"
            scope.scope.append(_CppSingleLine(scope, f'auto v{idx} = {funcname}(v{inp[0]});'))
        elif isinstance(op, BinaryConstOp):
            assert(op.__class__.__name__.endswith("Const"))
            thename = op.__class__.__name__.replace("Const", "")
            rhs = _value_to_float(op, elem_type)
            if not op.attrs.get("swap", False):
                scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}(v{inp[0]}, {rhs});"))
            else:
                scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}({rhs}, v{inp[0]});"))
        elif isinstance(op, SetInfOrNanToValue):
            thename = op.__class__.__name__
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}(v{inp[0]}, {_value_to_float(op, elem_type)});"))
        elif isinstance(op, BinaryElementwiseOp):
            thename = op.__class__.__name__
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}(v{inp[0]}, v{inp[1]});"))
        elif isinstance(op, UnaryElementwiseOp):
            thename = op.__class__.__name__
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}(v{inp[0]});"))
        elif isinstance(op, ForeachBackWindow):
            window = op.attrs["window"]
            seg_end = op.attrs.get("segment_end", 0)
            copy_prev_body = op.attrs.get("copy_prev_body", False)
            the_for = _CppFor(scope, f"for(int iter = {window - 1};iter >= {seg_end};iter--) ")
            if copy_prev_body:
                for line in prev_for.body.scope:
                    the_for.body.scope.append(_CppSingleLine(the_for.body, line.line))
            scope.scope.append(the_for)
            loop_to_cpp_loop[op] = the_for.body
            prev_for = the_for
        elif isinstance(op, IterValue):
            loop = op.inputs[0]
            buf_name = _get_buffer_name(op.inputs[1], inp[1])
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {buf_name}.getWindow(i, iter);"))
        elif isinstance(op, ReductionOp):
            thename = op.__class__.__name__
            if isinstance(op, ReduceDecayLinear):
                thename = f'{thename}<{elem_type}, {simd_lanes}, {op.attrs["window"]}>'
            else:
                thename = f'{thename}<{elem_type}, {simd_lanes}>'
            loop_op = op.inputs[0] if isinstance(op.inputs[0], ForeachBackWindow) else op.inputs[0].get_parent()
            loop_body = loop_to_cpp_loop[loop_op]
            loop_var_idx = f.get_op_idx(loop_op)
            loop = loop_body.parent_for
            # insert a var definition before the for-loop
            loop_parent = loop.parent
            assert(isinstance(loop_parent, _CppScope))
            init_val = "" if len(op.inputs) == 1 else f"v{inp[1]}"
            loop_parent.scope.insert(loop_parent.scope.index(loop), _CppSingleLine(loop_parent, f"{thename} v{idx}{{{init_val}}};"))
            # insert a step in the for-loop
            loop_body.scope.append(_CppSingleLine(loop_body, f"v{idx}.step(v{inp[0]}, iter);"))
        elif isinstance(op, BackRef):
            assert(op.get_parent() is None)
            buf_name = _get_buffer_name(op.inputs[0], inp[0])
            funcname = "windowedRef"
            scope.scope.append(_CppSingleLine(scope, f'auto v{idx} = {funcname}<{elem_type}, {simd_lanes}, {op.attrs["window"]}>({buf_name}, i);'))
        elif isinstance(op, WindowedQuantile):
            assert(op.get_parent() is None)
            buf_name = _get_buffer_name(op.inputs[0], inp[0])
            funcname = "windowedQuantile"
            scope.scope.append(_CppSingleLine(scope, f'auto v{idx} = {funcname}<{elem_type}, {simd_lanes}, {op.attrs["window"]}>({buf_name}, i, {_float_value_to_float(op.attrs["q"], elem_type)});'))
        elif isinstance(op, FastWindowedSum):
            if stream_mode: raise RuntimeError(f"Stream Mode does not support {op.__class__.__name__}")
            assert(op.get_parent() is None)
            buf_name = _get_buffer_name(op.inputs[0], inp[0])
            window = op.attrs["window"]
            toplevel.scope.insert(-1, _CppSingleLine(toplevel, f"FastWindowedSum<{elem_type}, {simd_lanes}, {window}> sum_{idx};"))
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = sum_{idx}.step({buf_name}, v{inp[0]}, i);"))
        elif isinstance(op, ExpMovingAvg):
            if stream_mode: raise RuntimeError(f"Stream Mode does not support {op.__class__.__name__}")
            assert(op.get_parent() is None)
            window = op.attrs["window"]
            toplevel.scope.insert(-1, _CppSingleLine(toplevel, f"ExpMovingAvg<{elem_type}, {simd_lanes}, {window}> ema_{idx};"))
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = ema_{idx}.step(v{inp[0]}, i);"))
        elif isinstance(op, WindowedLinearRegression):
            if stream_mode: raise RuntimeError(f"Stream Mode does not support {op.__class__.__name__}")
            assert(op.get_parent() is None)
            buf_name = _get_buffer_name(op.inputs[0], inp[0])
            window = op.attrs["window"]
            toplevel.scope.insert(-1, _CppSingleLine(toplevel, f"WindowedLinearRegression<{elem_type}, {simd_lanes}, {window}> linear_{idx};"))
            scope.scope.append(_CppSingleLine(scope, f"const auto& v{idx} = linear_{idx}.step({buf_name}, v{inp[0]}, i);"))
        elif isinstance(op, Select):
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = Select(v{inp[0]}, v{inp[1]}, v{inp[2]});"))
        else:
            raise RuntimeError("Cannot generate " + str(op))
    return header + str(toplevel)