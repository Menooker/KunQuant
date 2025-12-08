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

def _float_value_to_float(v: Union[float, str], dtype: str) -> str:
    ret = str(v)
    if 'inf' == ret:
        return f"std::numeric_limits<{dtype}>::infinity()"
    if '-inf' == ret:
        return f"-std::numeric_limits<{dtype}>::infinity()"
    if 'nan' in ret.lower():
        return f"std::numeric_limits<{dtype}>::quiet_NaN()"
    if '.' not in ret and 'e' not in ret:
        ret += '.'
    if dtype == "float":
        ret += 'f'
    return ret

def _value_to_float(op: OpBase, dtype: str) -> str:
    return _float_value_to_float(op.attrs["value"], dtype)

def _is_cross_sectional(f: Function) -> GenericCrossSectionalOp:
    for op in f.ops:
        if isinstance(op, Input) or isinstance(op, Output):
            continue
        if isinstance(op, GenericCrossSectionalOp):
            return op
    return None

def _generate_cross_sectional_func_name(op: GenericCrossSectionalOp, inputs: List[Tuple[Input, bool]], outputs: List[Tuple[Output, bool]]) -> str:
    name = []
    for idx, (inp, buf_kind) in enumerate(inputs):
        layout = inp.attrs["layout"]
        name.append(layout)
    for idx, (outp, is_tmp) in enumerate(outputs):
        layout = outp.attrs["layout"]
        name.append(layout)
    return f"{op.__class__.__name__}_{'_'.join(name)}"

def codegen_cpp(prefix: str, f: Function, input_name_to_idx: Dict[str, int], inputs: List[Tuple[Input, bool]], outputs: List[Tuple[Output, bool]], options: dict, stream_mode: bool, query_temp_buffer_id, stream_window_size: Dict[str, int], generated_cross_sectional_func: Set[str], elem_type: str, simd_lanes: int, aligned: bool, static: bool) -> Tuple[str, str]:
    if len(f.ops) == 3 and isinstance(f.ops[1], SimpleCrossSectionalOp):
        return "", f'''static auto stage_{prefix}__{f.name} = {f.ops[1].__class__.__name__}Stocks<Mapper{f.ops[0].attrs["layout"]}<{elem_type}, {simd_lanes}>, Mapper{f.ops[2].attrs["layout"]}<{elem_type}, {simd_lanes}>>;'''
    
    is_cross_sectional = _is_cross_sectional(f)
    time_or_stock, ctx_or_stage = ("__time_idx", "RuntimeStage *stage") if is_cross_sectional else ("__stock_idx", "Context* __ctx")
    func_name = _generate_cross_sectional_func_name(is_cross_sectional, inputs, outputs) if is_cross_sectional else f.name
    header = f'''{"static " if static else ""}void stage_{prefix}__{func_name}({ctx_or_stage}, size_t {time_or_stock}, size_t __total_time, size_t __start, size_t __length) '''
    if static:
        decl = ""
    else:
        decl = f'''extern void stage_{prefix}__{func_name}({ctx_or_stage}, size_t {time_or_stock}, size_t __total_time, size_t __start, size_t __length);'''
    if is_cross_sectional:
        decl = f"{decl}\nstatic auto stage_{prefix}__{f.name} = stage_{prefix}__{func_name};"
        if func_name in generated_cross_sectional_func:
            return "", decl
        generated_cross_sectional_func.add(func_name)
        lines = []
        for idx, (inp, buf_kind) in enumerate(inputs):
            name = inp.attrs["name"]
            layout = inp.attrs["layout"]
            holder = f"{make_indents(1)}CrossSectionalDataHolder<Mapper{layout}<{elem_type}, {simd_lanes}>, ExtractInputBuffer> holder_input_{idx}{{stage, {idx}, __total_time, __start}};"
            lines.append(holder)
        for idx, (outp, is_tmp) in enumerate(outputs):
            name = outp.attrs["name"]
            layout = outp.attrs["layout"]
            holder = f"{make_indents(1)}CrossSectionalDataHolder<Mapper{layout}<{elem_type}, {simd_lanes}>, ExtractOutputBuffer> holder_output_{idx}{{stage, {idx}, __total_time, __start}};"
            lines.append(holder)
        lines.append(f'{make_indents(1)}auto time_end = std::min(__start + ({time_or_stock} + 1) * time_stride, __start + __length);')
        lines.append(f'{make_indents(1)}auto num_stocks = stage->ctx->stock_count;')      
        lines.append(f'{make_indents(1)}using T = {elem_type};')
        lines.append(is_cross_sectional.generate_head())
        lines.append(f'{make_indents(1)}for (size_t t = __start + ({time_or_stock}) * time_stride; t < time_end; t++) {{')
        for idx, (inp, buf_kind) in enumerate(inputs):
            lines.append(f'{make_indents(2)}auto input_{idx} = holder_input_{idx}.accessor(t);')
        for idx, (outp, is_tmp) in enumerate(outputs):
            lines.append(f'{make_indents(2)}auto output_{idx} = holder_output_{idx}.accessor(t);')

        lines.append(is_cross_sectional.generate_body())
        lines.append(f'{make_indents(1)}}}')
        src = "\n".join(lines)
        return f'''{header} {{
{src}
}}''', decl

    toplevel = _CppScope(None)
    buffer_type: Dict[OpBase, str] = dict()
    ptrname = "" if elem_type == "float" else "D"
    for inp, buf_kind in inputs:
        name = inp.attrs["name"]
        layout = inp.attrs["layout"]
        idx_in_ctx = input_name_to_idx[name]
        # if is user input, the time should start from __start and length of time is __total_time
        not_user_input = buf_kind != "INPUT" or inp.attrs.get("single_value", False)
        start_str = "0" if not_user_input else "__start"
        total_str = "__length" if not_user_input else "__total_time"
        if stream_mode:
            window_size = stream_window_size.get(name, 1)
            buffer_type[inp] = f"StreamWindow<{elem_type}, {simd_lanes}, {window_size}>"
            code = f"StreamWindow<{elem_type}, {simd_lanes}, {window_size}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].stream_buf{ptrname}, __stock_idx, __ctx->stock_count}};"
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
            code = f"StreamWindow<{elem_type}, {simd_lanes}, {window_size}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].stream_buf{ptrname}, __stock_idx, __ctx->stock_count}};"
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
                code = f"StreamWindow<{elem_type}, {simd_lanes}, {window}> temp_{idx}{{__ctx->buffers[{query_temp_buffer_id(bufname, window)}].stream_buf{ptrname}, __stock_idx, __ctx->stock_count}};"
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
            str_i = "i"
            if op.attrs.get("single_value", False):
                str_i = '0'
            name = op.attrs["name"]
            if not aligned and op.attrs["layout"] == "TS":
                mask_str = ", mask"
            else:
                mask_str = ""
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = buf_{name}.step({str_i}{mask_str});"))
        elif isinstance(op, Output):
            name = op.attrs["name"]
            if not aligned and op.attrs["layout"] == "TS":
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
        elif isinstance(op, WindowLoopIndex):
            loop = op.inputs[0]
            scope.scope.append(_CppSingleLine(scope, f' kun_simd::vec<{elem_type}, {simd_lanes}> v{idx} {{ {elem_type}({window} - 1 - iter) }};'))
        elif isinstance(op, ReductionOp):
            loop_op = op.inputs[0] if isinstance(op.inputs[0], ForeachBackWindow) else op.inputs[0].get_parent()
            loop_body = loop_to_cpp_loop[loop_op]
            loop = loop_body.parent_for
            # insert a var definition before the for-loop
            loop_parent = loop.parent
            assert(isinstance(loop_parent, _CppScope))
            vargs = [f"v{inpv}" for inpv in inp]
            loop_parent.scope.insert(loop_parent.scope.index(loop), _CppSingleLine(loop_parent, op.generate_init_code(idx, elem_type, simd_lanes, vargs, aligned)))
            # insert a step in the for-loop
            loop_body.scope.append(_CppSingleLine(loop_body, op.generate_step_code(idx, "iter", vargs)))
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
        elif isinstance(op, GloablStatefulOpTrait):
            if stream_mode: raise RuntimeError(f"Stream Mode does not support {op.__class__.__name__}")
            assert(op.get_parent() is None)
            args = {}
            if isinstance(op, WindowedTrait):
                buf_name = _get_buffer_name(op.inputs[0], inp[0])
                args["buf_name"] = buf_name
            vargs = [f"v{inpv}" for inpv in inp]
            toplevel.scope.insert(-1, _CppSingleLine(toplevel, op.generate_init_code(idx, elem_type, simd_lanes, vargs, aligned)))
            scope.scope.append(_CppSingleLine(scope, op.generate_step_code(idx, "i", vargs, **args)))
        elif isinstance(op, Select):
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = Select(v{inp[0]}, v{inp[1]}, v{inp[2]});"))
        elif isinstance(op, SetAccumulator):
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = SetAccumulator(v{inp[0]}, v{inp[1]}, v{inp[2]});"))
        elif isinstance(op, ReturnFirstValue):
            scope.scope.append(_CppSingleLine(scope, f"auto& v{idx} = v{inp[0]};"))
        else:
            raise RuntimeError(f"Cannot generate {op} of function {f}")
    return header + str(toplevel), decl