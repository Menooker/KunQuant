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

vector_len = 8

def codegen_cpp(f: Function, input_stride: int, output_stride: int, input_name_to_idx: Dict[str, int], inputs: List[Input], outputs: List[Output]) -> str:
    if len(f.ops) == 3 and isinstance(f.ops[1], Rank):
        return f'''static auto stage_{f.name} = RankST8sTimeStride8;'''
    header = f'''void stage_{f.name}(Context* __ctx, size_t __stock_idx, size_t __total_time, size_t __start, size_t __length) '''
    toplevel = _CppScope(None)
    buffer_type: Dict[OpBase, str] = dict()
    # currently only support ST8s format
    assert(input_stride == vector_len)
    for inp in inputs:
        name = inp.attrs["name"]
        idx_in_ctx = input_name_to_idx[name]
        buffer_type[inp] = f"Input<{input_stride}>"
        code = f"Input<{input_stride}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].ptr + __stock_idx * __total_time * {input_stride} +  __start * {input_stride} }};"
        toplevel.scope.append(_CppSingleLine(toplevel, code))
    assert(output_stride == vector_len)
    for idx, outp in enumerate(outputs):
        name = outp.attrs["name"]
        idx_in_ctx = input_name_to_idx[name]
        buffer_type[outp] = f"Output<{output_stride}>"
        code = f"Output<{output_stride}> buf_{name}{{__ctx->buffers[{idx_in_ctx}].ptr + __stock_idx * __length * {input_stride}}};"
        toplevel.scope.append(_CppSingleLine(toplevel, code))
    for op in f.ops:
        if op.get_parent() is None and isinstance(op, WindowedTempOutput):
            window = op.attrs["window"]
            idx = f.get_op_idx(op)
            buffer_type[op] = f"OutputWindow<{window}>"
            code = f"OutputWindow<{window}> temp_{idx}{{}};"
            toplevel.scope.append(_CppSingleLine(toplevel, code))

    top_for = _CppFor(toplevel, "for(size_t i = 0;i < __length;i++) ")
    toplevel.scope.append(top_for)
    top_body = top_for.body
    cur_body = top_body
    loop_to_cpp_loop: Dict[ForeachBackWindow, _CppScope] = {None: top_body}
    for op in f.ops:
        idx = f.get_op_idx(op)
        inp = [f.get_op_idx(inpv) for inpv in op.inputs]
        scope = loop_to_cpp_loop[op.get_parent()]
        if isinstance(op, Input):
            name = op.attrs["name"]
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = buf_{name}.step(i);"))
        elif isinstance(op, Output):
            name = op.attrs["name"]
            scope.scope.append(_CppSingleLine(scope, f"buf_{name}.store(i, v{inp[0]});"))
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = v{inp[0]};"))
        elif isinstance(op, WindowedTempOutput):
            scope.scope.append(_CppSingleLine(scope, f"temp_{idx}.store(i, v{inp[0]});"))
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = v{inp[0]};"))
        elif isinstance(op, ConstantOp):
            scope.scope.append(_CppSingleLine(scope, f'auto v{idx} = constVec({op.attrs["value"]});'))
        elif isinstance(op, BinaryConstOp):
            assert(op.__class__.__name__.endswith("Const"))
            thename = op.__class__.__name__.replace("Const", "")
            rhs = op.attrs["value"]
            if not op.attrs.get("swap", False):
                scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}(v{inp[0]}, {rhs});"))
            else:
                scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}({rhs}, v{inp[0]});"))
        elif isinstance(op, BinaryElementwiseOp):
            thename = op.__class__.__name__
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}(v{inp[0]}, v{inp[1]});"))
        elif isinstance(op, UnaryElementwiseOp):
            thename = op.__class__.__name__
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {thename}(v{inp[0]});"))
        elif isinstance(op, ForeachBackWindow):
            window = op.attrs["window"]
            the_for = _CppFor(scope, f"for(size_t idx_{idx} = 0;idx_{idx} < {window};idx_{idx}++) ")
            scope.scope.append(the_for)
            loop_to_cpp_loop[op] = the_for.body
        elif isinstance(op, IterValue):
            loop = op.inputs[0]
            buf_name = _get_buffer_name(op.inputs[1], inp[1])
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = {buf_name}.getWindow(i, idx_{f.get_op_idx(loop)});"))
        elif isinstance(op, ReductionOp):
            thename = op.__class__.__name__
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
            loop_body.scope.append(_CppSingleLine(loop_body, f"v{idx}.step(v{inp[0]}, idx_{loop_var_idx});"))
        elif isinstance(op, BackRef):
            assert(op.get_parent() is None)
            buf_name = _get_buffer_name(op.inputs[0], inp[0])
            scope.scope.append(_CppSingleLine(scope, f'auto v{idx} = windowedRef<{op.attrs["window"]}>({buf_name}, i);'))
        elif isinstance(op, FastWindowedSum):
            assert(op.get_parent() is None)
            buf_name = _get_buffer_name(op.inputs[0], inp[0])
            window = op.attrs["window"]
            toplevel.scope.insert(-1, _CppSingleLine(toplevel, f"FastWindowedSum<{window}> sum_{idx};"))
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = sum_{idx}.step({buf_name}, v{inp[0]}, i);"))
        elif isinstance(op, Select):
            scope.scope.append(_CppSingleLine(scope, f"auto v{idx} = Select(v{inp[0]}, v{inp[1]}, v{inp[2]});"))
        else:
            raise RuntimeError("Cannot generate " + str(op))
    return header + str(toplevel)