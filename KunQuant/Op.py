from os import path
from typing import List, Dict, Literal, Optional, Tuple, Union
from abc import ABC, abstractmethod
from collections import OrderedDict
import threading
import KunQuant
import zlib


class _builder(threading.local):
    ops: List['OpBase']
    loop: 'ForeachBackWindow'

    def __init__(self):
        self.ops = None
        self.loop = None


_tls = _builder()


def append_current_builder(op: 'OpBase') -> None:
    if _tls.ops is None:
        return
    _tls.ops.append(op)


class Builder:
    def __init__(self, loop=None) -> None:
        self.ops = []
        self.loop = loop
        self.old_ops = None
        self.old_loop = None

    def set_loop(self, loop) -> None:
        self.loop = loop
        _tls.loop = loop

    def __enter__(self):
        self.old_ops = _tls.ops
        self.old_loop = _tls.loop
        _tls.ops = self.ops
        _tls.loop = self.loop

    def __exit__(self, exception_type, exception_value, exception_traceback):
        _tls.ops = self.old_ops
        _tls.loop = self.old_loop


def make_indents(indent: int, spaces=2) -> str:
    return " "*(indent*spaces)


def print_attrs(attr: OrderedDict) -> str:
    inner = ",".join([f"{kv[0]}:{kv[1]}" for kv in attr.items()])
    if not inner:
        return inner
    return f"{{{inner}}}"


_clazzBackWindow = None
_clazzWindowedTrait = None
_empty_dict = {}

def _hash_combine(x: int, y: int) -> int:
    return (x * 23 + y + 1 + (x>>16) + (y>>16)*7) % (2 << 64)

def traverse_replace_map(op: 'OpBase', replace_map: Dict['OpBase', 'OpBase']) -> 'OpBase':
    found = replace_map.get(op, None)
    if not found:
        return op
    if found == op:
        return op
    return traverse_replace_map(found, replace_map)

class AcceptSingleValueInputTrait(ABC):
    @abstractmethod
    def get_single_value_input_id() -> int:
        pass

class OpBase:
    def __init__(self, inputs: List['OpBase'], attrs: Union[List[Tuple[str, object]], OrderedDict, None]) -> None:
        for i in inputs:
            if not isinstance(i, OpBase):
                raise RuntimeError("Bad inputs, given " + str(type(i)))
        self.inputs = inputs
        self._parent_loop: 'ForeachBackWindow' = _tls.loop
        if attrs is not None:
            self.attrs = OrderedDict(attrs)
        else:
            self.attrs = OrderedDict()
        append_current_builder(self)

    def replace_inputs(self, replace_map: Dict['OpBase', 'OpBase']):
        for idx, inp in enumerate(self.inputs):
            self.inputs[idx] = traverse_replace_map(inp, replace_map)
        if self._parent_loop in replace_map:
            self._parent_loop = traverse_replace_map(
                self._parent_loop, replace_map)

    def set_parent(self, loop: 'ForeachBackWindow') -> None:
        if loop is not None and not isinstance(loop, _clazzBackWindow):
            raise RuntimeError("set_parent failed")
        self._parent_loop = loop

    def get_parent(self) -> 'ForeachBackWindow':
        return self._parent_loop

    def attrs_str(self) -> str:
        return print_attrs(self.attrs)

    def fast_hash(self, cache: Optional[Dict['OpBase', int]] = None, **kwargs) -> int:
        if cache is not None:
            c = cache.get(self)
            if c is not None:
                return c
        ret = zlib.adler32(self.__class__.__name__.encode())
        attr_data = print_attrs(self.attrs).encode()
        ret = _hash_combine(ret, (zlib.adler32(attr_data) << 32) + zlib.crc32(attr_data))
        ret = _hash_combine(ret, 114514)
        for arg, subkwargs in self.get_args(True, **kwargs):
            ret = _hash_combine(ret, arg.fast_hash(cache, **subkwargs))
        ret = _hash_combine(ret, 1818910)
        if cache is not None:
            cache[self] = ret
        return ret
    
    def items(self, **kwargs):
        '''
        returns an generator expr for all contents of the op and its dependencies
        '''
        yield self.__class__
        yield self.attrs
        yield "["
        for arg, subkwargs in self.get_args(True, **kwargs):
            yield from arg.items(**subkwargs)
        yield "]"

    def get_args(self, identity: bool, **kwargs) -> List[Tuple['OpBase', dict]]:
        assert(len(kwargs) == 0)
        return [(v, _empty_dict) for v in self.inputs]

    def to_string(self, indent: int, identity: bool, **kwargs) -> str:
        selfname = self.__class__.__name__
        indents = make_indents(indent)
        args = [arg.to_string(indent+1, identity, **subkwargs) for arg, subkwargs in self.get_args(identity, **kwargs)]
        args = ",\n".join(args)
        return f'''{indents}{selfname}@{print_attrs(self.attrs)}(
{args}
{indents})'''

    def __str__(self) -> str:
        return self.to_string(0, False)

    def fast_str(self) -> str:
        selfname = self.__class__.__name__
        args = ",".join([str(id(v)) for v in self.inputs])
        return f"{selfname}@{id(self)}@{print_attrs(self.attrs)}({args})"

    def hash_hex(self, cache: Optional[Dict['OpBase', int]] = None) -> str:
        out = self.fast_hash(cache)
        return f"{out:016x}"

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        if isinstance(self, _clazzWindowedTrait):
            _clazzWindowedTrait.verify(self, func)
        allowed_single_value = -1
        if isinstance(self, AcceptSingleValueInputTrait):
            allowed_single_value = self.get_single_value_input_id()
        for idx, inp in enumerate(self.inputs):
            if isinstance(inp, Input):
                if allowed_single_value != idx and inp.attrs.get("single_value", False):
                    raise RuntimeError("verify() failed: using single value input: " + str(self))
            # if an input is in a loop, we must be in it too
            loop = inp.get_parent()
            if loop is not None:
                if loop != self._parent_loop:
                    raise RuntimeError(
                        "verify() failed: referencing cross loop results: " + str(self))

    def _build_op(self, other, TBinary, TConst, isreverse):
        if type(other) in [int, float]:
            if isreverse:
                return TConst(self, other, [("swap", True)])
            return TConst(self, other)
        if isinstance(other, OpBase):
            if isreverse:
                return TBinary(other, self)
            return TBinary(self, other)
        raise RuntimeError("Don't know how to build binary " + str(type(other)))

    def _build_op2(self, other, TBinary):
        if type(other) in [int, float]:
            other = ConstantOp(other)
        if isinstance(other, OpBase):
            return TBinary(self, other)
        raise RuntimeError("Don't know how to build binary " + str(type(other)))

    def __sub__(self, other):
        from KunQuant.ops.ElewiseOp import SubConst, Sub
        return self._build_op(other, Sub, SubConst, False)

    def __add__(self, other):
        from KunQuant.ops.ElewiseOp import Add, AddConst
        return self._build_op(other, Add, AddConst, False)

    def __radd__(self, other):
        from KunQuant.ops.ElewiseOp import Add, AddConst
        return self._build_op(other, Add, AddConst, False)

    def __rsub__(self, other):
        from KunQuant.ops.ElewiseOp import Sub, SubConst
        return self._build_op(other, Sub, SubConst, True)

    def __rmul__(self, other):
        from KunQuant.ops.ElewiseOp import Mul, MulConst
        return self._build_op(other, Mul, MulConst, True)

    def __mul__(self, other):
        from KunQuant.ops.ElewiseOp import Mul, MulConst
        return self._build_op(other, Mul, MulConst, False)
    
    def __truediv__ (self, other):
        from KunQuant.ops.ElewiseOp import Div, DivConst
        return self._build_op(other, Div, DivConst, False)
    
    def __rtruediv__ (self, other):
        from KunQuant.ops.ElewiseOp import Div, DivConst
        return self._build_op(other, Div, DivConst, True)

    def __lt__(self, other):
        from KunQuant.ops.ElewiseOp import LessThan, LessThanConst
        return self._build_op(other, LessThan, LessThanConst, False)

    def __ge__(self, other):
        from KunQuant.ops.ElewiseOp import GreaterEqual
        return self._build_op2(other, GreaterEqual)
    
    def __gt__(self, other):
        from KunQuant.ops.ElewiseOp import GreaterThan
        return self._build_op2(other, GreaterThan)

    def __le__(self, other):
        from KunQuant.ops.ElewiseOp import LessEqual
        return self._build_op2(other, LessEqual)
    
    def __or__(self, other):
        from KunQuant.ops.ElewiseOp import Or
        return self._build_op2(other, Or)

    def __invert__(self):
        from KunQuant.ops.ElewiseOp import Not
        return Not(self)

    def __and__(self, other):
        from KunQuant.ops.ElewiseOp import And
        return self._build_op2(other, And) 

class GraphSourceTrait:
    '''
    The "source" of a graph, like input and constant ops. They have no inputs.
    '''
    pass

class ConstantOp(OpBase, GraphSourceTrait):
    def __init__(self, v: Union[float, Literal['nan']]) -> None:
        super().__init__([], [("value", v)])
        self.check()

    def check(self):
        if self.attrs['value'] != self.attrs['value']:
            raise RuntimeError("ConstantOp value cannot be float('nan'), use 'nan' string literal instead")
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        self.check()
        return super().verify(func)

class UnaryElementwiseOp(OpBase):
    def __init__(self, lhs: OpBase, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([lhs], attrs)


class BinaryElementwiseOp(OpBase):
    def __init__(self, lhs: OpBase,  rhs: OpBase, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([lhs, rhs], attrs)


class CompositiveOp(OpBase):
    @abstractmethod
    def decompose(self, options: dict) -> List[OpBase]:
        pass


class WindowedDataSourceOp(OpBase):
    '''
    The ops that can be an input of WindowedTrait. It provides a window of data
    '''
    pass


class Input(WindowedDataSourceOp, GraphSourceTrait):
    def __init__(self, name: str) -> None:
        super().__init__([], [("name", name)])

    def to_string(self, indent: int, identity: bool) -> str:
        return "{}input({})".format(make_indents(indent), self.attrs["name"])

class SinkOpTrait:
    '''
    The "sink" of a graph, like "output" op. Should keep ops extending this class even if no reference to these ops
    '''
    pass


class Output(WindowedDataSourceOp, SinkOpTrait):
    def __init__(self, inp: OpBase, name: str = "") -> None:
        super().__init__([inp], [("name", name)])


class WindowedTempOutput(WindowedDataSourceOp):
    '''
    Mark that we need a windowed buffer of previous data of the input
    '''
    def __init__(self, inp: OpBase, window: int) -> None:
        super().__init__([inp], [("window", window)])


class WindowedTrait:
    '''
    The ops that require a window of inputs of previous data.
    '''
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        if not func.strict_window:
            return
        for inp in self.inputs:
            if not isinstance(inp, WindowedDataSourceOp):
                raise RuntimeError(
                    "bad input for this op. The input should be windowed: " + str(inp) + "\n This op = " + str(self))
            if not isinstance(inp, Input) and not isinstance(inp, Output) and inp.attrs["window"] < self.required_input_window():
                raise RuntimeError(
                    "bad input for this op. The window does not match: " + str(inp) + "\n This op = " + str(self))
    def required_input_window(self) -> int:
        return self.attrs["window"]

_clazzWindowedTrait = WindowedTrait


class ForeachBackWindow(OpBase, WindowedTrait):
    '''
    A for-loop to iterate the input ops (must be windowed inputs) and reduce outputs
    inp: A windowed input
    window: for-loop length in window size
    args: optional other windowed inputs to iterate
    '''
    def __init__(self, inp: WindowedTrait, window: int, *args) -> None:
        inputs = [inp]
        if args:
            inputs.extend(args)
        super().__init__(inputs, [("window", window)])

    def get_args(self, identity: bool, **kwargs) -> List[Tuple['OpBase', bool, dict]]:
        if len(kwargs) == 0:
            return super().get_args(identity, **kwargs)
        assert(kwargs["display"] in self.inputs)
        return [(kwargs["display"], _empty_dict)]

    def __enter__(self):
        assert(_tls.loop is None)
        _tls.loop = self
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        assert(_tls.loop == self)
        _tls.loop = None


class IterValue(OpBase):
    '''
    Gets the current iteration value of the ForeachBackWindow
    loop: the loop
    src: the specific input of the loop to iterate. For example,
    itr = ForeachBackWindow(X, window = 10, Y)
    xItr = IterValue(itr, X) # the current value of X in the window in this iteration
    yItr = IterValue(itr, Y) # the current value of Y in the window in this iteration
    '''
    def __init__(self, loop: ForeachBackWindow, src: OpBase) -> None:
        super().__init__([loop, src], None)
        self.set_parent(loop)

    def get_args(self, identity: bool, **kwargs) -> str:
        assert (len(self.inputs) == 2)
        if not identity:
            return super().get_args(identity, **kwargs)
        # tell input[0] (the loop) only print self.input[1]
        return [(self.inputs[0], {'display':self.inputs[1]}), (self.inputs[1], _empty_dict)]

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        super().verify(func)
        if not isinstance(self.inputs[0], ForeachBackWindow):
            raise RuntimeError("Bad IterValue: " + str(self))


class WindowLoopIndex(OpBase):
    '''
    Get the current index of the ForEachWindow loop, starting from 0 to window-1. 0 for the oldest data
    and window-1 for the latest data
    '''
    def __init__(self, forwindow: ForeachBackWindow) -> None:
        super().__init__([forwindow], [])
        self.set_parent(forwindow)

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        super().verify(func)
        if not isinstance(self.inputs[0], ForeachBackWindow):
            raise RuntimeError("Bad WindowLoopIndex, input must be a forloop: " + str(self))


_clazzBackWindow = ForeachBackWindow

class BoolOpTrait:
    '''
    The ops that have boolean output type
    '''
    pass


class GenericCppCodegenTrait:
    '''
    The interface for generating C++ code for the op
    '''
    def get_func_or_class_template_name(self) -> str:
        '''
        return the function or state class template name for the op
        '''
        return f"{self.__class__.__name__}"


    def get_func_or_class_full_name(self, elem_type: str, simd_lanes: int) -> str:
        '''
        return the full function name or state class name for the op, with the template parameters
        '''
        if "window" in self.attrs:
            return f"{self.get_func_or_class_template_name()}<{elem_type}, {simd_lanes}, {self.attrs['window']}>"
        return f"{self.get_func_or_class_template_name()}<{elem_type}, {simd_lanes}>"
    
    def generate_step_code(self, idx: str, time_idx: str, inputs: List[str], **kwargs) -> str:
        '''
        generate the code for the step of the op
        idx: the output variable name index
        time_idx: the time index variable name, e.g. "i"
        inputs: the input variables of the op
        kwargs: additional arguments, e.g. buf_name for WindowedTrait
        '''
        raise NotImplementedError(f"generate_step_code not implemented, op = {self.__class__.__name__}")


class StatefulOpTrait(GenericCppCodegenTrait):
    '''
    The ops that have an internal state
    '''
    def get_state_variable_name_prefix(self) -> str:
        '''
        return the prefix of the state variable name, for better readability
        '''
        return "v"
    def generate_init_code(self, idx: str, elem_type: str, simd_lanes: int, inputs: List[str], aligned: bool) -> str:
        '''
        generate the code for the initialization of the state variable
        idx: the output variable name index
        elem_type: the element type of the state variable
        simd_lanes: SIMD lanes
        inputs: the input variables of the op
        '''
        return f"{self.get_func_or_class_full_name(elem_type, simd_lanes)} {self.get_state_variable_name_prefix()}{idx};"
    

class GloablStatefulOpTrait(StatefulOpTrait):
    '''
    The ops that have an internal state, and the state is carried between different time steps
    '''
    pass

class ReductionOp(OpBase, StatefulOpTrait):
    '''
    Base class of all reduction ops. A reduction op takes inputs that is originated from a IterValue. The input must be in a loop (v.get_parent() is a loop). The data produced
    by a ReductionOp should be used outside of the loop
    '''
    def __init__(self, v: OpBase, init_val: OpBase = None, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([v] if init_val is None else [v, init_val], attrs)
    def get_loop(self) -> ForeachBackWindow:
        inp = self.inputs[0]
        # The inputs must be in a loop. we must be in a parent of it
        loop = inp.get_parent()
        # if directly using a for-each var, the loop is itself
        if isinstance(inp, ForeachBackWindow):
            loop = inp
        return loop
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        assert(self.inputs.__len__() <= 2)
        loop = self.get_loop()
        if loop is None or loop == self._parent_loop:
            raise RuntimeError(
                f"verify() failed: ReductionOp using non-loop result: {self}\nself._parent_loop = {self._parent_loop}\nloop = {loop}")
        if self._parent_loop != loop.get_parent():
            raise RuntimeError(
                f"verify() failed: ReductionOp not in parent of input: {self}\nself._parent_loop = {self._parent_loop}\nloop.get_parent() = {loop.get_parent()}")
    
    def generate_init_code(self, idx: str, elem_type: str, simd_lanes: int, inputs: List[str], aligned: bool) -> str:
        init_val = "" if len(self.inputs) == 1 else inputs[1]
        return f"{self.get_func_or_class_full_name(elem_type, simd_lanes)} {self.get_state_variable_name_prefix()}{idx}{{{init_val}}};"
    
    def generate_step_code(self, idx: str, time_idx: str, inputs: List[str]) -> str:
        return f"{self.get_state_variable_name_prefix()}{idx}.step({inputs[0]}, {time_idx});"
    
class CrossSectionalOp(OpBase):
    pass

class SimpleCrossSectionalOp(CrossSectionalOp):
    '''
    The cross sectional ops that are implemented in pure C++
    '''
    def __init__(self, v: OpBase, attrs=None) -> None:
        super().__init__([v], attrs)

class Rank(SimpleCrossSectionalOp):
    '''
    the cross sectional rank among different stocks. Between [0, 1]
    Similar to df.rank(axis=1, pct=True, method="average")
    '''
    pass

class Scale(SimpleCrossSectionalOp):
    '''
    the cross sectionally scale different stocks, to make sum([abs(stock[i]) for i in stock]) == 1
    Similar to df.div(df.abs().sum(axis=1), axis=0)
    '''
    pass

# if __name__ == "__main__":
#     inp1 = Input("a")
#     inp2 = Input("b")
#     v1 = Mul(inp1, inp2)
#     v2 = AddConst(v1, 10)
#     out = Output(v2)
#     print(out)
