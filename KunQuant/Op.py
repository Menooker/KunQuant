from typing import List, Dict, Tuple, Union
from abc import ABC, abstractmethod
from collections import OrderedDict
import threading
import KunQuant

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
    def __init__(self, loop = None) -> None:
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
    

def make_indents(indent: int, spaces = 2) -> str:
    return " "*(indent*spaces)

def print_attrs(attr: OrderedDict) -> str:
    inner = ",".join([f"{kv[0]}:{kv[1]}" for kv in attr.items()])
    if not inner:
        return inner
    return f"{{{inner}}}"

_clazzBackWindow = None
_clazzWindowedTrait = None

def traverse_replace_map(op: 'OpBase', replace_map: Dict['OpBase', 'OpBase']) -> 'OpBase':
    found = replace_map.get(op, None)
    if not found:
        return op
    if found == op:
        return op
    return traverse_replace_map(found, replace_map)

class OpBase:
    def __init__(self, inputs: List['OpBase'], attrs: Union[List[Tuple[str, object]], OrderedDict, None]) -> None:
        self.inputs = inputs
        self._parent_loop: 'ForeachBackWindow' = _tls.loop
        if attrs is not None:
            self.attrs =  OrderedDict(attrs)
        else:
            self.attrs =  OrderedDict()
        append_current_builder(self)

    def replace_inputs(self, replace_map: Dict['OpBase', 'OpBase']):
        for idx, inp in enumerate(self.inputs):
            self.inputs[idx] = traverse_replace_map(inp, replace_map)
        if self._parent_loop in replace_map:
            self._parent_loop = traverse_replace_map(self._parent_loop, replace_map)

    def set_parent(self, loop: 'ForeachBackWindow') -> None:
        if loop is not None and not isinstance(loop, _clazzBackWindow):
            raise RuntimeError("set_parent failed")
        self._parent_loop = loop

    def get_parent(self) -> 'ForeachBackWindow':
        return self._parent_loop

    def attrs_str(self) -> str:
        return print_attrs(self.attrs)

    def to_string(self, indent: int) -> str:
        selfname = self.__class__.__name__
        indents = make_indents(indent)
        args = f",\n".join([v.to_string(indent+1) for v in self.inputs])
        return f'''{indents}{selfname}@{print_attrs(self.attrs)}(
{args}
{indents})'''

    def __str__(self) -> str:
        return self.to_string(0)

    def fast_str(self) -> str:
        selfname = self.__class__.__name__
        args = ",".join([str(id(v)) for v in self.inputs])
        return f"{selfname}@{id(self)}@{print_attrs(self.attrs)}({args})"
    
    def hash_hex(self) -> str:
        v = str(self)
        out = 0
        for c in v:
            out = (out * 23 + ord(c)) % (2<<32)
        return f"{out:08x}"

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        if isinstance(self, _clazzWindowedTrait):
            _clazzWindowedTrait.verify(self, func)
        for inp in self.inputs:
            # if an input is in a loop, we must be in it too
            loop = inp.get_parent()
            if loop is not None:
                if loop != self._parent_loop:
                    raise RuntimeError("verify() failed: referencing cross loop results: " + str(self))

class UnaryElementwiseOp(OpBase):
    def __init__(self, lhs: OpBase, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([lhs], attrs)

class BinaryElementwiseOp(OpBase):
    def __init__(self, lhs: OpBase,  rhs: OpBase, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([lhs, rhs], attrs)

class CompositiveOp(OpBase):
    @abstractmethod
    def decompose(self) -> List[OpBase]:
        pass

class WindowedDataSourceOp(OpBase):
    pass

class Input(WindowedDataSourceOp):
    def __init__(self, name: str) -> None:
        super().__init__([], [("name", name)])

    def to_string(self, indent: int) -> str:
        return "{}input({})".format(make_indents(indent), self.attrs["name"])

# Should keep Ops extending this class even if no reference to these ops
class SinkOpTrait:
    pass

class Output(WindowedDataSourceOp, SinkOpTrait):
    def __init__(self, inp: OpBase, name: str = "") -> None:
        super().__init__([inp], [("name", name)])

class WindowedTempOutput(WindowedDataSourceOp):
    def __init__(self, inp: OpBase, window: int) -> None:
        super().__init__([inp], [("window", window)])

class WindowedTrait:        
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        if not func.strict_window:
            return
        for inp in self.inputs:
            if not isinstance(inp, WindowedDataSourceOp):
                raise RuntimeError("bad input for this op. The input should be windowed: " + str(inp) + "\n This op = " + str(self))
            if not isinstance(inp, Input) and not isinstance(inp, Output) and inp.attrs["window"] < self.attrs["window"]:
                raise RuntimeError("bad input for this op. The window does not match: " + str(inp) + "\n This op = " + str(self))

_clazzWindowedTrait = WindowedTrait

class ForeachBackWindow(OpBase, WindowedTrait):
    def __init__(self, inp: WindowedTempOutput, window: int, *args) -> None:
        inputs = [inp]
        if args:
            inputs.extend(args)
        super().__init__(inputs, [("window", window)])

class IterValue(OpBase):
    def __init__(self, loop: ForeachBackWindow, src: OpBase) -> None:
        super().__init__([loop, src], None)
        self.set_parent(loop)
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        super().verify(func)
        if not isinstance(self.inputs[0], ForeachBackWindow):
            raise RuntimeError("Bad IterValue: " + str(self))

_clazzBackWindow = ForeachBackWindow

class StatefulOpTrait:
    pass

class ReductionOp(OpBase, StatefulOpTrait):
    def __init__(self, v: OpBase, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([v], attrs)

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        for inp in self.inputs:
            # The inputs must be in a loop. we must be in a parent of it
            loop = inp.get_parent()
            # if directly using a for-each var, the loop is itself
            if isinstance(inp, ForeachBackWindow):
                loop = inp
            if loop is None or loop == self._parent_loop:
                raise RuntimeError(f"verify() failed: ReductionOp using non-loop result: {self}\nself._parent_loop = {self._parent_loop}\nloop = {loop}")
            if self._parent_loop != loop.get_parent():
                raise RuntimeError(f"verify() failed: ReductionOp not in parent of input: {self}\nself._parent_loop = {self._parent_loop}\nloop.get_parent() = {loop.get_parent()}")

# the rank among different stocks. Between [0, 1]
class Rank(OpBase):
    def __init__(self, v: OpBase) -> None:
        super().__init__([v], None)



# if __name__ == "__main__":
#     inp1 = Input("a")
#     inp2 = Input("b")
#     v1 = Mul(inp1, inp2)
#     v2 = AddConst(v1, 10)
#     out = Output(v2)
#     print(out)