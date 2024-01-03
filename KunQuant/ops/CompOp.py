from .ReduceOp import ReduceAdd, ReduceArgMax
from KunQuant.Op import OpBase, CompositiveOp, WindowedTrait, ForeachBackWindow, WindowedTempOutput, Builder
from .ElewiseOp import DivConst, Sub, Mul, Sqrt, SubConst
from collections import OrderedDict
from typing import Union, List, Tuple

class WindowedCompositiveOp(CompositiveOp, WindowedTrait):
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

class WindowedSum(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            v2 = ReduceAdd(v1)
        return b.ops

class WindowedAvg(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedSum(self.inputs[0], self.attrs["window"])
            v1 = DivConst(v0, self.attrs["window"])
        return b.ops

class WindowedStddev(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        window = self.attrs["window"]
        b = Builder(self.get_parent())
        with b:
            avg = WindowedAvg(self.inputs[0], window)
            v0 = WindowedTempOutput(self.inputs[0], window)
            each = ForeachBackWindow(v0, window)
            b.set_loop(each)
            diff = Sub(each, avg)
            sqr = Mul(diff, diff)
            b.set_loop(self.get_parent())
            vsum = ReduceAdd(sqr)
            out = Sqrt(DivConst(vsum, window - 1))
        return b.ops
    
class TsArgMax(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            v2 = ReduceArgMax(v1)
            v3 = SubConst(v2, self.attrs["window"] - 1, True)
        return b.ops