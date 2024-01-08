from .ReduceOp import ReduceAdd, ReduceArgMax, ReduceRank
from KunQuant.Op import OpBase, CompositiveOp, WindowedTrait, ForeachBackWindow, WindowedTempOutput, Builder, IterValue
from .ElewiseOp import DivConst, Sub, Mul, Sqrt, SubConst, Div, CmpOp
from collections import OrderedDict
from typing import Union, List, Tuple

class WindowedCompositiveOp(CompositiveOp, WindowedTrait):
    def __init__(self, v: OpBase, window: int, v2 = None) -> None:
        inputs = [v]
        if v2 is not None:
            inputs.append(v2)
        super().__init__(inputs, [("window", window)])

class WindowedSum(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            itr = IterValue(v1, v0)
            v2 = ReduceAdd(itr)
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
            diff = Sub(IterValue(each, v0), avg)
            sqr = Mul(diff, diff)
            b.set_loop(self.get_parent())
            vsum = ReduceAdd(sqr)
            out = Sqrt(DivConst(vsum, window - 1))
        return b.ops
    
class WindowedCorrelation(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        window = self.attrs["window"]
        x = self.inputs[0]
        y = self.inputs[1]
        b = Builder(self.get_parent())
        with b:
            avgX = WindowedAvg(x, window)
            avgY = WindowedAvg(y, window)
            wX = WindowedTempOutput(x, window)
            wY = WindowedTempOutput(y, window)
            each = ForeachBackWindow(wX, window, wY)
            b.set_loop(each)
            diffX = Sub(IterValue(each, wX), avgX)
            diffY = Sub(IterValue(each, wY), avgY)
            sqrX = Mul(diffX, diffX)
            sqrY = Mul(diffY, diffY)
            xy = Mul(diffX, diffY)
            b.set_loop(self.get_parent())
            vsum1 = ReduceAdd(xy)
            vsum_x = Sqrt(ReduceAdd(sqrX))
            vsum_y = Sqrt(ReduceAdd(sqrY))
            sum_xy = Mul(vsum_x, vsum_y)
            out = Div(vsum1, sum_xy)
        return b.ops
class TsArgMax(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            v2 = ReduceArgMax(IterValue(v1, v0))
            v3 = SubConst(v2, self.attrs["window"], True)
        return b.ops

class TsRank(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            v2 = ReduceRank(IterValue(v1, v0), self.inputs[0])
        return b.ops