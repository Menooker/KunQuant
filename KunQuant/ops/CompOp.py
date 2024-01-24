from .ReduceOp import ReduceAdd, ReduceArgMax, ReduceRank, ReduceMin, ReduceMax, ReduceDecayLinear
from KunQuant.Op import ConstantOp, OpBase, CompositiveOp, WindowedTrait, ForeachBackWindow, WindowedTempOutput, Builder, IterValue
from .ElewiseOp import And, DivConst, GreaterThan, LessThan, Or, Select, SetInfOrNanToValue, Sub, Mul, Sqrt, SubConst, Div, CmpOp, Exp, Log
from collections import OrderedDict
from typing import Union, List, Tuple
import math

class WindowedCompositiveOp(CompositiveOp, WindowedTrait):
    def __init__(self, v: OpBase, window: int, v2 = None) -> None:
        inputs = [v]
        if v2 is not None:
            inputs.append(v2)
        super().__init__(inputs, [("window", window)])

class WindowedReduce(WindowedCompositiveOp):
    def make_reduce(self, v: OpBase) -> OpBase:
        raise RuntimeError("Not implemented")
    
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            itr = IterValue(v1, v0)
            v2 = self.make_reduce(itr)
        return b.ops

class WindowedSum(WindowedReduce):
    def make_reduce(self, v: OpBase) -> OpBase:
        return ReduceAdd(v)
    
class WindowedMin(WindowedReduce):
    def make_reduce(self, v: OpBase) -> OpBase:
        return ReduceMin(v)
    
class WindowedMax(WindowedReduce):
    def make_reduce(self, v: OpBase) -> OpBase:
        return ReduceMax(v)

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

class WindowedCovariance(WindowedCompositiveOp):
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
            xy = Mul(diffX, diffY)
            b.set_loop(self.get_parent())
            vsum1 = ReduceAdd(xy) / (window-1)
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

class ClipZero(CompositiveOp):
    def __init__(self, v: OpBase, eps: float) -> None:
        inputs = [v]
        super().__init__(inputs, [("value", eps)])

    def decompose(self) -> List[OpBase]:
        eps = self.attrs["value"]
        inp = self.inputs[0]
        b = Builder(self.get_parent())
        with b:
            v0 = LessThan(inp, ConstantOp(eps))
            v1 = GreaterThan(inp, ConstantOp(-eps))
            out = Select(And(v0, v1), ConstantOp(0), inp)
        return b.ops

class DecayLinear(WindowedCompositiveOp):
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        window = self.attrs["window"]
        with b:
            v0 = WindowedTempOutput(self.inputs[0], window)
            v1 = ForeachBackWindow(v0, window)
            v2 = ReduceDecayLinear(IterValue(v1, v0), None, [("window", window)])
        return b.ops
    
class Pow(CompositiveOp):
    def __init__(self, base: OpBase, expo: OpBase) -> None:
        inputs = [base, expo]
        super().__init__(inputs, None)

    def decompose(self) -> List[OpBase]:
        # pow(x,y) = exp(log(x)*y)
        b = Builder(self.get_parent())
        (base, expo) = self.inputs
        if isinstance(base, ConstantOp):
            basev = base.attrs["value"]
            ln_base = math.log(basev)
            if abs(ln_base-0) < 1e-5:
                with b:
                    ConstantOp(1)
                return b.ops
            elif abs(ln_base-1) < 1e-5:
                with b:
                    Exp(expo)
                return b.ops
            else:
                with b:
                    Exp(expo * ln_base)
                    return b.ops
        if isinstance(expo, ConstantOp):
            expov = expo.attrs["value"]
            if expov == 0.5:
                with b:
                    Sqrt(base)
                    return b.ops
            elif int(expov) == expov and expov <= 1024 and expov >= 0:
                # pow(x, 5) >>>>  x1=x*x x2=x1*x2 out=x2*x
                with b:
                    cur_mul = base
                    cur_exp = 1
                    remain = int(expov)
                    is_set = remain & cur_exp
                    if is_set:
                        curv = base
                    else:
                        curv = 1
                    remain = remain & ~cur_exp
                    while remain:
                        cur_exp *= 2
                        cur_mul = cur_mul * cur_mul
                        is_set = remain & cur_exp
                        if is_set:
                            curv = curv * cur_mul
                        remain = remain & ~cur_exp
                return b.ops
            else:
                with b:
                    Exp(Log(base) * expov)
                    return b.ops
        with b:
            Exp(expo * Log(base))
            return b.ops
