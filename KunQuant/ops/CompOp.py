from .ReduceOp import ReduceAdd, ReduceMul, ReduceArgMax, ReduceRank, ReduceMin, ReduceMax, ReduceDecayLinear, ReduceArgMin
from KunQuant.Op import ConstantOp, OpBase, CompositiveOp, WindowedTrait, ForeachBackWindow, WindowedTempOutput, Builder, IterValue
from .ElewiseOp import And, DivConst, GreaterThan, LessThan, Or, Select, SetInfOrNanToValue, Sub, Mul, Sqrt, SubConst, Div, CmpOp, Exp, Log, Min, Max
from .MiscOp import WindowedLinearRegression, WindowedLinearRegressionResiImpl, WindowedLinearRegressionRSqaureImpl, WindowedLinearRegressionSlopeImpl
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
    '''
    Sum of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).sum()
    '''
    def make_reduce(self, v: OpBase) -> OpBase:
        return ReduceAdd(v)

class WindowedProduct(WindowedReduce):
    '''
    Product of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).product()
    '''
    def make_reduce(self, v: OpBase) -> OpBase:
        return ReduceMul(v)

class WindowedMin(WindowedReduce):
    '''
    Min of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).min()
    '''
    def make_reduce(self, v: OpBase) -> OpBase:
        return ReduceMin(v)
    
class WindowedMax(WindowedReduce):
    '''
    Max of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).max()
    '''
    def make_reduce(self, v: OpBase) -> OpBase:
        return ReduceMax(v)

class WindowedAvg(WindowedCompositiveOp):
    '''
    Average of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).mean()
    '''
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedSum(self.inputs[0], self.attrs["window"])
            v1 = DivConst(v0, self.attrs["window"])
        return b.ops

class WindowedStddev(WindowedCompositiveOp):
    '''
    Unbiased standard deviation of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).std()
    '''
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
    '''
    Unbiased estimated covariance of a rolling look back window of two inputs, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).cov(y)
    '''
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
    '''
    Correlation of a rolling look back window of two inputs, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).corr(y)
    '''
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


class WindowedSkew(WindowedCompositiveOp):
    '''
    Unbiased estimated skewness of a rolling look back window of input, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).skew()
    The bias adjustion factor is math.sqrt(window-1)*window/(window-2)
    '''
    def decompose(self) -> List[OpBase]:
        window = self.attrs["window"]
        x = self.inputs[0]
        b = Builder(self.get_parent())
        with b:
            avgX = WindowedAvg(x, window)
            wX = WindowedTempOutput(x, window)
            each = ForeachBackWindow(wX, window)
            b.set_loop(each)
            diffX = Sub(IterValue(each, wX), avgX)
            x2 = Mul(diffX, diffX)
            x3 = x2 * diffX
            b.set_loop(self.get_parent())
            vsum_x2 = ReduceAdd(x2)
            vsum_x3 = ReduceAdd(x3)
            vsum_x3 / (vsum_x2 * Sqrt(vsum_x2)) * ConstantOp(math.sqrt(window-1)*window/(window-2))
        return b.ops

class WindowedKurt(WindowedCompositiveOp):
    '''
    Unbiased estimated kurtosis of a rolling look back window of input, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).kurt()
    '''
    def decompose(self) -> List[OpBase]:
        n = self.attrs["window"]
        x = self.inputs[0]
        b = Builder(self.get_parent())
        with b:
            avgX = WindowedAvg(x, n)
            wX = WindowedTempOutput(x, n)
            each = ForeachBackWindow(wX, n)
            b.set_loop(each)
            diffX = Sub(IterValue(each, wX), avgX)
            x2 = Mul(diffX, diffX)
            x4 = x2 * x2
            b.set_loop(self.get_parent())
            vsum_x2 = ReduceAdd(x2)
            vsum_x4 = ReduceAdd(x4)
            adjfactor = n*(n-1)*(n+1) / (n-2) / (n-3)
            adjoffset = 3*(n-1)*(n-1) / (n-2) / (n-3)
            vsum_x4 / (vsum_x2 * vsum_x2) * adjfactor - adjoffset
        return b.ops


class TsArgMax(WindowedCompositiveOp):
    '''
    ArgMax in a rolling look back window, including the current newest data.
    The result should be the index of the max element in the rolling window. The index of the oldest element of the rolling window is 1.
    Similar to df.rolling(window).apply(np.argmax) + 1 
    '''
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            v2 = ReduceArgMax(IterValue(v1, v0))
            v3 = SubConst(v2, self.attrs["window"], True)
        return b.ops
    
class TsArgMin(WindowedCompositiveOp):
    '''
    ArgMin in a rolling look back window, including the current newest data.
    The result should be the index of the min element in the rolling window. The index of the oldest element of the rolling window is 1.
    Similar to df.rolling(window).apply(np.argmin) + 1 
    '''
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            v2 = ReduceArgMin(IterValue(v1, v0))
            v3 = SubConst(v2, self.attrs["window"], True)
        return b.ops

class TsRank(WindowedCompositiveOp):
    '''
    Time series rank of the newest data in a rolling look back window, including the current newest data.
    Let num_values_less = the number of values in rolling window that is less than the current newest data.
    Let num_values_eq = the number of values in rolling window that is equal to the current newest data.
    rank = num_values_less + (num_values_eq + 1) / 2
    Similar to df.rolling(window).rank()
    '''
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            v0 = WindowedTempOutput(self.inputs[0], self.attrs["window"])
            v1 = ForeachBackWindow(v0, self.attrs["window"])
            v2 = ReduceRank(IterValue(v1, v0), self.inputs[0])
        return b.ops

class Clip(CompositiveOp):
    '''
    Elementwisely clip the input value `v` with a given positive constant `eps`:
    Clip(v) = max(min(v, eps), -eps)
    The output will be between [-eps, +eps]
    '''
    def __init__(self, v: OpBase, eps: float) -> None:
        inputs = [v]
        super().__init__(inputs, [("value", eps)])

    def decompose(self) -> List[OpBase]:
        eps = self.attrs["value"]
        inp = self.inputs[0]
        b = Builder(self.get_parent())
        with b:
            v0 = Min(inp, ConstantOp(eps))
            out = Max(v0, ConstantOp(-eps))
        return b.ops

class DecayLinear(WindowedCompositiveOp):
    '''
    Weighted average in a rolling look back window, including the current newest data.
    The weight decreases linearly and the newer value has the higher weight.
    step_size = 1.0 / ((1.0 + window) * window / 2)
    weight[i] = (i+1) * step_size
    DecayLinear = sum([weight[i] for i in 0 to window])
    '''
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        window = self.attrs["window"]
        with b:
            v0 = WindowedTempOutput(self.inputs[0], window)
            v1 = ForeachBackWindow(v0, window)
            v2 = ReduceDecayLinear(IterValue(v1, v0), None, [("window", window)])
        return b.ops

def _handle_special_pow(base, expov):
    if expov == 0:
        return ConstantOp(1)
    elif expov < 0:
        intpart = _handle_special_pow(base, -expov)
        return ConstantOp(1) / intpart
    elif expov == 0.5:
        return Sqrt(base)
    elif abs(expov - int(expov) - 0.5) < 1e-5:
        intpart = _handle_special_pow(base, int(expov))
        if intpart is None:
            return None
        return Sqrt(base) *intpart
    elif int(expov) == expov and expov <= 1024 and expov >= 0:
        # pow(x, 5) >>>>  x1=x*x x2=x1*x2 out=x2*x
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
        return curv
    return None

class Pow(CompositiveOp):
    '''
    elementwise math function power: base ** expo
    '''
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
            with b:
                if _handle_special_pow(base, expov) is None:
                    Exp(Log(base) * expov)
            return b.ops
        with b:
            Exp(expo * Log(base))
            return b.ops


class WindowedLinearRegressionBase(WindowedCompositiveOp):
    def make_extract(self, v: OpBase) -> OpBase:
        raise RuntimeError("Not implemented")
    
    def decompose(self) -> List[OpBase]:
        b = Builder(self.get_parent())
        window = self.attrs["window"]
        with b:
            v0 = WindowedLinearRegression(self.inputs[0], window)
            v1 = self.make_extract(v0, self.inputs[0])
        return b.ops

class WindowedLinearRegressionRSqaure(WindowedLinearRegressionBase):
    '''
    Rsquare value of windowed linear regression. Implementation see
    https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/_libs/rolling.pyx#L137
    '''
    def make_extract(self, regr: OpBase, inp: OpBase) -> OpBase:
        return WindowedLinearRegressionRSqaureImpl(regr)

class WindowedLinearRegressionSlope(WindowedLinearRegressionBase):
    '''
    Slope value of windowed linear regression. Implementation see
    https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/_libs/rolling.pyx#L49
    '''
    def make_extract(self, regr: OpBase, inp: OpBase) -> OpBase:
        return WindowedLinearRegressionSlopeImpl(regr)

class WindowedLinearRegressionResi(WindowedLinearRegressionBase):
    '''
    Residuals of windowed linear regression. Implementation see
    https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/_libs/rolling.pyx#L91
    '''
    def make_extract(self, regr: OpBase, inp: OpBase) -> OpBase:
        return WindowedLinearRegressionResiImpl(regr, inp)