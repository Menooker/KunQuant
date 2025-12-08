from .ReduceOp import ReduceAdd, ReduceMul, ReduceArgMax, ReduceRank, ReduceMin, ReduceMax, ReduceDecayLinear, ReduceArgMin
from KunQuant.Op import ConstantOp, OpBase, CompositiveOp, WindowedTrait, ForeachBackWindow, WindowedTempOutput, Builder, IterValue, WindowLoopIndex
from .ElewiseOp import And, DivConst, GreaterThan, LessThan, Or, Select, SetInfOrNanToValue, Sub, Mul, Sqrt, SubConst, Div, CmpOp, Exp, Log, Min, Max, Equals, Abs
from .MiscOp import Accumulator, BackRef, SetAccumulator, WindowedLinearRegression, WindowedLinearRegressionResiImpl, WindowedLinearRegressionRSqaureImpl, WindowedLinearRegressionSlopeImpl, ReturnFirstValue
from collections import OrderedDict
from typing import Union, List, Tuple, Dict
import math

def _is_fast_stat(opt: dict, attrs: dict) -> bool:
    return not opt.get("no_fast_stat", True) and not attrs.get("no_fast_stat", False)

class WindowedCompositiveOp(CompositiveOp, WindowedTrait):
    def __init__(self, v: OpBase, window: int, v2 = None) -> None:
        inputs = [v]
        if v2 is not None:
            inputs.append(v2)
        super().__init__(inputs, [("window", window)])

class WindowedReduce(WindowedCompositiveOp):
    def make_reduce(self, v: OpBase) -> OpBase:
        raise RuntimeError("Not implemented")
    
    def decompose(self, options: dict) -> List[OpBase]:
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

# small - sum
def _kahan_sub(mask: OpBase, sum: OpBase, small: OpBase, compensation: OpBase) -> Union[OpBase, OpBase]:
    y = small - compensation
    t = y - sum
    compensation = SetAccumulator(compensation, mask, t + sum - y)
    t = ReturnFirstValue([t, compensation])
    return t, compensation

def _avg_remove_window(oldx: OpBase, window: int):
    '''
    remove a value from a rolling average.
    Return observed count accumulator, new observed value, can_update (old not nan and new observed not 0),
        mean_accumulator, new mean, compensation before remove, delta (oldx - oldmean)
    '''
    notnan_old = Equals(oldx, oldx)
    new_observed_acc = Accumulator(notnan_old, f"avg_obs_{window}")
    new_observed = Select(notnan_old, new_observed_acc - 1, new_observed_acc)
    new_observed_not_0 = Abs(new_observed) > 0.001
    can_update = And(new_observed_not_0, notnan_old)
    mean_acc = Accumulator(oldx, f"avg_{window}")
    mean = mean_acc
    compensation_remove = Accumulator(mean, f"avg_comp_remove_{window}")
    # delta = oldx - mean
    delta, new_comp = _kahan_sub(can_update, mean, oldx, compensation_remove)
    mean = Select(can_update, mean - delta / new_observed, mean)
    
    return new_observed_acc, new_observed, can_update, mean_acc, mean, compensation_remove, delta

def _update_accumulator(acc: Accumulator, mask: OpBase, current_value: OpBase, update_value: OpBase):
    return SetAccumulator(acc, mask, Select(mask, update_value, current_value))

def _avg_add_window(x: OpBase, window: int, observed_acc: Accumulator, new_observed: OpBase, mean: OpBase, mean_acc: Accumulator):
    '''
    add a value from a rolling average.
    Return new observed value, notnan, new mean, compensation before add, delta (x - oldmean)
    '''
    notnan = Equals(x, x)
    new_observed = _update_accumulator(observed_acc, notnan, new_observed, new_observed + 1)
    new_observed_not_0 = Abs(new_observed) > 0.001
    compensation_add = None
    compensation_add = Accumulator(new_observed_not_0, f"avg_comp_add_{window}")
    # delta = x - mean
    delta, _ = _kahan_sub(notnan, mean, x, compensation_add)
    mean = _update_accumulator(mean_acc, notnan, mean, Select(new_observed_not_0, mean + delta / new_observed, ConstantOp(0)))
    return new_observed, notnan, mean, compensation_add, delta

def with_same_opt_flag(target: OpBase, current: dict) -> bool:
    if 'no_fast_stat' in current:
        target.attrs['no_fast_stat'] = current['no_fast_stat']
    return target

class WindowedAvg(WindowedCompositiveOp):
    '''
    Average of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).mean()
    '''
    def decompose(self, options: dict) -> List[OpBase]:
        b = Builder(self.get_parent())
        x = self.inputs[0]
        with b:
            if _is_fast_stat(options, self.attrs):
                # remove
                window = self.attrs["window"]
                oldx = BackRef(x, window)
                new_observed_acc, new_observed, notnan_old, mean_acc, mean, compensation_remove, _ = _avg_remove_window(oldx, window)

                # add
                new_observed, notnan, mean, compensation_add, _ = _avg_add_window(x, window, new_observed_acc, new_observed, mean, mean_acc)
                # if new_observed != window, return NAN
                Select(Abs(new_observed - window) < 0.001, mean, ConstantOp('nan'))

            else:
                v0 = WindowedSum(x, self.attrs["window"])
                v1 = DivConst(v0, self.attrs["window"])
        return b.ops

def _stddev_remove_window(oldx: OpBase, window: int):
    '''
    remove a value from a rolling average.
    Return observed count accumulator, new observed value, can_update,
        mean_accumulator, new mean, compensation before remove, var_accumulator, new var
    '''
    observed_acc, new_observed, can_update, mean_acc, mean, compensation_remove, _ = _avg_remove_window(oldx, window)
    prev_mean = mean_acc - compensation_remove
    var_acc = Accumulator(mean, f"var_{window}")
    var = var_acc
    var = Select(can_update, var - (oldx - prev_mean) * (oldx - mean), var)
    return observed_acc, new_observed, can_update, mean_acc, mean, compensation_remove, var_acc, var

def _stddev_add_window(x: OpBase, window: int, observed_acc: Accumulator, new_observed: OpBase, oldmean: OpBase, mean_acc: Accumulator, var: OpBase, var_acc: Accumulator):
    '''
    add a value from a rolling average.
    Return new observed value, notnan, new mean, new var
    '''
    new_observed, notnan, new_mean, compensation_add, _ = _avg_add_window(x, window, observed_acc, new_observed, oldmean, mean_acc)
    prev_mean = oldmean - compensation_add
    var = _update_accumulator(var_acc, notnan, var, var + (x - prev_mean) * (x - new_mean))
    return new_observed, notnan, new_mean, var

class WindowedVar(WindowedCompositiveOp):
    '''
    Unbiased variance of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).var()
    '''
    def decompose(self, options: dict) -> List[OpBase]:
        window = self.attrs["window"]
        b = Builder(self.get_parent())
        with b:
            if _is_fast_stat(options, self.attrs):
                x = self.inputs[0]
                # remove
                oldx = BackRef(x, window)
                observed_acc, new_observed, notnan_old, mean_acc, mean, compensation_remove, var_acc, var = _stddev_remove_window(oldx, window)

                # add
                new_observed, notnan, mean, var = _stddev_add_window(x, window, observed_acc, new_observed, mean, mean_acc, var, var_acc)
                # if new_observed != window, return NAN
                var = DivConst(var, window - 1)
                out = Select(Abs(new_observed - window) < 0.001, var, ConstantOp('nan'))
            else:
                avg = with_same_opt_flag(WindowedAvg(self.inputs[0], window), self.attrs)
                v0 = WindowedTempOutput(self.inputs[0], window)
                each = ForeachBackWindow(v0, window)
                b.set_loop(each)
                diff = Sub(IterValue(each, v0), avg)
                sqr = Mul(diff, diff)
                b.set_loop(self.get_parent())
                vsum = ReduceAdd(sqr)
                out = DivConst(vsum, window - 1)
        return b.ops
    
class WindowedStddev(WindowedCompositiveOp):
    '''
    Unbiased standard deviation of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).std()
    '''
    def decompose(self, options: dict) -> List[OpBase]:
        window = self.attrs["window"]
        b = Builder(self.get_parent())
        with b:
            var = with_same_opt_flag(WindowedVar(self.inputs[0], window), self.attrs)
            Sqrt(var)
        return b.ops

class WindowedCovariance(WindowedCompositiveOp):
    '''
    Unbiased estimated covariance of a rolling look back window of two inputs, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).cov(y)
    '''
    def decompose(self, options: dict) -> List[OpBase]:
        window = self.attrs["window"]
        x = self.inputs[0]
        y = self.inputs[1]
        b = Builder(self.get_parent())
        with b:
            if _is_fast_stat(options, self.attrs):
                mean_x_y = WindowedAvg(x * y, window)
                mean_x = WindowedAvg(x, window)
                mean_y = WindowedAvg(y, window)
                numerator = (mean_x_y - mean_x * mean_y) * (
                    window / (window - 1)
                )
            else:
                avgX = with_same_opt_flag(WindowedAvg(x, window), self.attrs)
                avgY = with_same_opt_flag(WindowedAvg(y, window), self.attrs)
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
    def decompose(self, options: dict) -> List[OpBase]:
        window = self.attrs["window"]
        x = self.inputs[0]
        y = self.inputs[1]
        b = Builder(self.get_parent())
        with b:
            if _is_fast_stat(options, self.attrs):
                mean_x_y = WindowedAvg(x * y, window)
                mean_x = WindowedAvg(x, window)
                mean_y = WindowedAvg(y, window)
                x_var = WindowedVar(x, window)
                y_var = WindowedVar(y, window)
                numerator = (mean_x_y - mean_x * mean_y) * (
                    window / (window - 1)
                )
                denominator = Sqrt(x_var * y_var)
                result = numerator / denominator
            else:
                avgX = with_same_opt_flag(WindowedAvg(x, window), self.attrs)
                avgY = with_same_opt_flag(WindowedAvg(y, window), self.attrs)
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

def _make_m2_m3_acc_skew(oldx: OpBase, window: int):
    m2_acc = Accumulator(oldx, f"skew_m2_{window}")
    m3_acc = Accumulator(oldx, f"skew_m3_{window}")
    return m2_acc, m3_acc

def _remove_skew_update(m2_acc: Accumulator, m3_acc: Accumulator, notnan_old: OpBase, delta_mean: OpBase, n: OpBase):
    delta_n = delta_mean / n
    term1 = delta_n * delta_mean * (n + 1.0)
    m3_update = delta_n * (term1 * (n + 2.0) - 3.0 * m2_acc)
    m3 = Select(notnan_old, m3_acc - m3_update, m3_acc)
    m2 = Select(notnan_old, m2_acc - term1, m2_acc)
    return m2, m3, delta_n, term1

def _add_skew_update(m2_acc: Accumulator, m3_acc: Accumulator, notnan: OpBase, delta_mean: OpBase, n: OpBase, m2: OpBase, m3: OpBase):
    delta_n = delta_mean / n
    term1 = delta_mean * delta_n * (n - 1.0)

    m3_update = delta_n * (term1 * (n - 2.0) - 3.0 * m2)
    m3 = _update_accumulator(m3_acc, notnan, m3, m3 + m3_update)
    m2 = _update_accumulator(m2_acc, notnan, m2, m2 + term1)
    return m2, m3, delta_n, term1

class WindowedSkew(WindowedCompositiveOp):
    '''
    Unbiased estimated skewness of a rolling look back window of input, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).skew()
    The bias adjustion factor is math.sqrt(window-1)*window/(window-2)
    '''
    def decompose(self, options: dict) -> List[OpBase]:
        window = self.attrs["window"]
        x = self.inputs[0]
        b = Builder(self.get_parent())
        with b:
            if _is_fast_stat(options, self.attrs):
                # remove_skew of https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx
                # remove
                window = self.attrs["window"]
                oldx = BackRef(x, window)
                new_observed_acc, n, notnan_old, mean_acc, mean, compensation_remove, delta_mean = _avg_remove_window(oldx, window)
                m2_acc, m3_acc = _make_m2_m3_acc_skew(oldx, window)
                m2, m3, _, _ = _remove_skew_update(m2_acc, m3_acc, notnan_old, delta_mean, n)

                # add
                n, notnan, mean, compensation_add, delta_mean = _avg_add_window(x, window, new_observed_acc, n, mean, mean_acc)
                m2, m3, _, _ = _add_skew_update(m2_acc, m3_acc, notnan, delta_mean, n, m2, m3)

                moments_ratio = m3 / (m2 * Sqrt(m2))
                correction = window * math.sqrt((window - 1)) / (window - 2)
                result = moments_ratio * correction
                result = ReturnFirstValue([result, mean])

                # if new_observed != window, return NAN
                # see calc_skew of https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx
                Select(And(Abs(n - window) < 0.001, m2 > n * 1e-14), result, ConstantOp('nan'))
            else:
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
    def decompose(self, options: dict) -> List[OpBase]:
        n = self.attrs["window"]
        x = self.inputs[0]
        b = Builder(self.get_parent())
        with b:
            if _is_fast_stat(options, self.attrs):
                # remove_kurt of https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx
                # remove
                window = self.attrs["window"]
                oldx = BackRef(x, window)
                new_observed_acc, n, notnan_old, mean_acc, mean, compensation_remove, delta_mean = _avg_remove_window(oldx, window)
                m2_acc, m3_acc = _make_m2_m3_acc_skew(oldx, window)
                m4_acc = Accumulator(oldx, f"kurt_m4_{window}")
                m2, m3, delta_n, term1 = _remove_skew_update(m2_acc, m3_acc, notnan_old, delta_mean, n)
                m4_update = delta_n * (
                        4 * m3_acc
                        + delta_n * (
                            6 * m2_acc
                            - term1 * (n * n + 3 * n + 3)
                            )
                        )
                m4 = Select(notnan_old, m4_acc + m4_update, m4_acc)

                # add
                n, notnan, mean, compensation_add, delta_mean = _avg_add_window(x, window, new_observed_acc, n, mean, mean_acc)
                old_m2 = m2
                old_m3 = m3
                m2, m3, delta_n, term1 = _add_skew_update(m2_acc, m3_acc, notnan, delta_mean, n, m2, m3)
                m4_update = delta_n * (
                        -4 * old_m3
                        + delta_n * (
                            6 * old_m2 + term1 * (n * n - 3 * n + 3)
                            )
                        )
                m4 = _update_accumulator(m4_acc, notnan, m4, m4 + m4_update)

                moments_ratio = m4 / (m2 * m2)
                term1 = (window * (window + 1)) * moments_ratio
                term2 = 3 * (window - 1)
                inner = term1 - term2

                correction = (window - 1) / ((window - 2) * (window - 3))
                result = correction * inner
                # result does not directly depend on mean and m3, so we need to keep the dependency of them
                result = ReturnFirstValue([result, mean, m3])
                Select(And(Abs(n - window) < 0.001, m2 > n * 1e-14), result, ConstantOp('nan'))
            else:
                avgX = WindowedAvg(x, n)
                wX = WindowedTempOutput(x, n)
                with ForeachBackWindow(wX, n) as each:
                    diffX = Sub(IterValue(each, wX), avgX)
                    x2 = Mul(diffX, diffX)
                    x4 = x2 * x2
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
    def decompose(self, options: dict) -> List[OpBase]:
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
    def decompose(self, options: dict) -> List[OpBase]:
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
    def decompose(self, options: dict) -> List[OpBase]:
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

    def decompose(self, options: dict) -> List[OpBase]:
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
    def decompose(self, options: dict) -> List[OpBase]:
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

    def decompose(self, options: dict) -> List[OpBase]:
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
    
    def decompose(self, options: dict) -> List[OpBase]:
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


class WindowedMaxDrawdown(WindowedCompositiveOp):
    '''
    Max Drawdown in a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    maxdrawdown = (max(hwm) - v) / max(hwm)
    where hwm is the highest of the input v in the rolling window and v is the lowest value of input with
    index larger than or equal to the index of highest value.
    '''
    def decompose(self, options: dict) -> List[OpBase]:
        b = Builder(self.get_parent())
        window = self.attrs["window"]
        v = self.inputs[0]
        with b:
            peak = WindowedMax(v, window)
            max_bar_index = TsArgMax(v, window) - 1 # tsargmax starts from 1
            inf = ConstantOp(float('inf'))
            with ForeachBackWindow(v, window) as each:
                index = WindowLoopIndex(each)
                filtered = Select(index >= max_bar_index, IterValue(each, v), inf)
            trough = ReduceMin(filtered)
            (peak - trough) / peak
        return b.ops