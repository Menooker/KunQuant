# Documentation for operators

This document provides detailed interfaces and semantics of operators of KunQuant.

There are two basic categories for operators of KunQuant:

 * `user ops` that are designed to be directly used by the users of KunQaunt to compose factor expressions
 * `internal ops` that are defined for internal use of KunQuant that is not recommended used for the end-users

## User ops

Currently there are several kinds of `User ops`:
 * Basic ops
 * Elementwise ops
 * Compositive ops
 * Cross Sectional ops
 * Miscellaneous ops

We will discuss about them in this section.

### Basic ops

#### OpBase

This is the base class of all ops. All operators inherit from this class.

Defined at `KunQuant.Op`

Interfaces

```python
class OpBase:
    # the inputs of the op
    inputs: List['OpBase']
    # the attributes
    attrs: OrderedDict[str, object]
    def __init__(self, inputs: List['OpBase'], attrs: Union[List[Tuple[str, object]], OrderedDict, None]) -> None:
        '''
        Construct an Op from a list of inputs and attributes.
        '''
        pass

    def replace_inputs(self, replace_map: Dict['OpBase', 'OpBase']):
        '''
        Replace ops in self.inputs with the ops in replace_map
        '''
        pass

    def set_parent(self, loop: 'ForeachBackWindow') -> None:
        '''
        Set parent loop
        '''
        pass

    def get_parent(self) -> 'ForeachBackWindow':
        '''
        Get parent loop
        '''
        pass

    def to_string(self, indent: int, identity: bool, **kwargs) -> str:
        '''
        Get the string representation of the op and its dependency ops.
        indent: the level of indention
        identity: only print related inputs in ForeachBackWindow
        '''
        pass

    def __str__(self) -> str:
        '''
        Get the string representation of the op and its dependency ops.
        '''
        pass

    def fast_str(self) -> str:
        '''
        Get the string representation of the op. Don't print the dependency ops.
        '''
        pass

    def hash_hex(self) -> str:
        '''
        Get the hex hash string for the op and all its dependencies
        '''
        pass

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        '''
        Verify the op
        '''
        pass
```

`OpBase` also overloads `+ - * / >= > <= < & |` operators, so users can use `a + b` or `a + 2` to represent `Add(a, b)` or `Add(a, ConstantOp(2))` ops, if `a` and `b` are ops.  

#### Constant Op

This op represents a constant value that is known at compile time.

Defined at `KunQuant.Op`

Interfaces

```python
class ConstantOp(OpBase, GraphSourceTrait):
    def __init__(self, v: float) -> None:
        pass
```

#### Input Op

This op represents a named input (for example, "open", "high").

Defined at `KunQuant.Op`

```python
class Input(WindowedDataSourceOp, GraphSourceTrait):
    def __init__(self, name: str) -> None:
        super().__init__([], [("name", name)])
```

#### Output Op

This op represents a named output (for example, "alpha101"). The input of this op are marked as output of the factor library with the given name.

Defined at `KunQuant.Op`

```python
class Output(WindowedDataSourceOp, SinkOpTrait):
    def __init__(self, inp: OpBase, name: str = "") -> None:
        super().__init__([inp], [("name", name)])
```


### Elementwise ops

Elementwise ops are operations whose an output value only depends on the newest values at time series of the inputs. We further define base classes `UnaryElementwiseOp` and `BinaryElementwiseOp` for Elementwise ops that have 1 or 2 inputs.

Defined at `KunQuant.Op`

```python
class UnaryElementwiseOp(OpBase):
    def __init__(self, lhs: OpBase, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([lhs], attrs)


class BinaryElementwiseOp(OpBase):
    def __init__(self, lhs: OpBase,  rhs: OpBase, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        super().__init__([lhs, rhs], attrs)
```

Subclasses Ops of `BinaryElementwiseOp`, defined in `KunQuant.ops.ElewiseOp` (also can be found in `KunQuant.ops`):

 * Math ops: Add, Sub, Mul, Div, Or, And, Min, Max
 * Comparison ops: GreaterThan, GreaterEqual, LessThan, LessEqual, Equals


Subclasses Ops of `UnaryElementwiseOp`, defined in `KunQuant.ops.ElewiseOp` (also can be found in `KunQuant.ops`):

 * Math ops: Sqrt (square root), Log (logarithm to the base of the mathematical constant e), Abs (absolute value), Sign (+1/0/-1 for positive/zero/negative inputs), Not (logical not), Exp (exponential function to the base of the mathematical constant e)
 * SetInfOrNanToValue (If input is inf or Nan, set output to a given constant value)

KunQuant also supports `Select` op, defined in `KunQuant.ops.ElewiseOp` (also can be found in `KunQuant.ops`). It accepts a boolean predicate `cond` and two values `true_v` and `false_v`. When a value of `cond` is true, the output of the `Select` op will be the value of `true_v`. Otherwise, the output of the op will be `false_v`.

```python
class Select(OpBase):
    def __init__(self, cond: OpBase, true_v: OpBase, false_v: OpBase) -> None:
        pass
```

There are some deprecated ops defined in `KunQuant.ElewiseOp`, which are based on `BinaryConstOp`, to represent a "binary elementwise" op, whose one of the input is constant. Users should avoid using these ops, and use subclasses of `BinaryElementwiseOp` and `ConstantOp` instead.

Interface of `BinaryConstOp`

```python
class BinaryConstOp(UnaryElementwiseOp):
    def __init__(self, inp: OpBase, v: float, swap: bool = False) -> None:
        '''
        Deprecated. The base class binary ops whose one of the input is constant.
        inp: the input
        v: the constant input
        swap: if false, the constant `v` is on the right hand side. Otherwise, swap the two inputs 
        '''
        pass
```

Subclasses of it `AddConst`, `SubConst`, `MulConst`, `DivConst`, `GreaterThanConst`, `LessThanConst`.

### Compositive ops

This kind of ops are complex ops that can be further decomposed into smaller ops. They are defined in `KunQuant.ops.CompOp` and can also be imported in `KunQuant.ops`.

Most of these ops are windowed ops, whose output depends on a window of previous data in time series.

Single operand windowed ops:

```python
class WindowedSum(WindowedReduce):
    '''
    Sum of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).sum()
    '''

class WindowedProduct(WindowedReduce):
    '''
    Product of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).product()
    '''

class WindowedMin(WindowedReduce):
    '''
    Min of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).min()
    '''
    
class WindowedMax(WindowedReduce):
    '''
    Max of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).max()
    '''

class WindowedAvg(WindowedCompositiveOp):
    '''
    Average of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).mean()
    '''

class WindowedStddev(WindowedCompositiveOp):
    '''
    Unbiased standard deviation of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).std()
    '''

class TsArgMax(WindowedCompositiveOp):
    '''
    ArgMax in a rolling look back window, including the current newest data.
    The result should be the index of the max element in the rolling window. The index of the oldest element of the rolling window is 1.
    Similar to df.rolling(window).apply(np.argmax) + 1 
    '''
    
class TsArgMin(WindowedCompositiveOp):
    '''
    ArgMin in a rolling look back window, including the current newest data.
    The result should be the index of the min element in the rolling window. The index of the oldest element of the rolling window is 1.
    Similar to df.rolling(window).apply(np.argmin) + 1 
    '''

class TsRank(WindowedCompositiveOp):
    '''
    Time series rank of the newest data in a rolling look back window, including the current newest data.
    Let num_values_less = the number of values in rolling window that is less than the current newest data.
    Let num_values_eq = the number of values in rolling window that is equal to the current newest data.
    rank = num_values_less + (num_values_eq + 1) / 2
    Similar to df.rolling(window).rank()
    '''

class DecayLinear(WindowedCompositiveOp):
    '''
    Weighted average in a rolling look back window, including the current newest data.
    The weight decreases linearly and the newer value has the higher weight.
    step_size = 1.0 / ((1.0 + window) * window / 2)
    weight[i] = (i+1) * step_size
    DecayLinear = sum([weight[i] for i in 0 to window])
    '''

class WindowedLinearRegressionRSqaure(WindowedLinearRegressionBase):
    '''
    Rsquare value of windowed linear regression. Implementation see
    https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/_libs/rolling.pyx#L137
    '''

class WindowedLinearRegressionSlope(WindowedLinearRegressionBase):
    '''
    Slope value of windowed linear regression. Implementation see
    https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/_libs/rolling.pyx#L49
    '''

class WindowedLinearRegressionResi(WindowedLinearRegressionBase):
    '''
    Residuals of windowed linear regression. Implementation see
    https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/_libs/rolling.pyx#L91
    '''

class WindowedKurt(WindowedCompositiveOp):
    '''
    Unbiased estimated kurtosis of a rolling look back window of input, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).kurt()
    '''

class WindowedSkew(WindowedCompositiveOp):
    '''
    Unbiased estimated skewness of a rolling look back window of input, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).skew()
    The bias adjustion factor is math.sqrt(window-1)*window/(window-2)
    '''

class WindowedMaxDrawdown(WindowedCompositiveOp):
    '''
    Max Drawdown in a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    maxdrawdown = (max(hwm) - v) / max(hwm)
    where hwm is the highest of the input v in the rolling window and v is the lowest value of input with
    index larger than or equal to the index of highest value.
    '''
```

Multiple operands windowed ops: 

```python
class WindowedCovariance(WindowedCompositiveOp):
    '''
    Unbiased estimated covariance of a rolling look back window of two inputs, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).cov(y)
    '''

class WindowedCorrelation(WindowedCompositiveOp):
    '''
    Correlation of a rolling look back window of two inputs, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).corr(y)
    '''
```

Note that the above to construct above ops, users should write the code like `WindowedCorrelation(input1, window, input2)` (note the order of the operands).

Elementwise ops

```python
class Clip(CompositiveOp):
    '''
    Elementwisely clip the input value `v` with a given positive constant `eps`:
    Clip(v) = max(min(v, eps), -eps)
    The output will be between [-eps, +eps]
    '''

class Pow(CompositiveOp):
    '''
    elementwise math function power: base ** expo
    '''
```

### Cross Sectional ops

Most ops in KunQuant are time-series ops. That is, the output of these ops only depends on the current or previous data in time-series of the same stock. However, KunQuant also supports Cross Sectional ops, that output of these ops depends on the data across different stocks at the same time.

Defined at `KunQuant.Op`

```python
class Rank(CrossSectionalOp):
    '''
    the cross sectional rank among different stocks. Between [0, 1]
    Similar to df.rank(axis=1, pct=True, method="average")
    '''
    pass

class Scale(CrossSectionalOp):
    '''
    the cross sectionally scale different stocks, to make sum([abs(stock[i]) for i in stock]) == 1
    Similar to df.div(df.abs().sum(axis=1), axis=0)
    '''
    pass

class GenericCrossSectionalOp(CrossSectionalOp):
    '''
    Cross sectional op with customized C++ implementation.
    generate_body() should return a C++ source code string. The C++ code should iterate on the
    stocks at the same point of "time" to compute the output.
    '''
    def generate_body(self) -> str:
        '''
        Predefined types and variables:
        `T`: the datatype, float or double
        `num_stocks`: the number of stocks
        `input_{N}`: the array-like accessor for the input data at the current point of time. `input_0` should be the first input
            To access the data of a stock, use `input_X[i]` for `i` in 0 to `num_stocks`
        `output_0`: the array-like accessor for the output data
        '''
        raise NotImplementedError("GenericCrossSectionalOp must be specialized")

    def generate_head(self) -> str:
        '''
        Predefined types and variables:
        `T`: the datatype, float or double
        `num_stocks`: the number of stocks
        '''
        raise NotImplementedError("GenericCrossSectionalOp must be specialized")
```

You can define your own cross-sectional operator by injecting C++ code via `GenericCrossSectionalOp`. `DiffWithWeightedSum` is an example for defining a custom cross-sectional operator by extending `GenericCrossSectionalOp`.

### Miscellaneous ops

Defined at `KunQuant.Op.MiscOp` (also found in `KunQuant.Op`)

```python
class BackRef(OpBase, WindowedTrait):
    '''
    Gets the data in `window` rows ago
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        pass

class WindowedQuantile(OpBase, WindowedTrait):
    '''
    Quantile in `window` rows ago.
    Similar to pd.rolling(window).quantile(q, interpolation='linear')
    '''
    def __init__(self, v: OpBase, window: int, q: float) -> None:
        pass

class ExpMovingAvg(OpBase, GloablStatefulOpTrait):
    '''
    Exponential Moving Average (EMA)
    Similar to pd.DataFrame.ewm(span=window, adjust=False, ignore_na=True).mean()
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        pass
class DiffWithWeightedSum(GenericCrossSectionalOp):
    '''
    Compute cross sectional weighted sum (of all stocks) and compute the difference
    of each stock data and the sum. Similar to numpy code:
    v2 = np.sum(v * w, axis=1)
    result = v - v2.reshape((-1, 1))
    '''
    def __init__(self, v: OpBase, w: OpBase) -> None:
        super().__init__([v, w], None)
```

## Internal ops

This section is for internal development of KunQuant.

### Traits

A trait is marked on class of ops to indicate some special properties of the ops.

Traits defined in `KunQuant.Op`:


```python
class GraphSourceTrait:
    '''
    The "source" of a graph, like input and constant ops. They have no inputs.
    '''
    pass

class WindowedDataSourceOp(OpBase):
    '''
    The ops that can be an input of WindowedTrait. It provides a window of data
    '''
    pass

class SinkOpTrait:
    '''
    The "sink" of a graph, like "output" op. Should keep ops extending this class even if no reference to these ops
    '''
    pass

class WindowedTrait:
    '''
    The ops that require a window of inputs of previous data.
    '''
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        pass
    def required_input_window(self) -> int:
        return self.attrs["window"]


class StatefulOpTrait:
    '''
    The ops that have an internal state
    '''
    pass


class CrossSectionalOp(OpBase):
    def __init__(self, v: OpBase) -> None:
        pass
```

### Internal ops

Ops defined in `KunQuant.Op`:

```python
class WindowedTempOutput(WindowedDataSourceOp):
    '''
    Mark that we need a windowed buffer of previous data of the input
    '''
    def __init__(self, inp: OpBase, window: int) -> None:
        super().__init__([inp], [("window", window)])

class ForeachBackWindow(OpBase, WindowedTrait):
    '''
    A for-loop to iterate the input ops (must be windowed inputs) and reduce outputs
    inp: A windowed input
    window: for-loop length in window size
    args: optional other windowed inputs to iterate
    '''
    def __init__(self, inp: WindowedTrait, window: int, *args) -> None:
        pass

    def print_args(self, indent: int, identity: bool, **kwargs) -> str:
        pass

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
        pass

    def print_args(self, indent: int, identity: bool, **kwargs) -> str:
        pass

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        pass

class WindowLoopIndex(OpBase):
    '''
    Get the current index of the ForEachWindow loop, starting from 0 to window-1. 0 for the oldest data
    and window-1 for the latest data
    '''
    def __init__(self, forwindow: ForeachBackWindow) -> None:
        pass

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        pass

class ReductionOp(OpBase, StatefulOpTrait):
    '''
    Base class of all reduction ops. A reduction op takes inputs that is originated from a IterValue. The input must be in a loop (v.get_parent() is a loop). The data produced
    by a ReductionOp should be used outside of the loop
    '''
    def __init__(self, v: OpBase, init_val: OpBase = None, attrs: Union[List[Tuple[str, object]], OrderedDict, None] = None) -> None:
        pass

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        pass
```


Ops defined in `KunQuant.ops.ReduceOp`:

```python
class ReduceAdd(ReductionOp):
    pass

class ReduceMul(ReductionOp):
    pass

class ReduceMin(ReductionOp):
    pass

class ReduceMax(ReductionOp):
    pass

class ReduceArgMax(ReductionOp):
    pass

class ReduceArgMin(ReductionOp):
    pass

class ReduceRank(ReductionOp):
    pass

class ReduceDecayLinear(ReductionOp):
    pass
```


Ops defined in `KunQuant.ops.MiscOp`:

```python
class FastWindowedSum(OpBase, WindowedTrait, GloablStatefulOpTrait):
    '''
    Fast sum for windowed sum without reduction loop
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        pass

    def required_input_window(self) -> int:
        pass

class WindowedLinearRegression(OpBase, WindowedTrait, GloablStatefulOpTrait):
    '''
    Compute states of Windowed Linear Regression
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        pass

    def required_input_window(self) -> int:
        pass

class WindowedLinearRegressionImplBase(OpBase):
    def __init__(self, v: OpBase) -> None:
        pass
    
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        pass


class WindowedLinearRegressionRSqaureImpl(OpBase):
    '''
    Compute RSqaure of Windowed Linear Regression
    '''
    pass

class WindowedLinearRegressionSlopeImpl(OpBase):
    '''
    Compute RSqaure of Windowed Linear Regression
    '''
    pass

class WindowedLinearRegressionResiImpl(OpBase):
    '''
    Compute RSqaure of Windowed Linear Regression
    '''
    pass
```