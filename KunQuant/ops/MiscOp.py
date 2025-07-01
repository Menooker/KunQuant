import KunQuant
from KunQuant.Op import OpBase, WindowedTrait, SinkOpTrait, CrossSectionalOp, GloablStatefulOpTrait, UnaryElementwiseOp, BinaryElementwiseOp
from typing import List

class BackRef(OpBase, WindowedTrait):
    '''
    Gets the data in `window` rows ago
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])
        
    def required_input_window(self) -> int:
        return self.attrs["window"] + 1

class WindowedQuantile(OpBase, WindowedTrait):
    '''
    Quantile in `window` rows ago.
    Similar to pd.rolling(window).quantile(q, interpolation='linear')
    '''
    def __init__(self, v: OpBase, window: int, q: float) -> None:
        super().__init__([v], [("window", window), ("q", q)])
        
    def required_input_window(self) -> int:
        return self.attrs["window"] + 1

class FastWindowedSum(OpBase, WindowedTrait, GloablStatefulOpTrait):
    '''
    Fast sum for windowed sum without reduction loop
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

    def required_input_window(self) -> int:
        return self.attrs["window"] + 1
    
    def get_state_variable_name_prefix(self) -> str:
        return "sum_"
    
    def generate_step_code(self, idx: str, time_idx: str, inputs: List[str], buf_name: str) -> str:
        return f"auto v{idx} = sum_{idx}.step({buf_name}, {inputs[0]}, {time_idx});"


class ExpMovingAvg(OpBase, GloablStatefulOpTrait):
    '''
    Exponential Moving Average (EMA)
    Similar to pd.DataFrame.ewm(span=window, adjust=False, ignore_na=True).mean()
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

    def get_state_variable_name_prefix(self) -> str:
        return "ema_"
    
    def generate_step_code(self, idx: str, time_idx: str, inputs: List[str]) -> str:
        return f"auto v{idx} = ema_{idx}.step({inputs[0]}, {time_idx});"

class WindowedLinearRegression(OpBase, WindowedTrait, GloablStatefulOpTrait):
    '''
    Compute states of Windowed Linear Regression
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

    def required_input_window(self) -> int:
        return self.attrs["window"] + 1
    
    def get_state_variable_name_prefix(self) -> str:
        return "linear_"
    
    def generate_step_code(self, idx: str, time_idx: str, inputs: List[str], buf_name: str) -> str:
        return f"const auto& v{idx} = linear_{idx}.step({buf_name}, {inputs[0]}, {time_idx});"
    
class WindowedLinearRegressionImplBase(OpBase):
    def __init__(self, v: OpBase) -> None:
        super().__init__([v])
    
    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        if len(self.inputs) < 1 or not isinstance(self.inputs[0], WindowedLinearRegression):
            raise RuntimeError("WindowedLinearRegressionImpl expects WindowedLinearRegression Op as input")
        return super().verify(func)

class WindowedLinearRegressionConsumerTrait:
    pass

class WindowedLinearRegressionRSqaureImpl(UnaryElementwiseOp, WindowedLinearRegressionConsumerTrait):
    '''
    Compute RSqaure of Windowed Linear Regression
    '''
    pass

class WindowedLinearRegressionSlopeImpl(UnaryElementwiseOp, WindowedLinearRegressionConsumerTrait):
    '''
    Compute RSqaure of Windowed Linear Regression
    '''
    pass

class WindowedLinearRegressionResiImpl(BinaryElementwiseOp, WindowedLinearRegressionConsumerTrait):
    '''
    Compute RSqaure of Windowed Linear Regression
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


class DiffWithWeightedSum(GenericCrossSectionalOp):
    '''
    Compute cross sectional weighted sum (of all stocks) and compute the difference
    of each stock data and the sum. Similar to numpy code:
    v2 = np.sum(v * w, axis=1)
    result = v - v2.reshape((-1, 1))
    '''
    def __init__(self, v: OpBase, w: OpBase) -> None:
        super().__init__([v, w], None)

    def generate_body(self) -> str:
        return """
        T sum = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            sum += input_0[i] * input_1[i];
        }
        for (size_t i = 0; i < num_stocks; i++) {
            output_0[i] = input_0[i] - sum;
        }
        """

    def generate_head(self) -> str:
        return ""


class GenericPartition(OpBase, SinkOpTrait):
    pass
