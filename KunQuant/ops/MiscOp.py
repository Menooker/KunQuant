import KunQuant
from KunQuant.Op import AcceptSingleValueInputTrait, Input, OpBase, WindowedTrait, SinkOpTrait, CrossSectionalOp, GloablStatefulOpTrait, UnaryElementwiseOp, BinaryElementwiseOp
from typing import List, Union

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

class Accumulator(OpBase, GloablStatefulOpTrait):
    '''
    Accumulator is a stateful op that accumulates the input value over time.
    It can be used to compute running totals, moving averages, etc.'''
    def __init__(self, v: OpBase, name: str) -> None:
        super().__init__([v], [("name", name)])
    def get_state_variable_name_prefix(self) -> str:
        return "accu_"
    
    def generate_step_code(self, idx: str, time_idx: str, inputs: List[str]) -> str:
        return f"auto v{idx} = accu_{idx}.asValue();"


    def verify(self, func) -> None:
        num_set = None
        for user in func.op_to_id[self].uses:
            if isinstance(user, SetAccumulator):
                if num_set is not None:
                    raise RuntimeError(f"Accumulator {self.attrs['name']} can only be used with one SetAccumulator: " + str(user) + "\nand  " + str(num_set) )
                else:
                    num_set = user
        if num_set is None:
            raise RuntimeError(f"Accumulator {self.attrs['name']} is not used with any SetAccumulator")
        return super().verify(func)

class SetAccumulator(OpBase):
    '''
    Set the value of an Accumulator to a value, if mask is set. Otherwise, it does nothing.
    '''
    def __init__(self, accu: OpBase, mask: OpBase, value: OpBase) -> None:
        super().__init__([accu, mask, value], [])
        self.check()
    
    def check(self):
        if not isinstance(self.inputs[0], Accumulator):
            raise RuntimeError("SetAccumulator expects the first input to be an Accumulator op")

    def verify(self, func: 'KunQuant.Stage.Function') -> None:
        self.check()
        return super().verify(func)
    
class ReturnFirstValue(OpBase):
    '''
    Return the first value of the input. It is used keep the dependency of the input op, like SetAccumulator.
    '''
    def __init__(self, v: List[OpBase]) -> None:
        super().__init__(v, [])
    

class ExpMovingAvg(OpBase, GloablStatefulOpTrait, AcceptSingleValueInputTrait):
    '''
    Exponential Moving Average (EMA)
    Similar to pd.DataFrame.ewm(span=window, adjust=False, ignore_na=True).mean()
    optional parameter: init_val, the initial values for EMA. It must be an Input op with attr
    {"single_value":True}. The name of the Input op should starts with "__init".
    It should be an input of shape (num_stocks,)
    '''
    def __init__(self, v: OpBase, window: int, init_val: Union[Input, None] = None) -> None:
        args = [v]
        if init_val is not None:
            if not isinstance(init_val, Input) or not init_val.attrs.get("single_value", False) or not init_val.attrs.get('name', '').startswith("__init") :
                raise RuntimeError("EMA expects init_val to be Input op with single_value=True and name starting with __init")
            args.append(init_val)
        super().__init__(args, [("window", window)])

    def get_single_value_input_id(self) -> int:
        return 1

    def get_state_variable_name_prefix(self) -> str:
        return "ema_"

    def generate_init_code(self, idx: str, elem_type: str, simd_lanes: int, inputs: List[str], aligned: bool) -> str:
        initv = "NAN"
        if len(self.inputs) == 2:
            mask_or_empty = ", mask" if not aligned else ""
            initv = f"buf_{self.inputs[1].attrs['name']}.step(0{mask_or_empty})"
        return f"{self.get_func_or_class_full_name(elem_type, simd_lanes)} {self.get_state_variable_name_prefix()}{idx} {{ {initv} }};"
    
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
