import KunQuant
from KunQuant.Op import OpBase, WindowedTrait, SinkOpTrait, StatefulOpTrait, GloablStatefulOpTrait, UnaryElementwiseOp, BinaryElementwiseOp

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

class ExpMovingAvg(OpBase, GloablStatefulOpTrait):
    '''
    Exponential Moving Average (EMA)
    Similar to pd.DataFrame.ewm(span=window, adjust=False, ignore_na=True).mean()
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

class WindowedLinearRegression(OpBase, WindowedTrait, GloablStatefulOpTrait):
    '''
    Compute states of Windowed Linear Regression
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

    def required_input_window(self) -> int:
        return self.attrs["window"] + 1

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

class GenericPartition(OpBase, SinkOpTrait):
    pass