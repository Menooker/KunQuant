from KunQuant.Op import OpBase, WindowedTrait, SinkOpTrait, StatefulOpTrait

class BackRef(OpBase, WindowedTrait):
    '''
    Gets the data in `window` rows ago
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])
        
    def required_input_window(self) -> int:
        return self.attrs["window"] + 1

class FastWindowedSum(OpBase, WindowedTrait, StatefulOpTrait):
    '''
    Fast sum for windowed sum without reduction loop
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

    def required_input_window(self) -> int:
        return self.attrs["window"] + 1

class ExpMovingAvg(OpBase, StatefulOpTrait):
    '''
    Exponential Moving Average (EMA)
    Similar to pd.DataFrame.ewm(span=window, adjust=False).mean()
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])


class GenericPartition(OpBase, SinkOpTrait):
    pass