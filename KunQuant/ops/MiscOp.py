from KunQuant.Op import OpBase, WindowedTrait, SinkOpTrait, StatefulOpTrait

class BackRef(OpBase, WindowedTrait):
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])
        
    def required_input_window(self) -> int:
        return self.attrs["window"] + 1

class FastWindowedSum(OpBase, WindowedTrait, StatefulOpTrait):
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

    def required_input_window(self) -> int:
        return self.attrs["window"] + 1

class GenericPartition(OpBase, SinkOpTrait):
    pass