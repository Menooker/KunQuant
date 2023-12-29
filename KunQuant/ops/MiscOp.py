from KunQuant.Op import OpBase, WindowedTrait, SinkOpTrait, StatefulOpTrait

class BackRef(OpBase, WindowedTrait):
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

class FastWindowedSum(OpBase, WindowedTrait, StatefulOpTrait):
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

class GenericPartition(OpBase, SinkOpTrait):
    pass