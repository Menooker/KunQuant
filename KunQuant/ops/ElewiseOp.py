from KunQuant.Op import UnaryElementwiseOp, BinaryElementwiseOp, OpBase

class BinaryConstOp(UnaryElementwiseOp):
    def __init__(self, inp: OpBase,  v: float) -> None:
        super().__init__(inp, [("value", v)])

class AddConst(BinaryConstOp):
    pass

class DivConst(BinaryConstOp):
    pass

class Mul(BinaryElementwiseOp):
    pass

class Add(BinaryElementwiseOp):
    pass

class Sub(BinaryElementwiseOp):
    pass

class Div(BinaryElementwiseOp):
    pass


class Sqrt(UnaryElementwiseOp):
    pass