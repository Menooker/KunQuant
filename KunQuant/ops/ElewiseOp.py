from KunQuant.Op import UnaryElementwiseOp, BinaryElementwiseOp, OpBase

class BinaryConstOp(UnaryElementwiseOp):
    def __init__(self, inp: OpBase, v: float, swap: bool = False) -> None:
        attrs = [("value", v)]
        if swap:
            attrs.append(("swap", swap))
        super().__init__(inp, attrs)

class AddConst(BinaryConstOp):
    pass

class SubConst(BinaryConstOp):
    pass

class MulConst(BinaryConstOp):
    pass

class DivConst(BinaryConstOp):
    pass

class Mul(BinaryElementwiseOp):
    pass

class Add(BinaryElementwiseOp):
    pass

class Sub(BinaryElementwiseOp):
    pass

class Or(BinaryElementwiseOp):
    pass

class And(BinaryElementwiseOp):
    pass

class Div(BinaryElementwiseOp):
    pass

class CmpOp(BinaryElementwiseOp):
    pass

class GreaterThan(CmpOp):
    pass

class GreaterEqual(CmpOp):
    pass

class LessThan(CmpOp):
    pass

class LessEqual(CmpOp):
    pass

class GreaterThanConst(BinaryConstOp):
    pass

class LessThanConst(BinaryConstOp):
    pass

class Sqrt(UnaryElementwiseOp):
    pass

class Log(UnaryElementwiseOp):
    pass

class Abs(UnaryElementwiseOp):
    pass

class Sign(UnaryElementwiseOp):
    pass

class Not(UnaryElementwiseOp):
    pass

class SetInfOrNanToValue(UnaryElementwiseOp):
    def __init__(self, lhs: OpBase, value: float = 0.0) -> None:
        super().__init__(lhs, [("value", value)])

class Select(OpBase):
    def __init__(self, cond: OpBase, true_v: OpBase, false_v: OpBase) -> None:
        super().__init__([cond, true_v, false_v], None)