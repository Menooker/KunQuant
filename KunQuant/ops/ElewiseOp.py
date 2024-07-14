from KunQuant.Op import UnaryElementwiseOp, BinaryElementwiseOp, OpBase, BoolOpTrait

class BinaryConstOp(UnaryElementwiseOp):
    def __init__(self, inp: OpBase, v: float, swap: bool = False) -> None:
        '''
        Deprecated. The base class binary ops whose one of the input is constant.
        inp: the input
        v: the constant input
        swap: if false, the constant `v` is on the right hand side. Otherwise, swap the two inputs 
        '''
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

class Or(BinaryElementwiseOp, BoolOpTrait):
    pass

class And(BinaryElementwiseOp, BoolOpTrait):
    pass

class Max(BinaryElementwiseOp):
    pass

class Min(BinaryElementwiseOp):
    pass

class Div(BinaryElementwiseOp):
    pass

class CmpOp(BinaryElementwiseOp, BoolOpTrait):
    pass

class GreaterThan(CmpOp):
    pass

class GreaterEqual(CmpOp):
    pass

class LessThan(CmpOp):
    pass

class LessEqual(CmpOp):
    pass

class Equals(CmpOp):
    pass

class GreaterThanConst(BinaryConstOp, BoolOpTrait):
    pass

class LessThanConst(BinaryConstOp, BoolOpTrait):
    pass

class Sqrt(UnaryElementwiseOp):
    '''
    Square root of input
    '''
    pass

class Log(UnaryElementwiseOp):
    '''
    logarithm to the base of the mathematical constant e
    '''
    pass

class Abs(UnaryElementwiseOp):
    '''
    absolute value
    '''
    pass

class Sign(UnaryElementwiseOp):
    '''
    +1/0/-1 for positive/zero/negative inputs
    '''
    pass

class Not(UnaryElementwiseOp, BoolOpTrait):
    '''
    logical not
    '''
    pass

class Exp(UnaryElementwiseOp):
    '''
    exponential function to the base of the mathematical constant e
    '''
    pass

class SetInfOrNanToValue(UnaryElementwiseOp):
    '''
    If input is inf or Nan, set output to `value`
    '''
    def __init__(self, lhs: OpBase, value: float = 0.0) -> None:
        super().__init__(lhs, [("value", value)])

class Select(OpBase):
    def __init__(self, cond: OpBase, true_v: OpBase, false_v: OpBase) -> None:
        super().__init__([cond, true_v, false_v], None)