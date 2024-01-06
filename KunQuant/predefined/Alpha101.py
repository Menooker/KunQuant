from KunQuant.Op import *
from KunQuant.ops import *
from dataclasses import dataclass

@dataclass
class AllData:
    open: OpBase = None
    close: OpBase = None
    high: OpBase = None
    low: OpBase = None
    volume: OpBase = None
    vwap: OpBase = None

def stddev(v: OpBase, window: int) -> OpBase:
    return WindowedStddev(v, window)

def returns(v: OpBase) -> OpBase:
    prev1 = BackRef(v, 1)
    return SubConst(Div(v, prev1), 1.0)

def ts_argmax(v: OpBase, window: int) -> OpBase:
    return TsArgMax(v, window)

def correlation(v1: OpBase, v2: OpBase, window: int) -> OpBase:
    return WindowedCorrelation(v1, window, v2)

def delta(v1: OpBase, window: int) -> OpBase:
    return Sub(v1, BackRef(v1, window))

def alpha001(d: AllData):
    inner = d.close
    ret = returns(inner)
    cond = LessThanConst(ret, 0.0)
    sel = Select(cond, stddev(ret, 20), d.close)
    sel = Mul(sel, sel)
    Output(Rank(ts_argmax(sel, 5)), "alpha001")

def alpha002(d: AllData):
    v = MulConst(WindowedCorrelation(Rank(delta(Log(d.volume), 2)), 6, Rank(Div(Sub(d.close, d.open), d.open))), -1)
    Output(SetInfOrNanToZero(v), "alpha002")

def alpha006(d: AllData):
    dopen = d.open
    vol = d.volume
    v1 = MulConst(WindowedCorrelation(dopen, 10, vol), -1)
    Output(SetInfOrNanToZero(v1), "alpha006")