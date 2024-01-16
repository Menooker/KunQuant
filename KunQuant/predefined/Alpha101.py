from KunQuant.Op import *
from KunQuant.ops import *

class AllData:
    def __init__(self, open: OpBase, close: OpBase = None, high: OpBase = None, low: OpBase = None, volume: OpBase = None, amount: OpBase = None, vwap: OpBase = None) -> None:
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.amount = amount
        if vwap is None:
            self.vwap = Div(self.amount, AddConst(self.volume, 1))
        else:
            self.vwap = vwap
        self.returns = returns(close)

def stddev(v: OpBase, window: int) -> OpBase:
    return WindowedStddev(v, window)

def returns(v: OpBase) -> OpBase:
    prev1 = BackRef(v, 1)
    return SubConst(Div(v, prev1), 1.0)

def ts_argmax(v: OpBase, window: int) -> OpBase:
    return TsArgMax(v, window)

def ts_rank(v: OpBase, window: int) -> OpBase:
    return TsRank(v, window)

def ts_sum(v: OpBase, window: int) -> OpBase:
    return WindowedSum(v, window)

def ts_min(v: OpBase, window: int) -> OpBase:
    return WindowedMin(v, window)

def ts_max(v: OpBase, window: int) -> OpBase:
    return WindowedMax(v, window)

def correlation(v1: OpBase, v2: OpBase, window: int) -> OpBase:
    return WindowedCorrelation(v1, window, v2)

def delta(v1: OpBase, window: int) -> OpBase:
    return Sub(v1, BackRef(v1, window))

def rank(v: OpBase)-> OpBase:
    return Rank(v)

def sign(v: OpBase)-> OpBase:
    return Sign(v)

def covariance(v: OpBase, v2: OpBase, window: int) -> OpBase:
    return WindowedCovariance(v, window, v2)

delay = BackRef

def alpha001(d: AllData):
    inner = d.close
    cond = LessThanConst(d.returns, 0.0)
    sel = Select(cond, stddev(d.returns, 20), d.close)
    sel = Mul(sel, sel)
    Output(Rank(ts_argmax(sel, 5)), "alpha001")

def alpha002(d: AllData):
    v = MulConst(WindowedCorrelation(Rank(delta(Log(d.volume), 2)), 6, Rank(Div(Sub(d.close, d.open), d.open))), -1)
    Output(SetInfOrNanToZero(v), "alpha002")

def alpha003(d: AllData):
    df = MulConst(correlation(Rank(d.open), Rank(d.volume), 10), -1)
    Output(SetInfOrNanToZero(df), "alpha003")

def alpha004(d: AllData):
    df = MulConst(ts_rank(Rank(d.low), 9), -1)
    Output(df, "alpha004")

def alpha005(d: AllData):
    v1 = Rank(Sub(d.open, DivConst(WindowedSum(d.vwap, 10), 10)))
    v2 = MulConst(Abs(Rank(Sub(d.close, d.vwap))), -1)
    Output(Mul(v1, v2), "alpha005")

def alpha006(d: AllData):
    dopen = d.open
    vol = d.volume
    v1 = MulConst(WindowedCorrelation(dopen, 10, vol), -1)
    Output(SetInfOrNanToZero(v1), "alpha006")

def alpha007(d: AllData):
    adv20 = WindowedAvg(d.volume, 20)
    alpha = MulConst(Mul(ts_rank(Abs(delta(d.close, 7)), 60), sign(delta(d.close, 7))), -1)
    Output(Select(GreaterEqual(adv20, d.volume), ConstantOp(-1), alpha), "alpha007")

def alpha008(d: AllData):
    v = rank(Sub(Mul(ts_sum(d.open, 5), ts_sum(d.returns, 5)), BackRef(Mul(ts_sum(d.open, 5), ts_sum(d.returns, 5)), 10)))
    Output(MulConst(v, -1), "alpha008")

def alpha009(d: AllData):
    delta_close = delta(d.close, 1)
    cond_1 = GreaterThan(ts_min(delta_close, 5), ConstantOp(0))
    cond_2 = LessThan(ts_max(delta_close, 5), ConstantOp(0))
    alpha = MulConst(delta_close, -1)
    alpha = Select(Or(cond_1, cond_2), delta_close, alpha)
    Output(alpha, "alpha009")

def alpha010(d: AllData):
    delta_close = delta(d.close, 1)
    cond_1 = GreaterThan(ts_min(delta_close, 4), ConstantOp(0))
    cond_2 = LessThan(ts_max(delta_close, 4), ConstantOp(0))
    alpha = MulConst(delta_close, -1)
    alpha = Select(Or(cond_1, cond_2), delta_close, alpha)
    Output(alpha, "alpha010")

def alpha011(d: AllData):
    v = ((rank(ts_max((d.vwap - d.close), 3)) + rank(ts_min((d.vwap - d.close), 3))) *rank(delta(d.volume, 3)))
    Output(v, "alpha011")

# Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
def alpha012(d: AllData):
    v = sign(delta(d.volume, 1)) * (-1 * delta(d.close, 1))
    Output(v, "alpha012")

def alpha013(d: AllData):
    # alpha013 has rank(cov(rank(X), rank(Y))). Output of cov seems to have very similar results
    # like 1e-6 and 0. Thus the rank result will be different from pandas's reference
    Output(-1 * rank(covariance(rank(d.close), rank(d.volume), 5)), "alpha013")

def alpha014(d: AllData):
    df = SetInfOrNanToZero(correlation(d.open, d.volume, 10))
    Output(-1 * (rank(delta(d.returns, 3)) * df), "alpha014")

def alpha015(d: AllData):
    # due to corr on Rank data, the rank result will be different from pandas's reference
    df = SetInfOrNanToZero(correlation(rank(d.high), rank(d.volume), 3))
    Output(-1 * ts_sum(rank(df), 3), "alpha015")

def alpha016(d: AllData):
    Output(-1 * rank(covariance(rank(d.high), rank(d.volume), 5)), "alpha016")

def alpha017(d: AllData):
    adv20 = WindowedAvg(d.volume, 20)
    Output(-1 * (rank(ts_rank(d.close, 10)) *
                    rank(delta(delta(d.close, 1), 1)) *
                    rank(ts_rank((d.volume / adv20), 5))), "alpha017")

def alpha018(d: AllData):
    df = correlation(d.close, d.open, 10)
    df = SetInfOrNanToZero(df)
    Output(-1 * (rank((stddev(Abs((d.close - d.open)), 5) + (d.close - d.open)) +
                        df)), "alpha018")

def alpha019(d: AllData):
    Output((-1 * sign((d.close - delay(d.close, 7)) + delta(d.close, 7))) *
            (1 + rank(1 + ts_sum(d.returns, 250))), "alpha019")

# Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
def alpha020(d: AllData):
    Output(-1 * (rank(d.open - delay(d.high, 1)) *
                    rank(d.open - delay(d.close, 1)) *
                    rank(d.open - delay(d.low, 1))), "alpha020")

all_alpha = [alpha001, alpha002, alpha003, alpha004, alpha005,
    #           alpha006, alpha007, alpha008, alpha009, alpha010,
    # alpha011, alpha012, alpha013, alpha014, alpha015, alpha016, alpha017, alpha018, alpha019, alpha020
    ]