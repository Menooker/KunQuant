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

def sma(v: OpBase, window: int) -> OpBase:
    return WindowedAvg(v, window)

scale = Scale

delay = BackRef
decay_linear = DecayLinear

def alpha001(d: AllData):
    inner = d.close
    cond = LessThanConst(d.returns, 0.0)
    sel = Select(cond, stddev(d.returns, 20), d.close)
    sel = Mul(sel, sel)
    return Rank(ts_argmax(sel, 5))

def alpha002(d: AllData):
    v = MulConst(WindowedCorrelation(Rank(delta(Log(d.volume), 2)), 6, Rank(Div(Sub(d.close, d.open), d.open))), -1)
    return SetInfOrNanToValue(v)

def alpha003(d: AllData):
    df = MulConst(correlation(Rank(d.open), Rank(d.volume), 10), -1)
    return SetInfOrNanToValue(df)

def alpha004(d: AllData):
    df = MulConst(ts_rank(Rank(d.low), 9), -1)
    return df

def alpha005(d: AllData):
    v1 = Rank(Sub(d.open, DivConst(WindowedSum(d.vwap, 10), 10)))
    v2 = MulConst(Abs(Rank(Sub(d.close, d.vwap))), -1)
    return Mul(v1, v2)

def alpha006(d: AllData):
    dopen = d.open
    vol = d.volume
    v1 = MulConst(WindowedCorrelation(dopen, 10, vol), -1)
    return SetInfOrNanToValue(v1)

def alpha007(d: AllData):
    adv20 = WindowedAvg(d.volume, 20)
    alpha = MulConst(Mul(ts_rank(Abs(delta(d.close, 7)), 60), sign(delta(d.close, 7))), -1)
    return Select(GreaterEqual(adv20, d.volume), ConstantOp(-1), alpha)

def alpha008(d: AllData):
    v = rank(Sub(Mul(ts_sum(d.open, 5), ts_sum(d.returns, 5)), BackRef(Mul(ts_sum(d.open, 5), ts_sum(d.returns, 5)), 10)))
    return MulConst(v, -1)

def alpha009(d: AllData):
    delta_close = delta(d.close, 1)
    cond_1 = GreaterThan(ts_min(delta_close, 5), ConstantOp(0))
    cond_2 = LessThan(ts_max(delta_close, 5), ConstantOp(0))
    alpha = MulConst(delta_close, -1)
    alpha = Select(Or(cond_1, cond_2), delta_close, alpha)
    return alpha

def alpha010(d: AllData):
    delta_close = delta(d.close, 1)
    cond_1 = GreaterThan(ts_min(delta_close, 4), ConstantOp(0))
    cond_2 = LessThan(ts_max(delta_close, 4), ConstantOp(0))
    alpha = MulConst(delta_close, -1)
    alpha = Select(Or(cond_1, cond_2), delta_close, alpha)
    return alpha

def alpha011(d: AllData):
    v = ((rank(ts_max((d.vwap - d.close), 3)) + rank(ts_min((d.vwap - d.close), 3))) *rank(delta(d.volume, 3)))
    return v

# Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
def alpha012(d: AllData):
    v = sign(delta(d.volume, 1)) * (-1 * delta(d.close, 1))
    return v

def alpha013(d: AllData):
    # alpha013 has rank(cov(rank(X), rank(Y))). Output of cov seems to have very similar results
    # like 1e-6 and 0. Thus the rank result will be different from pandas's reference
    return -1 * rank(covariance(rank(d.close), rank(d.volume), 5))

def alpha014(d: AllData):
    df = SetInfOrNanToValue(correlation(d.open, d.volume, 10))
    return -1 * (rank(delta(d.returns, 3)) * df)

def alpha015(d: AllData):
    # due to corr on Rank data, the rank result will be different from pandas's reference
    df = SetInfOrNanToValue(correlation(rank(d.high), rank(d.volume), 3))
    return -1 * ts_sum(rank(df), 3)

def alpha016(d: AllData):
    return -1 * rank(covariance(rank(d.high), rank(d.volume), 5))

def alpha017(d: AllData):
    adv20 = WindowedAvg(d.volume, 20)
    return -1 * (rank(ts_rank(d.close, 10)) *
                    rank(delta(delta(d.close, 1), 1)) *
                    rank(ts_rank((d.volume / adv20), 5)))

def alpha018(d: AllData):
    df = correlation(d.close, d.open, 10)
    df = SetInfOrNanToValue(df)
    return -1 * (rank((stddev(Abs((d.close - d.open)), 5) + (d.close - d.open)) +
                        df))

def alpha019(d: AllData):
    return (-1 * sign((d.close - delay(d.close, 7)) + delta(d.close, 7))) * (1 + rank(1 + ts_sum(d.returns, 250)))

# Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
def alpha020(d: AllData):
    return -1 * (rank(d.open - delay(d.high, 1)) *
                    rank(d.open - delay(d.close, 1)) *
                    rank(d.open - delay(d.low, 1)))

def alpha021(d: AllData):
    # d = (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))
    # c = ((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8)))
    # b = (c ? 1 : d)
    # a = (((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2))
    # (a? (-1 * 1) : b)
    c = WindowedAvg(d.close,2) < WindowedAvg(d.close, 8) - stddev(d.close, 8)
    adv20 = WindowedAvg(d.volume, 20)
    dd = d.volume / adv20>=1
    a = WindowedAvg(d.close, 8) + stddev(d.close, 8) < WindowedAvg(d.close,2)
    out = Select(~a & (c | dd), ConstantOp(1), ConstantOp(-1))
    return out

def alpha022(self: AllData):
    df = correlation(self.high, self.volume, 5)
    df = SetInfOrNanToValue(df)
    return -1 * delta(df, 5) * rank(stddev(self.close, 20))

# Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
def alpha023(self: AllData):
    cond = sma(self.high, 20) < self.high
    alpha = -1 * SetInfOrNanToValue(delta(self.high, 2))
    return Select(cond, alpha, ConstantOp(0))

def alpha024(self: AllData):
    cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
    alpha = -1 * delta(self.close, 3)
    return Select(cond, -1 * (self.close - ts_min(self.close, 100)), alpha)

def alpha025(self: AllData):
    adv20 = sma(self.volume, 20)
    return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))

def alpha026(self: AllData):
    df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
    df = SetInfOrNanToValue(df)
    return -1 * ts_max(df, 3)

def alpha027(self: AllData):
    alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
    return Select(alpha > 0.5, ConstantOp(-1), ConstantOp(1))

# Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
def alpha028(self: AllData):
    adv20 = sma(self.volume, 20)
    df = correlation(adv20, self.low, 5)
    df = SetInfOrNanToValue(df)
    return scale(((df + ((self.high + self.low) / 2)) - self.close))

# Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
def alpha029(self: AllData):
    return (ts_min(rank(rank(scale(Log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
            ts_rank(delay((-1 * self.returns), 6), 5))

# Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
def alpha030(self: AllData):
    delta_close = delta(self.close, 1)
    inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
    return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

# Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
def alpha031(self: AllData):
    adv20 = sma(self.volume, 20)
    df = correlation(adv20, self.low, 12)
    df = SetInfOrNanToValue(df)   
    p1=rank(rank(rank(DecayLinear((-1 * rank(rank(delta(self.close, 10)))), 10)))) 
    p2=rank((-1 * delta(self.close, 3)))
    p3=sign(scale(df))
    return p1+p2+p3
    
# Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
def alpha032(self):
    return scale(((sma(self.close, 7) / 7) - self.close)) + (20 * scale(correlation(self.vwap, delay(self.close, 5),230)))

def alpha033(self: AllData):
    return rank((self.open / self.close) - 1)

def alpha034(self: AllData):
    inner = stddev(self.returns, 2) / stddev(self.returns, 5)
    inner = SetInfOrNanToValue(inner, 1.0)
    return rank(2 - rank(inner) - rank(delta(self.close, 1)))

def alpha035(self: AllData):
    return ((ts_rank(self.volume, 32) *
                (1 - ts_rank(self.close + self.high - self.low, 16))) *
            (1 - ts_rank(self.returns, 32)))

def alpha036(self: AllData):
    adv20 = sma(self.volume, 20)
    return (((((2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15))) + (0.7 * rank((self.open- self.close)))) + (0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))) + rank(Abs(correlation(self.vwap,adv20, 6)))) + (0.6 * rank((((sma(self.close, 200) / 200) - self.open) * (self.close - self.open)))))

def alpha037(self: AllData):
    return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

# Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
def alpha038(self: AllData):
    inner = self.close / self.open
    inner = SetInfOrNanToValue(inner, 1.0)
    return -1 * rank(ts_rank(self.open, 10)) * rank(inner)

# Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
def alpha039(self: AllData):
    adv20 = sma(self.volume, 20)
    return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear((self.volume / adv20), 9))))) *
            (1 + rank(sma(self.returns, 250))))

# Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
def alpha040(self: AllData):
    return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

# Alpha#41	 (((high * low)^0.5) - vwap)
def alpha041(self: AllData):
    return pow((self.high * self.low),0.5) - self.vwap

# Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
def alpha042(self: AllData):
    return rank((self.vwap - self.close)) / rank((self.vwap + self.close))
    
# Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
def alpha043(self: AllData):
    adv20 = sma(self.volume, 20)
    return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

# Alpha#44	 (-1 * correlation(high, rank(volume), 5))
def alpha044(self):
    df = correlation(self.high, rank(self.volume), 5)
    df = SetInfOrNanToValue(df)
    return -1 * df


def alpha057(self: AllData):
    return (0 - (1 * ((self.close - self.vwap) / DecayLinear(rank(ts_argmax(self.close, 30)), 2))))

all_alpha = [alpha001, alpha002, alpha003, alpha004, alpha005, alpha006, alpha007, alpha008, alpha009, alpha010,
    alpha011, alpha012, alpha013, alpha014, alpha015, alpha016, alpha017, alpha018, alpha019, alpha020, alpha021,
    alpha022, alpha023, alpha024, alpha025, alpha026, alpha027, alpha028, alpha029, alpha030, alpha031, alpha032,
    alpha033, alpha034, alpha035, alpha036, alpha037, alpha038, alpha039, alpha040,  alpha042, alpha043,
    alpha044,
    alpha057
    ]