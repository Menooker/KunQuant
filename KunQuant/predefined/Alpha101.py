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
            self.vwap = Div(self.amount, AddConst(self.volume, 0.0000001))
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

def ts_argmin(v: OpBase, window: int) -> OpBase:
    return TsArgMin(v, window)

def ts_rank(v: OpBase, window: int) -> OpBase:
    return TsRank(v, window)

def ts_sum(v: OpBase, window: int) -> OpBase:
    return WindowedSum(v, window)

def ts_min(v: OpBase, window: int) -> OpBase:
    return WindowedMin(v, window)

def ts_max(v: OpBase, window: int) -> OpBase:
    return WindowedMax(v, window)

def correlation(v1: OpBase, v2: OpBase, window: int, no_optimization: bool = False) -> OpBase:
    ret = WindowedCorrelation(v1, window, v2)
    if no_optimization:
        ret.attrs['no_fast_stat'] = True
    return ret

def delta(v1: OpBase, window: int = 1) -> OpBase:
    return Sub(v1, BackRef(v1, window))

def rank(v: OpBase)-> OpBase:
    return Rank(v)

def sign(v: OpBase)-> OpBase:
    return Sign(v)

def covariance(v: OpBase, v2: OpBase, window: int) -> OpBase:
    return WindowedCovariance(v, window, v2)

def sma(v: OpBase, window: int, no_optimization: bool = False) -> OpBase:
    ret = WindowedAvg(v, window)
    if no_optimization:
        ret.attrs['no_fast_stat'] = True
    return ret

def bool_to_10(v: OpBase) -> OpBase:
    return Select(v, ConstantOp(1), ConstantOp(0))

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
    df = SetInfOrNanToValue(correlation(rank(d.high), rank(d.volume), 3, no_optimization=True))
    return -1 * ts_sum(rank(df), 3)

def alpha016(d: AllData):
    return -1 * rank(covariance(rank(d.high), rank(d.volume), 5))

def alpha017(d: AllData):
    adv20 = WindowedAvg(d.volume, 20)
    return -1 * (rank(ts_rank(d.close, 10)) *
                    rank(delta(delta(d.close, 1), 1)) *
                    rank(ts_rank(SetInfOrNanToValue(d.volume / adv20), 5)))

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
    df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5, no_optimization=True)
    df = SetInfOrNanToValue(df)
    return -1 * ts_max(df, 3)

def alpha027(self: AllData):
    alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2, no_optimization=True) / 2.0))
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
    return ((1.0 - rank(inner)) * SetInfOrNanToValue(ts_sum(self.volume, 5) / ts_sum(self.volume, 20), 1))

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
    return -1 * rank(stddev(self.high, 10)) * SetInfOrNanToValue(correlation(self.high, self.volume, 10), 1.0)

# Alpha#41	 (((high * low)^0.5) - vwap)
def alpha041(self: AllData):
    return Pow((self.high * self.low), ConstantOp(0.5)) - self.vwap

# Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
def alpha042(self: AllData):
    return rank((self.vwap - self.close)) / rank((self.vwap + self.close))
    
# Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
def alpha043(self: AllData):
    adv20 = sma(self.volume, 20)
    return ts_rank(SetInfOrNanToValue(self.volume / adv20), 20) * ts_rank((-1 * delta(self.close, 7)), 8)

# Alpha#44	 (-1 * correlation(high, rank(volume), 5))
def alpha044(self: AllData):
    df = correlation(self.high, rank(self.volume), 5)
    df = SetInfOrNanToValue(df)
    return -1 * df

# Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
def alpha045(self: AllData):
    df = correlation(self.close, self.volume, 2)
    df = SetInfOrNanToValue(df)
    return -1 * (rank(sma(delay(self.close, 5), 20)) * df *
                    rank(SetInfOrNanToValue(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2), 1)))

# Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
def alpha046(self: AllData):
    inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
    alpha = (-1 * delta(self.close, 1))
    alpha = Select(inner<0, ConstantOp(1), alpha)
    alpha = Select(inner>0.25, ConstantOp(-1), alpha)
    # alpha[inner < 0] = 1
    # alpha[inner > 0.25] = -1
    return alpha

# Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
def alpha047(self: AllData):
    adv20 = sma(self.volume, 20)
    return SetInfOrNanToValue(((rank((1 / self.close)) * self.volume / adv20) * ((self.high * rank((self.high - self.close))) / (sma(self.high, 5) /5))) - rank((self.vwap - delay(self.vwap, 5))))

# Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha049(self: AllData):
    inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
    alpha = (-1 * delta(self.close))
    # alpha[inner < -0.1] = 1
    alpha = Select(inner < -0.1, ConstantOp(1), alpha)
    return alpha

# Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
def alpha050(self: AllData):
    df = SetInfOrNanToValue(correlation(rank(self.volume), rank(self.vwap), 5))
    return (-1 * ts_max(rank(df), 5))
    # return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))

# Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha051(self: AllData):
    inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
    alpha = (-1 * delta(self.close))
    alpha = Select(inner < -0.05, ConstantOp(1), alpha)
    return alpha

# Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
def alpha052(self: AllData):
    return (((-1 * delta(ts_min(self.low, 5), 5)) *
                rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))
    
# Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
def alpha053(self: AllData):
    inner = (self.close - self.low)
    inner = Select(Equals(inner, ConstantOp(0)), ConstantOp(0.0001), inner)
    return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

# Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
def alpha054(self: AllData):
    inner = (self.low - self.high)
    inner = Select(Equals(inner, ConstantOp(0)), ConstantOp(0.0001), inner)
    return -1 * (self.low - self.close) * (Pow(self.open, ConstantOp(5))) / (inner * Pow(self.close, ConstantOp(5)))

def alpha055(self: AllData):
    divisor = (ts_max(self.high, 12) - ts_min(self.low, 12))
    divisor = Select(Equals(divisor, ConstantOp(0)), ConstantOp(0.0001), divisor)
    inner = (self.close - ts_min(self.low, 12)) / (divisor)
    df = correlation(rank(inner), rank(self.volume), 6)
    return -1 * SetInfOrNanToValue(df)

def alpha057(self: AllData):
    return (0 - (1 * ((self.close - self.vwap) / DecayLinear(rank(ts_argmax(self.close, 30)), 2))))

# Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
def alpha060(self: AllData):
    divisor = (self.high - self.low)
    divisor = Select(Equals(divisor, ConstantOp(0)), ConstantOp(0.0001), divisor)
    inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
    return 0 - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))

# Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
def alpha061(self: AllData):
    adv180 = sma(self.volume, 180)
    return bool_to_10(rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18)))

# Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
def alpha062(self: AllData):
    adv20 = sma(self.volume, 20)
    v1 = rank(correlation(self.vwap, sma(adv20, 22), 10))
    v2 = (rank(self.open) +rank(self.open))
    v3 = (rank(((self.high + self.low) / 2)) + rank(self.high))
    v4 = bool_to_10(v2 < v3)
    v5 = bool_to_10(v1 < rank(v4))
    return (v5 * -1)

# Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
def alpha064(self: AllData):
    adv120 = sma(self.volume, 120)
    a = rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13),sma(adv120, 13), 17))
    b = rank(delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 -0.178404))), 4))
    c = bool_to_10(a < b)
    return (c * -1)

# Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
def alpha065(self: AllData):
    adv60 = sma(self.volume, 60)
    a = rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60,9), 6))
    b = rank((self.open - ts_min(self.open, 14)))
    return (bool_to_10(a < b) * -1)
    
# Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
def alpha066(self: AllData):
    return ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(SetInfOrNanToValue((((self.low* 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)

# Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
def alpha068(self: AllData):
    adv15 = sma(self.volume, 15)
    a = ts_rank(correlation(rank(self.high), rank(adv15), 9), 14)
    b = rank(delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))
    return (bool_to_10(a < b) * -1)

# Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
def alpha071(self: AllData):
    adv180 = sma(self.volume, 180)
    p1=ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18), 4), 16)
    inner = Pow(rank(((self.low + self.open) - (self.vwap +self.vwap))), ConstantOp(2))
    p2=ts_rank(decay_linear(inner, 16), 4)
    return Max(p1, p2)
    #return max(ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16), ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4))

# Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
def alpha072(self: AllData):
    adv40 = sma(self.volume, 40)
    a = rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 9), 10)) + 0.0001
    b = rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7),3)) + 0.0001
    return (a / b)    

# Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
def alpha073(self: AllData):
    p1=rank(decay_linear(delta(self.vwap, 5), 3))
    p2=ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17)
    return -1* Max(p1, p2)

# Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
def alpha074(self: AllData):
    adv30 = sma(self.volume, 30)
    a = rank(correlation(self.close, sma(adv30, 37), 15))
    b = rank(correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11, no_optimization=True))
    return (bool_to_10(a < b)* -1)

# Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
def alpha075(self: AllData):
    adv50 = sma(self.volume, 50)
    return bool_to_10(rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50),12)))

def alpha077(self: AllData):
    adv40 = sma(self.volume, 40)
    p1=rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20))
    p2=rank(decay_linear(SetInfOrNanToValue(correlation(((self.high + self.low) / 2), adv40, 3), 1), 6))
    return Min(p1, p2)

# Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
def alpha078(self: AllData):
    adv40 = sma(self.volume, 40)
    a = rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20),ts_sum(adv40,20), 7))
    b = rank(SetInfOrNanToValue(correlation(rank(self.vwap), rank(self.volume), 6)))
    return Pow(a, b)

# Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
def alpha081(self: AllData):
    adv10 = sma(self.volume, 10)
    inner = Pow(rank(correlation(self.vwap, ts_sum(adv10, 50),8)), ConstantOp(4))
    a = rank(Log(WindowedProduct(rank(inner), 15)))
    b = rank(correlation(rank(self.vwap), rank(self.volume), 5))
    return (bool_to_10(a < b) * -1)

# Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
def alpha083(self: AllData):
    return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (((self.high -self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))

# Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
def alpha084(self: AllData):
    return Pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close,5))

# Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
def alpha085(self: AllData):
    adv30 = sma(self.volume, 30)
    base = rank(SetInfOrNanToValue(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30,10), 1))
    expo = rank(SetInfOrNanToValue(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10),7), 1))
    return Pow(base, expo)

def alpha086(self: AllData):
    adv20 = sma(self.volume, 20)
    a = ts_rank(correlation(self.close, sma(adv20, 15), 6), 20)
    b = rank(((self.open+ self.close) - (self.vwap +self.open)))
    return (bool_to_10(a < b) * -1)

def alpha088(self: AllData):
    adv60 = sma(self.volume, 60)
    p1=rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))),8))
    p2=ts_rank(decay_linear(SetInfOrNanToValue(correlation(ts_rank(self.close, 8), ts_rank(adv60,21), 8)), 7), 3)
    return Min(p2, p1)

def alpha092(self: AllData):
    adv30 = sma(self.volume, 30)
    p1=ts_rank(decay_linear(bool_to_10((((self.high + self.low) / 2) + self.close) < (self.low + self.open)), 15),19)
    p2=ts_rank(decay_linear(SetInfOrNanToValue(correlation(rank(self.low), rank(adv30), 8)), 7),7)
    return Min(p2, p1)

def alpha094(self: AllData):
    adv60 = sma(self.volume, 60)
    base = rank((self.vwap - ts_min(self.vwap, 12)))
    expo = ts_rank(SetInfOrNanToValue(correlation(ts_rank(self.vwap,20), ts_rank(adv60, 4), 18)), 3)
    return ((Pow(base, expo) * -1))

# Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
def alpha095(self: AllData):
    adv40 = sma(self.volume, 40)
    return bool_to_10(rank((self.open - ts_min(self.open, 12))) < ts_rank(Pow(rank(correlation(sma(((self.high + self.low)/ 2), 19), sma(adv40, 19), 13)), ConstantOp(5)), 12))

def alpha096(self: AllData):
    adv60 = sma(self.volume, 60)
    p1=ts_rank(decay_linear(SetInfOrNanToValue(correlation(rank(self.vwap), rank(self.volume), 4)),4), 8)
    p2=ts_rank(decay_linear(ts_argmax(SetInfOrNanToValue(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4)), 13), 14), 13)
    return -1* Max(p1, p2)

def alpha098(self: AllData):
    adv5 = sma(self.volume, 5)
    adv15 = sma(self.volume, 15)
    return (rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5), 7)) -rank(decay_linear(ts_rank(ts_argmin(SetInfOrNanToValue(correlation(rank(self.open), rank(adv15), 21)), 9),7), 8)))
    
def alpha099(self: AllData):
    adv60 = sma(self.volume, 60)
    return (bool_to_10(rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) <rank(correlation(self.low, self.volume, 6))) * -1)

def alpha101(self: AllData):
    return (self.close - self.open) /((self.high - self.low) + 0.001)

all_alpha = [alpha001, alpha002, alpha003, alpha004, alpha005, alpha006, alpha007, alpha008, alpha009, alpha010,
    alpha011, alpha012, alpha013, alpha014, alpha015, alpha016, alpha017, alpha018, alpha019, alpha020, alpha021,
    alpha022, alpha023, alpha024, alpha025, alpha026, alpha027, alpha028, alpha029, alpha030, alpha031, alpha032,
    alpha033, alpha034, alpha035, alpha036, alpha037, alpha038, alpha039, alpha040, alpha041, alpha042, alpha043,
    alpha044, alpha045, alpha046, alpha047, alpha049, alpha050, alpha051, alpha052, alpha053, alpha054, alpha055,
    alpha057, alpha060, alpha061, alpha062, alpha064, alpha065, alpha066, alpha068, alpha071, alpha072, alpha073,
    alpha074, alpha075, alpha077, alpha078, alpha081, alpha083, alpha084, alpha085, alpha086, alpha088, alpha092,
    alpha094, alpha095, alpha096, alpha098, alpha099, alpha101
    ]