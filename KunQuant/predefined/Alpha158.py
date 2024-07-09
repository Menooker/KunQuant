from KunQuant.Op import *
from KunQuant.ops import *

def _bool_to_10(v: OpBase) -> OpBase:
    return Select(v, ConstantOp(1), ConstantOp(0))

class AllData:
    def __init__(self, open: OpBase, close: OpBase = None, high: OpBase = None, low: OpBase = None, volume: OpBase = None, amount: OpBase = None, vwap: OpBase = None) -> None:
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.amount = amount
        if vwap is None:
            self.vwap = Div(self.amount, AddConst(self.volume, 0.000001))

    def build(self, config: dict):
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                (self.close - self.open)/self.open,
                (self.high-self.low)/self.open,
                (self.close-self.open)/(self.high-self.low+1e-12),
                (self.high-Max(self.open, self.close))/self.open,
                (self.high-Max(self.open, self.close))/(self.high-self.low+1e-12),
                (Min(self.open, self.close)-self.low)/self.open,
                (Min(self.open, self.close)-self.low)/(self.high-self.low+1e-12),
                (2*self.close-self.high-self.low)/self.open,
                (2*self.close-self.high-self.low)/(self.high-self.low+1e-12),
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", [("OPEN", self.open), ("HIGH",self.high), ("LOW",self.low)])
            for field_name, field in feature:
                field_name = field_name.lower()
                fields += [BackRef(field, d)/self.close if d != 0 else field/self.close for d in windows]
                names += [field_name.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += [BackRef(self.volume, d)/(self.volume+1e-12) if d != 0 else self.volume/(self.volume+1e-12) for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field

            def use(x):
                return x not in exclude and (include is None or x in include)

            # Some factor ref: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                fields += [BackRef(self.close, d)/self.close for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                fields += [WindowedAvg(self.close, d)/self.close for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                fields += [WindowedStddev(self.close, d)/self.close for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                fields += [WindowedLinearRegressionSlope(self.close, d)/self.close for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                fields += [WindowedLinearRegressionRSqaure(self.close, d) for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                fields += [WindowedLinearRegressionResi(self.close, d)/self.close for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                # The max price for past d days, divided by latest close price to remove unit
                fields += [WindowedMax(self.high, d)/self.close for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                # The low price for past d days, divided by latest close price to remove unit
                fields += [WindowedMin(self.low, d)/self.close for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                # Used with MIN and MAX
                fields += [WindowedQuantile(self.close, d, 0.8)/self.close for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                fields += [WindowedQuantile(self.close, d, 0.2)/self.close for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                fields += [TsRank(self.close, d)/d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                # Represent the price position between upper and lower resistent price for past d days.
                fields += [Clip((self.close-WindowedMin(self.low, d))/(WindowedMax(self.high, d)-WindowedMin(self.low, d)+1e-12), 10) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                # The number of days between current date and previous highest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += [TsArgMax(self.high, d)/d for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                # The number of days between current date and previous lowest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += [TsArgMin(self.low, d)/d for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                # The time period between previous lowest-price date occur after highest price date.
                # Large value suggest downward momemtum.
                fields += [(TsArgMax(self.high, d)-TsArgMin(self.low, d))/d for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                # The correlation between absolute close price and log scaled trading volume
                fields += [SetInfOrNanToValue(WindowedCorrelation(self.close, d, Log(self.volume+1))) for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                # The correlation between price change ratio and volume change ratio
                fields += [SetInfOrNanToValue(WindowedCorrelation(self.close/BackRef(self.close,1), d, Log(self.volume/(BackRef(self.volume, 1)+1e-12)+1))) for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                # The percentage of days in past d days that price go up.
                fields += [WindowedAvg(_bool_to_10(self.close>BackRef(self.close, 1)), d) for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                # The percentage of days in past d days that price go down.
                fields += [WindowedAvg(_bool_to_10(self.close<BackRef(self.close, 1)), d) for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                # The diff between past up day and past down day
                fields += [WindowedAvg(_bool_to_10(self.close>BackRef(self.close, 1)), d)-WindowedAvg(_bool_to_10(self.close<BackRef(self.close, 1)), d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    WindowedSum(Max(self.close-BackRef(self.close, 1), ConstantOp(0)), d)/(WindowedSum(Abs(self.close-BackRef(self.close, 1)), d)+1e-12)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    WindowedSum(Max(BackRef(self.close, 1)-self.close, ConstantOp(0)), d)/(WindowedSum(Abs(self.close-BackRef(self.close, 1)), d)+1e-12)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    (WindowedSum(Max(self.close-BackRef(self.close, 1), ConstantOp(0)), d)-WindowedSum(Max(BackRef(self.close, 1)-self.close, ConstantOp(0)), d))
                    /(WindowedSum(Abs(self.close-BackRef(self.close, 1)), d)+1e-12)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
                fields += [WindowedAvg(self.volume, d)/(self.volume+1e-12) for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                # The standard deviation for volume in past d days.
                fields += [WindowedStddev(self.volume, d)/(self.volume+1e-12) for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # The volume weighted price change volatility
                fields += [
                    WindowedStddev(Abs(self.close/BackRef(self.close, 1)-1)*self.volume, d)/(WindowedAvg(Abs(self.close/BackRef(self.close, 1)-1)*self.volume, d)+1e-12)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                # The total volume increase / the absolute total volume changed
                fields += [
                    WindowedSum(Max(self.volume-BackRef(self.volume, 1), ConstantOp(0)), d)/(WindowedSum(Abs(self.volume-BackRef(self.volume, 1)), d)+1e-12)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                # The total volume increase / the absolute total volume changed
                # Can be derived from VSUMP by VSUMN = 1 - VSUMP
                fields += [
                    WindowedSum(Max(BackRef(self.volume, 1)-self.volume, ConstantOp(0)), d)/(WindowedSum(Abs(self.volume-BackRef(self.volume, 1)), d)+1e-12)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                # The diff ratio between total volume increase and total volume decrease
                # RSI indicator for volume
                fields += [
                    (WindowedSum(Max(self.volume-BackRef(self.volume, 1), ConstantOp(0)), d)-WindowedSum(Max(BackRef(self.volume, 1)-self.volume, ConstantOp(0)), d))
                    /(WindowedSum(Abs(self.volume-BackRef(self.volume, 1)), d)+1e-12)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        return fields, names