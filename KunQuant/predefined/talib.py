'''
TA-Lib compatible indicators implemented as KunQuant composite ops.
'''
from typing import List

from KunQuant.Op import OpBase, CompositiveOp, Builder, WindowedTrait, ConstantOp
from KunQuant.ops.ElewiseOp import Max, Min, Abs, Select, Equals, And, Or, Not
from KunQuant.ops.CompOp import WindowedSum
from KunQuant.ops.MiscOp import (
    Accumulator,
    BackRef,
    ExpMovingAvg,
    ReturnFirstValue,
    SetAccumulator,
)


class TRANGE(CompositiveOp):
    '''
    True Range. TA-Lib compatible: returns NaN on the first bar since there
    is no preceding close.
    TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
    '''
    def __init__(self, high: OpBase, low: OpBase, close: OpBase) -> None:
        super().__init__([high, low, close], None)

    def decompose(self, options: dict) -> List[OpBase]:
        b = Builder(self.get_parent())
        high, low, close = self.inputs
        with b:
            prev_close = BackRef(close, 1)
            hl = high - low
            hc = Abs(high - prev_close)
            lc = Abs(low - prev_close)
            prev_close_valid = Equals(prev_close, prev_close)
            tr_full = Max(hl, Max(hc, lc))
            Select(prev_close_valid, tr_full, ConstantOp('nan'))
        return b.ops


class ATR(CompositiveOp):
    '''
    Average True Range with Wilder smoothing, TA-Lib compatible.
    Implemented as an exponential moving average of TRANGE with smoothing
    factor alpha = 1/window (i.e. span = 2*window - 1). To match TA-Lib's
    ATR exactly, the EMA is seeded at bar `window` with the SMA of
    TRANGE[1..window]; bars before that emit NaN.
    '''
    def __init__(self, high: OpBase, low: OpBase, close: OpBase, window: int) -> None:
        super().__init__([high, low, close], [("window", window)])

    def decompose(self, options: dict) -> List[OpBase]:
        b = Builder(self.get_parent())
        window = self.attrs["window"]
        high, low, close = self.inputs
        with b:
            tr = TRANGE(high, low, close)

            mask_true = Equals(ConstantOp(0), ConstantOp(0))
            cnt_acc = Accumulator(high, f"atr_cnt_{window}")
            prev_cnt = cnt_acc
            new_cnt = prev_cnt + 1
            set_cnt = SetAccumulator(cnt_acc, mask_true, new_cnt)

            sma_seed = WindowedSum(tr, window) / window
            is_seed_bar = Equals(new_cnt, ConstantOp(window + 1))
            is_post_seed = new_cnt > window + 1
            tr_seeded = Select(is_seed_bar, sma_seed,
                               Select(is_post_seed, tr, ConstantOp('nan')))

            ema = ExpMovingAvg(tr_seeded, 2 * window - 1)
            ReturnFirstValue([ema, set_cnt])
        return b.ops


class SAR(CompositiveOp):
    '''
    Parabolic SAR (Stop And Reverse), TA-Lib compatible. Output for bar 0
    is NaN. From bar 1 onward the algorithm matches ta_func/ta_SAR.c:
      - Trend at bar 1 is seeded via Wilder MINUS_DM on bars 0..1
        (DOWN if -DM > 0 else UP).
      - Init SAR is low[0] (UP) or high[0] (DOWN); init EP is high[1] or low[1].
      - The first iteration "cheats": prevHigh=newHigh=high[1] and
        prevLow=newLow=low[1].
      - On reversal and continuation the new SAR is capped within the last
        two bars' lows (uptrend) or highs (downtrend).
    State is carried across bars via Accumulator/SetAccumulator.
    '''
    def __init__(self, high: OpBase, low: OpBase,
                 af_init: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> None:
        super().__init__([high, low],
                         [("af_init", af_init), ("af_step", af_step), ("af_max", af_max)])

    def decompose(self, options: dict) -> List[OpBase]:
        b = Builder(self.get_parent())
        with b:
            high = self.inputs[0]
            low = self.inputs[1]
            af_init = self.attrs["af_init"]
            af_step = self.attrs["af_step"]
            af_max = self.attrs["af_max"]

            mask_true = Equals(ConstantOp(0), ConstantOp(0))

            cnt_acc = Accumulator(high, "sar_cnt")
            prev_cnt = cnt_acc
            set_cnt = SetAccumulator(cnt_acc, mask_true, prev_cnt + 1)
            is_bar_0 = Equals(prev_cnt, ConstantOp(0))
            is_bar_1 = Equals(prev_cnt, ConstantOp(1))

            sar_acc = Accumulator(high, "sar_value")
            ep_acc = Accumulator(high, "sar_ep")
            af_acc = Accumulator(high, "sar_af")

            prev_sar = sar_acc
            prev_ep = ep_acc
            prev_af = af_acc

            high_prev = BackRef(high, 1)
            low_prev = BackRef(low, 1)

            # Wilder MINUS_DM on bars 0..1 to seed the trend at bar 1
            diff_m = low_prev - low
            diff_p = high - high_prev
            minus_dm = Select(And(diff_m > 0, diff_p < diff_m), diff_m, ConstantOp(0.0))
            init_is_long_bool = Not(minus_dm > 0)
            init_sar = Select(init_is_long_bool, low_prev, high_prev)
            init_ep = Select(init_is_long_bool, high, low)

            cur_sar = Select(is_bar_1, init_sar, prev_sar)
            cur_ep = Select(is_bar_1, init_ep, prev_ep)
            cur_af = Select(is_bar_1, ConstantOp(af_init), prev_af)
            # Trend invariant: cur_sar < cur_ep in UP, cur_sar > cur_ep in DOWN.
            is_up = cur_sar < cur_ep

            # ---- Trend-relative inputs (single unified computation below) ----
            # friend: side EP advances toward (high in UP, low in DOWN)
            # enemy: side that triggers reversal (low in UP, high in DOWN)
            friend = Select(is_up, high, low)
            enemy = Select(is_up, low, high)
            friend_prev = Select(is_up, high_prev, low_prev)
            enemy_prev = Select(is_up, low_prev, high_prev)
            # bar 1 "cheat": prev == new
            friend_prev_eff = Select(is_bar_1, friend, friend_prev)
            enemy_prev_eff = Select(is_bar_1, enemy, enemy_prev)

            # Trend-direction max/min: "toward friend" = max in UP, min in DOWN
            def toward_friend(a, b):
                return Select(is_up, Max(a, b), Min(a, b))
            def toward_enemy(a, b):
                return Select(is_up, Min(a, b), Max(a, b))

            # Reversal: enemy has crossed cur_sar in the bearish direction.
            # UP wants `enemy <= cur_sar`; DOWN wants `enemy >= cur_sar`. Swap the
            # comparison operands via Select so the comparison direction is shared
            # (KunQuant's Select can't take three bool operands).
            reversal = Select(is_up, enemy, cur_sar) <= Select(is_up, cur_sar, enemy)
            # EP update: friend exceeds cur_ep in trend direction.
            # UP: friend > cur_ep; DOWN: friend < cur_ep, i.e. cur_ep > friend.
            ep_changed = Select(is_up, friend, cur_ep) > Select(is_up, cur_ep, friend)

            new_ep_cont = Select(ep_changed, friend, cur_ep)
            new_af_cont = Select(ep_changed,
                                 Min(cur_af + af_step, ConstantOp(af_max)),
                                 cur_af)
            next_sar_cont = cur_sar + new_af_cont * (new_ep_cont - cur_sar)
            next_sar_cont = toward_enemy(next_sar_cont, enemy_prev_eff)
            next_sar_cont = toward_enemy(next_sar_cont, enemy)

            sar_rev = toward_friend(cur_ep, friend_prev_eff)
            sar_rev = toward_friend(sar_rev, friend)
            next_sar_rev = sar_rev + ConstantOp(af_init) * (enemy - sar_rev)
            next_sar_rev = toward_friend(next_sar_rev, friend_prev_eff)
            next_sar_rev = toward_friend(next_sar_rev, friend)

            output_bar = Select(reversal, sar_rev, cur_sar)
            next_sar = Select(reversal, next_sar_rev, next_sar_cont)
            next_ep = Select(reversal, enemy, new_ep_cont)
            next_af = Select(reversal, ConstantOp(af_init), new_af_cont)

            output = Select(is_bar_0, ConstantOp('nan'), output_bar)

            # Bar 0: BackRef is NaN, so written values may be NaN — override.
            # Defaults are chosen so the bar 1 trend invariant (sar < ep iff UP) is
            # established by the explicit init Selects above, not by these stores.
            store_sar = Select(is_bar_0, ConstantOp(0.0), next_sar)
            store_ep = Select(is_bar_0, ConstantOp(0.0), next_ep)
            store_af = Select(is_bar_0, ConstantOp(af_init), next_af)

            set_sar = SetAccumulator(sar_acc, mask_true, store_sar)
            set_ep = SetAccumulator(ep_acc, mask_true, store_ep)
            set_af = SetAccumulator(af_acc, mask_true, store_af)

            ReturnFirstValue([output, set_cnt, set_sar, set_ep, set_af])
        return b.ops
