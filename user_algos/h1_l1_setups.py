"""
H1 Long & L1 Short — Al Brooks First Pullback in Trend

H1 Long: First pullback buy in an established bull trend (7+ of last 10 bars above EMA).
L1 Short: First pullback short in an established bear trend (7+ of last 10 bars below EMA).

Backtested over 7 years (2019–2026), 91 tickers:
  H1 Long:  3,609 trades, 70.6% WR, PF 2.45, +1,510R
  L1 Short: 2,055 trades, 61.5% WR, PF 1.62, +486R
"""

from typing import List
from algo_engine import Bar, Setup


def detect_h1_long_first_pullback(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    H1 Long — First pullback buy in a bull trend.

    Entry: Buy 1 tick above the pullback bar's high after a bull break bar.
    Stop: Below the recent swing low (min of last 3 bars).
    Context: 7+ of last 10 bars closing above EMA-20.
    """
    setups = []
    n = len(bars)
    if n < 20:
        return setups

    last_long = -10

    for i in range(5, n):
        b = bars[i]
        prev = bars[i - 1]
        prev2 = bars[i - 2] if i >= 2 else prev

        # Bull trend: 7+ of last 10 bars above EMA
        above_count = sum(1 for j in range(max(0, i - 10), i) if bars[j].close > ema[j])
        if above_count >= 7 and i - last_long >= 5:
            # Pullback: prev bar is bear or made a lower low
            is_pullback = prev.is_bear or prev.low < prev2.low
            # At least 2 bull bars in the 3 bars before the pullback
            bars_before = bars[max(0, i - 4):i - 1]
            bull_before = sum(1 for bb in bars_before if bb.is_bull)

            if is_pullback and bull_before >= 2:
                # Current bar breaks above prev high and is bull
                if b.high > prev.high and b.is_bull:
                    # Not too many recent pullbacks (fresh trend)
                    recent_pbs = sum(1 for j in range(max(0, i - 8), i - 1)
                                     if bars[j].is_bear and bars[j].close > ema[j])
                    if recent_pbs <= 2:
                        entry = round(prev.high + 0.01, 2)
                        stop = round(min(bars[j].low for j in range(max(0, i - 2), i + 1)) - 0.01, 2)
                        risk = entry - stop

                        if risk > 0 and risk < entry * 0.15:
                            setups.append(Setup(
                                setup_name="H1 Long (First Pullback Buy)",
                                entry_bar=b.idx,
                                entry_price=entry,
                                order_type="Stop",
                                confidence=0.75,
                            ))
                            last_long = i

    return setups


def detect_l1_short_first_pullback(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    L1 Short — First pullback short in a bear trend.

    Entry: Sell 1 tick below the pullback bar's low after a bear break bar.
    Stop: Above the recent swing high (max of last 3 bars).
    Context: 7+ of last 10 bars closing below EMA-20.
    """
    setups = []
    n = len(bars)
    if n < 20:
        return setups

    last_short = -10

    for i in range(5, n):
        b = bars[i]
        prev = bars[i - 1]
        prev2 = bars[i - 2] if i >= 2 else prev

        # Bear trend: 7+ of last 10 bars below EMA
        below_count = sum(1 for j in range(max(0, i - 10), i) if bars[j].close < ema[j])
        if below_count >= 7 and i - last_short >= 5:
            # Pullback: prev bar is bull or made a higher high
            is_pullback = prev.is_bull or prev.high > prev2.high
            # At least 2 bear bars in the 3 bars before the pullback
            bars_before = bars[max(0, i - 4):i - 1]
            bear_before = sum(1 for bb in bars_before if bb.is_bear)

            if is_pullback and bear_before >= 2:
                # Current bar breaks below prev low and is bear
                if b.low < prev.low and b.is_bear:
                    # Not too many recent pullbacks (fresh trend)
                    recent_pbs = sum(1 for j in range(max(0, i - 8), i - 1)
                                     if bars[j].is_bull and bars[j].close < ema[j])
                    if recent_pbs <= 2:
                        entry = round(prev.low - 0.01, 2)
                        stop = round(max(bars[j].high for j in range(max(0, i - 2), i + 1)) + 0.01, 2)
                        risk = stop - entry

                        if risk > 0 and risk < entry * 0.15:
                            setups.append(Setup(
                                setup_name="L1 Short (First Pullback Sell)",
                                entry_bar=b.idx,
                                entry_price=entry,
                                order_type="Stop",
                                confidence=0.75,
                            ))
                            last_short = i

    return setups
