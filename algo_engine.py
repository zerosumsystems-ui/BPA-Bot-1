"""
algo_engine.py — Fast Al Brooks Price Action Pattern Detection Engine

Runs on raw OHLC data (no LLM calls). Detects setups, day types, and
market cycles using programmatic rules derived from Al Brooks' methodology.

Usage:
    from algo_engine import analyze_bars
    result = analyze_bars(df)  # df has columns: Open, High, Low, Close, Volume
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────── BAR CLASSIFICATION ──────────────────────────────

@dataclass
class Bar:
    """Classify a single 5-min bar using Al Brooks bar anatomy."""
    idx: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bull(self) -> bool:
        return self.close > self.open

    @property
    def is_bear(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        if self.range == 0:
            return True
        return self.body / self.range < 0.15

    @property
    def upper_tail(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_tail(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

    @property
    def closes_near_high(self) -> bool:
        """Close is in top 30% of range."""
        if self.range == 0:
            return False
        return (self.close - self.low) / self.range > 0.7

    @property
    def closes_near_low(self) -> bool:
        """Close is in bottom 30% of range."""
        if self.range == 0:
            return False
        return (self.high - self.close) / self.range > 0.7

    def is_big(self, avg_range: float) -> bool:
        """Bar range is > 1.5x average range."""
        return self.range > avg_range * 1.5

    def is_strong_bull(self, avg_range: float) -> bool:
        return self.is_bull and self.closes_near_high and self.body > self.range * 0.5

    def is_strong_bear(self, avg_range: float) -> bool:
        return self.is_bear and self.closes_near_low and self.body > self.range * 0.5

    def is_inside(self, prev: "Bar") -> bool:
        return self.high <= prev.high and self.low >= prev.low

    def is_outside(self, prev: "Bar") -> bool:
        return self.high > prev.high and self.low < prev.low


def bars_from_df(df: pd.DataFrame) -> list[Bar]:
    """Convert OHLCV DataFrame to list of Bar objects."""
    bars = []
    for i, row in df.iterrows():
        bars.append(Bar(
            idx=len(bars) + 1,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row.get("Volume", 0)),
        ))
    return bars


# ─────────────────────────── EMA COMPUTATION ─────────────────────────────────

def compute_ema(bars: list[Bar], period: int = 20) -> list[float]:
    """Compute EMA from bar closes."""
    closes = [b.close for b in bars]
    ema = [closes[0]]
    k = 2 / (period + 1)
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[-1] * (1 - k))
    return ema


# ─────────────────────────── SWING DETECTION ─────────────────────────────────

def find_swing_highs(bars: list[Bar], lookback: int = 3) -> list[int]:
    """Find swing high bar indices (local peaks)."""
    swings = []
    for i in range(lookback, len(bars) - lookback):
        if all(bars[i].high >= bars[i - j].high for j in range(1, lookback + 1)) and \
           all(bars[i].high >= bars[i + j].high for j in range(1, lookback + 1)):
            swings.append(i)
    return swings


def find_swing_lows(bars: list[Bar], lookback: int = 3) -> list[int]:
    """Find swing low bar indices (local troughs)."""
    swings = []
    for i in range(lookback, len(bars) - lookback):
        if all(bars[i].low <= bars[i - j].low for j in range(1, lookback + 1)) and \
           all(bars[i].low <= bars[i + j].low for j in range(1, lookback + 1)):
            swings.append(i)
    return swings


# ─────────────────────────── PATTERN DETECTORS ───────────────────────────────

@dataclass
class Setup:
    setup_name: str
    entry_bar: int
    entry_price: float
    order_type: str  # "Stop" or "Limit"
    confidence: float = 0.5


def detect_high_low_flags(bars: list[Bar], ema: list[float]) -> list[Setup]:
    """
    Detect High 1-4 (bull flags) and Low 1-4 (bear flags).

    High 1: First bar whose high goes above the high of a prior bar
            after a pullback in a bull trend (price above EMA).
    High 2: Second such occurrence → higher probability.
    Low 1/2: Mirror for bear flags below EMA.
    """
    setups = []
    if len(bars) < 10:
        return setups

    bull_count = 0  # Counts consecutive "high" breakout attempts above EMA
    bear_count = 0

    for i in range(2, len(bars)):
        bar = bars[i]
        prev = bars[i - 1]
        avg_range = np.mean([b.range for b in bars[max(0, i-20):i]]) if i > 0 else bar.range

        # Bull flags: price is above EMA (uptrend context)
        if bar.close > ema[i]:
            # Pullback bar (bear bar or bar with lower low)
            if prev.is_bear or prev.low < bars[i-2].low:
                # Next bar breaks above → High N
                if bar.high > prev.high and bar.is_bull:
                    bull_count += 1
                    flag_name = f"High {min(bull_count, 4)} Bull Flag"
                    conf = 0.45 + (min(bull_count, 2) * 0.1)  # H2 > H1
                    setups.append(Setup(
                        setup_name=flag_name,
                        entry_bar=bar.idx,
                        entry_price=round(prev.high + 0.01, 2),
                        order_type="Stop",
                        confidence=round(conf, 2),
                    ))
            else:
                # Not a pullback → reset or keep counting
                if bar.is_bull and bar.closes_near_high:
                    pass  # Trend continuation, don't reset
                else:
                    bull_count = 0

        # Bear flags: price is below EMA (downtrend context)
        if bar.close < ema[i]:
            if prev.is_bull or prev.high > bars[i-2].high:
                if bar.low < prev.low and bar.is_bear:
                    bear_count += 1
                    flag_name = f"Low {min(bear_count, 4)} Bear Flag"
                    conf = 0.45 + (min(bear_count, 2) * 0.1)
                    setups.append(Setup(
                        setup_name=flag_name,
                        entry_bar=bar.idx,
                        entry_price=round(prev.low - 0.01, 2),
                        order_type="Stop",
                        confidence=round(conf, 2),
                    ))
            else:
                if bar.is_bear and bar.closes_near_low:
                    pass
                else:
                    bear_count = 0

    return setups


def detect_double_bottoms_tops(bars: list[Bar], swing_lows: list[int], swing_highs: list[int]) -> list[Setup]:
    """Detect Double Bottom and Double Top patterns from swing points."""
    setups = []

    # Double Bottoms: two swing lows at similar price
    for i in range(1, len(swing_lows)):
        lo1 = bars[swing_lows[i - 1]]
        lo2 = bars[swing_lows[i]]
        spread = abs(lo1.low - lo2.low)
        avg = (lo1.low + lo2.low) / 2
        if avg > 0 and spread / avg < 0.003:  # Within 0.3%
            # Higher low variant
            name = "Double Bottom"
            if lo2.low > lo1.low:
                name = "Higher Low Double Bottom"
            elif lo2.low < lo1.low:
                name = "Lower Low Double Bottom"
            setups.append(Setup(
                setup_name=name,
                entry_bar=lo2.idx,
                entry_price=round(lo2.high + 0.01, 2),
                order_type="Stop",
                confidence=0.55,
            ))

    # Double Tops: two swing highs at similar price
    for i in range(1, len(swing_highs)):
        hi1 = bars[swing_highs[i - 1]]
        hi2 = bars[swing_highs[i]]
        spread = abs(hi1.high - hi2.high)
        avg = (hi1.high + hi2.high) / 2
        if avg > 0 and spread / avg < 0.003:
            name = "Double Top"
            if hi2.high < hi1.high:
                name = "Lower High Double Top"
            elif hi2.high > hi1.high:
                name = "Higher High Double Top"
            setups.append(Setup(
                setup_name=name,
                entry_bar=hi2.idx,
                entry_price=round(hi2.low - 0.01, 2),
                order_type="Stop",
                confidence=0.55,
            ))

    return setups


def detect_wedges(bars: list[Bar], swing_lows: list[int], swing_highs: list[int]) -> list[Setup]:
    """
    Detect Wedge Bottoms (3 pushes down) and Wedge Tops (3 pushes up).
    Al Brooks: Every pattern is built from Wedges, Doubles, and Breakouts.
    """
    setups = []

    # Wedge Bottom: 3 consecutive lower swing lows
    for i in range(2, len(swing_lows)):
        lo1 = bars[swing_lows[i - 2]].low
        lo2 = bars[swing_lows[i - 1]].low
        lo3 = bars[swing_lows[i]].low
        if lo3 < lo2 < lo1:
            entry_bar = bars[swing_lows[i]]
            setups.append(Setup(
                setup_name="Wedge Bottom",
                entry_bar=entry_bar.idx,
                entry_price=round(entry_bar.high + 0.01, 2),
                order_type="Stop",
                confidence=0.60,
            ))

    # Wedge Top: 3 consecutive higher swing highs
    for i in range(2, len(swing_highs)):
        hi1 = bars[swing_highs[i - 2]].high
        hi2 = bars[swing_highs[i - 1]].high
        hi3 = bars[swing_highs[i]].high
        if hi3 > hi2 > hi1:
            entry_bar = bars[swing_highs[i]]
            setups.append(Setup(
                setup_name="Wedge Top",
                entry_bar=entry_bar.idx,
                entry_price=round(entry_bar.low - 0.01, 2),
                order_type="Stop",
                confidence=0.60,
            ))

    return setups


def detect_breakouts(bars: list[Bar], ema: list[float]) -> list[Setup]:
    """
    Detect strong breakouts — consecutive big bars closing near extreme,
    above/below recent trading range and EMA.
    """
    setups = []
    if len(bars) < 10:
        return setups

    for i in range(5, len(bars)):
        bar = bars[i]
        prev = bars[i - 1]
        avg_range = np.mean([b.range for b in bars[max(0, i-20):i]])
        if avg_range == 0:
            continue

        # Bull breakout: 2+ consecutive strong bull bars
        if (bar.is_strong_bull(avg_range) and prev.is_strong_bull(avg_range)):
            # Confirm it's above recent highs
            recent_high = max(b.high for b in bars[max(0, i-10):i-1])
            if bar.close > recent_high and bar.close > ema[i]:
                setups.append(Setup(
                    setup_name="Breakout (BO)",
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2),
                    order_type="Stop",
                    confidence=0.65,
                ))

        # Bear breakout
        if (bar.is_strong_bear(avg_range) and prev.is_strong_bear(avg_range)):
            recent_low = min(b.low for b in bars[max(0, i-10):i-1])
            if bar.close < recent_low and bar.close < ema[i]:
                setups.append(Setup(
                    setup_name="Breakout (BO)",
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2),
                    order_type="Stop",
                    confidence=0.65,
                ))

    return setups


def detect_ii_ioi(bars: list[Bar]) -> list[Setup]:
    """Detect ii (inside-inside) and ioi (inside-outside-inside) patterns."""
    setups = []
    for i in range(2, len(bars)):
        b0, b1, b2 = bars[i-2], bars[i-1], bars[i]

        # ii: two consecutive inside bars
        if i >= 3:
            b_prev = bars[i-3]
            if b1.is_inside(b_prev) and b2.is_inside(b1):
                direction = "Buy" if b2.is_bull else "Sell"
                setups.append(Setup(
                    setup_name="ii Pattern",
                    entry_bar=b2.idx,
                    entry_price=round(b2.high + 0.01 if direction == "Buy" else b2.low - 0.01, 2),
                    order_type="Stop",
                    confidence=0.50,
                ))

        # ioi: inside, outside, inside
        if i >= 3:
            b_prev = bars[i-3]
            if b1.is_inside(b_prev) and b0.is_outside(b1) and b2.is_inside(b0):
                setups.append(Setup(
                    setup_name="ioi Pattern",
                    entry_bar=b2.idx,
                    entry_price=round(b2.high + 0.01, 2),
                    order_type="Stop",
                    confidence=0.50,
                ))

    return setups


def detect_ema_gap_bars(bars: list[Bar], ema: list[float]) -> list[Setup]:
    """
    20-Gap Bar: bar entirely above/below EMA (gap between bar low/high and EMA).
    Strong trend signal per Al Brooks.
    """
    setups = []
    for i in range(1, len(bars)):
        bar = bars[i]
        # 20-Gap Bar Buy: bar low is above EMA → strong bull
        if bar.low > ema[i] and bar.is_bull and bars[i-1].low > ema[i-1]:
            setups.append(Setup(
                setup_name="20-Gap Bar Buy",
                entry_bar=bar.idx,
                entry_price=round(bar.high + 0.01, 2),
                order_type="Stop",
                confidence=0.55,
            ))
        # 20-Gap Bar Sell: bar high is below EMA → strong bear
        elif bar.high < ema[i] and bar.is_bear and bars[i-1].high < ema[i-1]:
            setups.append(Setup(
                setup_name="20-Gap Bar Sell",
                entry_bar=bar.idx,
                entry_price=round(bar.low - 0.01, 2),
                order_type="Stop",
                confidence=0.55,
            ))
    return setups


def detect_spike_and_channel(bars: list[Bar], ema: list[float]) -> list[Setup]:
    """
    Spike and Channel: strong move (spike) followed by a weaker
    trending channel in the same direction.
    """
    setups = []
    if len(bars) < 20:
        return setups

    avg_range = np.mean([b.range for b in bars[:20]])
    if avg_range == 0:
        return setups

    # Look for spike in first 15 bars
    for i in range(3, min(15, len(bars))):
        bar = bars[i]
        # Bull spike: big bull bar(s) at the start
        if bar.is_big(avg_range) and bar.is_bull and bar.closes_near_high:
            # Check for channel after spike
            channel_bars = bars[i+1:min(i+20, len(bars))]
            if len(channel_bars) < 5:
                continue
            bull_count = sum(1 for b in channel_bars if b.is_bull)
            if bull_count > len(channel_bars) * 0.5:
                # Weaker trend continuing → spike and channel bull
                setups.append(Setup(
                    setup_name="Spike and Channel Bull",
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2),
                    order_type="Stop",
                    confidence=0.55,
                ))
            break  # Only detect once

        # Bear spike
        if bar.is_big(avg_range) and bar.is_bear and bar.closes_near_low:
            channel_bars = bars[i+1:min(i+20, len(bars))]
            if len(channel_bars) < 5:
                continue
            bear_count = sum(1 for b in channel_bars if b.is_bear)
            if bear_count > len(channel_bars) * 0.5:
                setups.append(Setup(
                    setup_name="Spike and Channel Bear",
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2),
                    order_type="Stop",
                    confidence=0.55,
                ))
            break

    return setups


# ─────────────────────────── DAY TYPE CLASSIFIER ─────────────────────────────

def classify_day_type(bars: list[Bar], ema: list[float]) -> str:
    """Classify the overall day type based on price action structure."""
    if len(bars) < 10:
        return "Trading Range Day"

    n = len(bars)
    first = bars[0]
    last = bars[-1]

    # Calculate trend metrics
    total_range = max(b.high for b in bars) - min(b.low for b in bars)
    net_move = last.close - first.open
    if total_range == 0:
        return "Trading Range Day"

    trend_ratio = abs(net_move) / total_range
    bull_bars = sum(1 for b in bars if b.is_bull)
    bear_bars = sum(1 for b in bars if b.is_bear)
    bull_ratio = bull_bars / n

    # Bars above/below EMA
    above_ema = sum(1 for i, b in enumerate(bars) if b.close > ema[i])
    above_ratio = above_ema / n

    # Check for big opening spike
    first_5_range = max(b.high for b in bars[:5]) - min(b.low for b in bars[:5])
    opening_spike = first_5_range > total_range * 0.4

    # Strong trend from open
    if trend_ratio > 0.7 and opening_spike:
        if net_move > 0:
            return "Bull Trend From The Open"
        else:
            return "Bear Trend From The Open"

    # Small pullback trend: strong directional, mostly on one side of EMA
    if trend_ratio > 0.5 and above_ratio > 0.8 and bull_ratio > 0.6:
        return "Small Pullback Bull Trend"
    if trend_ratio > 0.5 and above_ratio < 0.2 and bull_ratio < 0.4:
        return "Small Pullback Bear Trend"

    # Spike and channel detection
    if opening_spike and trend_ratio > 0.3:
        if net_move > 0:
            return "Spike and Channel Bull Trend"
        else:
            return "Spike and Channel Bear Trend"

    # Broad channel
    if trend_ratio > 0.3 and not opening_spike:
        if net_move > 0 and above_ratio > 0.6:
            return "Broad Bull Channel"
        elif net_move < 0 and above_ratio < 0.4:
            return "Broad Bear Channel"

    # Trading range: net move is small relative to total range
    if trend_ratio < 0.3:
        return "Trading Range Day"

    # Trending trading range
    if 0.3 <= trend_ratio <= 0.5:
        if net_move > 0:
            return "Trending Trading Range Day (Bull)"
        else:
            return "Trending Trading Range Day (Bear)"

    return "Trading Range Day"


def classify_market_cycle(bars: list[Bar], ema: list[float]) -> str:
    """Classify the current market cycle phase."""
    if len(bars) < 10:
        return "Trading Range"

    n = len(bars)
    avg_range = np.mean([b.range for b in bars])

    # Check last 10 bars for current state
    recent = bars[-10:]
    recent_ema = ema[-10:]

    # Breakout: recent big bars with momentum
    big_recent = sum(1 for b in recent if b.is_big(avg_range))
    if big_recent >= 2:
        return "Breakout (Spike)"

    # Micro channel: very tight, almost no overlap
    overlaps = 0
    for i in range(1, len(recent)):
        if recent[i].low < recent[i-1].high and recent[i].high > recent[i-1].low:
            overlaps += 1
    if overlaps < 3:
        return "Micro Channel"

    # Tight channel
    above = sum(1 for i, b in enumerate(recent) if b.close > recent_ema[i])
    if above >= 8 or above <= 2:
        return "Tight Channel (Small PB Trend)"

    # Broad channel
    total_range = max(b.high for b in recent) - min(b.low for b in recent)
    net = recent[-1].close - recent[0].close
    if total_range > 0 and abs(net) / total_range > 0.3:
        if net > 0:
            return "Broad Bull Channel"
        return "Broad Bear Channel"

    return "Trading Range"


# ─────────────────────────── MAIN ANALYSIS FUNCTION ──────────────────────────

def analyze_bars(df: pd.DataFrame) -> dict:
    """
    Run the full Al Brooks analysis on an OHLCV DataFrame.
    Returns a dict matching the Gemini vision output format.

    ~10-50ms execution time vs 5-15s for Gemini API call.
    """
    bars = bars_from_df(df)
    if len(bars) < 5:
        return {
            "day_type": "N/A",
            "market_cycle": "N/A",
            "reasoning": "Insufficient data",
            "setups": [],
            "action": "Wait / No Trade",
            "confidence": 0.0,
        }

    ema = compute_ema(bars)
    swing_lows = find_swing_lows(bars)
    swing_highs = find_swing_highs(bars)

    # Detect all patterns
    all_setups: list[Setup] = []
    all_setups.extend(detect_high_low_flags(bars, ema))
    all_setups.extend(detect_double_bottoms_tops(bars, swing_lows, swing_highs))
    all_setups.extend(detect_wedges(bars, swing_lows, swing_highs))
    all_setups.extend(detect_breakouts(bars, ema))
    all_setups.extend(detect_ii_ioi(bars))
    all_setups.extend(detect_ema_gap_bars(bars, ema))
    all_setups.extend(detect_spike_and_channel(bars, ema))

    # Sort by confidence, take top 5
    all_setups.sort(key=lambda s: s.confidence, reverse=True)
    top_setups = all_setups[:5]

    # Classify day type and market cycle
    day_type = classify_day_type(bars, ema)
    market_cycle = classify_market_cycle(bars, ema)

    # Determine action from highest-confidence setup
    action = "Wait / No Trade"
    overall_conf = 0.0
    if top_setups:
        best = top_setups[0]
        overall_conf = best.confidence
        buy_keywords = ["Bull", "Bottom", "Buy", "High", "Breakout"]
        sell_keywords = ["Bear", "Top", "Sell", "Low"]
        if any(kw in best.setup_name for kw in buy_keywords):
            action = "Buy"
        elif any(kw in best.setup_name for kw in sell_keywords):
            action = "Sell"

    # Build reasoning string
    reasoning_parts = [f"Day type: {day_type}.", f"Market cycle: {market_cycle}."]
    if top_setups:
        reasoning_parts.append(
            f"Best setup: {top_setups[0].setup_name} at bar {top_setups[0].entry_bar} "
            f"(confidence {top_setups[0].confidence:.0%})."
        )
        last_bar = bars[-1]
        if last_bar.close > ema[-1]:
            reasoning_parts.append("Price is above the 20 EMA, bullish bias.")
        else:
            reasoning_parts.append("Price is below the 20 EMA, bearish bias.")

    return {
        "day_type": day_type,
        "market_cycle": market_cycle,
        "reasoning": " ".join(reasoning_parts),
        "setups": [
            {
                "setup_name": s.setup_name,
                "entry_bar": s.entry_bar,
                "entry_price": s.entry_price,
                "order_type": s.order_type,
            }
            for s in top_setups
        ],
        "action": action,
        "confidence": round(overall_conf, 2),
    }


# ─────────────────────────── CLI TEST ────────────────────────────────────────

if __name__ == "__main__":
    import time
    import yfinance as yf

    ticker = "AAPL"
    print(f"Fetching 5-min data for {ticker}...")
    df = yf.download(ticker, period="1d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    start = time.perf_counter()
    result = analyze_bars(df)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\n⚡ Analysis completed in {elapsed:.1f}ms")
    print(f"Day Type: {result['day_type']}")
    print(f"Market Cycle: {result['market_cycle']}")
    print(f"Action: {result['action']} (confidence: {result['confidence']:.0%})")
    print(f"Reasoning: {result['reasoning']}")
    print(f"\nTop Setups:")
    for s in result["setups"]:
        print(f"  • {s['setup_name']} @ bar {s['entry_bar']} — {s['order_type']} @ ${s['entry_price']}")
