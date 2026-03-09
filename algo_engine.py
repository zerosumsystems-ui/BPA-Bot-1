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
from typing import Optional, Any
import importlib
import pkgutil
from pathlib import Path

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
    ema_20: float = 0.0

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def body_size(self) -> float:
        return self.body
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
    setup_name: str = ""
    signal_bar: int = 0
    entry_bar: int = 0
    entry_price: float = 0.0
    order_type: str = "Stop"
    confidence: float = 0.5
    target_price: float = 0.0
    stop_loss: float = 0.0
    
    # Optional kwargs from dynamic user algos
    setup_type: str = ""
    direction: int = 1
    notes: str = ""
    index: int = 0
    bar: Any = None
    
    def __post_init__(self):
        # Support legacy / dynamic user algos that pass name as setup_type
        if self.setup_type and not self.setup_name:
            self.setup_name = self.setup_type
            
        # If user algo passed 'index' (the signal bar)
        if self.index > 0:
            if self.signal_bar == 0:
                self.signal_bar = self.index
            if self.entry_bar == 0:
                self.entry_bar = self.index + 1
                
        # Try to infer entry_price if not provided but bar is
        if self.entry_price == 0.0 and hasattr(self.bar, 'high'):
            self.entry_price = round(self.bar.high + 0.01, 2) if self.direction == 1 else round(self.bar.low - 0.01, 2)


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
        
        # Bull flags: price is above EMA (uptrend context)
        if bar.close > ema[i]:
            # Pullback criteria: the prior bar must have made a lower low or been a bear bar
            is_pullback_bar = prev.is_bear or (prev.low < bars[i-2].low)
            
            # Setup Trigger: The current bar breaks above the high of the pullback bar
            if is_pullback_bar and bar.high > prev.high and bar.is_bull:
                # To simplify counting, we will classify the depth of the pullback by looking back 5 bars.
                # If price has been generally going down for a few bars, it's a higher-order flag (H2, H3).
                num_recent_pullbacks = sum(1 for b in bars[max(0, i-5):i] if b.is_bear or b.low < bars[b.idx-1].low)
                
                flag_num = 1
                if num_recent_pullbacks >= 3:
                    flag_num = 3
                elif num_recent_pullbacks >= 1:
                    flag_num = 2
                    
                flag_name = f"High {flag_num} Bull Flag"
                conf = 0.45 + (flag_num * 0.1) # H2/H3 have higher probability
                
                setups.append(Setup(
                    setup_name=flag_name,
                    entry_bar=bar.idx,
                    entry_price=round(prev.high + 0.01, 2),
                    order_type="Stop",
                    confidence=round(min(0.85, conf), 2),
                ))

        # Bear flags: price is below EMA (downtrend context)
        if bar.close < ema[i]:
            # Pullback criteria: the prior bar must have made a higher high or been a bull bar
            is_pullback_bar = prev.is_bull or (prev.high > bars[i-2].high)
            
            # Setup Trigger: The current bar breaks below the low of the pullback bar
            if is_pullback_bar and bar.low < prev.low and bar.is_bear:
                num_recent_pullbacks = sum(1 for b in bars[max(0, i-5):i] if b.is_bull or b.high > bars[b.idx-1].high)
                
                flag_num = 1
                if num_recent_pullbacks >= 3:
                    flag_num = 3
                elif num_recent_pullbacks >= 1:
                    flag_num = 2
                    
                flag_name = f"Low {flag_num} Bear Flag"
                conf = 0.45 + (flag_num * 0.1)
                
                setups.append(Setup(
                    setup_name=flag_name,
                    entry_bar=bar.idx,
                    entry_price=round(prev.low - 0.01, 2),
                    order_type="Stop",
                    confidence=round(min(0.85, conf), 2),
                ))

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

        # Bull Breakout: Either a massive single bar, or two consecutive moderately strong bars
        recent_high = max(b.high for b in bars[max(0, i-10):i-1]) if i >= 10 else max(b.high for b in bars[0:i-1])
        
        # 1. Single Massive Bar Breakout
        single_massive_bull = bar.is_strong_bull(avg_range) and bar.is_big(avg_range) and bar.close > recent_high
        
        # 2. Consecutive Strong Bars Breakout
        consecutive_bulls = bar.is_strong_bull(avg_range) and prev.is_strong_bull(avg_range) and bar.close > recent_high

        if (single_massive_bull or consecutive_bulls) and bar.close > ema[i]:
            setups.append(Setup(
                setup_name="Bull Breakout (BO)",
                entry_bar=bar.idx,
                entry_price=round(bar.close, 2),
                order_type="Stop",  # Often entered at market or stop above the breakout bar
                confidence=0.65 if consecutive_bulls else 0.55,
            ))

        # Bear Breakout
        recent_low = min(b.low for b in bars[max(0, i-10):i-1]) if i >= 10 else min(b.low for b in bars[0:i-1])
        
        # 1. Single Massive Bar Breakout
        single_massive_bear = bar.is_strong_bear(avg_range) and bar.is_big(avg_range) and bar.close < recent_low
        
        # 2. Consecutive Strong Bars Breakout
        consecutive_bears = bar.is_strong_bear(avg_range) and prev.is_strong_bear(avg_range) and bar.close < recent_low

        if (single_massive_bear or consecutive_bears) and bar.close < ema[i]:
            setups.append(Setup(
                setup_name="Bear Breakout (BO)",
                entry_bar=bar.idx,
                entry_price=round(bar.close, 2),
                order_type="Stop",
                confidence=0.65 if consecutive_bears else 0.55,
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


def detect_opening_reversals_and_spikes(bars: list[Bar], ema: list[float]) -> list[Setup]:
    """
    Opening Reversals and Spikes: High momentum setups occurring in the first 
    1-12 bars of the day (first hour).
    - Spike / Opening BO: A massive trend bar breaking out of the opening range.
    - Opening Reversal: A strong move in one direction that immediately reverses 
      with equal or greater force in the opposite direction.
    """
    setups = []
    if len(bars) < 5:
        return setups

    # Analyze the first hour of trading (bars 1 through 12 roughly)
    for i in range(1, min(15, len(bars))):
        bar = bars[i]
        prev = bars[i - 1]
        
        # Calculate a running average range based on prior days if possible, or just the opening volatility
        lookback_bars = bars[max(0, i-20):i] if i > 0 else [bar]
        avg_range = np.mean([b.range for b in lookback_bars]) if lookback_bars else bar.range
        if avg_range == 0:
            continue

        # For the first 5 bars, any solid trend bar relative to the open is a spike. 
        # After bar 5, demand a 1.25x explosive move to classify as a new spike.
        threshold = 1.0 if i < 5 else 1.25
        
        # 1. Opening Breakout / Spike (Bull)
        # In the first 25 minutes (bars 1-4), the 9:30 bar can have an insanely massive 
        # range (e.g. $3.00), masking a perfectly valid $1.50 spike on bar 2.
        # We drop the strict relative size requirement early on if it's a solid trend bar.
        is_spike_bull = bar.is_strong_bull(avg_range) and (bar.range >= (avg_range * threshold) or (i < 5 and bar.range >= avg_range * 0.5))
        is_spike_bear = bar.is_strong_bear(avg_range) and (bar.range >= (avg_range * threshold) or (i < 5 and bar.range >= avg_range * 0.5))
        
        prev_is_spike_bull = prev.is_strong_bull(avg_range) and (prev.range >= (avg_range * threshold) or (i < 5 and prev.range >= avg_range * 0.5))
        prev_is_spike_bear = prev.is_strong_bear(avg_range) and (prev.range >= (avg_range * threshold) or (i < 5 and prev.range >= avg_range * 0.5))

        if is_spike_bull:
            # Check if this spike is reversing a prior immediate bear attempt (Opening Reversal Bull)
            if prev_is_spike_bear:
                setups.append(Setup(
                    setup_name="Opening Reversal (Bull)",
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2), # Enter at market or stop above the reversal bar
                    order_type="Stop", 
                    confidence=0.70, # ORs are high probability traps
                ))
            else:
                # Regular Opening Spike / Breakout
                setups.append(Setup(
                    setup_name="Spike and Channel Bull", # Keeping legacy name for now to match UI enums
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2),
                    order_type="Stop/Market",  # Buy the close of the strong spike
                    confidence=0.60,
                ))
                
        # 2. Opening Breakout / Spike (Bear)
        if is_spike_bear:
            # Check if this spike is reversing a prior immediate bull attempt (Opening Reversal Bear)
            if prev_is_spike_bull:
                 setups.append(Setup(
                    setup_name="Opening Reversal (Bear)",
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2),
                    order_type="Stop",
                    confidence=0.70,
                ))
            else:
                # Regular Opening Spike / Breakout
                setups.append(Setup(
                    setup_name="Spike and Channel Bear",
                    entry_bar=bar.idx,
                    entry_price=round(bar.close, 2),
                    order_type="Stop/Market",
                    confidence=0.60,
                ))

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


# ─────────────────────────── REASONING GENERATORS ────────────────────────────

def _get_pattern_reason(setup: Setup, bars: list[Bar], ema: list[float]) -> str:
    """Generate reason 1: why this specific pattern is a good trade."""
    bar_idx = min(setup.entry_bar - 1, len(bars) - 1)
    bar = bars[bar_idx]
    name = setup.setup_name

    if "Bull Flag" in name or "High" in name:
        if bar.is_strong_bull(np.mean([b.range for b in bars])):
            return "a strong bull signal bar closing near its high with good follow-through potential"
        return "representing a pullback buy in a bull trend with a stop entry above the prior bar"

    if "Bear Flag" in name or "Low" in name:
        if bar.is_strong_bear(np.mean([b.range for b in bars])):
            return "a strong bear signal bar closing near its low with good follow-through potential"
        return "representing a pullback sell in a bear trend with a stop entry below the prior bar"

    if "Double Bottom" in name:
        return "confirming a failed attempt to break lower, creating a strong reversal setup"
    if "Double Top" in name:
        return "confirming a failed attempt to break higher, creating a strong reversal setup"

    if "Wedge Bottom" in name:
        return "completing a 3-push reversal pattern with increasing exhaustion on each push down"
    if "Wedge Top" in name:
        return "completing a 3-push reversal pattern with increasing exhaustion on each push up"

    if "Breakout" in name:
        return "with consecutive strong bars closing near their extremes breaking out of the prior range"

    if "Spike and Channel" in name:
        return "transitioning from an initial strong spike into a weaker channel continuation"

    if "20-Gap Bar" in name:
        return "with bars entirely gapping away from the 20 EMA confirming strong trend momentum"

    if "ii" in name or "ioi" in name:
        return "a tight consolidation pattern signaling a breakout is imminent"

    return f"offering a {setup.confidence:.0%} probability entry at ${setup.entry_price}"


def _get_context_reason(bars: list[Bar], ema: list[float], day_type: str) -> str:
    """Generate reason 2: why the market context supports this trade."""
    last_bar = bars[-1]
    avg_range = np.mean([b.range for b in bars])

    # EMA relationship
    if last_bar.close > ema[-1]:
        ema_msg = "Price is above the 20 EMA acting as dynamic support"
    else:
        ema_msg = "Price is below the 20 EMA acting as dynamic resistance"

    # Trend strength
    bull_count = sum(1 for b in bars[-10:] if b.is_bull)
    if bull_count >= 7:
        trend_msg = "with strong bullish momentum in recent bars"
    elif bull_count <= 3:
        trend_msg = "with strong bearish momentum in recent bars"
    else:
        trend_msg = "in a mixed/two-sided trading environment"

    return f"{ema_msg}, {trend_msg}."


# ─────────────────────────── USER ALGO DISCOVERY ───────────────────────────────

USER_ALGO_FUNCTIONS = []

def load_user_algos():
    """Dynamically discover and load any algorithm starting with 'detect_' in user_algos/."""
    global USER_ALGO_FUNCTIONS
    USER_ALGO_FUNCTIONS.clear()
    
    try:
        # Try to import user_algos; will succeed if run from the same directory
        import user_algos
        
        # Load all modules in the user_algos package
        path_list = [str(Path(p).resolve()) for p in user_algos.__path__] if hasattr(user_algos, "__path__") else [str(Path(user_algos.__file__).parent.resolve())]
        for _, name, _ in pkgutil.iter_modules(path_list):
            module = importlib.import_module(f"user_algos.{name}")
            
            # Find all functions starting with "detect_"
            for attr_name in dir(module):
                if attr_name.startswith("detect_"):
                    func = getattr(module, attr_name)
                    if callable(func):
                        USER_ALGO_FUNCTIONS.append(func)
    except Exception as e:
        print(f"Warning: Failed to load user algorithms: {e}")

# Load them on import
load_user_algos()


# ─────────────────────────── CONTEXTUAL FILTRATION ────────────────────────────

def filter_by_context(setups: list[Setup], day_type: str, market_cycle: str) -> list[Setup]:
    """
    Applies Al Brooks' macro rules to reject setups that fight the current environment.
    """
    filtered = []
    
    # Keywords indicating a setup is a "fade" (counter-trend)
    fade_keywords = ["Fade", "Top", "Bottom", "Reversal", "Exhaustion", "Test"]
    
    # Keywords indicating a setup is a "trend continuation"
    trend_keywords = ["Flag", "Breakout", "Stairs", "H1", "L1", "Pullback"]

    for s in setups:
        setup_name = s.setup_type if hasattr(s, 'setup_type') else getattr(s, 'setup_name', '')
        
        # RULE 1: Do not fade a Strong Trend / Spike
        if "Spike" in market_cycle or "Trend" in day_type:
            if any(k in setup_name for k in fade_keywords):
                continue # Reject this setup
                
        # RULE 2: Do not trade Trend setups in a Tight Trading Range
        if "Tight" in market_cycle or "Trading Range" in day_type:
            if any(k in setup_name for k in trend_keywords) and "Major" not in setup_name:
                continue # Reject this setup (Major Trend Reversals are allowed to form in ranges)
                
        # RULE 3: Do not buy Highs / sell Lows in a Broad Trading Range
        if "Broad" in market_cycle:
            if "Breakout" in setup_name and "Test" not in setup_name:
                continue # Reject breakouts in ranges (they usually fail)
                
        filtered.append(s)
        
    return filtered


# ─────────────────────────── MARKET PRESSURE ANALYSIS ────────────────────────

def evaluate_market_pressure(bars: list[Bar]) -> str:
    """
    Analyzes the last 10 bars to determine who is making money (Stop vs Limit traders).
    This tells us whether the market is trending or trapping participants in a range.
    """
    if len(bars) < 10:
        return ""
        
    recent_bars = bars[-10:]
    stop_bull_wins = 0
    limit_bear_wins = 0
    
    for i in range(1, len(recent_bars)):
        curr = recent_bars[i]
        prev = recent_bars[i-1]
        
        # Did Stop Bulls buy the prior bar's high?
        if curr.high > prev.high:
            # If the current bar closes near its high or above the entry, Stop Bulls are making money
            if curr.close >= prev.high + (prev.range * 0.1): 
                stop_bull_wins += 1
            # If the current bar violently reverses and closes below its midpoint, Limit Bears are making money fading the high
            elif curr.close < curr.open and curr.close < curr.low + (curr.range * 0.5):
                limit_bear_wins += 1

    if stop_bull_wins > limit_bear_wins and stop_bull_wins >= 2:
        return "Stop Bulls are consistently generating profitable follow-through above prior bars."
    elif limit_bear_wins > stop_bull_wins and limit_bear_wins >= 2:
        return "Limit Bears are actively making money by trapping bulls and heavily fading breakouts above prior bars."
    else:
        return "Neither side is demonstrating consistent follow-through; market pressure is deadlocked."


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

    # Inject EMA into Bars for user algo compatibility
    for idx_in_list, b in enumerate(bars):
        b.ema_20 = ema[idx_in_list]

    # Detect all patterns
    all_setups: list[Setup] = []
    all_setups.extend(detect_high_low_flags(bars, ema))
    all_setups.extend(detect_double_bottoms_tops(bars, swing_lows, swing_highs))
    all_setups.extend(detect_wedges(bars, swing_lows, swing_highs))
    all_setups.extend(detect_breakouts(bars, ema))
    all_setups.extend(detect_ii_ioi(bars))
    all_setups.extend(detect_ema_gap_bars(bars, ema))
    all_setups.extend(detect_opening_reversals_and_spikes(bars, ema))

    # Run User Algos
    for user_func in USER_ALGO_FUNCTIONS:
        try:
            user_setups = user_func(bars, ema)
            if user_setups:
                all_setups.extend(user_setups)
        except Exception as e:
            print(f"Warning: User algo {user_func.__name__} crashed: {e}")

    # Classify day type and market cycle
    day_type = classify_day_type(bars, ema)
    market_cycle = classify_market_cycle(bars, ema)

    # Apply Contextual Filtration
    all_setups = filter_by_context(all_setups, day_type, market_cycle)

    # --- BEST-PER-BAR SELECTION (no confluence renaming) ---
    # Multiple detectors fire at the same entry bar. Keep ONLY the single
    # best (highest confidence) setup per bar so each trade is cleanly
    # attributable to one setup type. Give a small confidence boost when
    # multiple distinct families agree at the same bar.

    _FAMILY_KEYWORDS = [
        "Flag", "Double Bottom", "Double Top", "Wedge", "Breakout",
        "ii", "ioi", "Gap Bar", "Opening Reversal", "Spike",
        "MTR", "Trend Reversal", "Pullback", "Climax", "EMA",
        "Stairs", "H&S", "Head and Shoulders", "Triangle",
        "Range", "Ledge", "OO Pattern", "Cup", "Parabolic",
        "Failed BO", "Limit", "Fade",
    ]

    def _setup_family(name: str) -> str:
        for kw in _FAMILY_KEYWORDS:
            if kw.lower() in name.lower():
                return kw
        return name

    bar_map: dict[int, list[Setup]] = {}
    for s in all_setups:
        bar_map.setdefault(s.entry_bar, []).append(s)

    deduped: list[Setup] = []
    for eb, setups_at_bar in bar_map.items():
        # Count distinct families for confluence bonus
        families = set(_setup_family(s.setup_name) for s in setups_at_bar)
        # Pick the single best setup at this bar
        best = max(setups_at_bar, key=lambda s: s.confidence)
        # Small confidence boost when multiple families agree
        if len(families) >= 2:
            best.confidence = min(0.95, best.confidence + (len(families) - 1) * 0.02)
        deduped.append(best)

    all_setups = deduped

    # Sort by confidence, take all valid filtered setups
    all_setups.sort(key=lambda s: s.confidence, reverse=True)
    
    # --- Time-Based Deduplication ---
    # Don't show the exact same setup name if it just fired in the last 5 bars
    final_setups = []
    seen_names_recent_bars = {} # setup_name -> last_seen_index
    
    for s in all_setups:
        name = s.setup_name
        idx = s.entry_bar
        
        # We also want to skip minor variations of recent setups (e.g. if Breakout BO is in the name)
        is_spam_spam = False
        for seen_name, seen_idx in seen_names_recent_bars.items():
            if (idx - seen_idx) < 3: # In the last 3 bars
                # If they are mostly the same string (e.g. Breakout in both)
                if ("Breakout" in name and "Breakout" in seen_name) or name == seen_name:
                    is_spam_spam = True
                    # Reset the timer so it continues suppressing consecutive bars
                    seen_names_recent_bars[seen_name] = idx
                    break
        
        if not is_spam_spam:
            final_setups.append(s)
            seen_names_recent_bars[name] = idx
            
    top_setups = final_setups

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

    # Build reasoning with two specific reasons for the best trade
    reasoning = ""
    if top_setups:
        best = top_setups[0]
        last_bar = bars[-1]
        avg_range = np.mean([b.range for b in bars])

        # Reason 1: pattern-based
        reason1 = _get_pattern_reason(best, bars, ema)
        # Reason 2: context-based
        reason2 = _get_context_reason(bars, ema, day_type)
        # Reason 3: momentum/pressure-based
        reason3 = evaluate_market_pressure(bars)

        # Comparative Reasoning: Why is this one better than the others?
        comparative_reasoning = ""
        if len(top_setups) > 1:
            runners_up = top_setups[1:4] # Get next 3
            runner_up_names = [f"Bar {s.entry_bar} {s.setup_name}" for s in runners_up]
            runner_up_str = ", ".join(runner_up_names)
            
            # Formulate the comparison based on confidence delta
            margin = best.confidence - runners_up[0].confidence
            if margin >= 0.10:
                comparative_reasoning = f"This setup was mathematically far superior to alternatives ({runner_up_str}) because it aligned perfectly with the {market_cycle} context, granting it a {margin*100:.0f}% higher probability of follow-through."
            elif margin > 0:
                comparative_reasoning = f"This setup narrowly edged out alternatives ({runner_up_str}) due to slightly better structural context and momentum alignment at the entry bar."
            else:
               comparative_reasoning = f"This setup tied with other high-probability signals ({runner_up_str}) but was selected as the primary trade due to having the earliest clean entry in the trend."
        else:
            comparative_reasoning = "This was the only mathematically viable high-probability setup detected for the entire day."

        reasoning = (
            f"**🏆 Best Trade of the Day: Bar {best.entry_bar} {best.setup_name} ({best.order_type} Order, {best.confidence:.0%} Conf)**\n\n"
            f"**Comparative Analysis:**\n{comparative_reasoning}\n\n"
            f"**Contextual Breakdown:**\n"
            f"{reason1}. {reason2} {reason3}"
        )
    else:
        reasoning = f"No high-probability setups detected. Day type: {day_type}, Market cycle: {market_cycle}."

    # Build enriched setup list with stop/target levels
    enriched_setups = []
    for i, s in enumerate(top_setups):
        direction = "Long"
        sell_kw = ["Bear", "Top", "Sell", "Low", "Short"]
        if any(kw in s.setup_name for kw in sell_kw):
            direction = "Short"

        # Compute Al Brooks stop / target using signal bar
        sig_idx = s.signal_bar if s.signal_bar > 0 else max(0, s.entry_bar - 1)
        if 0 <= sig_idx < len(bars):
            sb = bars[sig_idx]
            tick = 0.01
            if direction == "Long":
                entry = round(sb.high + tick, 2)
                stop = round(sb.low - tick, 2)
                risk = max(round(entry - stop, 2), tick)
                scalp = round(entry + risk, 2)
                swing = round(entry + 2 * risk, 2)
            else:
                entry = round(sb.low - tick, 2)
                stop = round(sb.high + tick, 2)
                risk = max(round(stop - entry, 2), tick)
                scalp = round(entry - risk, 2)
                swing = round(entry - 2 * risk, 2)
        else:
            entry = s.entry_price
            stop = 0.0
            risk = 0.0
            scalp = 0.0
            swing = 0.0

        enriched_setups.append({
            "rank": i + 1,
            "setup_name": s.setup_name,
            "entry_bar": s.entry_bar,
            "entry_price": entry if entry else s.entry_price,
            "order_type": s.order_type,
            "confidence": s.confidence,
            "direction": direction,
            "stop_loss": stop,
            "risk": risk,
            "scalp_target": scalp,
            "swing_target": swing,
            "num_setups_on_bar": len(bar_map.get(s.entry_bar, [s])),
        })

    return {
        "day_type": day_type,
        "market_cycle": market_cycle,
        "reasoning": reasoning,
        "setups": enriched_setups,
        "action": action,
        "confidence": round(overall_conf, 2),
    }


# ─────────────────────────── CLI TEST ────────────────────────────────────────

if __name__ == "__main__":
    import time
    import os

    ticker = "AAPL"
    try:
        from data_source import get_data_source
        db_key = os.environ.get("DATABENTO_API_KEY", "")
        if not db_key:
            print("DATABENTO_API_KEY is not set. Please set it to run the CLI demo.")
            raise SystemExit(1)

        source = get_data_source(api_key=db_key)
        print(f"Fetching 5-min data for {ticker} via Databento...")
        df = source.fetch_historical(ticker)
        if df is None or df.empty:
            print("No data returned from Databento.")
            raise SystemExit(1)

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
    except Exception as e:
        print(f"Databento demo failed: {e}")
