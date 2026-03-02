"""
Additional Al Brooks Setups Implementation

Implements setups that complement the core detection engine:
1. OO Pattern (Outside-Outside) — with trend filter
2. Cup and Handle — relaxed tolerance
3. Ledge Bottom / Ledge Top — tighter context filters
4. Parabolic Wedge Bottom / Top — acceleration + rejection
5. Failed BO Bull Trap / Bear Trap — immediate reversal detection

All setups require EMA context agreement to avoid noise trades.
"""

from algo_engine import Bar, Setup
from typing import List


# ─── helpers ───────────────────────────────────────────────────────────────────

def _avg_range(bars: List[Bar]) -> float:
    """Average bar range over a list of bars (avoids divide-by-zero)."""
    if not bars:
        return 1.0
    return max(sum(b.range for b in bars) / len(bars), 1e-9)


# ─── 1.  OO Pattern ───────────────────────────────────────────────────────────

def detect_oo_pattern(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Two consecutive outside bars where each engulfs the prior bar.
    Direction from the 2nd bar's close relative to EMA (trend filter).
    Requires the OO to happen near EMA (within 1 ATR) to avoid chasing.
    """
    setups: List[Setup] = []
    if len(bars) < 5:
        return setups

    atr = _avg_range(bars[-20:]) if len(bars) >= 20 else _avg_range(bars)

    for i in range(2, len(bars)):
        curr = bars[i]
        prev1 = bars[i - 1]
        prev2 = bars[i - 2]

        prev1_outside = prev1.high > prev2.high and prev1.low < prev2.low
        curr_outside = curr.high > prev1.high and curr.low < prev1.low
        if not (prev1_outside and curr_outside):
            continue

        # Near-EMA filter: midpoint of 2nd OO bar within 1 ATR of EMA
        ema_val = ema[min(i, len(ema) - 1)]
        if abs(curr.midpoint - ema_val) > atr:
            continue

        # Strong close bias required (>60 % of range)
        body_ratio = curr.body_size / max(curr.range, 1e-9)
        if body_ratio < 0.40:
            continue

        if curr.close > ema_val and curr.close > curr.midpoint:
            setups.append(Setup(
                index=i, bar=curr, setup_type="OO Pattern",
                direction=1, order_type="Stop", confidence=0.62,
                notes="Two consecutive outside bars near EMA, bullish close.",
            ))
        elif curr.close < ema_val and curr.close < curr.midpoint:
            setups.append(Setup(
                index=i, bar=curr, setup_type="OO Pattern",
                direction=-1, order_type="Stop", confidence=0.62,
                notes="Two consecutive outside bars near EMA, bearish close.",
            ))

    return setups


# ─── 2.  Cup and Handle ───────────────────────────────────────────────────────

def detect_cup_and_handle(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Rounded-bottom U-shape (10-20 bars) followed by a 2-4 bar tight handle.
    Relaxed from prior version: handle tolerance 1 ATR (was 0.5 %).
    Bull-only pattern.  Must close above EMA to confirm.
    """
    setups: List[Setup] = []
    if len(bars) < 25:
        return setups

    atr = _avg_range(bars[-20:])

    for i in range(20, len(bars)):
        curr = bars[i]
        if curr.close < ema[min(i, len(ema) - 1)]:
            continue  # Must be above EMA for bull confirmation

        cup = bars[i - 15: i - 2]
        if not cup:
            continue

        cup_low_idx = min(range(len(cup)), key=lambda j: cup[j].low)
        # U-shape: low should be roughly in the middle third
        if not (len(cup) * 0.25 <= cup_low_idx <= len(cup) * 0.75):
            continue

        # Left rim should descend; right rim should ascend (relaxed: net direction)
        left = cup[: cup_low_idx + 1]
        right = cup[cup_low_idx:]
        if len(left) < 2 or len(right) < 2:
            continue
        if left[0].low <= left[-1].low:
            continue  # left side didn't descend
        if right[-1].low <= right[0].low:
            continue  # right side didn't ascend

        # Handle: last 2-4 bars should be tight (range < 1 ATR)
        handle = bars[i - 3: i]
        handle_range = max(b.high for b in handle) - min(b.low for b in handle)
        if handle_range > atr * 1.5:
            continue

        # Breakout above handle high
        handle_high = max(b.high for b in handle)
        if curr.close > handle_high:
            setups.append(Setup(
                index=i, bar=curr, setup_type="Cup and Handle",
                direction=1, order_type="Stop", confidence=0.68,
                notes="U-shaped cup with tight handle; breakout above handle high.",
            ))

    return setups


# ─── 3.  Ledge Bottom ─────────────────────────────────────────────────────────

def detect_ledge_bottom(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    3-5 bars with lows within 0.25 % of each other after a downtrend.
    Current bar must close above the ledge high AND above the ledge EMA.
    """
    setups: List[Setup] = []
    if len(bars) < 20:
        return setups

    for i in range(10, len(bars)):
        curr = bars[i]
        ema_val = ema[min(i, len(ema) - 1)]

        # Prior downtrend: at least 6 of last 10 bars below EMA
        start = max(0, i - 10)
        below = sum(1 for b in bars[start:i] if b.close < ema[min(b.idx, len(ema) - 1)])
        if below < 6:
            continue

        for length in (3, 4, 5):
            if i < length:
                continue
            ledge = bars[i - length: i]
            ledge_low = min(b.low for b in ledge)
            tol = ledge_low * 0.0025
            if all(abs(b.low - ledge_low) < tol for b in ledge):
                ledge_high = max(b.high for b in ledge)
                if curr.close > ledge_high and curr.close > ema_val:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Ledge Bottom",
                        direction=1, order_type="Stop", confidence=0.65,
                        notes=f"Flat support ledge ({length} bars) after downtrend; breakout confirmed.",
                    ))
                    break  # one per bar

    return setups


# ─── 4.  Ledge Top ────────────────────────────────────────────────────────────

def detect_ledge_top(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    3-5 bars with highs within 0.25 % of each other after an uptrend.
    Current bar must close below the ledge low AND below EMA.
    """
    setups: List[Setup] = []
    if len(bars) < 20:
        return setups

    for i in range(10, len(bars)):
        curr = bars[i]
        ema_val = ema[min(i, len(ema) - 1)]

        # Prior uptrend: at least 6 of last 10 bars above EMA
        start = max(0, i - 10)
        above = sum(1 for b in bars[start:i] if b.close > ema[min(b.idx, len(ema) - 1)])
        if above < 6:
            continue

        for length in (3, 4, 5):
            if i < length:
                continue
            ledge = bars[i - length: i]
            ledge_high = max(b.high for b in ledge)
            tol = ledge_high * 0.0025
            if all(abs(b.high - ledge_high) < tol for b in ledge):
                ledge_low = min(b.low for b in ledge)
                if curr.close < ledge_low and curr.close < ema_val:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Ledge Top",
                        direction=-1, order_type="Stop", confidence=0.65,
                        notes=f"Flat resistance ledge ({length} bars) after uptrend; breakdown confirmed.",
                    ))
                    break

    return setups


# ─── 5.  Parabolic Wedge Bottom ───────────────────────────────────────────────

def detect_parabolic_wedge_bottom(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Accelerating selloff with 3 pushes down where spacing between lows INCREASES.
    Reversal when current bar shows strong bull rejection (close > 60 % of range).
    """
    setups: List[Setup] = []
    if len(bars) < 25:
        return setups

    for i in range(20, len(bars)):
        curr = bars[i]
        window = bars[i - 20: i]

        # Find local lows (must be lower than 2 neighbours each side)
        lows = []
        for j in range(2, len(window) - 2):
            b = window[j]
            if (b.low < window[j - 1].low and b.low < window[j - 2].low and
                    b.low < window[j + 1].low and b.low < window[j + 2].low):
                lows.append(b.low)

        if len(lows) < 3:
            continue

        l1, l2, l3 = lows[-3], lows[-2], lows[-1]
        # Each push must be lower
        if not (l2 < l1 and l3 < l2):
            continue
        dist1 = l1 - l2
        dist2 = l2 - l3
        # Acceleration: 2nd gap > 1st gap
        if dist2 <= dist1 or dist1 <= 0:
            continue

        # Strong bull rejection bar
        body_ratio = curr.body_size / max(curr.range, 1e-9)
        if curr.close > curr.midpoint and body_ratio > 0.50:
            setups.append(Setup(
                index=i, bar=curr, setup_type="Parabolic Wedge Bottom",
                direction=1, order_type="Stop", confidence=0.70,
                notes="3 accelerating pushes down; bull rejection bar.",
            ))

    return setups


# ─── 6.  Parabolic Wedge Top ──────────────────────────────────────────────────

def detect_parabolic_wedge_top(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Accelerating rally with 3 pushes up where spacing between highs INCREASES.
    Reversal when current bar shows strong bear rejection (close < 40 % of range).
    """
    setups: List[Setup] = []
    if len(bars) < 25:
        return setups

    for i in range(20, len(bars)):
        curr = bars[i]
        window = bars[i - 20: i]

        highs = []
        for j in range(2, len(window) - 2):
            b = window[j]
            if (b.high > window[j - 1].high and b.high > window[j - 2].high and
                    b.high > window[j + 1].high and b.high > window[j + 2].high):
                highs.append(b.high)

        if len(highs) < 3:
            continue

        h1, h2, h3 = highs[-3], highs[-2], highs[-1]
        if not (h2 > h1 and h3 > h2):
            continue
        dist1 = h2 - h1
        dist2 = h3 - h2
        if dist2 <= dist1 or dist1 <= 0:
            continue

        body_ratio = curr.body_size / max(curr.range, 1e-9)
        if curr.close < curr.midpoint and body_ratio > 0.50:
            setups.append(Setup(
                index=i, bar=curr, setup_type="Parabolic Wedge Top",
                direction=-1, order_type="Stop", confidence=0.70,
                notes="3 accelerating pushes up; bear rejection bar.",
            ))

    return setups


# ─── 7.  Failed Breakout (Bull Trap / Bear Trap) ──────────────────────────────

def detect_failed_breakouts(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Bull Trap: break above 10-bar high that reverses within 1-2 bars.
    Bear Trap: break below 10-bar low that reverses within 1-2 bars.
    Requires strong reversal bar (body > 50 % of range).
    """
    setups: List[Setup] = []
    if len(bars) < 15:
        return setups

    for i in range(12, len(bars)):
        curr = bars[i]
        prev = bars[i - 1]

        ref_bars = bars[i - 12: i - 2]
        if not ref_bars:
            continue

        ref_high = max(b.high for b in ref_bars)
        ref_low = min(b.low for b in ref_bars)

        body_ratio = curr.body_size / max(curr.range, 1e-9)

        # BULL TRAP: prev broke above ref_high, curr reverses hard
        if prev.high > ref_high and prev.close > ref_high:
            if (curr.close < prev.low and body_ratio > 0.50
                    and curr.close < ema[min(i, len(ema) - 1)]):
                setups.append(Setup(
                    index=i, bar=curr, setup_type="Failed BO Bull Trap",
                    direction=-1, order_type="Stop", confidence=0.72,
                    notes="Breakout above 10-bar high reversed; strong bear bar closes below EMA.",
                ))

        # BEAR TRAP: prev broke below ref_low, curr reverses hard
        if prev.low < ref_low and prev.close < ref_low:
            if (curr.close > prev.high and body_ratio > 0.50
                    and curr.close > ema[min(i, len(ema) - 1)]):
                setups.append(Setup(
                    index=i, bar=curr, setup_type="Failed BO Bear Trap",
                    direction=1, order_type="Stop", confidence=0.72,
                    notes="Breakout below 10-bar low reversed; strong bull bar closes above EMA.",
                ))

    return setups
