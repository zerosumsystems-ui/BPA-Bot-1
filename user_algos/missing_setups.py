"""
Missing Al Brooks Setups Implementation

Implements the 7 missing setups from SETUP_OPTIONS:
1. OO Pattern (Outside-Outside)
2. Cup and Handle
3. Ledge Bottom
4. Ledge Top
5. Parabolic Wedge Bottom
6. Parabolic Wedge Top
7. Failed BO Bull Trap / Failed BO Bear Trap
"""

from algo_engine import Bar, Setup
from typing import List


def detect_oo_pattern(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects OO Pattern (Outside-Outside).

    Two consecutive outside bars where each bar's range engulfs the prior bar.
    Entry direction is from the 2nd bar's close.
    Entry is placed above/below the 2nd bar.

    Confidence: 0.55
    Order Type: Stop
    """
    setups = []

    if len(bars) < 3:
        return setups

    for i in range(2, len(bars)):
        curr = bars[i]
        prev1 = bars[i-1]
        prev2 = bars[i-2]

        # Check if prev1 is an outside bar relative to prev2
        prev1_outside = prev1.high > prev2.high and prev1.low < prev2.low

        # Check if curr is an outside bar relative to prev1
        curr_outside = curr.high > prev1.high and curr.low < prev1.low

        if prev1_outside and curr_outside:
            # Determine direction from current bar's close
            if curr.close > curr.midpoint:
                # Bull setup - entry above current bar
                setups.append(Setup(
                    index=i,
                    bar=curr,
                    setup_type="OO Pattern",
                    direction=1,
                    order_type="Stop",
                    confidence=0.55,
                    notes="Two consecutive outside bars, bullish close."
                ))
            elif curr.close < curr.midpoint:
                # Bear setup - entry below current bar
                setups.append(Setup(
                    index=i,
                    bar=curr,
                    setup_type="OO Pattern",
                    direction=-1,
                    order_type="Stop",
                    confidence=0.55,
                    notes="Two consecutive outside bars, bearish close."
                ))

    return setups


def detect_cup_and_handle(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Cup and Handle pattern.

    Rounded bottom (U-shape over 10-20 bars) where lows form a U shape
    (descend then ascend), followed by a 2-3 bar tight pullback (handle).

    Confidence: 0.70
    Order Type: Stop, Direction: Long
    """
    setups = []

    if len(bars) < 25:
        return setups

    for i in range(20, len(bars)):
        curr = bars[i]

        # Look back 15 bars for the cup formation
        cup_window = bars[i-15:i]

        # Find the lowest point in the cup
        cup_low = min(b.low for b in cup_window)
        cup_low_idx = min(range(len(cup_window)),
                         key=lambda j: cup_window[j].low)

        # Check if lows descend then ascend (U-shape)
        first_half = cup_window[:cup_low_idx+1]
        second_half = cup_window[cup_low_idx:]

        # First half should generally be descending
        first_descending = len(first_half) < 2 or all(
            first_half[j].low >= first_half[j+1].low
            for j in range(len(first_half)-1)
        )

        # Second half should generally be ascending
        second_ascending = len(second_half) < 2 or all(
            second_half[j].low <= second_half[j+1].low
            for j in range(len(second_half)-1)
        )

        if first_descending and second_ascending:
            # Now check for the handle (2-3 bar tight pullback)
            handle_window = bars[i-3:i]
            handle_high = max(b.high for b in handle_window)
            handle_low = min(b.low for b in handle_window)
            handle_range = handle_high - handle_low

            # Handle should be tight (less than 0.5% of price)
            if handle_range < (curr.close * 0.005):
                # Entry is above the handle high
                if curr.close > curr.midpoint:
                    setups.append(Setup(
                        index=i,
                        bar=curr,
                        setup_type="Cup and Handle",
                        direction=1,
                        order_type="Stop",
                        confidence=0.70,
                        notes="U-shaped cup followed by tight handle pullback."
                    ))

    return setups


def detect_ledge_bottom(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Ledge Bottom pattern.

    3+ consecutive bars with lows within 0.15% of each other (flat support).
    Requires prior downtrend (price below EMA).
    Entry is above the ledge high.

    Confidence: 0.65
    Order Type: Stop, Direction: Long
    """
    setups = []

    if len(bars) < 20:
        return setups

    for i in range(15, len(bars)):
        curr = bars[i]

        # Check for prior downtrend: majority of past 10 bars below EMA
        past_10 = bars[i-10:i]
        below_ema_count = sum(1 for b in past_10 if b.close < ema[b.idx])

        if below_ema_count >= 7:  # At least 70% below EMA
            # Look for 3+ bars with lows within 0.15%
            for ledge_len in range(3, 6):  # Check 3-5 bar ledges
                if i >= ledge_len:
                    ledge_bars = bars[i-ledge_len:i]
                    ledge_lows = [b.low for b in ledge_bars]

                    ledge_low = min(ledge_lows)
                    ledge_high = max(b.high for b in ledge_bars)

                    # Check if all lows are within 0.15% of the minimum low
                    tolerance = ledge_low * 0.0015
                    lows_aligned = all(
                        abs(low - ledge_low) < tolerance
                        for low in ledge_lows
                    )

                    if lows_aligned and curr.close > ledge_high:
                        setups.append(Setup(
                            index=i,
                            bar=curr,
                            setup_type="Ledge Bottom",
                            direction=1,
                            order_type="Stop",
                            confidence=0.65,
                            notes=f"Flat support ledge ({ledge_len} bars) after downtrend."
                        ))
                        break

    return setups


def detect_ledge_top(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Ledge Top pattern.

    3+ consecutive bars with highs within 0.15% of each other (flat resistance).
    Requires prior uptrend (price above EMA).
    Entry is below the ledge low.

    Confidence: 0.65
    Order Type: Stop, Direction: Short
    """
    setups = []

    if len(bars) < 20:
        return setups

    for i in range(15, len(bars)):
        curr = bars[i]

        # Check for prior uptrend: majority of past 10 bars above EMA
        past_10 = bars[i-10:i]
        above_ema_count = sum(1 for b in past_10 if b.close > ema[b.idx])

        if above_ema_count >= 7:  # At least 70% above EMA
            # Look for 3+ bars with highs within 0.15%
            for ledge_len in range(3, 6):  # Check 3-5 bar ledges
                if i >= ledge_len:
                    ledge_bars = bars[i-ledge_len:i]
                    ledge_highs = [b.high for b in ledge_bars]

                    ledge_high = max(ledge_highs)
                    ledge_low = min(b.low for b in ledge_bars)

                    # Check if all highs are within 0.15% of the maximum high
                    tolerance = ledge_high * 0.0015
                    highs_aligned = all(
                        abs(high - ledge_high) < tolerance
                        for high in ledge_highs
                    )

                    if highs_aligned and curr.close < ledge_low:
                        setups.append(Setup(
                            index=i,
                            bar=curr,
                            setup_type="Ledge Top",
                            direction=-1,
                            order_type="Stop",
                            confidence=0.65,
                            notes=f"Flat resistance ledge ({ledge_len} bars) after uptrend."
                        ))
                        break

    return setups


def detect_parabolic_wedge_bottom(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Parabolic Wedge Bottom.

    Accelerating selloff with 3 pushes down where the distance between lows
    INCREASES (parabolic acceleration). Reversal when acceleration stalls.

    Confidence: 0.70
    Order Type: Stop, Direction: Long
    """
    setups = []

    if len(bars) < 25:
        return setups

    for i in range(20, len(bars)):
        curr = bars[i]

        # Look back to find 3 distinct pushes (lows)
        past_20 = bars[i-20:i]

        # Simple approach: find local lows
        local_lows = []
        for j in range(2, len(past_20)-2):
            if past_20[j].low < past_20[j-1].low and \
               past_20[j].low < past_20[j-2].low and \
               past_20[j].low < past_20[j+1].low and \
               past_20[j].low < past_20[j+2].low:
                local_lows.append((j, past_20[j].low))

        # Need at least 3 distinct lows
        if len(local_lows) >= 3:
            low1_idx, low1_val = local_lows[-3]
            low2_idx, low2_val = local_lows[-2]
            low3_idx, low3_val = local_lows[-1]

            # Calculate distances between lows
            dist1 = abs(low1_val - low2_val)
            dist2 = abs(low2_val - low3_val)

            # Parabolic acceleration: second distance should be larger than first
            if dist2 > dist1 and dist1 > 0:
                # Check if current bar is rejecting further downside
                if curr.close > curr.midpoint and curr.body_size > curr.range * 0.3:
                    setups.append(Setup(
                        index=i,
                        bar=curr,
                        setup_type="Parabolic Wedge Bottom",
                        direction=1,
                        order_type="Stop",
                        confidence=0.70,
                        notes="3 pushes down with accelerating spacing; acceleration stalling."
                    ))

    return setups


def detect_parabolic_wedge_top(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Parabolic Wedge Top.

    Accelerating rally with 3 pushes up where the distance between highs
    INCREASES (parabolic acceleration). Reversal when acceleration stalls.

    Confidence: 0.70
    Order Type: Stop, Direction: Short
    """
    setups = []

    if len(bars) < 25:
        return setups

    for i in range(20, len(bars)):
        curr = bars[i]

        # Look back to find 3 distinct pushes (highs)
        past_20 = bars[i-20:i]

        # Simple approach: find local highs
        local_highs = []
        for j in range(2, len(past_20)-2):
            if past_20[j].high > past_20[j-1].high and \
               past_20[j].high > past_20[j-2].high and \
               past_20[j].high > past_20[j+1].high and \
               past_20[j].high > past_20[j+2].high:
                local_highs.append((j, past_20[j].high))

        # Need at least 3 distinct highs
        if len(local_highs) >= 3:
            high1_idx, high1_val = local_highs[-3]
            high2_idx, high2_val = local_highs[-2]
            high3_idx, high3_val = local_highs[-1]

            # Calculate distances between highs
            dist1 = abs(high1_val - high2_val)
            dist2 = abs(high2_val - high3_val)

            # Parabolic acceleration: second distance should be larger than first
            if dist2 > dist1 and dist1 > 0:
                # Check if current bar is rejecting further upside
                if curr.close < curr.midpoint and curr.body_size > curr.range * 0.3:
                    setups.append(Setup(
                        index=i,
                        bar=curr,
                        setup_type="Parabolic Wedge Top",
                        direction=-1,
                        order_type="Stop",
                        confidence=0.70,
                        notes="3 pushes up with accelerating spacing; acceleration stalling."
                    ))

    return setups


def detect_failed_breakouts(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Failed Breakout (Bull Trap / Bear Trap).

    Bull Trap: Breakout above recent 10-bar high that immediately reverses within 2 bars.
    Bear Trap: Breakout below recent 10-bar low that reverses within 2 bars.

    Confidence: 0.75
    Order Type: Stop
    """
    setups = []

    if len(bars) < 15:
        return setups

    for i in range(12, len(bars)):
        curr = bars[i]
        prev1 = bars[i-1] if i >= 1 else None
        prev2 = bars[i-2] if i >= 2 else None

        if not prev1 or not prev2:
            continue

        # Look back 10 bars for reference highs/lows
        lookback_window = bars[i-10:i-2]
        if not lookback_window:
            continue

        ref_high = max(b.high for b in lookback_window)
        ref_low = min(b.low for b in lookback_window)

        # BULL TRAP: prev1 broke above ref_high, but curr bar reverses down
        if prev1.high > ref_high and prev1.close > ref_high:
            if curr.close < prev1.close and curr.low < prev1.midpoint:
                setups.append(Setup(
                    index=i,
                    bar=curr,
                    setup_type="Failed BO Bull Trap",
                    direction=-1,
                    order_type="Stop",
                    confidence=0.75,
                    notes="Breakout above 10-bar high failed immediately; bull trap reversal."
                ))

        # BEAR TRAP: prev1 broke below ref_low, but curr bar reverses up
        if prev1.low < ref_low and prev1.close < ref_low:
            if curr.close > prev1.close and curr.high > prev1.midpoint:
                setups.append(Setup(
                    index=i,
                    bar=curr,
                    setup_type="Failed BO Bear Trap",
                    direction=1,
                    order_type="Stop",
                    confidence=0.75,
                    notes="Breakout below 10-bar low failed immediately; bear trap reversal."
                ))

    return setups
