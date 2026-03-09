from typing import List, Dict, Any
from algo_engine import Setup, Bar

def detect_bear_stairs(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Setup #3: Buying 3rd or 4th push down in Bear Stairs.
    A bear stairs pattern is a series of lower highs and lower lows, 
    but with deep pullbacks (overlapping bars) acting like a channel.
    The 3rd or 4th push is usually the final trap before a reversal.
    """
    setups = []
    if len(bars) < 20: return setups
    
    for i in range(15, len(bars)):
        curr = bars[i]
        
        # Look back to count distinct descending pushes (swing lows)
        past_15 = bars[i-15:i]
        
        # Extremely simplified push counter: counting local minima
        local_minima = 0
        for j in range(2, len(past_15) - 2):
            if past_15[j].low < past_15[j-1].low and past_15[j].low < past_15[j-2].low and \
               past_15[j].low < past_15[j+1].low and past_15[j].low < past_15[j+2].low:
                local_minima += 1
                
        # If we see 3 or 4 pushes down, and the current bar is a strong bull reversal
        if local_minima >= 3:
            if curr.close > curr.open and curr.close > curr.midpoint and curr.body_size > curr.range * 0.5:
                setups.append(Setup(
                    index=i, bar=curr, setup_name="Bear Stairs Reversal (3rd/4th Push)",
                    direction=1, confidence=0.75, notes="Exhaustion of sellers after 3+ pushes down in a stair-step channel."
                ))
    return setups

def _find_spike(bars: List[Bar], end_idx: int, min_bars: int = 3, body_ratio: float = 0.40):
    """
    Scan backwards from end_idx to find the most recent spike.
    Returns (direction, start, end, high, low, avg_range) or None.
    """
    for j in range(end_idx, max(min_bars - 1, end_idx - 50) - 1, -1):
        if j < min_bars:
            break

        # Bull spike
        count = 0
        for k in range(j, max(j - 8, -1), -1):
            b = bars[k]
            if b.is_bull and b.close > b.midpoint and b.range > 0 and b.body / b.range >= body_ratio:
                count += 1
            else:
                break
        if count >= min_bars:
            start = j - count + 1
            sl = bars[start:j + 1]
            return ("bull", start, j,
                    max(b.high for b in sl), min(b.low for b in sl),
                    sum(b.range for b in sl) / len(sl))

        # Bear spike
        count = 0
        for k in range(j, max(j - 8, -1), -1):
            b = bars[k]
            if b.is_bear and b.close < b.midpoint and b.range > 0 and b.body / b.range >= body_ratio:
                count += 1
            else:
                break
        if count >= min_bars:
            start = j - count + 1
            sl = bars[start:j + 1]
            return ("bear", start, j,
                    max(b.high for b in sl), min(b.low for b in sl),
                    sum(b.range for b in sl) / len(sl))

    return None


def detect_spike_buy_market(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    With-trend entry: buy/sell at the market immediately after a strong spike.

    Bull spike (3+ consecutive strong bull bars) → Buy at the close of the
    spike (next bar's open), stop below spike low, target 1:1.

    Bear spike → Sell at close of spike, stop above spike high, target 1:1.

    This is the simplest with-trend spike trade: you see strength, you join.
    """
    setups = []
    if len(bars) < 5:
        return setups

    MIN_SPIKE = 3
    BODY_RATIO = 0.40
    COOLDOWN = 5

    last_fire = -999

    for i in range(MIN_SPIKE, len(bars)):
        # Check if a spike just ended at bar i-1 (the previous bar is the
        # last bar of the spike, and bar i is the first bar after the spike)
        spike = _find_spike(bars, i - 1, min_bars=MIN_SPIKE, body_ratio=BODY_RATIO)
        if spike is None:
            continue

        direction, sp_start, sp_end, s_high, s_low, s_avg = spike

        # Spike must have JUST ended (sp_end == i-1)
        if sp_end != i - 1:
            continue

        if (i - last_fire) <= COOLDOWN:
            continue

        curr = bars[i]
        spike_size = s_high - s_low
        if spike_size <= 0:
            continue

        if direction == "bull":
            setups.append(Setup(
                index=i, bar=curr,
                setup_name="Bull Spike Buy",
                direction=1, confidence=0.75,
                notes=f"Buy at market after {sp_end - sp_start + 1}-bar bull spike. Stop below spike low.",
            ))
        else:
            setups.append(Setup(
                index=i, bar=curr,
                setup_name="Bear Spike Sell",
                direction=-1, confidence=0.75,
                notes=f"Sell at market after {sp_end - sp_start + 1}-bar bear spike. Stop above spike high.",
            ))
        last_fire = i

    return setups


def detect_spike_buy_pullback(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    With-trend entry: wait for a small pullback after a strong spike, then enter.

    After a bull spike, wait for price to pull back (bar with low < prior bar's low),
    then enter on the pullback bar if it closes in the trend direction.
    Stop below spike low, target 1:1.

    This is the scaling-in approach: wait for a better entry after the spike.
    """
    setups = []
    if len(bars) < 6:
        return setups

    MIN_SPIKE = 3
    BODY_RATIO = 0.40
    COOLDOWN = 5
    MAX_PB_BARS = 5   # pullback must happen within 5 bars of spike end

    last_fire = -999

    for i in range(MIN_SPIKE + 1, len(bars)):
        if (i - last_fire) <= COOLDOWN:
            continue

        curr = bars[i]

        # Look for a spike that ended 1-5 bars ago
        for lookback in range(1, MAX_PB_BARS + 1):
            sp_end_candidate = i - lookback
            if sp_end_candidate < MIN_SPIKE:
                break

            spike = _find_spike(bars, sp_end_candidate, min_bars=MIN_SPIKE, body_ratio=BODY_RATIO)
            if spike is None:
                continue

            direction, sp_start, sp_end, s_high, s_low, s_avg = spike
            if sp_end != sp_end_candidate:
                continue

            spike_size = s_high - s_low
            if spike_size <= 0:
                continue

            if direction == "bull":
                # Pullback: current bar's low < prior bar's low (price dipped)
                if curr.low < bars[i - 1].low:
                    # Pullback shouldn't give back more than 50% of spike
                    pullback_depth = s_high - curr.low
                    if pullback_depth <= spike_size * 0.50:
                        setups.append(Setup(
                            index=i, bar=curr,
                            setup_name="Bull Spike Pullback Buy",
                            direction=1, confidence=0.78,
                            notes=f"Pullback buy {lookback} bars after {sp_end - sp_start + 1}-bar bull spike. PB depth {pullback_depth:.2f}.",
                        ))
                        last_fire = i
                        break  # found a valid entry, stop looking
            else:
                # Pullback: current bar's high > prior bar's high (price bounced)
                if curr.high > bars[i - 1].high:
                    pullback_depth = curr.high - s_low
                    if pullback_depth <= spike_size * 0.50:
                        setups.append(Setup(
                            index=i, bar=curr,
                            setup_name="Bear Spike Pullback Sell",
                            direction=-1, confidence=0.78,
                            notes=f"Pullback sell {lookback} bars after {sp_end - sp_start + 1}-bar bear spike. PB depth {pullback_depth:.2f}.",
                        ))
                        last_fire = i
                        break

    return setups


def detect_h1_l1_in_strong_spike(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Setups #10 & #11: H1 in strong bull spike / L1 in strong bear spike.
    Only valid if the spike is freshly breaking out (not a climax).
    """
    setups = []
    if len(bars) < 5: return setups
    
    for i in range(4, len(bars)):
        curr = bars[i]
        prev1 = bars[i-1]
        prev2 = bars[i-2]
        prev3 = bars[i-3]
        
        # Bull Spike: 3 consecutive strong bull bars closing near their highs
        if prev3.close > prev3.open and prev2.close > prev2.open and prev1.close > prev1.open:
            if prev1.close > prev1.midpoint and prev2.close > prev2.midpoint:
                # The H1: A small pullback bar that doesn't reverse the whole spike
                if curr.low < prev1.low and curr.close > curr.open:
                    setups.append(Setup(index=i, bar=curr, setup_name="H1 in Strong Bull Spike", direction=1, confidence=0.85, notes="First pullback (H1) after 3-bar bull breakout spike."))
                    
        # Bear Spike: 3 consecutive strong bear bars closing near their lows
        elif prev3.close < prev3.open and prev2.close < prev2.open and prev1.close < prev1.open:
            if prev1.close < prev1.midpoint and prev2.close < prev2.midpoint:
                # The L1: A small pullback bar
                if curr.high > prev1.high and curr.close < curr.open:
                    setups.append(Setup(index=i, bar=curr, setup_name="L1 in Strong Bear Spike", direction=-1, confidence=0.85, notes="First pullback (L1) after 3-bar bear breakout spike."))
                    
    return setups


def detect_range_boundary_fades(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Setups #12 & #13: Short top of trading range / Buy bottom of trading range.
    *Especially* on the 2nd entry attempt.
    """
    setups = []
    if len(bars) < 30: return setups
    
    for i in range(25, len(bars)):
        curr = bars[i]
        
        # Define the range over the last 20 bars
        range_bars = bars[i-20:i-1]
        range_high = max(b.high for b in range_bars)
        range_low = min(b.low for b in range_bars)
        
        # Make sure it's actually a horizontal range (EMA is relatively flat)
        ema_start = range_bars[0].ema_20
        ema_end = range_bars[-1].ema_20
        if abs(ema_end - ema_start) < (range_high - range_low) * 0.2:
            
            # Fade the Top (Shorting)
            # If price pushes up to the old high, but immediately reverses into a bear bar
            if curr.high >= (range_high * 0.998) and curr.close < curr.midpoint:
                setups.append(Setup(index=i, bar=curr, setup_name="Trading Range Top Reversal", direction=-1, confidence=0.80, notes="Price rejected perfectly off the established 20-bar range ceiling."))
                
            # Fade the Bottom (Buying)
            # If price pushes down to the old low, but immediately reverses into a bull bar
            elif curr.low <= (range_low * 1.002) and curr.close > curr.midpoint:
                setups.append(Setup(index=i, bar=curr, setup_name="Trading Range Bottom Reversal", direction=1, confidence=0.80, notes="Price rejected perfectly off the established 20-bar range floor."))
                
    return setups
