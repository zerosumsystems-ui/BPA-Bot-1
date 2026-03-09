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


def _score_spike_strength(bars: List[Bar], sp_start: int, sp_end: int,
                          direction: str, context_bars: int = 20) -> int:
    """
    Score a spike 0-5 based on Al Brooks' signs of strength.

    1. BODY GAPS — bodies of consecutive spike bars don't overlap (measuring gaps)
    2. CLOSES NEAR EXTREMES — small tails in the trend direction (< 25% of range)
    3. SURPRISE SIZE — spike total range > 1.5× avg range of prior bars
    4. MINIMAL OVERLAP — bars barely overlap each other (< 30% overlap)
    5. CONSECUTIVE NEW EXTREMES — each bar makes a new high (bull) / new low (bear)
    """
    spike_bars = bars[sp_start:sp_end + 1]
    n = len(spike_bars)
    if n < 2:
        return 0

    score = 0

    # ── 1. Body gaps (measuring gaps) ──
    # Bull: close[k] > open[k+1] means bodies don't overlap
    # Bear: close[k] < open[k+1]
    body_gaps = 0
    for k in range(n - 1):
        if direction == "bull":
            # Body gap up: prior bar's close > next bar's open
            if spike_bars[k].close > spike_bars[k + 1].open:
                body_gaps += 1
        else:
            # Body gap down: prior bar's close < next bar's open
            if spike_bars[k].close < spike_bars[k + 1].open:
                body_gaps += 1
    if body_gaps > 0:
        score += 1

    # ── 2. Closes near extremes (small tails in trend direction) ──
    # Bull: small upper tail = close near high. Tail ratio = (high - close) / range
    # Bear: small lower tail = close near low. Tail ratio = (close - low) / range
    tail_ratios = []
    for b in spike_bars:
        if b.range <= 0:
            continue
        if direction == "bull":
            tail_ratios.append((b.high - b.close) / b.range)
        else:
            tail_ratios.append((b.close - b.low) / b.range)
    if tail_ratios and sum(tail_ratios) / len(tail_ratios) < 0.25:
        score += 1

    # ── 3. Surprise size — spike range vs context ──
    prior_start = max(0, sp_start - context_bars)
    if prior_start < sp_start:
        prior_bars = bars[prior_start:sp_start]
        if prior_bars:
            avg_prior_range = sum(b.range for b in prior_bars) / len(prior_bars)
            spike_total_range = max(b.high for b in spike_bars) - min(b.low for b in spike_bars)
            if avg_prior_range > 0 and spike_total_range > avg_prior_range * 1.5:
                score += 1

    # ── 4. Minimal overlap between spike bars ──
    # Overlap = how much each bar's range sits inside the prior bar's range
    # Bull: overlap = max(0, prior.high - curr.low) / curr.range
    # Low overlap means bars are stacking cleanly
    overlaps = []
    for k in range(1, n):
        curr_rng = spike_bars[k].range
        if curr_rng <= 0:
            continue
        if direction == "bull":
            overlap = max(0, spike_bars[k - 1].high - spike_bars[k].low) / curr_rng
        else:
            overlap = max(0, spike_bars[k].high - spike_bars[k - 1].low) / curr_rng
        overlaps.append(overlap)
    if overlaps and sum(overlaps) / len(overlaps) < 0.30:
        score += 1

    # ── 5. Consecutive new extremes ──
    # Bull: each bar's high > prior bar's high
    # Bear: each bar's low < prior bar's low
    new_extremes = 0
    for k in range(1, n):
        if direction == "bull" and spike_bars[k].high > spike_bars[k - 1].high:
            new_extremes += 1
        elif direction == "bear" and spike_bars[k].low < spike_bars[k - 1].low:
            new_extremes += 1
    if new_extremes == n - 1:  # every bar made a new extreme
        score += 1

    return score


def detect_spike_buy_market(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    With-trend entry: buy/sell at the market immediately after a strong spike.

    Each spike is scored 0-5 on Brooks' signs of strength:
      1. Body gaps (measuring gaps)
      2. Closes near extremes (small tails)
      3. Surprise size (> 1.5× avg prior range)
      4. Minimal overlap between bars
      5. Consecutive new extremes

    Score is included in the setup name so the backtester can filter by strength.
    """
    setups = []
    if len(bars) < 5:
        return setups

    MIN_SPIKE = 3
    BODY_RATIO = 0.40
    COOLDOWN = 5

    last_fire = -999

    for i in range(MIN_SPIKE, len(bars)):
        spike = _find_spike(bars, i - 1, min_bars=MIN_SPIKE, body_ratio=BODY_RATIO)
        if spike is None:
            continue

        direction, sp_start, sp_end, s_high, s_low, s_avg = spike

        if sp_end != i - 1:
            continue
        if (i - last_fire) <= COOLDOWN:
            continue

        curr = bars[i]
        spike_size = s_high - s_low
        if spike_size <= 0:
            continue

        strength = _score_spike_strength(bars, sp_start, sp_end, direction)
        n_bars = sp_end - sp_start + 1

        if direction == "bull":
            setups.append(Setup(
                index=i, bar=curr,
                setup_name=f"Bull Spike Buy S{strength}",
                direction=1, confidence=0.50 + strength * 0.10,
                notes=f"S{strength}/5 | {n_bars}-bar bull spike. Stop below spike low.",
            ))
        else:
            setups.append(Setup(
                index=i, bar=curr,
                setup_name=f"Bear Spike Sell S{strength}",
                direction=-1, confidence=0.50 + strength * 0.10,
                notes=f"S{strength}/5 | {n_bars}-bar bear spike. Stop above spike high.",
            ))
        last_fire = i

    return setups


def detect_spike_buy_pullback(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    With-trend entry: wait for a small pullback after a strong spike, then enter.

    Same spike strength scoring as detect_spike_buy_market.
    """
    setups = []
    if len(bars) < 6:
        return setups

    MIN_SPIKE = 3
    BODY_RATIO = 0.40
    COOLDOWN = 5
    MAX_PB_BARS = 5

    last_fire = -999

    for i in range(MIN_SPIKE + 1, len(bars)):
        if (i - last_fire) <= COOLDOWN:
            continue

        curr = bars[i]

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

            strength = _score_spike_strength(bars, sp_start, sp_end, direction)
            n_bars = sp_end - sp_start + 1

            if direction == "bull":
                if curr.low < bars[i - 1].low:
                    pullback_depth = s_high - curr.low
                    if pullback_depth <= spike_size * 0.50:
                        setups.append(Setup(
                            index=i, bar=curr,
                            setup_name=f"Bull Spike Pullback Buy S{strength}",
                            direction=1, confidence=0.50 + strength * 0.10,
                            notes=f"S{strength}/5 | PB buy {lookback}b after {n_bars}-bar bull spike. Depth {pullback_depth:.2f}.",
                        ))
                        last_fire = i
                        break
            else:
                if curr.high > bars[i - 1].high:
                    pullback_depth = curr.high - s_low
                    if pullback_depth <= spike_size * 0.50:
                        setups.append(Setup(
                            index=i, bar=curr,
                            setup_name=f"Bear Spike Pullback Sell S{strength}",
                            direction=-1, confidence=0.50 + strength * 0.10,
                            notes=f"S{strength}/5 | PB sell {lookback}b after {n_bars}-bar bear spike. Depth {pullback_depth:.2f}.",
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
