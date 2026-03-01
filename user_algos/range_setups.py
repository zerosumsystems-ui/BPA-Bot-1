import pandas as pd
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
                    index=i, bar=curr, setup_type="Bear Stairs Reversal (3rd/4th Push)",
                    direction=1, confidence=0.75, notes="Exhaustion of sellers after 3+ pushes down in a stair-step channel."
                ))
    return setups

def detect_spike_and_channel_exhaustion(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Setup #4: Spike-and-Channel at Measured Move Target Weakening.
    """
    # This requires deep measured-move projection logic. 
    # For now, we will flag late-stage channels that lose momentum.
    setups = []
    if len(bars) < 25: return setups
    
    for i in range(20, len(bars)):
        curr = bars[i]
        past_20 = bars[i-20:i]
        
        # Did a massive spike happen ~15-20 bars ago?
        early_bars = past_20[:5]
        late_bars = past_20[10:]
        
        early_range = max(b.high for b in early_bars) - min(b.low for b in early_bars)
        late_range = max(b.high for b in late_bars) - min(b.low for b in late_bars)
        
        # If the late channel is much slower/weaker than the initial spike
        if early_range > 0 and (late_range / len(late_bars)) < (early_range / len(early_bars)) * 0.5:
            # Bull Spike & Channel Exhaustion (Look for Bear signal)
            if curr.close < curr.open and curr.close < curr.midpoint:
                 setups.append(Setup(i, curr, "Bull Spike & Channel Top", -1, 0.70, "Momentum heavily decayed from original spike."))
            # Bear Spike & Channel Exhaustion (Look for Bull signal)
            elif curr.close > curr.open and curr.close > curr.midpoint:
                 setups.append(Setup(i, curr, "Bear Spike & Channel Bottom", 1, 0.70, "Momentum heavily decayed from original spike."))
                 
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
                    setups.append(Setup(i, curr, "H1 in Strong Bull Spike", 1, 0.85, "First pullback (H1) after 3-bar bull breakout spike."))
                    
        # Bear Spike: 3 consecutive strong bear bars closing near their lows
        elif prev3.close < prev3.open and prev2.close < prev2.open and prev1.close < prev1.open:
            if prev1.close < prev1.midpoint and prev2.close < prev2.midpoint:
                # The L1: A small pullback bar
                if curr.high > prev1.high and curr.close < curr.open:
                    setups.append(Setup(i, curr, "L1 in Strong Bear Spike", -1, 0.85, "First pullback (L1) after 3-bar bear breakout spike."))
                    
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
                setups.append(Setup(i, curr, "Trading Range Top Reversal", -1, 0.80, "Price rejected perfectly off the established 20-bar range ceiling."))
                
            # Fade the Bottom (Buying)
            # If price pushes down to the old low, but immediately reverses into a bull bar
            elif curr.low <= (range_low * 1.002) and curr.close > curr.midpoint:
                setups.append(Setup(i, curr, "Trading Range Bottom Reversal", 1, 0.80, "Price rejected perfectly off the established 20-bar range floor."))
                
    return setups
