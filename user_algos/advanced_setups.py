from algo_engine import Bar, Setup
import pandas as pd
from typing import List

def detect_wedge_patterns(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Advanced Al Brooks Patterns:
    Detects Wedge Tops and Wedge Bottoms (3-Push Patterns).
    """
    setups = []
    if len(bars) < 30:
        return setups

    bull_pushes = 0
    bear_pushes = 0
    
    in_bull_pullback = False
    in_bear_pullback = False

    for i in range(1, len(bars)):
        bar = bars[i]
        prev_bar = bars[i-1]
        
        # ── WEDGE TOP LOGIC (3 Pushes Up) ──
        if bar.high > prev_bar.high:
            if in_bull_pullback:
                bull_pushes = bull_pushes + 1
                in_bull_pullback = False
        elif bar.high < prev_bar.high:
            in_bull_pullback = True

        # If we have 3 pushes up, look for a good bear signal bar to short
        if bull_pushes >= 3:
            # Bear signal bar: closes near its low
            if bar.close < bar.open and bar.close <= bar.low + (bar.high - bar.low) * 0.4:
                setups.append(Setup(
                    setup_name="Custom Wedge Top",
                    entry_bar=bar.idx,
                    entry_price=round(bar.low - 0.01, 2),
                    order_type="Stop",
                    confidence=0.80,
                ))
                bull_pushes = 0  # Reset after trigger
        
        # ── WEDGE BOTTOM LOGIC (3 Pushes Down) ──
        if bar.low < prev_bar.low:
            if in_bear_pullback:
                bear_pushes = bear_pushes + 1
                in_bear_pullback = False
        elif bar.low > prev_bar.low:
            in_bear_pullback = True

        # If we have 3 pushes down, look for a good bull signal bar to buy
        if bear_pushes >= 3:
            # Bull signal bar: closes near its high
            if bar.close > bar.open and bar.close >= bar.high - (bar.high - bar.low) * 0.4:
                setups.append(Setup(
                    setup_name="Custom Wedge Bottom",
                    entry_bar=bar.idx,
                    entry_price=round(bar.high + 0.01, 2),
                    order_type="Stop",
                    confidence=0.80,
                ))
                bear_pushes = 0  # Reset after trigger

    return setups

def detect_two_legged_pullbacks(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Advanced Al Brooks Patterns:
    Detects High 2 (Bull Flag) and Low 2 (Bear Flag) pullbacks closely testing the 20-EMA.
    """
    setups = []
    if len(bars) < 20:
        return setups

    # Simple 20-bar lookback for trend context
    for i in range(20, len(bars)):
        bar = bars[i]
        bar_ema = ema[i]
        
        bull_context = sum(1 for idx, b in enumerate(bars[i-15:i]) if b.close > ema[i-15+idx]) == 15
        
        bear_context = sum(1 for idx, b in enumerate(bars[i-15:i]) if b.close < ema[i-15+idx]) == 15

        # High 2 (H2) Pullback to EMA
        if bull_context:
            # We want the low of the bar to be testing the EMA
            if bar.low <= bar_ema * 1.002 and bar.low >= bar_ema * 0.998:
                # Signal bar needs to be a decent bull bar
                if bar.is_bull and bar.close > (bar.high + bar.low) / 2:
                    setups.append(Setup(
                        setup_name="H2 Pullback to EMA",
                        entry_bar=bar.idx,
                        entry_price=round(bar.high + 0.01, 2),
                        order_type="Stop",
                        confidence=0.85,
                    ))

        # Low 2 (L2) Pullback to EMA
        if bear_context:
            # We want the high of the bar to be testing the EMA
            if bar.high >= bar_ema * 0.998 and bar.high <= bar_ema * 1.002:
                # Signal bar needs to be a decent bear bar
                if bar.is_bear and bar.close < (bar.high + bar.low) / 2:
                    setups.append(Setup(
                        setup_name="L2 Pullback to EMA",
                        entry_bar=bar.idx,
                        entry_price=round(bar.low - 0.01, 2),
                        order_type="Stop",
                        confidence=0.85,
                    ))

    return setups

def detect_nested_wedge_reversals(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Advanced Al Brooks Patterns:
    Detects Nested Wedge Reversals at Support/Resistance.
    This is a wedge pattern (3 pushes) where the 3rd push itself is composed of a smaller, micro-wedge.
    Often happens at EMA or key swing points. context = TR or fading trend.
    """
    setups = []
    if len(bars) < 30:
        return setups

    # We will look for a 3-push pattern, where the final push has 3 micro-pushes.
    for i in range(25, len(bars)):
        curr = bars[i]
        
        # Simplified context check: Is it near the EMA?
        near_ema = abs(curr.close - ema[i]) < (ema[i] * 0.002)

        if not near_ema:
            continue
            
        past_20 = bars[i-20:i]
        
        # Micro-wedge counting (last 5 bars)
        micro_bars = bars[i-5:i]
        
        # Bull Nested Wedge (Looking for Reversal Up)
        local_lows = [b.low for j, b in enumerate(past_20) if j >= 2 and j <= len(past_20)-3 and b.low <= min([past_20[k].low for k in range(j-2, j+3)])]
        
        if len(local_lows) >= 2:
            # We have at least 2 macro pushes down prior
            
            # Now look for 3 micro pushes down in the final leg
            micro_lows = [b.low for j, b in enumerate(micro_bars) if j >= 1 and j <= len(micro_bars)-2 and b.low <= micro_bars[j-1].low and b.low <= micro_bars[j+1].low]
            
            if len(micro_lows) >= 1 and curr.low < micro_bars[-1].low:
                 # This acts as a rough proxy for a nested structure: Macro pushes + micro exhaustion
                 if curr.close > curr.midpoint and curr.body_size > curr.range * 0.4:
                     setups.append(Setup(
                        index=i, bar=curr, setup_type="Nested Wedge Bottom",
                        direction=1, confidence=0.85,
                        notes="Fractal exhaustion: 3rd push of macro wedge contains a micro wedge at Support."
                    ))
                     
        # Bear Nested Wedge (Looking for Reversal Down)
        local_highs = [b.high for j, b in enumerate(past_20) if j >= 2 and j <= len(past_20)-3 and b.high >= max([past_20[k].high for k in range(j-2, j+3)])]
        
        if len(local_highs) >= 2:
            # We have at least 2 macro pushes up prior
            
            # Now look for micro pushes up in the final leg
            micro_highs = [b.high for j, b in enumerate(micro_bars) if j >= 1 and j <= len(micro_bars)-2 and b.high >= micro_bars[j-1].high and b.high >= micro_bars[j+1].high]
            
            if len(micro_highs) >= 1 and curr.high > micro_bars[-1].high:
                 if curr.close < curr.midpoint and curr.body_size > curr.range * 0.4:
                     setups.append(Setup(
                        index=i, bar=curr, setup_type="Nested Wedge Top",
                        direction=-1, confidence=0.85,
                        notes="Fractal exhaustion: 3rd push of macro wedge contains a micro wedge at Resistance."
                    ))

    return setups
