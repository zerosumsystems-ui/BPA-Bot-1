import pandas as pd
from typing import List, Dict, Any
from algo_engine import Setup, Bar

def detect_major_trend_reversal(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Al Brooks' #1 Rated Strategy: The Major Trend Reversal (MTR)
    
    Criteria:
    1. A strong, established trend is broken by a significant counter-trend move (breaking the trend line).
    2. The price weakly attempts to resume the original trend, failing to make a significant new extreme.
    3. A strong reversal signal bar forms, signaling the start of a completely new trend.
    """
    setups = []
    
    # We need a decent amount of history to confirm a "Major" trend
    if len(bars) < 30:
        return setups
        
    for i in range(25, len(bars)):
        current_bar = bars[i]

        # --- LOOKING FOR BULL REVERSAL (Bear Trend -> Bull Trend) ---
        # 1. MARKET CYCLE CONTEXT: Was there a strong bear trend recently? (e.g., prices consistently below 20 EMA)
        # Avoid taking MTRs on Trending Trading Range Days or pure TR days.
        past_15_bars = bars[i-20:i-5]
        past_30_bars = bars[i-30:i] if i >= 30 else bars[:i]
        recent_bars = bars[i-5:i]

        # Determine day type / market cycle: Is it a broad trading range?
        is_trading_range = abs(past_30_bars[-1].ema_20 - past_30_bars[0].ema_20) < past_30_bars[0].close * 0.002
        if is_trading_range:
            continue # MTR requires a preceding strong trend, not a trading range

        if sum(1 for b in past_15_bars if b.close < ema[b.idx]) >= 12: # 80% below EMA
            
            # 2. Was there a break of the trend line / EMA? (A strong bull push)
            recent_bars = bars[i-5:i]
            if any(b.close > ema[b.idx] for b in recent_bars):
                
                # 3. Did it weakly retest the lows? (Failing to crash further)
                # Ensure the current low isn't massively lower than the lowest low of the bear trend
                trend_low = min(b.low for b in past_15_bars)
                if current_bar.low >= (trend_low * 0.995): # Didn't break down significantly
                
                    # 4. Do we have a strong Bull Reversal Signal Bar right now?
                    if current_bar.close > current_bar.open and \
                       current_bar.close > current_bar.midpoint and \
                       current_bar.body_size > (current_bar.range * 0.4):
                        setups.append(Setup(
                            index=i,
                            bar=current_bar,
                            setup_type="Bull Major Trend Reversal",
                            direction=1,
                            confidence=0.85,
                            notes="Strong bear trend broken, weak resumption, robust bull signal bar."
                        ))
                        
        # --- LOOKING FOR BEAR REVERSAL (Bull Trend -> Bear Trend) ---
        # 1. MARKET CYCLE CONTEXT: Was there a strong bull trend recently? 
        elif sum(1 for b in past_15_bars if b.close > ema[b.idx]) >= 12: 
            
            # 2. Was there a break of the trend line / EMA? (A strong bear push)
            if any(b.close < ema[b.idx] for b in recent_bars):
                
                # 3. Did it weakly retest the highs?
                trend_high = max(b.high for b in past_15_bars)
                if current_bar.high <= (trend_high * 1.005): 
                
                    # 4. Do we have a strong Bear Reversal Signal Bar?
                    if current_bar.close < current_bar.open and \
                       current_bar.close < current_bar.midpoint and \
                       current_bar.body_size > (current_bar.range * 0.4):
                        setups.append(Setup(
                            index=i,
                            bar=current_bar,
                            setup_type="Bear Major Trend Reversal",
                            direction=-1,
                            confidence=0.85,
                            notes="Strong bull trend broken, weak resumption, robust bear signal bar."
                        ))
                        
    return setups

def detect_final_flag(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Al Brooks' Final Flag Reversals
    
    Criteria:
    1. A mature trend exists.
    2. A tight trading range (flag) forms.
    3. The price breaks out of the flag in the direction of the trend, but it fails immediately.
    """
    setups = []
    
    if len(bars) < 20: return setups
    
    for i in range(15, len(bars)):
        curr = bars[i]
        
        # Look back for a tight trading range (the 'Flag')
        flag_bars = bars[i-6:i-1]
        flag_high = max(b.high for b in flag_bars)
        flag_low = min(b.low for b in flag_bars)
        flag_range = flag_high - flag_low
        
        # If the flag is relatively tight compared to the price...
        if flag_range < (curr.close * 0.015): 
            
            # Final Bull Flag: Price breaks ABOVE the flat flag, then crashes
            if curr.high > flag_high and curr.close < flag_low:
                setups.append(Setup(
                    index=i,
                    bar=curr,
                    setup_type="Final Bull Flag Reversal",
                    direction=-1,
                    confidence=0.75,
                    notes="Failed upside breakout of a late-stage tight bull flag."
                ))
            
            # Final Bear Flag: Price breaks BELOW the flat flag, then rallies
            elif curr.low < flag_low and curr.close > flag_high:
                setups.append(Setup(
                    index=i,
                    bar=curr,
                    setup_type="Final Bear Flag Reversal",
                    direction=1,
                    confidence=0.75,
                    notes="Failed downside breakout of a late-stage tight bear flag."
                ))
                
    return setups

def detect_breakout_pullback(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Breakout Pullbacks (BOPB)
    
    Criteria:
    1. A strong flag or trading range is broken cleanly (the breakout).
    2. The price pulls back strictly to the breakout point (the test).
    3. The price forms a signal bar, predicting an explosive continuation.
    """
    setups = []
    
    if len(bars) < 15: return setups
    
    # Iterate through recent history
    for i in range(10, len(bars)):
        curr = bars[i]
        
        # Define a past 5-bar range that was cleanly broken
        base_bars = bars[i-8:i-3]
        base_top = max(b.high for b in base_bars)
        base_bot = min(b.low for b in base_bars)
        
        # Find the breakout bar (i-2 or i-3)
        bo_bar = bars[i-2]
        
        # Bull Breakout Pullback
        if bo_bar.close > base_top and bo_bar.range > (base_top - base_bot) * 0.5:
            # Did we just pull back and test the top of the old base?
            if curr.low <= base_top and curr.low >= (base_top * 0.998):
                if curr.close > curr.midpoint:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Bull Breakout Pullback",
                        direction=1, confidence=0.80, notes="Bulls successfully re-tested the breakout point."
                    ))
                    
        # Bear Breakout Pullback
        elif bo_bar.close < base_bot and bo_bar.range > (base_top - base_bot) * 0.5:
            # Did we just pull back and test the bottom of the old base?
            if curr.high >= base_bot and curr.high <= (base_bot * 1.002):
                if curr.close < curr.midpoint:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Bear Breakout Pullback",
                        direction=-1, confidence=0.80, notes="Bears successfully re-tested the breakout point."
                    ))

    return setups

def detect_exhaustive_climax_at_mm(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects the highest probability reversal setup: An Exhaustive Climax exactly at a Measured Move target.
    
    Criteria:
    1. A parabolic trend exists with a clear breakout point.
    2. We calculate the Measured Move (MM) target by doubling the size of the initial spike.
    3. The price hits the MM target and instantly prints the biggest bar of the entire trend (exhaustion).
    """
    setups = []
    if len(bars) < 30: return setups
    
    for i in range(25, len(bars)):
        curr = bars[i]
        
        # Analyze the past 20 bars for a trend and an initial spike
        trend_bars = bars[i-20:i]
        
        # Very simplified way to find an initial spike: the first 5 bars of the window
        spike_bars = trend_bars[0:5]
        spike_low = min(b.low for b in spike_bars)
        spike_high = max(b.high for b in spike_bars)
        spike_size = spike_high - spike_low
        
        if spike_size <= 0:
            continue
            
        # Analyze the rest of the trend to find the average bar size
        channel_bars = trend_bars[5:]
        avg_bar_size = sum(b.range for b in channel_bars) / len(channel_bars) if channel_bars else 0
        
        # Bull Climax
        if curr.close > ema[i]:
            # MM target based on adding the spike size to the top of the spike
            mm_target_up = spike_high + spike_size
            
            # If current bar hits the target, and is massively larger than average (exhaustion)
            if curr.high >= mm_target_up and curr.range > (avg_bar_size * 2.5):
                # Needs to be a strong reversal bar to enter (closing near low)
                if curr.close < curr.midpoint:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Exhaustive Bull Climax at MM",
                        direction=-1, confidence=0.90, 
                        notes="Hit exact mathematical target and printed massive exhaustion bar."
                    ))
                    
        # Bear Climax
        elif curr.close < ema[i]:
            # MM target based on subtracting spike size from the bottom of the spike
            mm_target_down = spike_low - spike_size
            
            # If current bar hits the target, and is massively larger than average (exhaustion)
            if curr.low <= mm_target_down and curr.range > (avg_bar_size * 2.5):
                # Needs to be a strong reversal bar to enter (closing near high)
                if curr.close > curr.midpoint:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Exhaustive Bear Climax at MM",
                        direction=1, confidence=0.90, 
                        notes="Hit exact mathematical target and printed massive exhaustion bar."
                    ))

    return setups

def detect_wedge_double_bottoms(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects a Double Bottom or Double Top specifically at the 3rd leg of a wedge.
    This is cited as the highest probability entry for a wedge reversal.
    """
    setups = []
    if len(bars) < 25: return setups
    
    for i in range(20, len(bars)):
        curr = bars[i]
        
        # Count pushes in the last 15 bars
        past_bars = bars[i-15:i]
        
        # Very simplified local extrema counting
        local_lows = []
        local_highs = []
        
        for j in range(2, len(past_bars)-2):
            b = past_bars[j]
            if b.low <= min(past_bars[k].low for k in range(j-2, j+3)):
                local_lows.append(b.low)
            if b.high >= max(past_bars[k].high for k in range(j-2, j+3)):
                local_highs.append(b.high)
                
        # Wedge Double Bottom (L3 DB)
        if len(local_lows) >= 2:
            last_low = local_lows[-1]
            
            # If the current bar is effectively a double bottom with the last swing low...
            # And it represents the 3rd overall push down (the wedge)
            if abs(curr.low - last_low) < (ema[i] * 0.001):
                if curr.close > curr.midpoint and curr.body_size > curr.range * 0.4:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Wedge Double Bottom (L3 DB)",
                        direction=1, confidence=0.90,
                        notes="Sellers gave up trying to break lower on the 3rd push."
                    ))
                    
        # Wedge Double Top (H3 DT)
        if len(local_highs) >= 2:
            last_high = local_highs[-1]
            
            # If the current bar is a double top with the last swing high on the 3rd push
            if abs(curr.high - last_high) < (ema[i] * 0.001):
                if curr.close < curr.midpoint and curr.body_size > curr.range * 0.4:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Wedge Double Top (H3 DT)",
                        direction=-1, confidence=0.90,
                        notes="Buyers gave up trying to break higher on the 3rd push."
                    ))

    return setups

def detect_20_gap_bar_pullback(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects the 20-Gap Bar Pullback Setup.
    When the market stays strictly above/below the EMA for 20+ consecutive bars, 
    the very first time it touches the EMA is an incredibly high-probability entry in the trend direction.
    """
    setups = []
    if len(bars) < 30: return setups
    
    for i in range(25, len(bars)):
        curr = bars[i]
        
        # Bull 20 Gap Bar: Price was entirely above EMA for 20+ bars
        past_20_bull = all(b.low > ema[b.idx-1] for b in bars[i-20:i])
        
        if past_20_bull:
            # Does the current bar finally touch the EMA?
            if curr.low <= ema[i]:
                # If it prints a bull reversal bar (a rejection of the moving average)
                if curr.close > curr.midpoint and curr.close > curr.open:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="20-Gap Bar EMA Touch (Bull)",
                        direction=1, confidence=0.85,
                        notes="First touch of the EMA after 20+ consecutive bars entirely above it."
                    ))
                    
        # Bear 20 Gap Bar: Price was entirely below EMA for 20+ bars
        past_20_bear = all(b.high < ema[b.idx-1] for b in bars[i-20:i])
        
        if past_20_bear:
            # Does the current bar finally touch the EMA?
            if curr.high >= ema[i]:
                # If it prints a bear reversal bar
                if curr.close < curr.midpoint and curr.close < curr.open:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="20-Gap Bar EMA Touch (Bear)",
                        direction=-1, confidence=0.85,
                        notes="First touch of the EMA after 20+ consecutive bars entirely below it."
                    ))

    return setups

def detect_ma_gap_bars(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Moving Average Gap Bars.
    A deep pullback inside a strong trend that crosses entirely over the EMA, 
    forming a visible 'gap' between the bar and the EMA, before reversing back.
    """
    setups = []
    if len(bars) < 20: return setups
    
    for i in range(15, len(bars)):
        curr = bars[i]
        
        # Deep Bull Pullback Gap Bar
        # Trend was up...
        if bars[i-10].ema_20 > bars[i-15].ema_20:
            # But the current bar is entirely BELOW the EMA (a deep pullback)
            if curr.high < ema[i]:
                # And prints a strong bull reversal bar
                if curr.close > curr.midpoint and curr.close > curr.open:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="EMA Gap Bar (Bull)",
                        direction=1, confidence=0.75,
                        notes="Deep pullback completely crossed the EMA before rejecting."
                    ))
                    
        # Deep Bear Pullback Gap Bar
        # Trend was down...
        elif bars[i-10].ema_20 < bars[i-15].ema_20:
            # But the current bar is entirely ABOVE the EMA
            if curr.low > ema[i]:
                # And prints a strong bear reversal bar
                if curr.close < curr.midpoint and curr.close < curr.open:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="EMA Gap Bar (Bear)",
                        direction=-1, confidence=0.75,
                        notes="Deep pullback completely crossed the EMA before rejecting."
                    ))

    return setups

def detect_consecutive_climaxes(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Consecutive Climax Flags.
    2 or 3 distinct climax/spike patterns in a row with minimal pullback.
    Signals massive, unsustainable momentum that usually leads to a brutal reversal.
    """
    setups = []
    if len(bars) < 20: return setups
    
    for i in range(15, len(bars)):
        curr = bars[i]
        past_10 = bars[i-10:i]
        avg_range = sum(b.range for b in bars[max(0, i-20):i]) / min(20, max(1, i))
        
        if avg_range == 0: continue
        
        # Count massive climax bars in the recent window
        bull_climaxes = [b for b in past_10 if b.is_bull and b.range > avg_range * 1.5 and b.close > b.open]
        bear_climaxes = [b for b in past_10 if b.is_bear and b.range > avg_range * 1.5 and b.close < b.open]
        
        if len(bull_climaxes) >= 3:
            # Check for a strong bearish reversal bar (the trap springing)
            if curr.close < curr.open and curr.close < curr.midpoint:
                 setups.append(Setup(
                    index=i, bar=curr, setup_type="Consecutive Buy Climaxes (Reversal)",
                    direction=-1, confidence=0.85,
                    notes="Unsustainable consecutive parabolic buy spikes reversing."
                ))
                 
        if len(bear_climaxes) >= 3:
            # Check for a strong bullish reversal bar
            if curr.close > curr.open and curr.close > curr.midpoint:
                 setups.append(Setup(
                    index=i, bar=curr, setup_type="Consecutive Sell Climaxes (Reversal)",
                    direction=1, confidence=0.85,
                    notes="Unsustainable consecutive parabolic sell spikes reversing."
                ))

    return setups
