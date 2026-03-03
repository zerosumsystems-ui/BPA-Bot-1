import pandas as pd
from typing import List, Dict, Any
from algo_engine import Setup, Bar

def detect_weak_pullback_fades(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Setups #2 & #3 (Limit Entries): 
    Buy below weak L1/L2 in new bull trend.
    Short above weak H1/H2 in new bear trend.
    (This is fading the early counter-trend traders).
    """
    setups = []
    if len(bars) < 20: return setups
    
    for i in range(15, len(bars)):
        curr = bars[i]
        
        # Check current trend via EMA slope over last 10 bars
        ema_now = ema[i]
        ema_10_ago = bars[i-10].ema_20
        
        # New Strong Bull Trend (EMA sloping up sharply)
        if ema_now > ema_10_ago * 1.002:
            # We are looking for a WEAK L1 or L2 (a small bear bar trying to pull back)
            if curr.close < curr.open and curr.range < (ema[i] * 0.002): # Very small body/range
                # Limit Order Location: Buy *below* this weak bear bar's low
                setups.append(Setup(
                    setup_name="Fade Weak L1/L2",
                    entry_bar=curr.idx,
                    entry_price=round(curr.low - 0.01, 2),
                    order_type="Limit",
                    confidence=0.75,
                ))
                
        # New Strong Bear Trend (EMA sloping down sharply)
        elif ema_now < ema_10_ago * 0.998:
            # We are looking for a WEAK H1 or H2 (a small bull bar trying to pull back)
            if curr.close > curr.open and curr.range < (ema[i] * 0.002):
                # Limit Order Location: Short *above* this weak bull bar's high
                setups.append(Setup(
                    setup_name="Fade Weak H1/H2",
                    entry_bar=curr.idx,
                    entry_price=round(curr.high + 0.01, 2),
                    order_type="Limit",
                    confidence=0.75,
                ))
                
    return setups

def detect_quiet_flag_ma_entries(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Setup #4 (Limit Entries):
    Buy/Sell at or below/above prior bar in a quiet flag right at the moving average.
    """
    setups = []
    if len(bars) < 15: return setups
    
    for i in range(10, len(bars)):
        curr = bars[i]
        prev = bars[i-1]
        
        # Define a "quiet flag" - 4 bars with very low volatility
        flag_bars = bars[i-4:i]
        avg_range = sum(b.range for b in flag_bars) / 4
        
        # If the average range of the last 4 bars is extremely small (quiet)
        if avg_range < (ema[i] * 0.0015):
            
            # If price is resting almost exactly on the EMA
            if abs(curr.close - ema[i]) < avg_range:
                
                # Check overall macro trend over last 10 bars
                if ema[i] > bars[i-10].ema_20:
                    setups.append(Setup(
                        setup_name="Quiet Bull Flag at MA",
                        entry_bar=curr.idx,
                        entry_price=round(prev.low, 2),
                        order_type="Limit",
                        confidence=0.80,
                    ))
                elif ema[i] < bars[i-10].ema_20:
                    setups.append(Setup(
                        setup_name="Quiet Bear Flag at MA",
                        entry_bar=curr.idx,
                        entry_price=round(prev.high, 2),
                        order_type="Limit",
                        confidence=0.80,
                    ))
                    
    return setups

def detect_weak_breakout_tests(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Setups #6 & #7 (Limit Entries):
    Buy/Sell on Breakout Tests (Stop-running setups).
    Buying the close of a weak bear breakout bar expecting failure.
    """
    setups = []
    if len(bars) < 15: return setups
    
    for i in range(10, len(bars)):
        curr = bars[i]
        prev = bars[i-1]
        
        # Did the current bar just break out of a 5-bar local extreme?
        local_high = max(b.high for b in bars[i-6:i-1])
        local_low = min(b.low for b in bars[i-6:i-1])
        
        # Weak Bear Breakout Failure (Buy the close)
        # It broke the local low, but closed poorly (long tail, closed above its midpoint)
        if curr.low < local_low and curr.close > curr.midpoint:
            setups.append(Setup(
                setup_name="Weak Bear Breakout Test",
                entry_bar=curr.idx,
                entry_price=round(curr.close, 2),
                order_type="Limit",
                confidence=0.85,
            ))
            
        # Weak Bull Breakout Failure (Sell the close)
        # It broke the local high, but closed poorly (long tail, closed below midpoint)
        elif curr.high > local_high and curr.close < curr.midpoint:
            setups.append(Setup(
                setup_name="Weak Bull Breakout Test",
                entry_bar=curr.idx,
                entry_price=round(curr.close, 2),
                order_type="Limit",
                confidence=0.85,
            ))
            
    return setups
