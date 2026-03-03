"""
TEMPLATE ALGORITHM
------------------
Welcome to the algorithmic sandbox! 

You can write any trading logic you want in this file (or create new files in this folder).
As long as your function name starts with `detect_` and follows the correct input/output format, 
the main engine will automatically find it and test it!

To write a rule, you use the `Bar` object which gives you access to:
  - b.open, b.high, b.low, b.close
  - b.range (high - low)
  - b.body (abs(close - open))
  - b.is_bull, b.is_bear

And you return a list of `Setup` objects to tell the engine where to enter.
"""

from algo_engine import Bar, Setup
import pandas as pd

def detect_al_brooks_setups(bars: list[Bar], ema: list[float]) -> list[Setup]:
    """
    Al Brooks Price Action Strategy:
    Detects High 1 / High 2 Bull Flags and Low 1 / Low 2 Bear Flags during trends.
    """
    setups = []
    
    if len(bars) < 20:
        return setups

    # Use a faster 9-EMA to establish short-term 'trend' confirmation
    closes = pd.Series([b.close for b in bars])
    fast_ema = closes.ewm(span=9, adjust=False).mean()
    
    # State tracking variables for Al Brooks swing counting
    highest_high = -1.0
    lowest_low = float('inf')
    
    h_count = 0
    l_count = 0
    in_pullback = False

    for i in range(1, len(bars)):
        bar = bars[i]
        prev_bar = bars[i-1]
        
        fast = fast_ema.iloc[i]
        slow = ema[i]
        
        # ── BULL TREND LOGIC ──
        if fast > slow:
            # Reset Bear Counters
            l_count = 0
            lowest_low = float('inf')
            
            # Record swing highs
            if bar.high >= highest_high:
                highest_high = bar.high
                h_count = 0
                in_pullback = False
            
            # Start pullback counting
            if bar.high < prev_bar.high:
                in_pullback = True
                
            if in_pullback and bar.high > prev_bar.high:
                h_count += 1
                
                # High 1 or High 2 triggered!
                setup_name = f"Custom High {h_count} Bull Flag" if h_count <= 4 else "Custom High 4+ Bull Flag"
                
                # Ensure it's a valid Al Brooks signal (must close relatively strong)
                if bar.close > (bar.high - bar.low) / 2 + bar.low: 
                    setups.append(Setup(
                        setup_name=setup_name,
                        entry_bar=bar.idx,
                        entry_price=round(bar.high + 0.01, 2), # Buy exactly 1 tick above the high
                        order_type="Stop",
                        confidence=0.75,
                    ))
                    
                    # Reset counting after setup triggers so it searches for the next leg
                    highest_high = bar.high
                    h_count = 0
                    in_pullback = False

        # ── BEAR TREND LOGIC ──
        elif fast < slow:
            # Reset Bull Counters
            h_count = 0
            highest_high = -1.0
            
            # Record swing lows
            if bar.low <= lowest_low:
                lowest_low = bar.low
                l_count = 0
                in_pullback = False
                
            # Start pullback counting
            if bar.low > prev_bar.low:
                in_pullback = True
                
            if in_pullback and bar.low < prev_bar.low:
                l_count += 1
                
                # Low 1 or Low 2 triggered!
                setup_name = f"Custom Low {l_count} Bear Flag" if l_count <= 4 else "Custom Low 4+ Bear Flag"
                
                # Ensure it's a valid Al Brooks signal (must close relatively weak)
                if bar.close < (bar.high - bar.low) / 2 + bar.low:
                    setups.append(Setup(
                        setup_name=setup_name,
                        entry_bar=bar.idx,
                        entry_price=round(bar.low - 0.01, 2), # Sell exactly 1 tick below the low
                        order_type="Stop",
                        confidence=0.75,
                    ))
                    
                    # Reset counting after setup triggers
                    lowest_low = bar.low
                    l_count = 0
                    in_pullback = False

    return setups
