from algo_engine import Bar, Setup
from typing import List

def detect_htf_ema_opening_reversals(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Opening Reversals at the 60-Minute EMA.
    A premier setup where the market opens, tests a major higher-timeframe support/resistance
    level (the 60-min EMA), and immediately reverses.
    
    Note: For simplicity in the current 5-minute data architecture, we will approximate 
    the 60-minute EMA by tracking the moving average over a period roughly 12x longer 
    (12 5-min bars = 60 mins), or explicitly searching for the first 30-60 mins of the day.
    """
    setups = []
    
    # Need enough data to approximate a HTF EMA
    if len(bars) < 240: # Roughly a day or two of 5-min data
        return setups
        
    for i in range(200, len(bars)):
        curr = bars[i]
        
        # 1. CONTEXT: Is this the opening of the session? 
        # (Approximation: check if the time of the bar is within the first hour of trading)
        # Assuming typical US market hours (9:30 AM - 4:00 PM EST)
        
        # Extract hour/minute if available, else use a heuristic.
        # For this logic, we'll assume `curr.datetime` exists as a pandas timestamp or string
        # If it doesn't, we will fall back to just structural price action.
        is_opening_hour = False
        try:
            if hasattr(curr, 'datetime') and curr.datetime is not None:
                # pandas timestamp check
                if curr.datetime.hour == 9 and curr.datetime.minute >= 30:
                    is_opening_hour = True
                elif curr.datetime.hour == 10 and curr.datetime.minute <= 30:
                    is_opening_hour = True
            else:
                # If no datetime, we can't strictly enforce "opening" reversal, 
                # but we can still look for HTF EMA tests.
                is_opening_hour = True # Lenient fallback
        except:
            is_opening_hour = True # Lenient fallback
            
        if not is_opening_hour:
            continue
            
        # 2. Approximate 60-min EMA on a 5-min chart 
        # A standard 20-EMA on a 60-min chart is roughly a 240-EMA on a 5-min chart (20 * 12)
        # We will calculate a pseudo 240-period SMA/EMA for this check
        past_240 = bars[i-240:i]
        pseudo_htf_ema = sum(b.close for b in past_240) / len(past_240)
        
        # 3. Test & Reversal
        near_htf_ema = abs(curr.close - pseudo_htf_ema) < (pseudo_htf_ema * 0.003)
        
        if near_htf_ema:
            # Bull Reversal (Testing down into HTF Support)
            if curr.low <= pseudo_htf_ema and curr.close > curr.open and curr.close > curr.midpoint:
                 setups.append(Setup(
                    index=i, bar=curr, setup_type="HTF EMA Opening Reversal (Bull)",
                    direction=1, confidence=0.85,
                    notes="Market opened, tested the 60-min EMA support, and formed a bull reversal bar."
                ))
                 
            # Bear Reversal (Testing up into HTF Resistance)
            elif curr.high >= pseudo_htf_ema and curr.close < curr.open and curr.close < curr.midpoint:
                 setups.append(Setup(
                    index=i, bar=curr, setup_type="HTF EMA Opening Reversal (Bear)",
                    direction=-1, confidence=0.85,
                    notes="Market opened, tested the 60-min EMA resistance, and formed a bear reversal bar."
                ))

    return setups
