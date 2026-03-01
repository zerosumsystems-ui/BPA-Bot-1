import pandas as pd
from typing import List, Dict, Any
from algo_engine import Setup, Bar

def detect_head_and_shoulders(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Head and Shoulders Tops and Bottoms.
    A reversal pattern consisting of 3 pushes, where the middle push (the head)
    exceeds the outer two pushes (the shoulders), followed by a break of the neckline.
    """
    setups = []
    if len(bars) < 40: return setups
    
    for i in range(35, len(bars)):
        curr = bars[i]
        
        # Look for the last 3 major swing points
        past_30 = bars[i-30:i]
        
        # Extremely simplified local extrema detection for structural logic
        local_highs = []
        local_lows = []
        
        for j in range(2, len(past_30) - 2):
            if past_30[j].high >= max(b.high for b in past_30[j-2:j+3]):
                local_highs.append(past_30[j])
            if past_30[j].low <= min(b.low for b in past_30[j-2:j+3]):
                local_lows.append(past_30[j])
                
        # H&S Top (Reversal Down)
        if len(local_highs) >= 3 and len(local_lows) >= 2:
            left_shoulder = local_highs[-3]
            head = local_highs[-2]
            right_shoulder = local_highs[-1]
            
            # The head must be higher than both shoulders
            if head.high > left_shoulder.high and head.high > right_shoulder.high:
                
                # The right shoulder should generally be lower than the head, but test the neckline
                neckline_y1 = local_lows[-2].low
                neckline_y2 = local_lows[-1].low
                
                # If the current bar breaks below the right neckline
                if curr.low < neckline_y2 and curr.close < curr.open and curr.close < curr.midpoint:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Head and Shoulders Top",
                        direction=-1, confidence=0.75,
                        notes="3-push exhaustion pattern with a higher middle peak; broken neckline."
                    ))
                    
        # H&S Bottom (Reversal Up)
        if len(local_lows) >= 3 and len(local_highs) >= 2:
            left_shoulder = local_lows[-3]
            head = local_lows[-2]
            right_shoulder = local_lows[-1]
            
            if head.low < left_shoulder.low and head.low < right_shoulder.low:
                neckline_y1 = local_highs[-2].high
                neckline_y2 = local_highs[-1].high
                
                if curr.high > neckline_y2 and curr.close > curr.open and curr.close > curr.midpoint:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Head and Shoulders Bottom",
                        direction=1, confidence=0.75,
                        notes="3-push exhaustion pattern with a lower middle trough; broken neckline."
                    ))
                    
    return setups

def detect_expanding_triangles(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Expanding Triangles (Megaphones).
    A trading range that gets wider with both higher highs and lower lows.
    We fade the extremes.
    """
    setups = []
    if len(bars) < 30: return setups
    
    for i in range(25, len(bars)):
        curr = bars[i]
        past_20 = bars[i-20:i]
        
        local_highs = [b for j, b in enumerate(past_20) if j >= 2 and j <= len(past_20)-3 and b.high >= max(past_20[k].high for k in range(j-2, j+3))]
        local_lows = [b for j, b in enumerate(past_20) if j >= 2 and j <= len(past_20)-3 and b.low <= min(past_20[k].low for k in range(j-2, j+3))]
        
        # Require growing extremes
        if len(local_highs) >= 2 and len(local_lows) >= 2:
            if local_highs[-1].high > local_highs[-2].high + (ema[i] * 0.001) and \
               local_lows[-1].low < local_lows[-2].low - (ema[i] * 0.001):
                
                # Fade the expanding top
                if abs(curr.high - local_highs[-1].high) < (ema[i] * 0.002) and curr.close < curr.midpoint:
                     setups.append(Setup(
                        index=i, bar=curr, setup_type="Expanding Triangle Top (Fade)",
                        direction=-1, confidence=0.65,
                        notes="Fading the top edge of a broadening formation."
                    ))
                     
                # Fade the expanding bottom
                elif abs(curr.low - local_lows[-1].low) < (ema[i] * 0.002) and curr.close > curr.midpoint:
                     setups.append(Setup(
                        index=i, bar=curr, setup_type="Expanding Triangle Bottom (Fade)",
                        direction=1, confidence=0.65,
                        notes="Fading the bottom edge of a broadening formation."
                    ))

    return setups

def detect_shrinking_stairs(bars: List[Bar], ema: List[float]) -> List[Setup]:
    """
    Detects Shrinking Stairs / Decaying Momentum.
    A trend making new highs, but the physical size of consecutive pushes is shrinking.
    """
    setups = []
    if len(bars) < 25: return setups
    
    for i in range(20, len(bars)):
        curr = bars[i]
        past_20 = bars[i-20:i]
        
        # Find consecutive pushes up/down and measure their distance
        local_highs = [b for j, b in enumerate(past_20) if j >= 2 and j <= len(past_20)-3 and b.high >= max(past_20[k].high for k in range(j-2, j+3))]
        local_lows = [b for j, b in enumerate(past_20) if j >= 2 and j <= len(past_20)-3 and b.low <= min(past_20[k].low for k in range(j-2, j+3))]
        
        if len(local_highs) >= 3:
            push1 = local_highs[-2].high - local_highs[-3].high
            push2 = local_highs[-1].high - local_highs[-2].high
            
            # Shrinking momentum at the top
            if push1 > 0 and push2 > 0 and push2 < (push1 * 0.5):
                # Strong reversal bar
                if curr.close < curr.midpoint and curr.body_size > curr.range * 0.5 and curr.high >= local_highs[-1].high:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Shrinking Stairs Top Exhaustion",
                        direction=-1, confidence=0.70,
                        notes="Momentum decaying severely on consecutive new highs."
                    ))
                    
        if len(local_lows) >= 3:
            push1 = local_lows[-3].low - local_lows[-2].low
            push2 = local_lows[-2].low - local_lows[-1].low
            
            # Shrinking momentum at the bottom (distances measured downwards)
            if push1 > 0 and push2 > 0 and push2 < (push1 * 0.5):
                if curr.close > curr.midpoint and curr.body_size > curr.range * 0.5 and curr.low <= local_lows[-1].low:
                    setups.append(Setup(
                        index=i, bar=curr, setup_type="Shrinking Stairs Bottom Exhaustion",
                        direction=1, confidence=0.70,
                        notes="Momentum decaying severely on consecutive new lows."
                    ))

    return setups
