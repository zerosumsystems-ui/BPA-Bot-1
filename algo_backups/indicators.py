import pandas as pd
from typing import List, Dict, Tuple, Optional
from algo_engine import Bar

class BrooksIndicators:
    """
    Core Al Brooks Indicators:
    1. Bar Counter (H1-H4, L1-L4)
    2. Gaps (Measuring, Exhaustion, Breakaway, Negative)
    3. Advanced Market Cycle / Day Type (Continuous Assessment)
    """

    @staticmethod
    def count_pullback_bars(bars: List[Bar], ema: List[float]) -> List[Dict[str, any]]:
        """
        Implements Al Brooks' strict High/Low counting system for pullbacks.
        Returns a list of dictionaries mapping the bar index to its H/L count and state.
        
        Rules:
        - Uptrend: H1 is the first bar with a high > prior bar's high during a pullback.
        - Downtrend: L1 is the first bar with a low < prior bar's low during a pullback.
        """
        counts = []
        
        if len(bars) < 5:
            return counts
            
        in_bull_pullback = False
        in_bear_pullback = False
        bull_count = 0
        bear_count = 0
        
        for i in range(1, len(bars)):
            curr = bars[i]
            prev = bars[i-1]
            is_above_ema = curr.close > ema[i]
            
            # --- Bull Pullbacks (Counting H1, H2, H3, H4) ---
            if is_above_ema:
                # Triggering a pullback: price drops below prior low
                if curr.low < prev.low:
                    if not in_bull_pullback:
                        in_bull_pullback = True
                        bull_count = 0
                
                # Resuming the trend: price pushes above prior high
                if in_bull_pullback and curr.high > prev.high:
                    bull_count += 1
                    counts.append({
                        "index": i,
                        "type": f"H{bull_count}",
                        "is_active": True,
                        "bar": curr
                    })
                    # If we get a strong resumption, the pullback might be over
                    if curr.close > curr.open and curr.close > curr.midpoint:
                        in_bull_pullback = False
                        
            # --- Bear Pullbacks (Counting L1, L2, L3, L4) ---
            else:
                # Triggering a pullback: price rises above prior high
                if curr.high > prev.high:
                    if not in_bear_pullback:
                        in_bear_pullback = True
                        bear_count = 0
                        
                # Resuming the trend: price drops below prior low
                if in_bear_pullback and curr.low < prev.low:
                    bear_count += 1
                    counts.append({
                        "index": i,
                        "type": f"L{bear_count}",
                        "is_active": True,
                        "bar": curr
                    })
                    # If we get a strong resumption, the pullback might be over
                    if curr.close < curr.open and curr.close < curr.midpoint:
                        in_bear_pullback = False
                        
        return counts

    @staticmethod
    def detect_gaps(bars: List[Bar]) -> List[Dict[str, any]]:
        """
        Detects Al Brooks specific gaps.
        A "gap" in BPA isn't just an overnight opening gap; it's the space between
        overlapping bars, or the space between a breakout and a test.
        """
        gaps = []
        if len(bars) < 10:
            return gaps
            
        for i in range(5, len(bars)):
            curr = bars[i]
            
            # 1. Breakaway Gap: A strong bar breaks out, and the next bar's low (or high) 
            # does not overlap with the extreme of the bar *before* the breakout bar.
            # E.g., Bar 1: high of 10. Bar 2: huge breakout to 15. Bar 3: low is 12.
            # The gap is between 10 and 12.
            prev2 = bars[i-2]
            
            # Bull Breakaway Gap
            if curr.low > prev2.high and curr.is_bull:
                gaps.append({
                    "index": i,
                    "type": "Bull Breakaway Gap",
                    "top": curr.low,
                    "bottom": prev2.high,
                    "size": curr.low - prev2.high
                })
                
            # Bear Breakaway Gap
            elif curr.high < prev2.low and curr.is_bear:
                gaps.append({
                    "index": i,
                    "type": "Bear Breakaway Gap",
                    "top": prev2.low,
                    "bottom": curr.high,
                    "size": prev2.low - curr.high
                })
                
            # 2. Negative Gaps (Overlapping Bars = Trading Range Behavior)
            # If the current bar completely overlaps the prior bar's body, 
            # or if the last 3 bars have heavily overlapping bodies.
            if i >= 3:
                prev1 = bars[i-1]
                prev3 = bars[i-3]
                
                # Check for significant overlap among the last 3 bars
                highest_low = max(curr.low, prev1.low, prev2.low)
                lowest_high = min(curr.high, prev1.high, prev2.high)
                
                if lowest_high > highest_low:
                    gaps.append({
                        "index": i,
                        "type": "Negative Gap (Overlap)",
                        "size": lowest_high - highest_low,
                        "notes": "Market is acting like a trading range."
                    })

        return gaps

    @staticmethod
    def classify_advanced_market_cycle(bars: List[Bar], ema: List[float], current_index: int) -> str:
        """
        Continuously assess the market cycle (Spike -> Channel -> TR) up to the current index.
        This provides a granular, real-time context for algo triggers.
        """
        if current_index < 20:
            return "Too Early to Call"
            
        history = bars[max(0, current_index-20):current_index+1]
        
        # Calculate recent average range
        avg_range = sum(b.range for b in history) / len(history)
        if avg_range == 0: return "Tight Trading Range"
        
        # Check the last 5 bars for intense momentum (Spike / Breakout Phase)
        recent_5 = history[-5:]
        big_trend_bars = sum(1 for b in recent_5 if (b.is_strong_bull(avg_range) or b.is_strong_bear(avg_range)) and b.range > avg_range * 1.2)
        
        if big_trend_bars >= 2:
             # It's a spike. Is it breaking out or climaxing?
             # Climax if it's late in a trend
             trend_bars = sum(1 for b in history if b.close > ema[b.idx-1]) if recent_5[-1].is_bull else sum(1 for b in history if b.close < ema[b.idx-1])
             if trend_bars >= 15:
                 return "Exhaustion / Climax Breakout"
             return "Breakout (Spike) Phase"
             
        # Check for Broad vs Tight Channel
        ema_crosses = 0
        for j in range(1, len(history)):
            curr_b = history[j]
            prev_b = history[j-1]
            if (curr_b.close > ema[curr_b.idx-1] and prev_b.close <= ema[prev_b.idx-2]) or \
               (curr_b.close < ema[curr_b.idx-1] and prev_b.close >= ema[prev_b.idx-2]):
                ema_crosses += 1
                
        if ema_crosses == 0:
            return "Tight Channel (Strong Trend)"
        elif ema_crosses <= 3:
            return "Broad Channel (Weak Trend)"
        else:
            return "Trading Range"
