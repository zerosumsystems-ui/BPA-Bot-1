import pandas as pd
from algo_engine import analyze_bars
from data_source import get_data_source
import os
import datetime as dt


def test_breakouts_and_microchannels(ticker: str, days: int = 5):
    """
    Downloads recent 5-min data and scans exclusively for Breakouts
    and Microchannel exhaustion setups to demonstrate the bot's logic.
    """
    print(f"\n🚀 Scanning {days} days of {ticker} 5-min data for Breakouts and Microchannels...\n")
    try:
        db_key = os.environ.get("DATABENTO_API_KEY", "")
        if not db_key:
            print("DATABENTO_API_KEY is not set. Set it to run this test.")
            return
        source = get_data_source(api_key=db_key)
        end = dt.date.today().isoformat()
        start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        df = source.fetch_historical(ticker, start, end)
        if df is None or df.empty:
            print("No data found.")
            return

        # We'll run the analyzer on a rolling window to simulate live trading
        # Since analyze_bars just looks at the end of the provided dataframe,
        # we will feed it chunks of the day's data.
        
        found_setups = []
        
        # Start scanning after we have at least 30 bars of context for the day
        for i in range(30, len(df)):
            window = df.iloc[:i+1]
            result = analyze_bars(window)
            
            # The analyzer returns the 'best' setups at the current bar (index i)
            # We specifically want to highlight Breakouts, Spikes, and Microchannel logic
            for s in result.get('setups', []):
                name = s['setup_name'].lower()
                
                # Filter for the user's favorite setups
                is_breakout = "breakout" in name or "bo" in name.split()
                is_spike = "spike" in name
                is_microchannel = "micro" in name or "climax" in name or "shrinking" in name
                
                if is_breakout or is_spike or is_microchannel:
                    # Get the timestamp of the actual bar
                    timestamp = window.index[-1].strftime("%Y-%m-%d %H:%M")
                    
                    found_setups.append({
                        "Time": timestamp,
                        "Type": s['setup_name'],
                        "Price": s['entry_price'],
                        "Action": s['order_type'],
                        "Cycle": result['market_cycle'],
                        "Conf": s['confidence']
                    })
                    
        # Deduplicate continuous signals (bot might flag a breakout for 3 bars in a row)
        unique_signals = []
        last_types = []  # maintain an ordered list for time-based cooldowns
        
        for sig in found_setups:
            sig_type = sig["Type"]
            
            # Check if a very similar signal was fired recently
            # e.g., "Breakout (BO)" and "Confluence: Breakout (BO) + High 2 Bull"
            is_spam = False
            for recent_type in last_types[-8:]: # Look at the last 8 accepted signals (~40 minutes of action)
                # If they share a major keyword, consider it the same continuous setup
                if "Breakout" in sig_type and "Breakout" in recent_type:
                    is_spam = True
                    break
                if "Microchannel" in sig_type and "Microchannel" in recent_type:
                    is_spam = True
                    break
                if "Climax" in sig_type and "Climax" in recent_type:
                    is_spam = True
                    break
                if sig_type == recent_type:
                    is_spam = True
                    break
                    
            if not is_spam:
                unique_signals.append(sig)
                last_types.append(sig_type)
                
        # Print Results
        if not unique_signals:
            print(f"No clear breakouts or microchannel setups found in the last {days} days for {ticker}.")
        else:
            print("-" * 90)
            print(f"{'Time':<20} | {'Setup Types (Confluence)':<45} | {'Market Cycle Context':<20}")
            print("-" * 90)
            for sig in unique_signals:
                # Highlight the extremely high confidence ones
                star = "⭐ " if sig['Conf'] >= 0.80 else "  "
                print(f"{star}{sig['Time']:<18} | {sig['Type'][:43]:<45} | {sig['Cycle'][:20]:<20}")
            print("-" * 90)
            print(f"\nTotal High-Probability Breakout/Microchannel Setups Found: {len(unique_signals)}")

    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    # Test on a highly volatile moving stock where breakouts and channels are common
    test_breakouts_and_microchannels("NVDA", days=5)
    test_breakouts_and_microchannels("SPY", days=5)
