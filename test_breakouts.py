import pandas as pd
import sys
import os
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algo_engine import analyze_bars
from data_source import get_data_source

def test_breakouts_and_microchannels(ticker: str, days: int = 5):
    """
    Fetches recent 5-min data via Databento and scans for Breakouts
    and Microchannel exhaustion setups.
    """
    print(f"\nScanning {days} days of {ticker} 5-min data for Breakouts and Microchannels...\n")
    try:
        ds = get_data_source()
        today = dt.date.today()
        start_date = (today - dt.timedelta(days=days + 3)).isoformat()  # pad for weekends
        end_date = today.isoformat()

        df = ds.fetch_historical(ticker, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            print("No data found.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        found_setups = []

        for i in range(30, len(df)):
            window = df.iloc[:i+1]
            result = analyze_bars(window)

            for s in result.get('setups', []):
                name = s['setup_name'].lower()

                is_breakout = "breakout" in name or "bo" in name.split()
                is_spike = "spike" in name
                is_microchannel = "micro" in name or "climax" in name or "shrinking" in name

                if is_breakout or is_spike or is_microchannel:
                    timestamp = window.index[-1].strftime("%Y-%m-%d %H:%M")

                    found_setups.append({
                        "Time": timestamp,
                        "Type": s['setup_name'],
                        "Price": s['entry_price'],
                        "Action": s['order_type'],
                        "Cycle": result['market_cycle'],
                        "Conf": s['confidence']
                    })

        # Deduplicate continuous signals
        unique_signals = []
        last_types = []

        for sig in found_setups:
            sig_type = sig["Type"]

            is_spam = False
            for recent_type in last_types[-8:]:
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

        if not unique_signals:
            print(f"No clear breakouts or microchannel setups found in the last {days} days for {ticker}.")
        else:
            print("-" * 90)
            print(f"{'Time':<20} | {'Setup Types (Confluence)':<45} | {'Market Cycle Context':<20}")
            print("-" * 90)
            for sig in unique_signals:
                star = "** " if sig['Conf'] >= 0.80 else "   "
                print(f"{star}{sig['Time']:<18} | {sig['Type'][:43]:<45} | {sig['Cycle'][:20]:<20}")
            print("-" * 90)
            print(f"\nTotal High-Probability Breakout/Microchannel Setups Found: {len(unique_signals)}")

    except Exception as e:
        print(f"Error analyzing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_breakouts_and_microchannels("NVDA", days=5)
    test_breakouts_and_microchannels("SPY", days=5)
