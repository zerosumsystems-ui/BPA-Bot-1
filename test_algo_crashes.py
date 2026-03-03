import sys
import pandas as pd
from algo_engine import analyze_bars, USER_ALGO_FUNCTIONS
from data_source import DatabentoSource

def run_test():
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        ds = DatabentoSource(api_key=os.getenv("DATABENTO_API_KEY", ""))
        print("Fetching test data...")
        from app import fetch_chart_data_v2
        df = fetch_chart_data_v2("SPY", "2026-02-23", "2026-02-27")
        if df is None or df.empty:
            print("Failed to fetch data")
            sys.exit(1)
            
        print(f"Loaded {len(df)} bars. Running analysis engine with {len(USER_ALGO_FUNCTIONS)} user algorithms...")
        analysis = analyze_bars(df)
        
        setups = analysis.get("setups", [])
        print(f"Analysis complete. Found {len(setups)} setups.")
        for i, s in enumerate(setups[:5]):
            # Verify we can access new attributes without crash
            name = s.setup_name if hasattr(s, 'setup_name') else s.get('setup_name', 'Unknown')
        print(f"[{i}] Setup: {name} | Signal Bar: {getattr(s, 'signal_bar', 'N/A')} | Entry Bar: {getattr(s, 'entry_bar', 'N/A')}")
            
        if len(setups) > 5: print(f"... and {len(setups) - 5} more.")
        print("SUCCESS: Full engine execution completed without crashes.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
