import datetime as dt
from app import get_databento_key, get_sp500_tickers
import databento as db
import pandas as pd
import time
import sys

def main():
    tickers = get_sp500_tickers()
    batches = [tickers[i:i+50] for i in range(0, 100, 50)]  # test 2 batches of 50
    client = db.Historical(get_databento_key())

    today = dt.date.today()
    start = (today - dt.timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")
    end = today.strftime("%Y-%m-%dT00:00:00")

    for i, batch in enumerate(batches):
        print(f"Fetching batch {i} with {len(batch)} tickers...", flush=True)
        t0 = time.time()
        try:
            data = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=batch,
                stype_in="raw_symbol",
                schema="ohlcv-1m",
                start=start,
                end=end,
            )
            df = data.to_df()
            print(f"Batch {i} took {time.time()-t0:.2f}s, rows={len(df)}")
            if not df.empty and 'symbol' in df.columns:
                print(f"Unique symbols found: {df['symbol'].nunique()}")
        except Exception as e:
            print(f"Error in batch {i}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
