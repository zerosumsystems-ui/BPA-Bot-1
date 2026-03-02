"""
Confidence vs Profitability Analysis
Run locally:  cd /path/to/BPA-Bot-1 && python test_confidence.py

Tests whether the algo's confidence scores actually predict profitability.

Requirements: pip install yfinance pandas numpy scipy
"""
import pandas as pd
import numpy as np
import sys
import os
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

# Ensure we can import from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import run_backtest, run_multi_day_backtest
from data_source import YFinanceSource

source = YFinanceSource()

tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
end = dt.date.today()
start = end - dt.timedelta(days=45)

all_trades = []

print("Fetching data and running backtests...")
for ticker in tickers:
    try:
        df = source.fetch_historical(ticker, start.isoformat(), end.isoformat())
        if df is None or df.empty:
            print(f"  {ticker}: no data")
            continue
        df.index = pd.to_datetime(df.index)
        daily_dfs = {}
        for date, group in df.groupby(df.index.date):
            if len(group) >= 10:
                daily_dfs[str(date)] = group
        if not daily_dfs:
            print(f"  {ticker}: no valid trading days (need 10+ bars per day)")
            continue
        report = run_multi_day_backtest(daily_dfs, mode="scalp", ticker=ticker)
        for t in report["trades"]:
            t.ticker = ticker
        all_trades.extend(report["trades"])
        print(f"  {ticker}: {len(report['trades'])} trades across {len(daily_dfs)} days")
    except Exception as e:
        import traceback
        print(f"  {ticker}: error {e}")
        traceback.print_exc()

print(f"\nTotal trades: {len(all_trades)}\n")

if not all_trades:
    print("No trades generated!")
    sys.exit(1)

rows = []
for t in all_trades:
    rows.append({
        "ticker": t.ticker,
        "setup": t.setup_name,
        "direction": t.direction,
        "confidence": t.confidence,
        "pnl": t.pnl,
        "r_multiple": t.r_multiple,
        "is_winner": t.is_winner,
        "with_trend": t.with_trend,
        "exit_reason": t.exit_reason,
    })

tdf = pd.DataFrame(rows)

print("=" * 60)
print("CONFIDENCE DISTRIBUTION")
print("=" * 60)
print(tdf["confidence"].describe())
print(f"\nUnique values: {sorted(tdf['confidence'].unique())}")

print("\n" + "=" * 60)
print("CONFIDENCE BUCKETS vs PROFITABILITY")
print("=" * 60)
bins = [0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
labels = ["<0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"]
tdf["conf_bucket"] = pd.cut(tdf["confidence"], bins=bins, labels=labels, right=False)

summary = tdf.groupby("conf_bucket", observed=True).agg(
    trades=("pnl", "count"),
    win_rate=("is_winner", "mean"),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
    avg_r=("r_multiple", "mean"),
).round(4)
summary["win_rate"] = (summary["win_rate"] * 100).round(1).astype(str) + "%"
print(summary.to_string())

print("\n" + "=" * 60)
print("SETUPS BY AVG P&L (min 5 trades)")
print("=" * 60)
setup_stats = tdf.groupby("setup").agg(
    trades=("pnl", "count"),
    win_rate=("is_winner", "mean"),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
    avg_conf=("confidence", "mean"),
).round(4)
setup_stats = setup_stats[setup_stats["trades"] >= 5]
setup_stats["win_rate"] = (setup_stats["win_rate"] * 100).round(1).astype(str) + "%"
setup_stats = setup_stats.sort_values("avg_pnl", ascending=False)
print(setup_stats.to_string())

print("\n" + "=" * 60)
print("WITH TREND vs COUNTER TREND")
print("=" * 60)
for trend_val, trend_label in [(True, "With Trend"), (False, "Counter Trend")]:
    subset = tdf[tdf["with_trend"] == trend_val]
    if subset.empty:
        continue
    wr = subset["is_winner"].mean()
    ap = subset["pnl"].mean()
    tp = subset["pnl"].sum()
    print(f"\n{trend_label}: {len(subset)} trades | Win Rate: {wr:.1%} | Avg P&L: ${ap:.4f} | Total: ${tp:.2f}")
    bs = subset.groupby("conf_bucket", observed=True).agg(
        trades=("pnl", "count"),
        win_rate=("is_winner", "mean"),
        avg_pnl=("pnl", "mean"),
    ).round(4)
    bs["win_rate"] = (bs["win_rate"] * 100).round(1).astype(str) + "%"
    print(bs.to_string())

print("\n" + "=" * 60)
print("STATISTICAL CORRELATION")
print("=" * 60)
corr_pnl = tdf["confidence"].corr(tdf["pnl"])
corr_r = tdf["confidence"].corr(tdf["r_multiple"])
corr_win = tdf["confidence"].corr(tdf["is_winner"].astype(float))
print(f"Confidence <> P&L:        r = {corr_pnl:.4f}")
print(f"Confidence <> R-Multiple: r = {corr_r:.4f}")
print(f"Confidence <> Win Rate:   r = {corr_win:.4f}")

try:
    from scipy import stats
    r_pb, p_val = stats.pointbiserialr(tdf["is_winner"].astype(int), tdf["confidence"])
    print(f"\nPoint-biserial r = {r_pb:.4f}, p-value = {p_val:.6f}")
    print(f"Significant (p < 0.05): {'YES' if p_val < 0.05 else 'NO'}")
except ImportError:
    print("\n(install scipy for significance test: pip install scipy)")

print("\n" + "=" * 60)
print("EXIT REASON BREAKDOWN")
print("=" * 60)
exit_stats = tdf.groupby("exit_reason").agg(
    trades=("pnl", "count"),
    avg_pnl=("pnl", "mean"),
).round(4)
print(exit_stats.to_string())
