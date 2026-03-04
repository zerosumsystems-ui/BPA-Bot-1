"""
Confidence vs Profitability Analysis
Run locally:  cd /path/to/BPA-Bot-1 && python test_confidence.py

Tests whether the algo's confidence scores actually predict profitability.
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
from data_source import get_data_source

tickers = [
    # Ultra-liquid ETFs
    "SPY", "QQQ", "IWM",
    # Mega-cap tech (tightest spreads, highest volume)
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    # High-volume financials & industrials
    "JPM", "BAC", "GS", "XOM", "WMT",
    # Liquid mid-large caps
    "AMD", "INTC", "COIN", "UBER", "NFLX",
]

all_trades = []

print("Fetching data and running backtests (max 60 days of 5m data)...")

# Ensure Databento key is set
db_key = os.environ.get("DATABENTO_API_KEY", "")
if not db_key:
    print("DATABENTO_API_KEY is not set. Please set it to run this analysis.")
    sys.exit(1)

source = get_data_source(api_key=db_key)

for ticker in tickers:
    try:
        # Fetch last 60 days of 5m RTH bars from Databento
        end = dt.date.today().isoformat()
        start = (dt.date.today() - dt.timedelta(days=60)).isoformat()
        df = source.fetch_historical(ticker, start, end)
        if df is None or df.empty:
            print(f"  {ticker}: no data")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure required columns
        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            print(f"  {ticker}: missing columns {list(df.columns)}")
            continue

        df = df.dropna(subset=required)

        # Data_source returns Eastern-localized index already; group by date
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
        "entry_price": t.entry_price,
        "stop_loss": t.stop_loss,
        "scalp_target": t.scalp_target,
        "risk_per_share": t.risk_per_share,
        "risk": abs(t.entry_price - t.stop_loss),
        "reward": abs(t.scalp_target - t.entry_price),
        "mfe_r": t.mfe_r,       # Max Favorable Excursion in R-multiples
        "mae_r": t.mae_r,       # Max Adverse Excursion in R-multiples
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
    avg_risk=("risk", "mean"),
    avg_reward=("reward", "mean"),
    avg_conf=("confidence", "mean"),
).round(4)
setup_stats = setup_stats[setup_stats["trades"] >= 5]
setup_stats["win_rate"] = (setup_stats["win_rate"] * 100).round(1).astype(str) + "%"
setup_stats["rr_ratio"] = (setup_stats["avg_reward"] / setup_stats["avg_risk"].replace(0, np.nan)).round(2)
setup_stats["avg_risk"] = "$" + setup_stats["avg_risk"].round(2).astype(str)
setup_stats["avg_reward"] = "$" + setup_stats["avg_reward"].round(2).astype(str)
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

# ============================================================
# R:R SCENARIO ANALYSIS
# ============================================================
# Using MFE (Max Favorable Excursion) to simulate what would happen
# at different R:R targets. If mfe_r >= target_r, the trade WOULD
# have hit that target. Otherwise it hit stop (-1R).
# We exclude unfilled trades (exit_reason == "unfilled") from this analysis.

print("\n\n" + "=" * 80)
print("R:R SCENARIO ANALYSIS — WHAT IF YOU CHANGED THE TARGET?")
print("=" * 80)
print("Using MFE data to simulate each setup at different R:R ratios.")
print("Win = MFE reached target R. Loss = -1R (full stop).")

# ─── REALISTIC COST MODEL ───
# Commission: round-trip per share (entry + exit)
# Spread: estimated half-spread cost on each side (entry + exit)
# These are applied as a flat dollar cost per trade, reducing every trade's P&L.
COMMISSION_PER_SHARE = 0.005   # $0.005/share (e.g. IBKR tiered)
SHARES_PER_TRADE = 100         # assume 100 shares per trade
SPREAD_PER_SIDE = 0.01         # $0.01 (1 cent) half-spread per side for liquid stocks
ROUND_TRIP_COMMISSION = COMMISSION_PER_SHARE * SHARES_PER_TRADE * 2  # entry + exit
ROUND_TRIP_SPREAD = SPREAD_PER_SIDE * 2 * SHARES_PER_TRADE           # entry + exit spread
COST_PER_TRADE = (ROUND_TRIP_COMMISSION + ROUND_TRIP_SPREAD) / SHARES_PER_TRADE  # per-share cost

print(f"\nCost model: ${COMMISSION_PER_SHARE}/share commission × {SHARES_PER_TRADE} shares × 2 sides = ${ROUND_TRIP_COMMISSION:.2f}")
print(f"            ${SPREAD_PER_SIDE}/share spread × 2 sides × {SHARES_PER_TRADE} shares = ${ROUND_TRIP_SPREAD:.2f}")
print(f"            Total cost per trade (per share): ${COST_PER_TRADE:.4f}\n")

rr_ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# Filter out unfilled trades for scenario analysis
filled = tdf[tdf["exit_reason"] != "unfilled"].copy()

def simulate_rr(group, target_r, cost_per_share=COST_PER_TRADE):
    """Simulate a given R:R target using MFE data, with commission + spread."""
    trades = len(group)
    if trades == 0:
        return None
    wins = (group["mfe_r"] >= target_r).sum()
    losses = trades - wins
    win_rate = wins / trades
    avg_risk = group["risk"].mean()

    # Gross EV per trade in R-multiples
    gross_ev_r = (win_rate * target_r) - ((1 - win_rate) * 1.0)
    # Cost as fraction of risk (cost eats into every trade regardless of outcome)
    cost_in_r = cost_per_share / avg_risk if avg_risk > 0 else 0
    # Net EV after costs
    net_ev_r = gross_ev_r - cost_in_r
    net_ev_dollar = net_ev_r * avg_risk
    total_dollar = net_ev_dollar * trades

    return {
        "trades": trades,
        "win_rate": win_rate,
        "gross_ev_r": round(gross_ev_r, 4),
        "net_ev_r": round(net_ev_r, 4),
        "net_ev_dollar": round(net_ev_dollar, 4),
        "total": round(total_dollar, 2),
        "cost_r": round(cost_in_r, 4),
    }

# Per-setup scenario table
setups_list = filled.groupby("setup").filter(lambda x: len(x) >= 5)["setup"].unique()
setups_list = sorted(setups_list)

results_rows = []
for setup_name in setups_list:
    group = filled[filled["setup"] == setup_name]
    avg_risk = group["risk"].mean()
    for rr in rr_ratios:
        sim = simulate_rr(group, rr)
        if sim:
            results_rows.append({
                "setup": setup_name,
                "R:R": f"{rr}:1",
                "trades": sim["trades"],
                "win_rate": f"{sim['win_rate']:.1%}",
                "gross_EV(R)": sim["gross_ev_r"],
                "cost(R)": sim["cost_r"],
                "net_EV(R)": sim["net_ev_r"],
                "net_EV($)": sim["net_ev_dollar"],
                "total_pnl": sim["total"],
                "avg_risk": round(avg_risk, 2),
            })

rr_df = pd.DataFrame(results_rows)

# Show the best R:R for each setup (NET of costs)
print("=" * 80)
print("OPTIMAL R:R PER SETUP — NET OF COMMISSIONS + SPREAD")
print("=" * 80)
best_rr = []
for setup_name in setups_list:
    subset = rr_df[rr_df["setup"] == setup_name]
    if subset.empty:
        continue
    best_idx = subset["net_EV(R)"].idxmax()
    best_row = subset.loc[best_idx]
    best_rr.append({
        "setup": setup_name,
        "best_RR": best_row["R:R"],
        "trades": best_row["trades"],
        "win_rate": best_row["win_rate"],
        "gross_EV(R)": best_row["gross_EV(R)"],
        "cost(R)": best_row["cost(R)"],
        "net_EV(R)": best_row["net_EV(R)"],
        "net_EV($)": best_row["net_EV($)"],
        "total_pnl": best_row["total_pnl"],
    })
best_df = pd.DataFrame(best_rr).sort_values("net_EV(R)", ascending=False)
print(best_df.to_string(index=False))

# Full scenario grid for setups with positive NET EV at ANY R:R
print("\n" + "=" * 80)
print("FULL R:R GRID — SETUPS WITH POSITIVE NET EV AT ANY RATIO")
print("=" * 80)
positive_setups = rr_df[rr_df["net_EV(R)"] > 0]["setup"].unique()
for setup_name in sorted(positive_setups):
    subset = rr_df[rr_df["setup"] == setup_name]
    print(f"\n--- {setup_name} ---")
    print(subset[["R:R", "trades", "win_rate", "gross_EV(R)", "cost(R)", "net_EV(R)", "net_EV($)", "total_pnl"]].to_string(index=False))

# Overall portfolio: what if ALL setups used their optimal R:R?
print("\n" + "=" * 80)
print("PORTFOLIO SIMULATION: EACH SETUP AT ITS OPTIMAL R:R (NET OF COSTS)")
print("=" * 80)
total_trades = 0
total_pnl = 0
total_costs = 0
positive_setups_count = 0
for _, row in best_df.iterrows():
    if row["net_EV(R)"] > 0:
        total_trades += row["trades"]
        total_pnl += row["total_pnl"]
        positive_setups_count += 1

print(f"Setups with positive net EV at optimal R:R: {positive_setups_count}")
print(f"Total trades: {int(total_trades)}")
print(f"Total net P&L: ${total_pnl:.2f}")
if total_trades > 0:
    print(f"Avg net P&L/trade: ${total_pnl/total_trades:.4f}")
    print(f"Cost per trade: ${COST_PER_TRADE:.4f} (commission + spread)")
