#!/usr/bin/env python3
"""
forward_rs_returns.py — Where do top BPA RS stocks end up 1w, 2w, 1m later?

For each trading day:
  1. Compute BPA RS scores for all tickers using that day's 5-min bars
  2. Take the top 10 (strongest) and bottom 10 (weakest)
  3. Track their forward returns at 5, 10, 20 trading days
  4. Compare vs SPY (market benchmark)

Universe: ~100 liquid S&P 500 stocks + sector ETFs
Period:   ~60 calendar days (~42 trading days)
"""

import os
import sys
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_source import get_data_source, DatabentoSource
from algo_engine import bars_from_df, compute_ema
from daily_rs_rankings import compute_bpa_rs_score

logger = logging.getLogger(__name__)

# ─────────────── TICKER UNIVERSE ──────────────────────────────────────────────

LIQUID_UNIVERSE = [
    # Sector ETFs
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    # Mega cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Large cap tech
    "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN", "IBM", "INTC", "TXN", "QCOM", "INTU",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD", "LOW",
    # Industrials
    "CAT", "DE", "HON", "UPS", "BA", "GE", "RTX", "LMT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Other
    "DIS", "NFLX", "PYPL", "ABNB", "UBER", "NEE", "SO", "DUK",
    "AMT", "PLD", "CCI", "LIN", "SHW",
]


# ─────────────── DATA LOADING ────────────────────────────────────────────────

def _resolve_api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY", "")
    if key:
        return key
    for path in [
        os.path.expanduser("~/.streamlit/secrets.toml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".streamlit", "secrets.toml"),
    ]:
        try:
            import toml
            if os.path.exists(path):
                return toml.load(path).get("DATABENTO_API_KEY", "")
        except Exception:
            pass
    return ""


def fetch_and_process(source, tickers, start, end):
    """
    Bulk fetch 1-min data, produce:
      1. Per-ticker per-day 5-min DataFrames (for RS scoring)
      2. Per-ticker daily close series (for forward returns)

    Returns: (ticker_5min_days, daily_closes)
      ticker_5min_days: {ticker: {date: DataFrame}}
      daily_closes: {ticker: {date: float}}  — closing price on each day
    """
    print(f"\nBulk fetching {len(tickers)} tickers ({start} to {end})...")

    if not isinstance(source, DatabentoSource):
        print("ERROR: Only Databento source supported for bulk fetch")
        return {}, {}

    try:
        bulk_df = source.get_bulk_chart_data(tickers, start, end)
    except Exception as e:
        print(f"  Bulk fetch failed: {e}")
        return {}, {}

    if bulk_df is None or bulk_df.empty:
        print("  No data returned")
        return {}, {}

    ticker_5min_days = {}
    daily_closes = {}

    for sym, sym_group in bulk_df.groupby("symbol"):
        group = sym_group.drop(columns=["symbol", "BarNumber"], errors="ignore")
        group["_date"] = group.index.date
        dates = sorted(group["_date"].unique())

        ticker_5min_days[sym] = {}
        daily_closes[sym] = {}

        for d in dates:
            day_df = group[group["_date"] == d].drop(columns=["_date"])
            if len(day_df) >= 10:
                ticker_5min_days[sym][d] = day_df
                # Daily close = last bar's close
                daily_closes[sym][d] = float(day_df["Close"].iloc[-1])

    num_tickers = len(ticker_5min_days)
    num_days = max((len(v) for v in ticker_5min_days.values()), default=0)
    print(f"  Got data for {num_tickers} tickers across {num_days} trading days")

    return ticker_5min_days, daily_closes


# ─────────────── FORWARD RETURN ANALYSIS ──────────────────────────────────────

def run_forward_return_analysis(
    n_top: int = 10,
    n_bottom: int = 10,
    forward_periods: list[int] = [5, 10, 20],  # Trading days
    lookback_days: int = 70,  # Calendar days of data to fetch
):
    """
    Main analysis: compute RS each day, track forward returns for top/bottom stocks.
    """
    period_labels = {5: "1 Week", 10: "2 Weeks", 20: "1 Month"}

    print("=" * 90)
    print("  FORWARD RETURN ANALYSIS — WHERE DO TOP RS STOCKS GO?")
    print(f"  Tracking top {n_top} & bottom {n_bottom} RS stocks")
    print(f"  Forward periods: {', '.join(period_labels.get(p, f'{p}d') for p in forward_periods)}")
    print("=" * 90)

    # Date range
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=lookback_days)).isoformat()
    end = (today - datetime.timedelta(days=1)).isoformat()

    api_key = _resolve_api_key()
    source = get_data_source("databento", api_key=api_key)

    # Fetch data
    ticker_5min_days, daily_closes = fetch_and_process(source, LIQUID_UNIVERSE, start, end)
    if not ticker_5min_days:
        print("No data. Exiting.")
        return

    # Get all unique trading days (sorted)
    all_dates = set()
    for sym_dates in ticker_5min_days.values():
        all_dates.update(sym_dates.keys())
    sorted_dates = sorted(all_dates)
    date_index = {d: i for i, d in enumerate(sorted_dates)}

    print(f"\nTrading days available: {len(sorted_dates)}")
    print(f"  First: {sorted_dates[0]}  Last: {sorted_dates[-1]}")

    max_forward = max(forward_periods)
    # RS can be computed for days where we have enough forward data
    usable_rs_dates = sorted_dates[:len(sorted_dates) - max_forward]
    print(f"  Usable RS dates (with {max_forward}-day forward): {len(usable_rs_dates)}")

    if len(usable_rs_dates) < 3:
        print("Not enough trading days for forward analysis. Need more data.")
        return

    # ─── DAY-BY-DAY RS + FORWARD RETURNS ───
    # For each RS date, store: top 10 tickers, their RS scores, and forward returns
    all_observations = []  # List of dicts

    for rs_date in usable_rs_dates:
        di = date_index[rs_date]

        # 1. Compute RS scores for all tickers with data on this day
        day_scores = []
        for ticker, sym_dates in ticker_5min_days.items():
            if rs_date not in sym_dates:
                continue
            df = sym_dates[rs_date]
            try:
                score = compute_bpa_rs_score(df)
                score["ticker"] = ticker
                day_scores.append(score)
            except Exception:
                continue

        if len(day_scores) < n_top + n_bottom:
            continue

        # Sort by RS
        day_scores.sort(key=lambda x: x["rs_score"], reverse=True)
        avg_rs = np.mean([s["rs_score"] for s in day_scores])

        top_n = day_scores[:n_top]
        bot_n = day_scores[-n_bottom:]
        # Middle group for comparison
        mid_start = len(day_scores) // 2 - n_top // 2
        mid_n = day_scores[mid_start:mid_start + n_top]

        # 2. Compute forward returns for each group
        for group_name, group_scores in [("Top", top_n), ("Bottom", bot_n), ("Middle", mid_n)]:
            for score_data in group_scores:
                ticker = score_data["ticker"]

                # Get base close price on RS date
                base_close = daily_closes.get(ticker, {}).get(rs_date)
                if base_close is None or base_close <= 0:
                    continue

                obs = {
                    "rs_date": rs_date,
                    "ticker": ticker,
                    "rs_score": score_data["rs_score"],
                    "group": group_name,
                    "base_close": base_close,
                    "avg_market_rs": avg_rs,
                }

                # Forward returns at each period
                for period in forward_periods:
                    future_idx = di + period
                    if future_idx >= len(sorted_dates):
                        obs[f"fwd_{period}d_ret"] = None
                        continue

                    future_date = sorted_dates[future_idx]
                    future_close = daily_closes.get(ticker, {}).get(future_date)

                    if future_close is not None and future_close > 0:
                        ret_pct = ((future_close - base_close) / base_close) * 100
                        obs[f"fwd_{period}d_ret"] = round(ret_pct, 3)
                    else:
                        obs[f"fwd_{period}d_ret"] = None

                all_observations.append(obs)

    # Also compute SPY forward returns as benchmark
    spy_fwd = {}
    for rs_date in usable_rs_dates:
        di = date_index[rs_date]
        spy_close = daily_closes.get("SPY", {}).get(rs_date)
        if spy_close is None or spy_close <= 0:
            continue
        spy_fwd[rs_date] = {}
        for period in forward_periods:
            future_idx = di + period
            if future_idx < len(sorted_dates):
                future_date = sorted_dates[future_idx]
                future_close = daily_closes.get("SPY", {}).get(future_date)
                if future_close is not None and future_close > 0:
                    spy_fwd[rs_date][period] = ((future_close - spy_close) / spy_close) * 100

    if not all_observations:
        print("\nNo observations generated.")
        return

    print(f"\n  Total observations: {len(all_observations)}")
    print(f"  RS dates used: {len(usable_rs_dates)}")

    # ─── ANALYZE RESULTS ───
    df_obs = pd.DataFrame(all_observations)

    print(f"\n{'=' * 90}")
    print(f"  RESULTS — FORWARD RETURNS BY RS GROUP")
    print(f"{'=' * 90}")

    for period in forward_periods:
        col = f"fwd_{period}d_ret"
        label = period_labels.get(period, f"{period}d")

        print(f"\n  ── {label} ({period} trading days) Forward Returns ──")
        print(f"  {'Group':<12} {'N':>5}  {'Avg %':>8}  {'Med %':>8}  {'Win%':>6}  {'Best%':>8}  {'Worst%':>8}  {'Sharpe':>7}")
        print("  " + "-" * 75)

        for group in ["Top", "Middle", "Bottom"]:
            grp = df_obs[(df_obs["group"] == group) & (df_obs[col].notna())]
            if grp.empty:
                continue

            rets = grp[col].values
            n = len(rets)
            avg = np.mean(rets)
            med = np.median(rets)
            win_rate = (rets > 0).sum() / n * 100
            best = np.max(rets)
            worst = np.min(rets)
            std = np.std(rets)
            sharpe = avg / std if std > 0 else 0

            color_indicator = "+" if avg > 0 else "-"
            print(f"  {group + ' RS':<12} {n:>5}  {avg:>+7.2f}%  {med:>+7.2f}%  {win_rate:>5.1f}%  {best:>+7.2f}%  {worst:>+7.2f}%  {sharpe:>7.2f}")

        # SPY benchmark
        spy_rets = [spy_fwd[d].get(period) for d in usable_rs_dates if d in spy_fwd and period in spy_fwd.get(d, {})]
        spy_rets = [r for r in spy_rets if r is not None]
        if spy_rets:
            avg_spy = np.mean(spy_rets)
            med_spy = np.median(spy_rets)
            win_spy = sum(1 for r in spy_rets if r > 0) / len(spy_rets) * 100
            std_spy = np.std(spy_rets)
            sharpe_spy = avg_spy / std_spy if std_spy > 0 else 0
            print(f"  {'SPY (bench)':<12} {len(spy_rets):>5}  {avg_spy:>+7.2f}%  {med_spy:>+7.2f}%  {win_spy:>5.1f}%  {max(spy_rets):>+7.2f}%  {min(spy_rets):>+7.2f}%  {sharpe_spy:>7.2f}")

    # ─── ALPHA ANALYSIS ───
    print(f"\n{'=' * 90}")
    print(f"  ALPHA OVER SPY (Top RS vs Market)")
    print(f"{'=' * 90}")

    for period in forward_periods:
        col = f"fwd_{period}d_ret"
        label = period_labels.get(period, f"{period}d")

        top_grp = df_obs[(df_obs["group"] == "Top") & (df_obs[col].notna())]
        bot_grp = df_obs[(df_obs["group"] == "Bottom") & (df_obs[col].notna())]

        # Calculate per-date average returns
        top_by_date = top_grp.groupby("rs_date")[col].mean()
        bot_by_date = bot_grp.groupby("rs_date")[col].mean()

        # Compute alpha per date (top RS return - SPY return)
        alphas_top = []
        alphas_bot = []
        for d in usable_rs_dates:
            spy_r = spy_fwd.get(d, {}).get(period)
            if spy_r is None:
                continue
            if d in top_by_date.index:
                alphas_top.append(top_by_date[d] - spy_r)
            if d in bot_by_date.index:
                alphas_bot.append(bot_by_date[d] - spy_r)

        if alphas_top:
            avg_alpha_top = np.mean(alphas_top)
            alpha_positive = sum(1 for a in alphas_top if a > 0) / len(alphas_top) * 100
            print(f"\n  {label}: Top RS alpha = {avg_alpha_top:+.2f}% avg ({alpha_positive:.0f}% of days beat SPY)")

        if alphas_bot:
            avg_alpha_bot = np.mean(alphas_bot)
            alpha_negative = sum(1 for a in alphas_bot if a < 0) / len(alphas_bot) * 100
            print(f"  {label}: Bottom RS alpha = {avg_alpha_bot:+.2f}% avg ({alpha_negative:.0f}% of days trail SPY)")

    # ─── TOP RS TICKERS FREQUENCY ───
    top_tickers_freq = df_obs[df_obs["group"] == "Top"]["ticker"].value_counts()
    print(f"\n  MOST FREQUENT TOP 10 RS TICKERS:")
    for ticker, count in top_tickers_freq.head(15).items():
        avg_ret_5d = df_obs[(df_obs["ticker"] == ticker) & (df_obs["group"] == "Top") & (df_obs["fwd_5d_ret"].notna())]["fwd_5d_ret"].mean()
        avg_ret_20d = df_obs[(df_obs["ticker"] == ticker) & (df_obs["group"] == "Top") & (df_obs["fwd_20d_ret"].notna())]["fwd_20d_ret"].mean()
        print(f"    {ticker:<7} appeared {count:>3} times  |  Avg 1w: {avg_ret_5d:>+6.2f}%  |  Avg 1m: {avg_ret_20d:>+6.2f}%")

    # ─── GENERATE PDF ───
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "BPA_RS_Forward_Returns.pdf")
    generate_pdf(df_obs, spy_fwd, forward_periods, period_labels, usable_rs_dates, sorted_dates, pdf_path)

    print(f"\n{'=' * 90}")
    print(f"  Done — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 90}\n")

    return pdf_path


# ─────────────── PDF REPORT ───────────────────────────────────────────────────

def generate_pdf(df_obs, spy_fwd, forward_periods, period_labels, usable_rs_dates, sorted_dates, output_path):
    """Generate PDF with forward return analysis charts."""
    print(f"\nGenerating PDF report...")

    with PdfPages(output_path) as pdf:
        # ── Page 1: Summary table ──
        fig = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

        ax.text(0.5, 0.93, "BPA RELATIVE STRENGTH", transform=ax.transAxes,
                fontsize=24, fontweight="bold", color="white", ha="center")
        ax.text(0.5, 0.87, "FORWARD RETURN ANALYSIS", transform=ax.transAxes,
                fontsize=20, fontweight="bold", color="#FFA726", ha="center")
        ax.text(0.5, 0.81,
                f"Where do the Top 10 RS stocks end up 1 week, 2 weeks, 1 month later?",
                transform=ax.transAxes, fontsize=11, color="#aaaaaa", ha="center")

        n_dates = len(usable_rs_dates)
        n_obs = len(df_obs)
        ax.text(0.5, 0.76,
                f"{n_dates} RS dates  |  {n_obs} observations  |  {sorted_dates[0]} to {sorted_dates[-1]}",
                transform=ax.transAxes, fontsize=10, color="#777777", ha="center")

        # Results table
        y = 0.68
        for period in forward_periods:
            col = f"fwd_{period}d_ret"
            label = period_labels.get(period, f"{period}d")

            ax.text(0.5, y, f"── {label} ({period} trading days) ──",
                    transform=ax.transAxes, fontsize=12, fontweight="bold", color="#FFA726", ha="center")
            y -= 0.03

            header = f"  {'Group':<14} {'N':>5}  {'Avg Return':>10}  {'Median':>8}  {'Win Rate':>8}  {'Sharpe':>7}"
            ax.text(0.08, y, header, transform=ax.transAxes, fontsize=7.5,
                    color="#999999", ha="left", family="monospace")
            y -= 0.025

            for group in ["Top", "Middle", "Bottom"]:
                grp = df_obs[(df_obs["group"] == group) & (df_obs[col].notna())]
                if grp.empty:
                    continue
                rets = grp[col].values
                n = len(rets)
                avg = np.mean(rets)
                med = np.median(rets)
                wr = (rets > 0).sum() / n * 100
                std = np.std(rets)
                sharpe = avg / std if std > 0 else 0

                if avg > 1:
                    color = "#00C853"
                elif avg > 0:
                    color = "#69F0AE"
                elif avg > -1:
                    color = "#FFA726"
                else:
                    color = "#FF1744"

                line = f"  {group + ' RS':<14} {n:>5}  {avg:>+9.2f}%  {med:>+7.2f}%  {wr:>6.1f}%  {sharpe:>7.2f}"
                ax.text(0.08, y, line, transform=ax.transAxes, fontsize=7.5,
                        color=color, ha="left", family="monospace")
                y -= 0.022

            # SPY
            spy_rets = [spy_fwd[d].get(period) for d in usable_rs_dates
                       if d in spy_fwd and period in spy_fwd.get(d, {})]
            spy_rets = [r for r in spy_rets if r is not None]
            if spy_rets:
                avg_spy = np.mean(spy_rets)
                med_spy = np.median(spy_rets)
                wr_spy = sum(1 for r in spy_rets if r > 0) / len(spy_rets) * 100
                std_spy = np.std(spy_rets)
                sharpe_spy = avg_spy / std_spy if std_spy > 0 else 0
                line = f"  {'SPY (bench)':<14} {len(spy_rets):>5}  {avg_spy:>+9.2f}%  {med_spy:>+7.2f}%  {wr_spy:>6.1f}%  {sharpe_spy:>7.2f}"
                ax.text(0.08, y, line, transform=ax.transAxes, fontsize=7.5,
                        color="#42A5F5", ha="left", family="monospace")

            y -= 0.04

        # Key insight
        top_5d = df_obs[(df_obs["group"] == "Top") & (df_obs["fwd_5d_ret"].notna())]["fwd_5d_ret"]
        top_20d = df_obs[(df_obs["group"] == "Top") & (df_obs["fwd_20d_ret"].notna())]["fwd_20d_ret"]

        if not top_20d.empty:
            avg_1m = top_20d.mean()
            wr_1m = (top_20d > 0).sum() / len(top_20d) * 100
            if avg_1m > 0:
                ax.text(0.5, 0.06,
                        f"Top 10 RS stocks average {avg_1m:+.2f}% over 1 month ({wr_1m:.0f}% positive)",
                        transform=ax.transAxes, fontsize=11, fontweight="bold",
                        color="#00C853", ha="center")
            else:
                ax.text(0.5, 0.06,
                        f"Top 10 RS stocks average {avg_1m:+.2f}% over 1 month ({wr_1m:.0f}% positive)",
                        transform=ax.transAxes, fontsize=11, fontweight="bold",
                        color="#FF1744", ha="center")

        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # ── Page 2: Distribution charts ──
        fig, axes = plt.subplots(len(forward_periods), 1,
                                 figsize=(11, 3.5 * len(forward_periods)),
                                 facecolor="#1a1a2e")
        if len(forward_periods) == 1:
            axes = [axes]

        for idx, period in enumerate(forward_periods):
            ax = axes[idx]
            ax.set_facecolor("#1a1a2e")

            col = f"fwd_{period}d_ret"
            label = period_labels.get(period, f"{period}d")

            top_rets = df_obs[(df_obs["group"] == "Top") & (df_obs[col].notna())][col].values
            bot_rets = df_obs[(df_obs["group"] == "Bottom") & (df_obs[col].notna())][col].values
            mid_rets = df_obs[(df_obs["group"] == "Middle") & (df_obs[col].notna())][col].values

            if len(top_rets) > 0:
                bins = np.linspace(min(np.min(top_rets), np.min(bot_rets) if len(bot_rets) > 0 else 0) - 1,
                                   max(np.max(top_rets), np.max(bot_rets) if len(bot_rets) > 0 else 0) + 1,
                                   30)
                ax.hist(top_rets, bins=bins, alpha=0.6, color="#00C853", label=f"Top RS (avg {np.mean(top_rets):+.2f}%)", edgecolor="none")
                if len(bot_rets) > 0:
                    ax.hist(bot_rets, bins=bins, alpha=0.6, color="#FF1744", label=f"Bottom RS (avg {np.mean(bot_rets):+.2f}%)", edgecolor="none")
                if len(mid_rets) > 0:
                    ax.hist(mid_rets, bins=bins, alpha=0.4, color="#FFA726", label=f"Middle RS (avg {np.mean(mid_rets):+.2f}%)", edgecolor="none")

                ax.axvline(x=0, color="#555555", linewidth=1, linestyle="--")
                ax.axvline(x=np.mean(top_rets), color="#00C853", linewidth=2, linestyle="-", alpha=0.8)
                if len(bot_rets) > 0:
                    ax.axvline(x=np.mean(bot_rets), color="#FF1744", linewidth=2, linestyle="-", alpha=0.8)

            ax.set_title(f"{label} Forward Return Distribution", fontsize=13, color="white", fontweight="bold")
            ax.set_xlabel("Return %", fontsize=9, color="#aaaaaa")
            ax.set_ylabel("Count", fontsize=9, color="#aaaaaa")
            ax.legend(fontsize=8, facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
            ax.tick_params(colors="#999999")
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # ── Page 3: Average forward return by group over time ──
        fig, axes = plt.subplots(len(forward_periods), 1,
                                 figsize=(11, 3.5 * len(forward_periods)),
                                 facecolor="#1a1a2e")
        if len(forward_periods) == 1:
            axes = [axes]

        for idx, period in enumerate(forward_periods):
            ax = axes[idx]
            ax.set_facecolor("#1a1a2e")
            col = f"fwd_{period}d_ret"
            label = period_labels.get(period, f"{period}d")

            for group, color in [("Top", "#00C853"), ("Bottom", "#FF1744"), ("Middle", "#FFA726")]:
                grp = df_obs[(df_obs["group"] == group) & (df_obs[col].notna())]
                if grp.empty:
                    continue

                by_date = grp.groupby("rs_date")[col].mean().sort_index()
                dates = [d for d in by_date.index]
                vals = by_date.values
                cum_avg = np.cumsum(vals) / np.arange(1, len(vals) + 1)

                ax.plot(range(len(dates)), cum_avg, color=color, linewidth=2,
                        label=f"{group} RS (rolling avg)")
                ax.scatter(range(len(dates)), vals, color=color, s=15, alpha=0.4)

            # SPY benchmark line
            spy_rets_series = []
            spy_dates_used = []
            for d in usable_rs_dates:
                ret = spy_fwd.get(d, {}).get(period)
                if ret is not None:
                    spy_rets_series.append(ret)
                    spy_dates_used.append(d)
            if spy_rets_series:
                cum_spy = np.cumsum(spy_rets_series) / np.arange(1, len(spy_rets_series) + 1)
                ax.plot(range(len(spy_dates_used)), cum_spy, color="#42A5F5", linewidth=2,
                        linestyle="--", label="SPY (rolling avg)")

            ax.axhline(y=0, color="#555555", linewidth=0.5)
            ax.set_title(f"{label} Forward Return — Rolling Average by RS Group",
                         fontsize=13, color="white", fontweight="bold")
            ax.set_xlabel("RS Date Index", fontsize=9, color="#aaaaaa")
            ax.set_ylabel("Avg Forward Return %", fontsize=9, color="#aaaaaa")
            ax.legend(fontsize=8, facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
            ax.tick_params(colors="#999999")
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # ── Page 4: Alpha chart (Top RS - SPY) ──
        fig, ax = plt.subplots(1, 1, figsize=(11, 6), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        bar_width = 0.25
        x = np.arange(len(forward_periods))

        for gi, (group, color) in enumerate([("Top", "#00C853"), ("Bottom", "#FF1744")]):
            alphas = []
            for period in forward_periods:
                col = f"fwd_{period}d_ret"
                grp = df_obs[(df_obs["group"] == group) & (df_obs[col].notna())]
                if grp.empty:
                    alphas.append(0)
                    continue

                by_date = grp.groupby("rs_date")[col].mean()
                alpha_list = []
                for d in by_date.index:
                    spy_r = spy_fwd.get(d, {}).get(period)
                    if spy_r is not None:
                        alpha_list.append(by_date[d] - spy_r)

                alphas.append(np.mean(alpha_list) if alpha_list else 0)

            bars = ax.bar(x + gi * bar_width, alphas, bar_width, color=color, alpha=0.8,
                         label=f"{group} RS Alpha", edgecolor="none")

            for bar, alpha in zip(bars, alphas):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{alpha:+.2f}%", ha="center", fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels([period_labels.get(p, f"{p}d") for p in forward_periods])
        ax.axhline(y=0, color="#555555", linewidth=1)
        ax.set_title("Alpha Over SPY by RS Group", fontsize=16, color="white", fontweight="bold")
        ax.set_ylabel("Average Alpha (%)", fontsize=11, color="#aaaaaa")
        ax.legend(fontsize=10, facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
        ax.tick_params(colors="#999999", labelsize=11)
        ax.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # ── Page 5: Top RS ticker frequency ──
        fig = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")

        top_tickers = df_obs[df_obs["group"] == "Top"]["ticker"].value_counts().head(20)

        if not top_tickers.empty:
            colors_bar = []
            for ticker in top_tickers.index:
                avg_ret = df_obs[(df_obs["ticker"] == ticker) & (df_obs["group"] == "Top") &
                                (df_obs["fwd_20d_ret"].notna())]["fwd_20d_ret"].mean()
                if avg_ret > 2:
                    colors_bar.append("#00C853")
                elif avg_ret > 0:
                    colors_bar.append("#69F0AE")
                elif avg_ret > -2:
                    colors_bar.append("#FFA726")
                else:
                    colors_bar.append("#FF1744")

            bars = ax.barh(range(len(top_tickers)), top_tickers.values, color=colors_bar, edgecolor="none")
            ax.set_yticks(range(len(top_tickers)))
            ax.set_yticklabels(top_tickers.index, fontsize=9, color="#cccccc")
            ax.invert_yaxis()

            for i, (ticker, count) in enumerate(top_tickers.items()):
                avg_1m = df_obs[(df_obs["ticker"] == ticker) & (df_obs["group"] == "Top") &
                               (df_obs["fwd_20d_ret"].notna())]["fwd_20d_ret"].mean()
                ax.text(count + 0.2, i, f"{count} days  (1m avg: {avg_1m:+.1f}%)",
                        fontsize=8, color="#cccccc", va="center")

        ax.set_title("Most Frequent Top 10 RS Stocks", fontsize=16, color="white", fontweight="bold")
        ax.set_xlabel("Days in Top 10", fontsize=11, color="#aaaaaa")
        ax.tick_params(colors="#999999")
        ax.grid(True, alpha=0.2, axis="x")

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

    print(f"PDF saved: {output_path}")


# ─────────────── MAIN ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_forward_return_analysis()
