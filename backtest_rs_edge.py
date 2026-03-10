#!/usr/bin/env python3
"""
backtest_rs_edge.py — Does BPA Relative Strength predict next-day pullback profitability?

Hypothesis: Stocks with top RS scores (strong BPA trends) produce profitable
pullback entries the following day, while bottom RS stocks produce profitable
short entries on rallies.

Strategy:
  Day D:  Compute BPA RS scores for all tickers → rank
  Day D+1: For top N tickers → find pullback-to-EMA long entries
           For bottom N tickers → find rally-to-EMA short entries
  Track outcomes at 1:1, 2:1, 3:1 R/R

Also tests SCALING strategy:
  - Enter 50% at first pullback, 50% at deeper pullback
  - Breakeven / 1R from average entry

Universe: S&P 500 + sector ETFs (~100 most liquid tickers for speed)
Period:   ~20 trading days
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
from algo_engine import Bar, bars_from_df, compute_ema, find_swing_lows, find_swing_highs
from daily_rs_rankings import compute_bpa_rs_score, SECTOR_ETFS

logger = logging.getLogger(__name__)


# ─────────────── TICKER UNIVERSE (LIQUID SUBSET) ────────────────────────────

# Top ~85 S&P 500 by market cap + 15 sector ETFs = 100 tickers
LIQUID_UNIVERSE = [
    # Sector ETFs
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    # Mega cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    # Large cap tech
    "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN", "IBM", "INTC", "TXN", "QCOM", "INTU",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD", "LOW",
    # Industrials
    "CAT", "DE", "HON", "UPS", "BA", "GE", "RTX", "LMT", "MMM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Other
    "DIS", "NFLX", "PYPL", "ABNB", "UBER", "NEE", "SO", "DUK",
    "AMT", "PLD", "CCI", "LIN", "SHW",
]


# ─────────────── PULLBACK DETECTION ─────────────────────────────────────────

def find_pullback_entries(bars: list[Bar], ema: list[float], direction: str) -> list[dict]:
    """
    Find pullback-to-EMA entries in a day's 5-min bars.

    Long: price is trending above EMA → dips to test EMA → bull bar forms → buy stop above
    Short: price is trending below EMA → rallies to test EMA → bear bar forms → sell stop below

    Returns list of entry dicts with bar_idx, entry, stop, risk.
    """
    entries = []
    n = len(bars)
    if n < 15:
        return entries

    avg_range = np.mean([b.range for b in bars if b.range > 0])
    if avg_range == 0:
        return entries

    for i in range(10, n - 2):  # Leave room for entry bar + some follow-through
        b = bars[i]
        e = ema[i]

        if e <= 0 or b.range == 0:
            continue

        if direction == "Long":
            # Context: majority of recent bars above EMA (trend confirmation)
            above_count = sum(1 for j in range(max(0, i - 10), i) if bars[j].close > ema[j])
            if above_count < 6:
                continue

            # Bar tests EMA: low is within 0.3% of EMA (or slightly below)
            ema_dist = (b.low - e) / e
            if ema_dist > 0.003 or ema_dist < -0.005:
                continue

            # Signal bar: bull bar closing in upper half, near EMA
            if not b.is_bull:
                continue
            if b.close < (b.high + b.low) / 2:
                continue

            entry_price = round(b.high + 0.01, 2)
            # Stop below the pullback low (or EMA, whichever is lower)
            pullback_low = min(b.low, min(bars[j].low for j in range(max(0, i - 2), i + 1)))
            stop_price = round(pullback_low - 0.01, 2)
            risk = entry_price - stop_price

            # Sanity: risk should be 0.1% to 3% of price
            if risk <= 0 or risk > entry_price * 0.03 or risk < entry_price * 0.001:
                continue

            entries.append({
                "bar_idx": i,
                "entry": entry_price,
                "stop": stop_price,
                "risk": risk,
                "signal_bar": b,
            })

        elif direction == "Short":
            # Context: majority of recent bars below EMA
            below_count = sum(1 for j in range(max(0, i - 10), i) if bars[j].close < ema[j])
            if below_count < 6:
                continue

            # Bar tests EMA from below
            ema_dist = (b.high - e) / e
            if ema_dist < -0.003 or ema_dist > 0.005:
                continue

            # Signal bar: bear bar closing in lower half
            if not b.is_bear:
                continue
            if b.close > (b.high + b.low) / 2:
                continue

            entry_price = round(b.low - 0.01, 2)
            pullback_high = max(b.high, max(bars[j].high for j in range(max(0, i - 2), i + 1)))
            stop_price = round(pullback_high + 0.01, 2)
            risk = stop_price - entry_price

            if risk <= 0 or risk > entry_price * 0.03 or risk < entry_price * 0.001:
                continue

            entries.append({
                "bar_idx": i,
                "entry": entry_price,
                "stop": stop_price,
                "risk": risk,
                "signal_bar": b,
            })

    # Deduplicate: only keep entries that are at least 5 bars apart
    if len(entries) <= 1:
        return entries

    filtered = [entries[0]]
    for e in entries[1:]:
        if e["bar_idx"] - filtered[-1]["bar_idx"] >= 5:
            filtered.append(e)

    return filtered


# ─────────────── TRADE SIMULATION ────────────────────────────────────────────

def simulate_trade(bars: list[Bar], entry_info: dict, direction: str,
                   target_multiples: list[float] = [1.0, 2.0, 3.0]) -> dict:
    """
    Simulate a trade and return outcomes at each R:R target.
    Entry is a stop order above/below signal bar.
    """
    signal_idx = entry_info["bar_idx"]
    entry_price = entry_info["entry"]
    stop_price = entry_info["stop"]
    risk = entry_info["risk"]

    # Check if entry is triggered (next bar must trade through entry price)
    filled = False
    fill_idx = None
    for look in range(signal_idx + 1, min(signal_idx + 4, len(bars))):  # 3-bar lookhead
        b = bars[look]
        if direction == "Long" and b.high >= entry_price:
            filled = True
            fill_idx = look
            break
        elif direction == "Short" and b.low <= entry_price:
            filled = True
            fill_idx = look
            break

    if not filled:
        return {"filled": False}

    results = {"filled": True, "fill_bar": fill_idx, "entry": entry_price, "risk": risk}

    # Track MAE/MFE
    mae = 0.0  # Maximum adverse excursion (in R)
    mfe = 0.0  # Maximum favorable excursion (in R)

    for mult in target_multiples:
        if direction == "Long":
            target = entry_price + risk * mult
        else:
            target = entry_price - risk * mult

        hit_target = False
        hit_stop = False
        exit_price = None
        exit_reason = "eod"

        for i in range(fill_idx, len(bars)):
            b = bars[i]

            if direction == "Long":
                # Track MAE/MFE
                unrealized_low = (b.low - entry_price) / risk
                unrealized_high = (b.high - entry_price) / risk
                mae = min(mae, unrealized_low)
                mfe = max(mfe, unrealized_high)

                # Check stop first (conservative)
                if b.low <= stop_price:
                    hit_stop = True
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if b.high >= target:
                    hit_target = True
                    exit_price = target
                    exit_reason = f"{mult}R_target"
                    break
            else:
                unrealized_low = (entry_price - b.high) / risk
                unrealized_high = (entry_price - b.low) / risk
                mae = min(mae, unrealized_low)
                mfe = max(mfe, unrealized_high)

                if b.high >= stop_price:
                    hit_stop = True
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if b.low <= target:
                    hit_target = True
                    exit_price = target
                    exit_reason = f"{mult}R_target"
                    break

        # If neither hit by EOD, close at last bar's close
        if not hit_target and not hit_stop:
            exit_price = bars[-1].close

        if direction == "Long":
            pnl_r = (exit_price - entry_price) / risk
        else:
            pnl_r = (entry_price - exit_price) / risk

        results[f"{mult}R"] = {
            "hit_target": hit_target,
            "hit_stop": hit_stop,
            "eod": exit_reason == "eod",
            "exit_reason": exit_reason,
            "pnl_r": round(pnl_r, 3),
            "is_winner": hit_target,
        }

    results["mae_r"] = round(mae, 3)
    results["mfe_r"] = round(mfe, 3)

    return results


def simulate_scale_in_trade(bars: list[Bar], entry_info: dict, direction: str) -> dict:
    """
    Simulate a 2-level scaling strategy.

    Level 1: Enter at signal bar (pullback to EMA)
    Level 2: If price goes 0.5R against you, add equal size
    Average entry: midpoint of L1 and L2
    Stop: 1R below L2 entry (1.5R from L1)
    Target: 1R from average entry (breakeven for L1 + 1R for L2)
    """
    signal_idx = entry_info["bar_idx"]
    entry1 = entry_info["entry"]
    stop1 = entry_info["stop"]
    risk1 = entry_info["risk"]

    # Check if L1 fills
    filled = False
    fill_idx = None
    for look in range(signal_idx + 1, min(signal_idx + 4, len(bars))):
        b = bars[look]
        if direction == "Long" and b.high >= entry1:
            filled = True
            fill_idx = look
            break
        elif direction == "Short" and b.low <= entry1:
            filled = True
            fill_idx = look
            break

    if not filled:
        return {"filled": False}

    # Level 2 entry: 0.5R adverse from L1
    if direction == "Long":
        entry2 = entry1 - risk1 * 0.5
        avg_entry = (entry1 + entry2) / 2
        final_stop = stop1 - risk1 * 0.5  # Wider stop for scale-in
        avg_risk = avg_entry - final_stop
        be_target = avg_entry + avg_risk * 0.5  # ~Breakeven (small profit)
        r1_target = avg_entry + avg_risk * 1.0  # 1R from avg
        r2_target = avg_entry + avg_risk * 2.0  # 2R from avg
    else:
        entry2 = entry1 + risk1 * 0.5
        avg_entry = (entry1 + entry2) / 2
        final_stop = stop1 + risk1 * 0.5
        avg_risk = final_stop - avg_entry
        be_target = avg_entry - avg_risk * 0.5
        r1_target = avg_entry - avg_risk * 1.0
        r2_target = avg_entry - avg_risk * 2.0

    # Walk forward: check if L2 fills, then check targets
    l2_filled = False
    results = {"filled": True, "fill_bar": fill_idx, "entry1": entry1, "strategy": "scale_in"}

    for i in range(fill_idx, len(bars)):
        b = bars[i]

        if direction == "Long":
            # Check if L2 triggers
            if not l2_filled and b.low <= entry2:
                l2_filled = True
                results["l2_filled"] = True
                results["l2_bar"] = i

            # Check final stop
            if l2_filled and b.low <= final_stop:
                results["exit_reason"] = "stop"
                results["pnl_r"] = -1.5  # Lost 1R on L1 + 0.5R on L2 (relative to L1 risk)
                results["is_winner"] = False
                return results
            elif not l2_filled and b.low <= stop1:
                results["exit_reason"] = "stop_l1_only"
                results["pnl_r"] = -1.0  # Just L1 stopped
                results["is_winner"] = False
                results["l2_filled"] = False
                return results

            # Check targets
            if l2_filled:
                if b.high >= r1_target:
                    results["exit_reason"] = "1R_scale_target"
                    results["pnl_r"] = 1.0  # ~1R profit from avg
                    results["is_winner"] = True
                    return results
            else:
                # No scale, just L1 — check original 1R target
                if b.high >= entry1 + risk1:
                    results["exit_reason"] = "1R_l1_only"
                    results["pnl_r"] = 1.0
                    results["is_winner"] = True
                    results["l2_filled"] = False
                    return results
        else:
            if not l2_filled and b.high >= entry2:
                l2_filled = True
                results["l2_filled"] = True
                results["l2_bar"] = i

            if l2_filled and b.high >= final_stop:
                results["exit_reason"] = "stop"
                results["pnl_r"] = -1.5
                results["is_winner"] = False
                return results
            elif not l2_filled and b.high >= stop1:
                results["exit_reason"] = "stop_l1_only"
                results["pnl_r"] = -1.0
                results["is_winner"] = False
                results["l2_filled"] = False
                return results

            if l2_filled:
                if b.low <= r1_target:
                    results["exit_reason"] = "1R_scale_target"
                    results["pnl_r"] = 1.0
                    results["is_winner"] = True
                    return results
            else:
                if b.low <= entry1 - risk1:
                    results["exit_reason"] = "1R_l1_only"
                    results["pnl_r"] = 1.0
                    results["is_winner"] = True
                    results["l2_filled"] = False
                    return results

    # EOD close
    last = bars[-1].close
    if l2_filled:
        if direction == "Long":
            pnl = ((last - avg_entry) / avg_risk)
        else:
            pnl = ((avg_entry - last) / avg_risk)
    else:
        if direction == "Long":
            pnl = ((last - entry1) / risk1)
        else:
            pnl = ((entry1 - last) / risk1)

    results["exit_reason"] = "eod"
    results["pnl_r"] = round(pnl, 3)
    results["is_winner"] = pnl > 0
    results["l2_filled"] = l2_filled
    return results


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


def fetch_and_split_by_day(source, tickers, start, end):
    """
    Bulk fetch data, split into per-ticker per-day DataFrames.
    Returns: {ticker: {date: DataFrame}}
    """
    print(f"\nBulk fetching {len(tickers)} tickers ({start} to {end})...")

    if not isinstance(source, DatabentoSource):
        print("ERROR: Only Databento source supported for bulk fetch")
        return {}

    try:
        bulk_df = source.get_bulk_chart_data(tickers, start, end)
    except Exception as e:
        print(f"  Bulk fetch failed: {e}")
        return {}

    if bulk_df is None or bulk_df.empty:
        print("  No data returned")
        return {}

    ticker_days = {}
    for sym, sym_group in bulk_df.groupby("symbol"):
        group = sym_group.drop(columns=["symbol", "BarNumber"], errors="ignore")
        group["_date"] = group.index.date
        dates = sorted(group["_date"].unique())

        ticker_days[sym] = {}
        for d in dates:
            day_df = group[group["_date"] == d].drop(columns=["_date"])
            if len(day_df) >= 10:
                ticker_days[sym][d] = day_df

    num_tickers = len(ticker_days)
    num_days = max((len(v) for v in ticker_days.values()), default=0)
    print(f"  Got data for {num_tickers} tickers across {num_days} trading days")

    return ticker_days


# ─────────────── MAIN BACKTEST ───────────────────────────────────────────────

def run_rs_edge_backtest(
    n_top: int = 20,       # How many top RS tickers to trade long
    n_bottom: int = 20,    # How many bottom RS tickers to trade short
    target_multiples: list[float] = [1.0, 2.0, 3.0],
):
    """
    Main backtest: compute RS on day D, trade pullbacks on day D+1.
    """
    print("=" * 90)
    print("  BACKTEST: BPA RELATIVE STRENGTH → NEXT-DAY PULLBACK EDGE")
    print(f"  Hypothesis: Top RS stocks produce profitable long pullbacks")
    print(f"              Bottom RS stocks produce profitable short pullbacks")
    print(f"  Testing: {', '.join(f'{m}:1' for m in target_multiples)} R:R + Scale-in")
    print("=" * 90)

    # Date range: ~25 trading days
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=35)).isoformat()
    end = (today - datetime.timedelta(days=1)).isoformat()  # Yesterday (skip today if market is open)

    api_key = _resolve_api_key()
    source = get_data_source("databento", api_key=api_key)

    # Fetch data
    ticker_days = fetch_and_split_by_day(source, LIQUID_UNIVERSE, start, end)
    if not ticker_days:
        print("No data. Exiting.")
        return

    # Get all unique trading days
    all_dates = set()
    for sym_dates in ticker_days.values():
        all_dates.update(sym_dates.keys())
    sorted_dates = sorted(all_dates)
    print(f"\nTrading days available: {len(sorted_dates)}")
    print(f"  First: {sorted_dates[0]}  Last: {sorted_dates[-1]}")

    # ─── DAY-BY-DAY RS → NEXT-DAY TRADE ───
    all_trades = []
    all_scale_trades = []
    rs_history = []

    for day_idx in range(len(sorted_dates) - 1):
        rs_day = sorted_dates[day_idx]
        trade_day = sorted_dates[day_idx + 1]

        print(f"\n  Day {day_idx + 1}: RS on {rs_day} → Trade on {trade_day}")

        # 1. Compute RS for all tickers that have data on rs_day
        day_scores = []
        for ticker, sym_dates in ticker_days.items():
            if rs_day not in sym_dates:
                continue
            df = sym_dates[rs_day]
            try:
                score = compute_bpa_rs_score(df)
                score["ticker"] = ticker
                day_scores.append(score)
            except Exception:
                continue

        if len(day_scores) < n_top + n_bottom:
            print(f"    Only {len(day_scores)} tickers scored, skipping")
            continue

        # Sort by RS
        day_scores.sort(key=lambda x: x["rs_score"], reverse=True)

        # Record RS history
        avg_rs = np.mean([s["rs_score"] for s in day_scores])
        rs_history.append({"date": rs_day, "avg_rs": avg_rs, "n_scored": len(day_scores)})

        # Top N (long candidates) and Bottom N (short candidates)
        top_n = day_scores[:n_top]
        bot_n = day_scores[-n_bottom:]

        top_tickers = [s["ticker"] for s in top_n]
        bot_tickers = [s["ticker"] for s in bot_n]

        print(f"    Scored {len(day_scores)} tickers | Avg RS: {avg_rs:+.1f}")
        print(f"    Top {n_top}: {', '.join(top_tickers[:5])}...")
        print(f"    Bot {n_bottom}: {', '.join(bot_tickers[:5])}...")

        # 2. Find and simulate trades on trade_day
        day_trade_count = 0

        # LONGS on top RS stocks
        for score_data in top_n:
            ticker = score_data["ticker"]
            if trade_day not in ticker_days.get(ticker, {}):
                continue

            trade_df = ticker_days[ticker][trade_day]
            bars = bars_from_df(trade_df)
            ema = compute_ema(bars)

            entries = find_pullback_entries(bars, ema, "Long")
            if not entries:
                continue

            # Take first entry only (most conservative)
            entry_info = entries[0]

            # Standard trades at multiple R:R
            result = simulate_trade(bars, entry_info, "Long", target_multiples)
            if result.get("filled"):
                result["ticker"] = ticker
                result["rs_score"] = score_data["rs_score"]
                result["rs_rank"] = top_tickers.index(ticker) + 1
                result["direction"] = "Long"
                result["rs_day"] = rs_day
                result["trade_day"] = trade_day
                result["rs_group"] = "top"
                all_trades.append(result)
                day_trade_count += 1

            # Scale-in trade
            scale_result = simulate_scale_in_trade(bars, entry_info, "Long")
            if scale_result.get("filled"):
                scale_result["ticker"] = ticker
                scale_result["rs_score"] = score_data["rs_score"]
                scale_result["direction"] = "Long"
                scale_result["rs_day"] = rs_day
                scale_result["trade_day"] = trade_day
                scale_result["rs_group"] = "top"
                all_scale_trades.append(scale_result)

        # SHORTS on bottom RS stocks
        for score_data in bot_n:
            ticker = score_data["ticker"]
            if trade_day not in ticker_days.get(ticker, {}):
                continue

            trade_df = ticker_days[ticker][trade_day]
            bars = bars_from_df(trade_df)
            ema = compute_ema(bars)

            entries = find_pullback_entries(bars, ema, "Short")
            if not entries:
                continue

            entry_info = entries[0]

            result = simulate_trade(bars, entry_info, "Short", target_multiples)
            if result.get("filled"):
                result["ticker"] = ticker
                result["rs_score"] = score_data["rs_score"]
                result["rs_rank"] = n_bottom - bot_tickers.index(ticker)
                result["direction"] = "Short"
                result["rs_day"] = rs_day
                result["trade_day"] = trade_day
                result["rs_group"] = "bottom"
                all_trades.append(result)
                day_trade_count += 1

            scale_result = simulate_scale_in_trade(bars, entry_info, "Short")
            if scale_result.get("filled"):
                scale_result["ticker"] = ticker
                scale_result["rs_score"] = score_data["rs_score"]
                scale_result["direction"] = "Short"
                scale_result["rs_day"] = rs_day
                scale_result["trade_day"] = trade_day
                scale_result["rs_group"] = "bottom"
                all_scale_trades.append(scale_result)

        print(f"    Trades filled: {day_trade_count}")

    # ─── RESULTS ───
    print_results(all_trades, all_scale_trades, target_multiples, rs_history, sorted_dates)

    # Generate PDF
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "BPA_RS_Edge_Backtest.pdf")
    generate_results_pdf(all_trades, all_scale_trades, target_multiples, rs_history, pdf_path)

    return pdf_path


# ─────────────── RESULTS ANALYSIS ────────────────────────────────────────────

def analyze_trades(trades: list[dict], target_key: str) -> dict:
    """Compute stats for a set of trades at a specific R:R target."""
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "avg_r": 0, "total_r": 0}

    valid = [t for t in trades if target_key in t and t[target_key] is not None]
    if not valid:
        return {"n": 0, "wr": 0, "pf": 0, "avg_r": 0, "total_r": 0}

    n = len(valid)
    winners = [t for t in valid if t[target_key]["is_winner"]]
    losers = [t for t in valid if t[target_key]["hit_stop"]]
    eod = [t for t in valid if t[target_key]["eod"]]

    wr = len(winners) / n * 100 if n > 0 else 0

    pnl_list = [t[target_key]["pnl_r"] for t in valid]
    total_r = sum(pnl_list)
    avg_r = np.mean(pnl_list) if pnl_list else 0

    gross_profit = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    return {
        "n": n,
        "winners": len(winners),
        "losers": len(losers),
        "eod": len(eod),
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "avg_r": round(avg_r, 3),
        "total_r": round(total_r, 1),
        "ev_per_trade": round(avg_r, 3),
    }


def print_results(trades, scale_trades, target_multiples, rs_history, dates):
    """Print comprehensive results."""
    if not trades:
        print("\nNo trades generated.")
        return

    print(f"\n{'=' * 90}")
    print(f"  RESULTS — {len(trades)} trades over {len(dates)} trading days")
    print(f"{'=' * 90}")

    # Split by direction/group
    long_trades = [t for t in trades if t["direction"] == "Long"]
    short_trades = [t for t in trades if t["direction"] == "Short"]
    top_trades = [t for t in trades if t["rs_group"] == "top"]
    bot_trades = [t for t in trades if t["rs_group"] == "bottom"]

    # Overall results by R:R
    print(f"\n  {'Strategy':<25} {'N':>5} {'WR%':>7} {'PF':>6} {'AvgR':>7} {'TotalR':>8} {'EV/Trade':>9}")
    print("  " + "-" * 70)

    for mult in target_multiples:
        key = f"{mult}R"
        label = f"All trades @ {mult}:1"
        stats = analyze_trades(trades, key)
        print(f"  {label:<25} {stats['n']:>5} {stats['wr']:>6.1f}% {stats['pf']:>6.2f} {stats['avg_r']:>+7.3f} {stats['total_r']:>+8.1f} {stats['ev_per_trade']:>+9.3f}")

    # Scale-in results
    if scale_trades:
        scale_winners = [t for t in scale_trades if t.get("is_winner", False)]
        scale_n = len(scale_trades)
        scale_wr = len(scale_winners) / scale_n * 100 if scale_n > 0 else 0
        scale_pnl = [t.get("pnl_r", 0) for t in scale_trades]
        scale_total = sum(scale_pnl)
        scale_avg = np.mean(scale_pnl)
        gp = sum(p for p in scale_pnl if p > 0)
        gl = abs(sum(p for p in scale_pnl if p < 0))
        scale_pf = gp / gl if gl > 0 else float('inf') if gp > 0 else 0

        print(f"  {'Scale-In (2 levels)':<25} {scale_n:>5} {scale_wr:>6.1f}% {scale_pf:>6.2f} {scale_avg:>+7.3f} {scale_total:>+8.1f} {scale_avg:>+9.3f}")

    # Breakdown: TOP RS (Longs) vs BOTTOM RS (Shorts)
    print(f"\n  BREAKDOWN BY RS GROUP:")
    print(f"  {'Group':<25} {'N':>5} {'WR%':>7} {'PF':>6} {'AvgR':>7} {'TotalR':>8}")
    print("  " + "-" * 60)

    for mult in target_multiples:
        key = f"{mult}R"
        top_stats = analyze_trades(top_trades, key)
        bot_stats = analyze_trades(bot_trades, key)

        print(f"  {'Top RS (Long) @ ' + str(mult) + ':1':<25} {top_stats['n']:>5} {top_stats['wr']:>6.1f}% {top_stats['pf']:>6.2f} {top_stats['avg_r']:>+7.3f} {top_stats['total_r']:>+8.1f}")
        print(f"  {'Bot RS (Short) @ ' + str(mult) + ':1':<25} {bot_stats['n']:>5} {bot_stats['wr']:>6.1f}% {bot_stats['pf']:>6.2f} {bot_stats['avg_r']:>+7.3f} {bot_stats['total_r']:>+8.1f}")

    # MAE/MFE analysis
    if trades:
        maes = [t.get("mae_r", 0) for t in trades if "mae_r" in t]
        mfes = [t.get("mfe_r", 0) for t in trades if "mfe_r" in t]
        if maes and mfes:
            print(f"\n  MAE/MFE ANALYSIS:")
            print(f"    Avg MAE: {np.mean(maes):+.2f}R  (worst drawdown during trade)")
            print(f"    Avg MFE: {np.mean(mfes):+.2f}R  (best unrealized profit)")
            print(f"    MFE > 2R before stop: {sum(1 for m in mfes if m > 2) / len(mfes) * 100:.0f}% of trades")

    # Per-ticker breakdown (top performers)
    ticker_pnl = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in ticker_pnl:
            ticker_pnl[tk] = []
        # Use 1R result
        if "1.0R" in t and t["1.0R"]:
            ticker_pnl[tk].append(t["1.0R"]["pnl_r"])

    if ticker_pnl:
        ticker_stats = []
        for tk, pnls in ticker_pnl.items():
            if len(pnls) >= 2:
                ticker_stats.append((tk, len(pnls), sum(pnls), np.mean(pnls)))
        ticker_stats.sort(key=lambda x: x[2], reverse=True)

        if ticker_stats:
            print(f"\n  TOP TICKERS BY P&L (@ 1:1):")
            for tk, n, total, avg in ticker_stats[:10]:
                print(f"    {tk:<7} {n:>3} trades  Total: {total:>+5.1f}R  Avg: {avg:>+5.2f}R")

            print(f"\n  WORST TICKERS BY P&L (@ 1:1):")
            for tk, n, total, avg in ticker_stats[-5:]:
                print(f"    {tk:<7} {n:>3} trades  Total: {total:>+5.1f}R  Avg: {avg:>+5.2f}R")


# ─────────────── PDF REPORT ──────────────────────────────────────────────────

def generate_results_pdf(trades, scale_trades, target_multiples, rs_history, output_path):
    """Generate a PDF report of backtest results."""
    if not trades:
        print("No trades to report.")
        return

    print(f"\nGenerating PDF report...")

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

        ax.text(0.5, 0.92, "BPA RELATIVE STRENGTH EDGE", transform=ax.transAxes,
                fontsize=24, fontweight="bold", color="white", ha="center")
        ax.text(0.5, 0.86, "Next-Day Pullback Backtest Results", transform=ax.transAxes,
                fontsize=16, color="#FFA726", ha="center")

        n_days = len(set(t.get("trade_day") for t in trades))
        ax.text(0.5, 0.80,
                f"{len(trades)} trades  |  {n_days} trading days  |  {len(set(t['ticker'] for t in trades))} unique tickers",
                transform=ax.transAxes, fontsize=11, color="#aaaaaa", ha="center")

        # Results table
        y = 0.72
        ax.text(0.5, y, "STRATEGY COMPARISON", transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="white", ha="center")

        y -= 0.04
        header = f"{'Strategy':<30} {'N':>5}  {'WR%':>6}  {'PF':>5}  {'Avg R':>7}  {'Total R':>8}"
        ax.text(0.08, y, header, transform=ax.transAxes, fontsize=8,
                color="#999999", ha="left", family="monospace")

        long_trades = [t for t in trades if t["direction"] == "Long"]
        short_trades = [t for t in trades if t["direction"] == "Short"]

        strategies = []
        for mult in target_multiples:
            key = f"{mult}R"
            all_stats = analyze_trades(trades, key)
            long_stats = analyze_trades(long_trades, key)
            short_stats = analyze_trades(short_trades, key)
            strategies.append((f"All @ {mult}:1 R:R", all_stats))
            strategies.append((f"  Longs (Top RS) @ {mult}:1", long_stats))
            strategies.append((f"  Shorts (Bot RS) @ {mult}:1", short_stats))

        # Scale-in
        if scale_trades:
            s_winners = [t for t in scale_trades if t.get("is_winner")]
            s_n = len(scale_trades)
            s_wr = len(s_winners) / s_n * 100 if s_n else 0
            s_pnls = [t.get("pnl_r", 0) for t in scale_trades]
            s_total = sum(s_pnls)
            s_avg = np.mean(s_pnls) if s_pnls else 0
            gp = sum(p for p in s_pnls if p > 0)
            gl = abs(sum(p for p in s_pnls if p < 0))
            s_pf = gp / gl if gl > 0 else 0
            strategies.append(("Scale-In (2 levels)", {
                "n": s_n, "wr": round(s_wr, 1), "pf": round(s_pf, 2),
                "avg_r": round(s_avg, 3), "total_r": round(s_total, 1),
            }))

        for label, stats in strategies:
            y -= 0.028
            n = stats["n"]
            if n == 0:
                continue
            wr = stats["wr"]
            pf_val = stats["pf"]
            avg_r = stats["avg_r"]
            total_r = stats["total_r"]

            # Color by profitability
            if total_r > 5:
                color = "#00C853"
            elif total_r > 0:
                color = "#69F0AE"
            elif total_r > -5:
                color = "#FFA726"
            else:
                color = "#FF1744"

            line = f"{label:<30} {n:>5}  {wr:>5.1f}%  {pf_val:>5.2f}  {avg_r:>+7.3f}  {total_r:>+8.1f}"
            ax.text(0.08, y, line, transform=ax.transAxes, fontsize=7.5,
                    color=color, ha="left", family="monospace")

        # Key insight
        best_strat = max(strategies, key=lambda x: x[1].get("total_r", 0) if x[1]["n"] > 0 else -999)
        y -= 0.06
        if best_strat[1]["n"] > 0 and best_strat[1]["total_r"] > 0:
            ax.text(0.5, y,
                    f"Best Strategy: {best_strat[0]} — {best_strat[1]['total_r']:+.1f}R total, "
                    f"{best_strat[1]['wr']:.1f}% WR, {best_strat[1]['pf']:.2f} PF",
                    transform=ax.transAxes, fontsize=10, fontweight="bold", color="#00C853", ha="center")
        else:
            ax.text(0.5, y,
                    "No profitable strategy found — RS may not predict next-day pullbacks",
                    transform=ax.transAxes, fontsize=10, fontweight="bold", color="#FF1744", ha="center")

        # Equity curve (cumulative R)
        # Sort trades chronologically
        sorted_trades = sorted(trades, key=lambda t: (str(t.get("trade_day", "")), t.get("fill_bar", 0)))

        ax.text(0.5, 0.25, "CUMULATIVE P&L (1:1 R:R)", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color="white", ha="center")

        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # Page 2: Equity curve
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor="#1a1a2e")

        # Cumulative R curve
        ax1 = axes[0]
        ax1.set_facecolor("#1a1a2e")

        for mult in target_multiples:
            key = f"{mult}R"
            cum_r = []
            running = 0
            for t in sorted_trades:
                if key in t and t[key]:
                    running += t[key]["pnl_r"]
                    cum_r.append(running)

            if cum_r:
                color = {"1.0R": "#FFA726", "2.0R": "#00C853", "3.0R": "#42A5F5"}.get(key, "#cccccc")
                ax1.plot(cum_r, color=color, linewidth=1.5, label=f"{mult}:1 R:R")

        # Scale-in equity curve
        if scale_trades:
            sorted_scale = sorted(scale_trades, key=lambda t: (str(t.get("trade_day", ""))))
            cum_scale = []
            running = 0
            for t in sorted_scale:
                running += t.get("pnl_r", 0)
                cum_scale.append(running)
            if cum_scale:
                ax1.plot(cum_scale, color="#E040FB", linewidth=1.5, linestyle="--", label="Scale-In")

        ax1.axhline(y=0, color="#555555", linewidth=0.5)
        ax1.set_title("Cumulative P&L (R-multiples)", fontsize=13, color="white", fontweight="bold")
        ax1.set_xlabel("Trade #", fontsize=9, color="#aaaaaa")
        ax1.set_ylabel("Cumulative R", fontsize=9, color="#aaaaaa")
        ax1.legend(fontsize=8, facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
        ax1.tick_params(colors="#999999")
        ax1.grid(True, alpha=0.2)

        # Win rate by RS quintile
        ax2 = axes[1]
        ax2.set_facecolor("#1a1a2e")

        # Bin trades by RS score
        rs_scores = [t["rs_score"] for t in trades]
        if rs_scores:
            bins = np.percentile(rs_scores, [0, 20, 40, 60, 80, 100])
            bin_labels = ["Q1 (weakest)", "Q2", "Q3", "Q4", "Q5 (strongest)"]
            bin_wrs = []
            bin_ns = []

            for i in range(len(bins) - 1):
                bin_trades = [t for t in trades if bins[i] <= t["rs_score"] < bins[i + 1] + 0.01]
                if bin_trades:
                    # Use 1:1 R:R
                    winners = sum(1 for t in bin_trades if "1.0R" in t and t["1.0R"]["is_winner"])
                    wr = winners / len(bin_trades) * 100
                    bin_wrs.append(wr)
                    bin_ns.append(len(bin_trades))
                else:
                    bin_wrs.append(0)
                    bin_ns.append(0)

            colors = ["#FF1744", "#FF8A65", "#FFA726", "#69F0AE", "#00C853"]
            bars_plot = ax2.bar(bin_labels, bin_wrs, color=colors, alpha=0.8, edgecolor="none")

            # Add count labels
            for bar, n in zip(bars_plot, bin_ns):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"n={n}", ha="center", fontsize=8, color="#aaaaaa")

            ax2.axhline(y=50, color="#FFA726", linewidth=1, linestyle="--", alpha=0.5, label="50% WR")
            ax2.set_title("Win Rate by RS Quintile (@ 1:1 R:R)", fontsize=13, color="white", fontweight="bold")
            ax2.set_ylabel("Win Rate %", fontsize=9, color="#aaaaaa")
            ax2.tick_params(colors="#999999")
            ax2.set_ylim(0, max(bin_wrs) * 1.3 if bin_wrs else 100)
            ax2.legend(fontsize=8, facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # Page 3: Long vs Short breakdown
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor="#1a1a2e")

        for idx, (direction, dir_trades, title_str) in enumerate([
            ("Long", [t for t in trades if t["direction"] == "Long"], "LONGS (Top RS → Pullback Buy)"),
            ("Short", [t for t in trades if t["direction"] == "Short"], "SHORTS (Bottom RS → Rally Sell)"),
        ]):
            ax = axes[idx]
            ax.set_facecolor("#1a1a2e")

            if not dir_trades:
                ax.text(0.5, 0.5, f"No {direction} trades", transform=ax.transAxes,
                        fontsize=14, color="#666666", ha="center")
                continue

            # Cumulative R curves for this direction
            sorted_dir = sorted(dir_trades, key=lambda t: (str(t.get("trade_day", "")), t.get("fill_bar", 0)))

            for mult in target_multiples:
                key = f"{mult}R"
                cum_r = []
                running = 0
                for t in sorted_dir:
                    if key in t and t[key]:
                        running += t[key]["pnl_r"]
                        cum_r.append(running)

                if cum_r:
                    color = {"1.0R": "#FFA726", "2.0R": "#00C853", "3.0R": "#42A5F5"}.get(key, "#cccccc")
                    ax.plot(cum_r, color=color, linewidth=1.5, label=f"{mult}:1")

            stats_1r = analyze_trades(dir_trades, "1.0R")
            ax.set_title(f"{title_str}  |  {stats_1r['n']} trades  |  "
                         f"WR: {stats_1r['wr']}%  |  PF: {stats_1r['pf']}  |  "
                         f"Total: {stats_1r['total_r']:+.1f}R",
                         fontsize=11, color="white", fontweight="bold")
            ax.axhline(y=0, color="#555555", linewidth=0.5)
            ax.set_xlabel("Trade #", fontsize=8, color="#aaaaaa")
            ax.set_ylabel("Cumulative R", fontsize=8, color="#aaaaaa")
            ax.legend(fontsize=8, facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
            ax.tick_params(colors="#999999")
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

    print(f"PDF saved: {output_path}")


# ─────────────── MAIN ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_rs_edge_backtest()
