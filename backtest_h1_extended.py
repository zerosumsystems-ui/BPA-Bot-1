#!/usr/bin/env python3
"""
backtest_h1_extended.py — 7-Year H1/L1 Daily Chart Backtest (Month-by-Month)

Tests H1 (first pullback long), L1 (first pullback short), and Buy Below L1
(contrarian long fading L1 shorts) on daily bars across 91 liquid tickers
over ~7 years (Jan 2019 – Mar 2026).

Uses Databento ohlcv-1m resampled to daily bars (fetched in quarterly chunks
to handle API limits). Results broken down month-by-month with yearly subtotals.
"""

import os
import sys
import datetime
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algo_engine import Bar, bars_from_df, compute_ema, find_swing_lows, find_swing_highs

logger = logging.getLogger(__name__)

# ─────────────── UNIVERSE ─────────────────────────────────────────────────────

LIQUID_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN", "IBM", "INTC", "TXN", "QCOM", "INTU",
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD", "LOW",
    "CAT", "DE", "HON", "UPS", "BA", "GE", "RTX", "LMT",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "DIS", "NFLX", "PYPL", "ABNB", "UBER", "NEE", "SO", "DUK",
    "AMT", "PLD", "CCI", "LIN", "SHW",
]

SECTOR_MAP = {
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",
    "XLB": "Materials", "XLC": "Comm", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Tech", "XLP": "Staples", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLV": "Healthcare", "XLY": "Disc",
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "NVDA": "Tech",
    "META": "Tech", "TSLA": "Tech", "AVGO": "Tech", "ORCL": "Tech", "CRM": "Tech",
    "AMD": "Tech", "ADBE": "Tech", "CSCO": "Tech", "ACN": "Tech", "IBM": "Tech",
    "INTC": "Tech", "TXN": "Tech", "QCOM": "Tech", "INTU": "Tech",
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "BAC": "Financials",
    "WFC": "Financials", "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "SCHW": "Financials", "AXP": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "PFE": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "BMY": "Healthcare",
    "WMT": "Staples", "PG": "Staples", "KO": "Staples", "PEP": "Staples",
    "COST": "Staples", "MCD": "Disc", "NKE": "Disc", "SBUX": "Disc",
    "TGT": "Disc", "HD": "Disc", "LOW": "Disc",
    "CAT": "Industrials", "DE": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "BA": "Industrials", "GE": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "DIS": "Comm", "NFLX": "Comm", "PYPL": "Financials", "ABNB": "Disc",
    "UBER": "Disc", "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "LIN": "Materials", "SHW": "Materials",
}


# ─────────────── DATA FETCHING ────────────────────────────────────────────────

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


def fetch_daily_bars_bulk(tickers, start, end, api_key):
    """
    Fetch daily bars via ohlcv-1m resampled to daily, in quarterly chunks.
    Returns {ticker: DataFrame} with daily OHLCV.
    """
    import databento as db

    client = db.Historical(api_key)
    dataset = "XNAS.ITCH"

    # Split date range into ~90-day chunks
    start_dt = datetime.datetime.fromisoformat(start)
    end_dt = datetime.datetime.fromisoformat(end)

    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + datetime.timedelta(days=90), end_dt)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    print(f"\n  Fetching {len(tickers)} tickers in {len(chunks)} quarterly chunks")
    print(f"  Period: {start} to {end}")

    all_daily = {}  # {ticker: list of daily row dicts}

    batch_size = 50
    for ci, (c_start, c_end) in enumerate(chunks):
        s_str = c_start.strftime("%Y-%m-%dT00:00:00")

        # Databento end is exclusive — add 1 day
        target_e = c_end.date() + datetime.timedelta(days=1)
        today = datetime.date.today()
        if target_e > today:
            target_e = today
        e_str = f"{target_e.strftime('%Y-%m-%d')}T00:00:00"

        print(f"\n  Chunk {ci+1}/{len(chunks)}: {c_start.date()} → {c_end.date()}")

        # Batch tickers
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

        for bi, batch in enumerate(batches):
            retries = 3
            for attempt in range(retries):
                try:
                    data = client.timeseries.get_range(
                        dataset=dataset,
                        symbols=batch,
                        stype_in="raw_symbol",
                        schema="ohlcv-1m",
                        start=s_str,
                        end=e_str,
                    )
                    df = data.to_df()
                    if df is None or df.empty:
                        break

                    # Normalize
                    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
                    df = df.rename(columns=rename)
                    keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "symbol"] if c in df.columns]
                    df = df[keep].dropna(subset=["Open", "High", "Low", "Close"])

                    if df.empty:
                        break

                    # Timezone
                    if df.index.tzinfo is None:
                        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
                    else:
                        df.index = df.index.tz_convert("US/Eastern")

                    # Filter RTH
                    df = df.between_time("09:30", "15:59")

                    # Resample to daily per symbol
                    for sym, grp in df.groupby("symbol"):
                        daily = grp.drop(columns=["symbol"]).resample("1D").agg({
                            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
                        }).dropna()
                        daily = daily[daily["Volume"] > 0]

                        if sym not in all_daily:
                            all_daily[sym] = []
                        all_daily[sym].append(daily)

                    print(f"    Batch {bi+1}/{len(batches)}: OK ({len(df)} rows)")
                    break

                except Exception as ex:
                    err = str(ex)
                    if "429" in err or "502" in err or "503" in err or "timeout" in err.lower():
                        wait = 2 ** (attempt + 1)
                        print(f"    Batch {bi+1}: transient error, retry in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"    Batch {bi+1}: error — {err[:120]}")
                        break

    # Concat per ticker
    result = {}
    for sym, dfs in all_daily.items():
        combined = pd.concat(dfs).sort_index()
        # Deduplicate (overlap between chunks)
        combined = combined[~combined.index.duplicated(keep='first')]
        if len(combined) >= 30:
            result[sym] = combined

    print(f"\n  Final: {len(result)} tickers with 30+ daily bars")
    bar_counts = [len(df) for df in result.values()]
    if bar_counts:
        print(f"  Bars per ticker: min={min(bar_counts)}, max={max(bar_counts)}, avg={np.mean(bar_counts):.0f}")

    return result


# ─────────────── H1/L1 DETECTION ──────────────────────────────────────────────

def detect_h1_l1_setups(bars, ema):
    """Detect H1 long and L1 short setups on daily bars."""
    setups = []
    n = len(bars)
    if n < 20:
        return setups

    last_long = -10
    last_short = -10

    for i in range(5, n):
        b = bars[i]
        prev = bars[i-1]
        prev2 = bars[i-2] if i >= 2 else prev

        # H1 LONG
        above_count = sum(1 for j in range(max(0, i-10), i) if bars[j].close > ema[j])
        if above_count >= 7 and i - last_long >= 5:
            is_pullback = prev.is_bear or prev.low < prev2.low
            bars_before = bars[max(0, i-4):i-1]
            bull_before = sum(1 for bb in bars_before if bb.is_bull)

            if is_pullback and bull_before >= 2:
                if b.high > prev.high and b.is_bull:
                    recent_pbs = sum(1 for j in range(max(0, i-8), i-1)
                                     if bars[j].is_bear and bars[j].close > ema[j])
                    if recent_pbs <= 2:
                        entry = prev.high + 0.01
                        stop = min(prev.low, min(bars[j].low for j in range(max(0, i-2), i+1))) - 0.01
                        risk = entry - stop
                        if risk > 0 and risk < entry * 0.15:
                            setups.append({"bar_idx": i, "direction": "Long", "setup": "H1",
                                           "entry": round(entry, 2), "stop": round(stop, 2),
                                           "risk": round(risk, 2)})
                            last_long = i

        # L1 SHORT
        below_count = sum(1 for j in range(max(0, i-10), i) if bars[j].close < ema[j])
        if below_count >= 7 and i - last_short >= 5:
            is_pullback = prev.is_bull or prev.high > prev2.high
            bars_before = bars[max(0, i-4):i-1]
            bear_before = sum(1 for bb in bars_before if bb.is_bear)

            if is_pullback and bear_before >= 2:
                if b.low < prev.low and b.is_bear:
                    recent_pbs = sum(1 for j in range(max(0, i-8), i-1)
                                     if bars[j].is_bull and bars[j].close < ema[j])
                    if recent_pbs <= 2:
                        entry = prev.low - 0.01
                        stop = max(prev.high, max(bars[j].high for j in range(max(0, i-2), i+1))) + 0.01
                        risk = stop - entry
                        if risk > 0 and risk < entry * 0.15:
                            setups.append({"bar_idx": i, "direction": "Short", "setup": "L1",
                                           "entry": round(entry, 2), "stop": round(stop, 2),
                                           "risk": round(risk, 2)})
                            last_short = i

    return setups


def detect_buy_below_l1(bars, ema):
    """Buy Below L1 — Contrarian long that fades L1 short entries.

    When an L1 short triggers (bar breaks below prev.low in a weak downtrend),
    enter LONG at prev.low - 0.01 (same price shorts enter). Stop below the
    trigger bar's low. Thesis: L1 shorts in weak downtrends often trap bears.
    """
    setups = []
    n = len(bars)
    if n < 20:
        return setups

    last_signal = -10

    for i in range(5, n):
        b = bars[i]
        prev = bars[i-1]
        prev2 = bars[i-2] if i >= 2 else prev

        # Weak downtrend context (6-8 of last 10 below EMA — not overwhelming)
        below_count = sum(1 for j in range(max(0, i-10), i) if bars[j].close < ema[j])
        if 6 <= below_count <= 8 and i - last_signal >= 5:
            is_pullback = prev.is_bull or prev.high > prev2.high
            bars_before = bars[max(0, i-4):i-1]
            bear_before = sum(1 for bb in bars_before if bb.is_bear)

            if is_pullback and bear_before >= 1:
                if b.low < prev.low and b.is_bear:
                    # L1 short triggered — we BUY at the short entry price
                    entry = round(prev.low - 0.01, 2)
                    stop = round(b.low - 0.01, 2)
                    risk = entry - stop
                    if risk > 0 and risk < entry * 0.15:
                        setups.append({"bar_idx": i, "direction": "Long",
                                       "setup": "Buy_Below_L1",
                                       "entry": entry, "stop": stop,
                                       "risk": round(risk, 2)})
                        last_signal = i

    return setups


# ─────────────── TRADE SIMULATION ─────────────────────────────────────────────

def simulate_trade(bars, setup, targets=[1.0, 2.0, 3.0], hold_limit=20):
    """Simulate a daily bar trade with multiple R:R targets and hold limit."""
    si = setup["bar_idx"]
    entry = setup["entry"]
    stop = setup["stop"]
    risk = setup["risk"]
    direction = setup["direction"]

    # Check fill
    filled = False
    fill_idx = None
    for look in range(si + 1, min(si + 4, len(bars))):
        b = bars[look]
        if direction == "Long" and b.high >= entry:
            filled = True; fill_idx = look; break
        elif direction == "Short" and b.low <= entry:
            filled = True; fill_idx = look; break

    if not filled:
        return None

    result = {"filled": True, "fill_bar": fill_idx, "entry": entry, "risk": risk,
              "direction": direction, "setup": setup["setup"]}

    mae = 0.0
    mfe = 0.0

    for mult in targets:
        target = entry + risk * mult if direction == "Long" else entry - risk * mult
        hit_target = False
        hit_stop = False
        exit_price = None

        end_bar = min(fill_idx + hold_limit, len(bars))
        for j in range(fill_idx, end_bar):
            b = bars[j]
            if direction == "Long":
                mae = min(mae, (b.low - entry) / risk)
                mfe = max(mfe, (b.high - entry) / risk)
                if b.low <= stop:
                    hit_stop = True; exit_price = stop; break
                if b.high >= target:
                    hit_target = True; exit_price = target; break
            else:
                mae = min(mae, (entry - b.high) / risk)
                mfe = max(mfe, (entry - b.low) / risk)
                if b.high >= stop:
                    hit_stop = True; exit_price = stop; break
                if b.low <= target:
                    hit_target = True; exit_price = target; break

        if not hit_target and not hit_stop:
            exit_price = bars[min(end_bar - 1, len(bars) - 1)].close

        pnl = ((exit_price - entry) / risk if direction == "Long"
               else (entry - exit_price) / risk)

        result[f"{mult}R"] = {"hit_target": hit_target, "hit_stop": hit_stop,
                              "pnl_r": round(pnl, 3), "is_winner": hit_target}

    result["mae_r"] = round(mae, 3)
    result["mfe_r"] = round(mfe, 3)
    return result


# ─────────────── STATS HELPERS ────────────────────────────────────────────────

def calc_stats(trades, rkey="1.0R"):
    """Calculate comprehensive stats for a list of trades at a given R:R."""
    valid = [t for t in trades if rkey in t and t[rkey] is not None]
    if not valid:
        return {"n": 0, "wr": 0, "pf": 0, "total_r": 0, "ev": 0}

    n = len(valid)
    winners = sum(1 for t in valid if t[rkey]["is_winner"])
    losers = sum(1 for t in valid if t[rkey]["hit_stop"])
    wr = winners / n * 100

    pnls = [t[rkey]["pnl_r"] for t in valid]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0
    total = sum(pnls)
    ev = np.mean(pnls)

    # Max drawdown (in R)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = np.min(dd) if len(dd) > 0 else 0

    return {"n": n, "winners": winners, "losers": losers,
            "wr": round(wr, 1), "pf": round(pf, 2),
            "total_r": round(total, 1), "ev": round(ev, 3),
            "max_dd": round(max_dd, 1)}


def print_stats_row(label, trades, targets=[1.0, 2.0, 3.0]):
    """Print a stats row for a set of trades."""
    if not trades:
        print(f"  {label:<30} {'0 trades':>10}")
        return

    for mult in targets:
        rkey = f"{mult}R"
        s = calc_stats(trades, rkey)
        if s["n"] == 0:
            continue
        marker = "***" if s["pf"] >= 2.0 and s["wr"] >= 60 else "  *" if s["pf"] >= 1.5 else "   "
        if mult == targets[0]:
            print(f"  {label:<30} {s['n']:>4} | {mult:.0f}:1 {s['wr']:>5.1f}% WR  PF {s['pf']:>5.2f}  "
                  f"Total {s['total_r']:>+7.1f}R  EV {s['ev']:>+6.3f}R  MaxDD {s['max_dd']:>+6.1f}R {marker}")
        else:
            print(f"  {'':<30} {'':>4} | {mult:.0f}:1 {s['wr']:>5.1f}% WR  PF {s['pf']:>5.2f}  "
                  f"Total {s['total_r']:>+7.1f}R  EV {s['ev']:>+6.3f}R  MaxDD {s['max_dd']:>+6.1f}R {marker}")


def print_monthly_table(label, trades, rkey="1.0R"):
    """Print month-by-month stats with yearly subtotals and cumulative R."""
    if not trades:
        print(f"  {label}: 0 trades")
        return

    monthly = {}
    for t in trades:
        d = t.get("date", "")
        if d:
            month = d[:7]
            if month not in monthly:
                monthly[month] = []
            monthly[month].append(t)

    print(f"\n  {'=' * 95}")
    print(f"  {label} — MONTH-BY-MONTH @ {rkey}")
    print(f"  {'=' * 95}")
    header = f"  {'Month':<10} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'PF':>6} {'Total R':>8} {'EV/trade':>9} {'MaxDD':>7} {'Cum R':>8}"
    print(header)
    print(f"  {'─' * 93}")

    cum_r = 0.0
    current_year = None
    year_trades = []

    for month in sorted(monthly.keys()):
        year = month[:4]

        if current_year is not None and year != current_year and year_trades:
            ys = calc_stats(year_trades, rkey)
            print(f"  {'─' * 93}")
            print(f"  {current_year + ' TOTAL':<10} {ys['n']:>4} {ys['winners']:>4} {ys['losers']:>4} "
                  f"{ys['wr']:>5.1f}% {ys['pf']:>5.2f} {ys['total_r']:>+7.1f}R {ys['ev']:>+8.3f}R {ys['max_dd']:>+6.1f}R {cum_r:>+7.1f}R")
            print(f"  {'─' * 93}")
            year_trades = []

        current_year = year
        m_trades = monthly[month]
        year_trades.extend(m_trades)
        s = calc_stats(m_trades, rkey)
        if s["n"] == 0:
            continue

        cum_r += s["total_r"]
        star = " ***" if s["pf"] >= 2.0 and s["wr"] >= 60 else "  *" if s["pf"] >= 1.5 else ""

        print(f"  {month:<10} {s['n']:>4} {s['winners']:>4} {s['losers']:>4} "
              f"{s['wr']:>5.1f}% {s['pf']:>5.2f} {s['total_r']:>+7.1f}R {s['ev']:>+8.3f}R {s['max_dd']:>+6.1f}R {cum_r:>+7.1f}R{star}")

    # Last year subtotal
    if year_trades:
        ys = calc_stats(year_trades, rkey)
        print(f"  {'─' * 93}")
        print(f"  {current_year + ' TOTAL':<10} {ys['n']:>4} {ys['winners']:>4} {ys['losers']:>4} "
              f"{ys['wr']:>5.1f}% {ys['pf']:>5.2f} {ys['total_r']:>+7.1f}R {ys['ev']:>+8.3f}R {ys['max_dd']:>+6.1f}R {cum_r:>+7.1f}R")

    # Grand total
    gs = calc_stats(trades, rkey)
    print(f"  {'=' * 93}")
    print(f"  {'GRAND TOTAL':<10} {gs['n']:>4} {gs['winners']:>4} {gs['losers']:>4} "
          f"{gs['wr']:>5.1f}% {gs['pf']:>5.2f} {gs['total_r']:>+7.1f}R {gs['ev']:>+8.3f}R {gs['max_dd']:>+6.1f}R {cum_r:>+7.1f}R")
    print()


# ─────────────── PDF GENERATION ───────────────────────────────────────────────

def _render_monthly_pages(pdf, label, trades, color, rkey="1.0R"):
    """Render multi-page monthly breakdown for a setup type."""
    monthly = {}
    for t in trades:
        d = t.get("date", "")
        if d:
            month = d[:7]
            if month not in monthly:
                monthly[month] = []
            monthly[month].append(t)

    months_sorted = sorted(monthly.keys())
    rows_per_page = 30
    total_pages = max(1, (len(months_sorted) + rows_per_page - 1) // rows_per_page)

    for page_start in range(0, len(months_sorted), rows_per_page):
        page_months = months_sorted[page_start:page_start + rows_per_page]
        page_num = page_start // rows_per_page + 1

        fig = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

        ax.text(0.5, 0.96, f"MONTHLY BREAKDOWN — {label} @ {rkey} (Page {page_num}/{total_pages})",
                transform=ax.transAxes, fontsize=12, fontweight="bold", color=color, ha="center")

        y = 0.92
        header = f"  {'Month':<10} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'PF':>6} {'Total R':>8} {'EV':>8}"
        ax.text(0.05, y, header, transform=ax.transAxes, fontsize=7,
                color="#999999", ha="left", family="monospace")
        y -= 0.025

        current_year = None
        for month in page_months:
            year = month[:4]
            # Year separator
            if current_year is not None and year != current_year:
                ax.text(0.05, y, "  " + "─" * 65, transform=ax.transAxes, fontsize=6,
                        color="#555555", ha="left", family="monospace")
                y -= 0.02

            current_year = year
            m_trades = monthly[month]
            s = calc_stats(m_trades, rkey)
            if s["n"] == 0:
                continue

            row_color = "#00C853" if s["total_r"] > 0 else "#FF1744"
            line = (f"  {month:<10} {s['n']:>4} {s['winners']:>4} {s['losers']:>4} "
                    f"{s['wr']:>5.1f}% {s['pf']:>5.2f} {s['total_r']:>+7.1f}R {s['ev']:>+7.3f}R")
            ax.text(0.05, y, line, transform=ax.transAxes, fontsize=6.5,
                    color=row_color, ha="left", family="monospace")
            y -= 0.022

        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)


def generate_pdf(all_h1, all_l1, all_bbl1, daily_data, output_path):
    """Generate comprehensive PDF report."""
    print(f"\nGenerating PDF...")

    with PdfPages(output_path) as pdf:
        # ── Page 1: Summary ──
        fig = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

        ax.text(0.5, 0.93, "H1 / L1 DAILY CHART BACKTEST — 7 YEARS", transform=ax.transAxes,
                fontsize=24, fontweight="bold", color="white", ha="center")
        ax.text(0.5, 0.87, "Month-by-Month — Al Brooks First Pullback", transform=ax.transAxes,
                fontsize=16, color="#FFA726", ha="center")

        n_tickers = len(daily_data)
        bar_counts = [len(df) for df in daily_data.values()]
        sample_df = list(daily_data.values())[0]
        date_range = f"{sample_df.index[0].date()} to {sample_df.index[-1].date()}"

        ax.text(0.5, 0.80,
                f"{n_tickers} tickers  |  {date_range}  |  ~{max(bar_counts)} trading days",
                transform=ax.transAxes, fontsize=11, color="#aaaaaa", ha="center")

        # Summary stats
        y = 0.70
        for label, trades, color in [
            ("H1 Long", all_h1, "#00C853"),
            ("L1 Short", all_l1, "#FF1744"),
            ("Buy Below L1 → Long", all_bbl1, "#42A5F5"),
        ]:
            s1 = calc_stats(trades, "1.0R")
            s3 = calc_stats(trades, "3.0R")
            if s1["n"] == 0:
                continue

            ax.text(0.5, y, f"{label}: {s1['n']} trades", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", color=color, ha="center")
            y -= 0.035
            ax.text(0.5, y,
                    f"1:1 → {s1['wr']}% WR, PF {s1['pf']}, EV {s1['ev']:+.3f}R  |  "
                    f"3:1 → {s3['wr']}% WR, PF {s3['pf']}, Total {s3['total_r']:+.1f}R",
                    transform=ax.transAxes, fontsize=9, color="#cccccc", ha="center")
            y -= 0.06

        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # ── Page 2: Equity curves ──
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), facecolor="#1a1a2e")

        for idx, (label, trades, color) in enumerate([
            ("H1 Long", all_h1, "#00C853"),
            ("L1 Short", all_l1, "#FF1744"),
            ("Buy Below L1 → Long", all_bbl1, "#42A5F5"),
        ]):
            ax = axes[idx]
            ax.set_facecolor("#1a1a2e")

            if not trades:
                ax.text(0.5, 0.5, f"No {label} trades", transform=ax.transAxes,
                        fontsize=14, color="#666666", ha="center")
                continue

            sorted_t = sorted(trades, key=lambda t: t.get("date", ""))

            for mult, mc in [(1.0, "#FFA726"), (2.0, "#00C853"), (3.0, "#42A5F5")]:
                rkey = f"{mult}R"
                cum = []
                running = 0
                for t in sorted_t:
                    if rkey in t:
                        running += t[rkey]["pnl_r"]
                        cum.append(running)
                if cum:
                    ax.plot(cum, color=mc, linewidth=1.5, label=f"{mult:.0f}:1")

            s1 = calc_stats(trades, "1.0R")
            ax.set_title(f"{label} — {s1['n']} trades, {s1['wr']}% WR, PF {s1['pf']}",
                         fontsize=12, color=color, fontweight="bold")
            ax.axhline(y=0, color="#555555", linewidth=0.5)
            ax.set_xlabel("Trade #", fontsize=8, color="#aaaaaa")
            ax.set_ylabel("Cumulative R", fontsize=8, color="#aaaaaa")
            ax.legend(fontsize=8, facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
            ax.tick_params(colors="#999999")
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # ── Monthly breakdown pages (multi-page per setup) ──
        _render_monthly_pages(pdf, "H1 Long", all_h1, "#00C853")
        _render_monthly_pages(pdf, "L1 Short", all_l1, "#FF1744")
        _render_monthly_pages(pdf, "Buy Below L1", all_bbl1, "#42A5F5")

        # ── MAE/MFE histograms ──
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), facecolor="#1a1a2e")

        for idx, (label, trades, color) in enumerate([
            ("H1 Long", all_h1, "#00C853"),
            ("L1 Short", all_l1, "#FF1744"),
        ]):
            if not trades:
                continue

            maes = [t["mae_r"] for t in trades]
            mfes = [t["mfe_r"] for t in trades]

            ax_mae = axes[idx][0]
            ax_mae.set_facecolor("#1a1a2e")
            ax_mae.hist(maes, bins=30, color=color, alpha=0.7, edgecolor="none")
            ax_mae.axvline(x=np.mean(maes), color="white", linewidth=2, linestyle="--")
            ax_mae.set_title(f"{label} MAE (avg {np.mean(maes):+.2f}R)", fontsize=10, color="white")
            ax_mae.set_xlabel("MAE (R)", fontsize=8, color="#aaaaaa")
            ax_mae.tick_params(colors="#999999")

            ax_mfe = axes[idx][1]
            ax_mfe.set_facecolor("#1a1a2e")
            ax_mfe.hist(mfes, bins=30, color=color, alpha=0.7, edgecolor="none")
            ax_mfe.axvline(x=np.mean(mfes), color="white", linewidth=2, linestyle="--")
            pct_2r = sum(1 for m in mfes if m > 2) / len(mfes) * 100
            pct_3r = sum(1 for m in mfes if m > 3) / len(mfes) * 100
            ax_mfe.set_title(f"{label} MFE (avg {np.mean(mfes):+.2f}R, {pct_2r:.0f}%>2R, {pct_3r:.0f}%>3R)",
                             fontsize=10, color="white")
            ax_mfe.set_xlabel("MFE (R)", fontsize=8, color="#aaaaaa")
            ax_mfe.tick_params(colors="#999999")

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

    print(f"PDF saved: {output_path}")


# ─────────────── MAIN ─────────────────────────────────────────────────────────

def run_extended_backtest():
    """Main entry point."""
    print("=" * 95)
    print("  7-YEAR H1/L1 DAILY CHART BACKTEST — MONTH BY MONTH")
    print("  Al Brooks First Pullback — 91 Liquid US Stocks")
    print("  Jan 2019 → Mar 2026 (~1,750 trading days)")
    print("=" * 95)

    api_key = _resolve_api_key()
    if not api_key:
        print("ERROR: No Databento API key found")
        return

    # Fetch data — 7 years
    start = "2019-01-02"
    end = "2026-03-09"

    daily_data = fetch_daily_bars_bulk(LIQUID_UNIVERSE, start, end, api_key)
    if not daily_data:
        print("No data fetched.")
        return

    # Show date range
    sample = list(daily_data.values())[0]
    print(f"\n  Date range: {sample.index[0].date()} → {sample.index[-1].date()}")

    # ─── RUN DETECTION + SIMULATION ───
    all_h1 = []
    all_l1 = []
    all_bbl1 = []

    for ticker, daily_df in daily_data.items():
        bars = bars_from_df(daily_df)
        ema = compute_ema(bars)
        daily_dates = [d.date() if hasattr(d, "date") else d for d in daily_df.index]

        # H1/L1
        setups = detect_h1_l1_setups(bars, ema)
        for setup in setups:
            result = simulate_trade(bars, setup)
            if result is None:
                continue
            result["ticker"] = ticker
            result["sector"] = SECTOR_MAP.get(ticker, "Unknown")
            result["date"] = str(daily_dates[setup["bar_idx"]]) if setup["bar_idx"] < len(daily_dates) else ""

            if setup["setup"] == "H1":
                all_h1.append(result)
            else:
                all_l1.append(result)

        # Buy Below L1
        bbl1_setups = detect_buy_below_l1(bars, ema)
        for setup in bbl1_setups:
            result = simulate_trade(bars, setup)
            if result is None:
                continue
            result["ticker"] = ticker
            result["sector"] = SECTOR_MAP.get(ticker, "Unknown")
            result["date"] = str(daily_dates[setup["bar_idx"]]) if setup["bar_idx"] < len(daily_dates) else ""
            all_bbl1.append(result)

    # Sort by date
    all_h1.sort(key=lambda t: t.get("date", ""))
    all_l1.sort(key=lambda t: t.get("date", ""))
    all_bbl1.sort(key=lambda t: t.get("date", ""))

    # ─── RESULTS ───
    print(f"\n{'=' * 95}")
    print(f"  RESULTS — {len(daily_data)} tickers, ~{max(len(df) for df in daily_data.values())} trading days")
    print(f"{'=' * 95}")

    # ── Multi-target summary ──
    print(f"\n  ── MULTI-TARGET SUMMARY ──")
    print_stats_row("H1 Long (7yr)", all_h1)
    print()
    print_stats_row("L1 Short (7yr)", all_l1)
    print()
    print_stats_row("Buy Below L1 (7yr)", all_bbl1)

    # ── Month-by-month tables ──
    print_monthly_table("H1 LONG", all_h1, "1.0R")
    print_monthly_table("L1 SHORT", all_l1, "1.0R")
    print_monthly_table("BUY BELOW L1 (Contrarian Long)", all_bbl1, "1.0R")

    # By sector
    print(f"\n  ── H1 BY SECTOR ──")
    sectors = {}
    for t in all_h1:
        sec = t.get("sector", "Unknown")
        if sec not in sectors:
            sectors[sec] = []
        sectors[sec].append(t)

    sector_stats = []
    for sec, trades in sectors.items():
        s = calc_stats(trades, "1.0R")
        if s["n"] >= 5:
            sector_stats.append((sec, s))

    sector_stats.sort(key=lambda x: x[1]["total_r"], reverse=True)
    print(f"  {'Sector':<16} {'N':>4} {'WR%':>6} {'PF':>6} {'Total R':>8} {'EV':>7}")
    print("  " + "-" * 55)
    for sec, s in sector_stats:
        print(f"  {sec:<16} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['total_r']:>+7.1f}R {s['ev']:>+6.3f}R")

    # Top tickers
    print(f"\n  ── TOP H1 TICKERS (by total R @ 1:1) ──")
    by_ticker = {}
    for t in all_h1:
        tk = t["ticker"]
        if tk not in by_ticker:
            by_ticker[tk] = []
        by_ticker[tk].append(t)

    ticker_stats = []
    for tk, trades in by_ticker.items():
        s = calc_stats(trades, "1.0R")
        if s["n"] >= 3:
            ticker_stats.append((tk, s))

    ticker_stats.sort(key=lambda x: x[1]["total_r"], reverse=True)
    print(f"  {'Ticker':<7} {'N':>4} {'WR%':>6} {'PF':>6} {'Total R':>8} {'EV':>7}")
    print("  " + "-" * 45)
    for tk, s in ticker_stats[:15]:
        print(f"  {tk:<7} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['total_r']:>+7.1f}R {s['ev']:>+6.3f}R")

    print(f"\n  ── WORST H1 TICKERS ──")
    for tk, s in ticker_stats[-5:]:
        print(f"  {tk:<7} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['total_r']:>+7.1f}R {s['ev']:>+6.3f}R")

    # MAE/MFE
    print(f"\n  ── MAE/MFE ANALYSIS ──")
    for label, trades in [("H1 Long", all_h1), ("L1 Short", all_l1), ("Buy Below L1", all_bbl1)]:
        if not trades:
            continue
        maes = [t["mae_r"] for t in trades]
        mfes = [t["mfe_r"] for t in trades]
        pct_2r = sum(1 for m in mfes if m > 2) / len(mfes) * 100
        pct_3r = sum(1 for m in mfes if m > 3) / len(mfes) * 100
        print(f"  {label:<18}: MAE {np.mean(maes):+.2f}R  MFE {np.mean(mfes):+.2f}R  "
              f"MFE>2R: {pct_2r:.0f}%  MFE>3R: {pct_3r:.0f}%")

    # Longs vs Shorts comparison
    print(f"\n  ── LONGS vs SHORTS COMPARISON @ 1:1 ──")
    h1_s = calc_stats(all_h1, "1.0R")
    l1_s = calc_stats(all_l1, "1.0R")
    bbl1_s = calc_stats(all_bbl1, "1.0R")
    combined_long = all_h1 + all_bbl1
    cl_s = calc_stats(combined_long, "1.0R")

    print(f"  H1 Long:         {h1_s['n']:>4} trades, {h1_s['wr']}% WR, PF {h1_s['pf']}, Total {h1_s['total_r']:+.1f}R")
    print(f"  L1 Short:        {l1_s['n']:>4} trades, {l1_s['wr']}% WR, PF {l1_s['pf']}, Total {l1_s['total_r']:+.1f}R")
    print(f"  Buy Below L1:    {bbl1_s['n']:>4} trades, {bbl1_s['wr']}% WR, PF {bbl1_s['pf']}, Total {bbl1_s['total_r']:+.1f}R")
    print(f"  All Longs:       {cl_s['n']:>4} trades, {cl_s['wr']}% WR, PF {cl_s['pf']}, Total {cl_s['total_r']:+.1f}R")

    # Generate PDF
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "BPA_H1_7Year_Monthly_Backtest.pdf")
    generate_pdf(all_h1, all_l1, all_bbl1, daily_data, pdf_path)

    print(f"\n{'=' * 95}")
    print(f"  Done — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 95}\n")

    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_extended_backtest()
