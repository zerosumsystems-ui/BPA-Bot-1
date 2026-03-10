#!/usr/bin/env python3
"""
daily_rs_rankings.py — Daily Relative Strength Rankings (Full US Market Scan)
Based on Al Brooks Price Action Principles & 17-Step Trend Cycle Analysis

Scans the full S&P 1500 (S&P 500 + MidCap 400 + SmallCap 600) + sector ETFs
via Databento bulk fetch — ~1,500+ tickers.
Ranks tickers by BPA relative strength using:
  - Always-In Direction (state machine — who is winning NOW)
  - Micro Channel detection (strongest trend signal)
  - Climax/Exhaustion warnings (prevent chasing blow-offs)
  - Day type & market cycle classification
  - EMA relationship (position + consecutive streak)
  - Momentum quality (centered body ratio, direction strength)
  - Swing structure (higher highs/lows)
  - Bar overlap (trend vs range fingerprint)
  - Trend ratio & volume confirmation

Outputs: ranked table, sector heatmap, charts PDF (top 20 + bottom 20).
"""

import os
import sys
import io
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.backends.backend_pdf import PdfPages

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_source import get_data_source, DatabentoSource
from algo_engine import (
    Bar, bars_from_df, compute_ema, find_swing_lows, find_swing_highs,
    classify_day_type, classify_market_cycle, analyze_bars,
)

logger = logging.getLogger(__name__)

# ─────────────────────── TICKER UNIVERSE ──────────────────────────────────────

# Sector ETFs for rotation analysis (always included)
SECTOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

# GICS sector mapping for any ticker
GICS_SECTORS = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLF": "Financials",
    "XLV": "Healthcare", "XLI": "Industrials", "XLB": "Materials",
    "XLRE": "Real Estate", "XLU": "Utilities",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",
}

# Sector mapping for individual stocks (populated dynamically from S&P 500 scrape)
STOCK_SECTOR_MAP = {}


def _scrape_sp_index(url: str, label: str) -> tuple[list[str], dict[str, str]]:
    """Scrape an S&P index page from Wikipedia for tickers + GICS sectors."""
    import requests as _req

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        resp = _req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        sector_map = {}
        for _, row in df.iterrows():
            sym = str(row["Symbol"]).replace(".", "-")
            sector = str(row.get("GICS Sector", "Unknown"))
            sector_map[sym] = sector
        print(f"  {label}: {len(tickers)} tickers")
        return tickers, sector_map
    except Exception as e:
        logger.warning(f"Failed to scrape {label}: {e}")
        return [], {}


def get_sp1500_tickers_with_sectors() -> tuple[list[str], dict[str, str]]:
    """
    Scrape the full S&P 1500 Composite from Wikipedia:
    S&P 500 + S&P MidCap 400 + S&P SmallCap 600 = ~1,500 tickers.
    """
    urls = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P MidCap 400"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P SmallCap 600"),
    ]
    all_tickers = []
    all_sectors = {}
    for url, label in urls:
        tickers, sectors = _scrape_sp_index(url, label)
        all_tickers.extend(tickers)
        all_sectors.update(sectors)

    # Deduplicate (preserving order)
    seen = set()
    deduped = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped, all_sectors


# ─────────────── AL BROOKS SCORING CONSTANTS ─────────────────────────────────

CYCLE_POSITION = {
    "Trading Range": 1,
    "Breakout (Spike)": 4,
    "Micro Channel": 6,
    "Tight Channel (Small PB Trend)": 8,
    "Broad Bull Channel": 10,
    "Broad Bear Channel": 10,
    "N/A": 0,
}

DAY_TYPE_SCORE = {
    "Bull Trend From The Open": 10,
    "Small Pullback Bull Trend": 9,
    "Spike and Channel Bull Trend": 8,
    "Broad Bull Channel": 6,
    "Trending Trading Range Day (Bull)": 4,
    "Trading Range Day": 0,
    "Trending Trading Range Day (Bear)": -4,
    "Broad Bear Channel": -6,
    "Spike and Channel Bear Trend": -8,
    "Small Pullback Bear Trend": -9,
    "Bear Trend From The Open": -10,
}


# ─────────────── NEW BPA SCORING COMPONENTS ──────────────────────────────────

def compute_always_in_direction(bars: list[Bar]) -> int:
    """
    Determine Always-In Direction — Brooks' #1 concept.
    Returns: +1 (AI Long), -1 (AI Short), 0 (Ambiguous)

    Logic: Track the most recent strong signal bar. When price closes beyond
    the opposite extreme of the last signal bar, AI direction flips.
    """
    if len(bars) < 5:
        return 0

    avg_range = np.mean([b.range for b in bars])
    ai_dir = 0  # Start ambiguous

    for i in range(1, len(bars)):
        b = bars[i]
        prev = bars[i - 1]

        # Strong bull signal: large body, closing near high
        if (b.is_bull and b.body > b.range * 0.5 and b.body > avg_range * 0.5
                and b.closes_near_high):
            ai_dir = 1

        # Strong bear signal: large body, closing near low
        elif (b.is_bear and b.body > b.range * 0.5 and b.body > avg_range * 0.5
              and b.closes_near_low):
            ai_dir = -1

        # Flip on strong reversal: close beyond prior strong bar's extreme
        if ai_dir == 1 and b.is_bear and b.close < prev.low:
            ai_dir = -1
        elif ai_dir == -1 and b.is_bull and b.close > prev.high:
            ai_dir = 1

    return ai_dir


def compute_micro_channel_score(bars: list[Bar]) -> float:
    """
    Detect micro channel — strongest possible trend signal.
    Bull MC: consecutive bars where each low >= prior low.
    Bear MC: consecutive bars where each high <= prior high.
    Returns score from -15 to +15.
    """
    if len(bars) < 10:
        return 0.0

    recent = bars[-20:] if len(bars) >= 20 else bars

    # Bull micro channel: longest run of rising lows
    bull_run = 0
    max_bull = 0
    for i in range(1, len(recent)):
        if recent[i].low >= recent[i - 1].low - 0.001:  # tiny tolerance
            bull_run += 1
            max_bull = max(max_bull, bull_run)
        else:
            bull_run = 0

    # Bear micro channel: longest run of falling highs
    bear_run = 0
    max_bear = 0
    for i in range(1, len(recent)):
        if recent[i].high <= recent[i - 1].high + 0.001:
            bear_run += 1
            max_bear = max(max_bear, bear_run)
        else:
            bear_run = 0

    if max_bull >= 8:
        return min(15.0, max_bull * 2.0)
    elif max_bear >= 8:
        return max(-15.0, -max_bear * 2.0)
    elif max_bull >= 5:
        return max_bull * 1.0
    elif max_bear >= 5:
        return -max_bear * 1.0

    return 0.0


def compute_climax_warning(bars: list[Bar]) -> float:
    """
    Detect climax/exhaustion — prevent chasing blow-off moves.
    3+ consecutive big bars closing near extreme = climax warning.
    Parabolic acceleration (each bar bigger) = additional penalty.
    Returns penalty from -15 to 0 (always reduces absolute RS).
    """
    if len(bars) < 10:
        return 0.0

    recent = bars[-10:]
    avg_range = np.mean([b.range for b in bars])

    # Count climax bars (big, closing near extreme in one direction)
    bull_climax = 0
    bear_climax = 0
    for b in recent:
        if b.range > avg_range * 1.3 and b.is_bull and b.closes_near_high:
            bull_climax += 1
        elif b.range > avg_range * 1.3 and b.is_bear and b.closes_near_low:
            bear_climax += 1

    # Parabolic acceleration: last 3 bars each bigger than prior
    parabolic = 0
    if len(recent) >= 3:
        r3 = recent[-3:]
        if r3[1].range > r3[0].range and r3[2].range > r3[1].range:
            parabolic = -3

    penalty = 0.0
    if bull_climax >= 3:
        penalty = -(bull_climax * 3 + abs(parabolic))
    elif bear_climax >= 3:
        penalty = -(bear_climax * 3 + abs(parabolic))

    return max(-15.0, penalty)


def compute_ema_streak(bars: list[Bar], ema: list[float]) -> float:
    """
    Count consecutive closes above/below EMA from the current bar backward.
    Long streaks = much stronger trend signal than scattered crossings.
    Returns -12 to +12.
    """
    if not bars or not ema:
        return 0.0

    n = len(bars)
    streak = 0
    above = bars[-1].close > ema[-1]

    for i in range(n - 1, -1, -1):
        if above and bars[i].close > ema[i]:
            streak += 1
        elif not above and bars[i].close < ema[i]:
            streak += 1
        else:
            break

    if streak >= 20:
        score = 12.0
    elif streak >= 15:
        score = 9.0
    elif streak >= 10:
        score = 6.0
    elif streak >= 5:
        score = 3.0
    else:
        score = 0.0

    return score if above else -score


def compute_bar_overlap(bars: list[Bar]) -> float:
    """
    Measure pairwise bar overlap — structural fingerprint of trend vs range.
    Low overlap = real move, high overlap = churning noise.
    Returns -10 to +10 (signed by net direction).
    """
    if len(bars) < 15:
        return 0.0

    recent = bars[-15:]
    avg_range = np.mean([b.range for b in recent])
    if avg_range == 0:
        return 0.0

    overlaps = []
    for i in range(1, len(recent)):
        overlap = max(0, min(recent[i].high, recent[i - 1].high) - max(recent[i].low, recent[i - 1].low))
        overlaps.append(overlap / avg_range)

    avg_overlap = np.mean(overlaps) if overlaps else 0.5

    # Low overlap (< 0.3) = strong trend, high overlap (> 0.7) = range
    raw = (0.5 - avg_overlap) * 20  # -10 to +10

    # Sign by net direction
    net = recent[-1].close - recent[0].open
    if net < 0:
        raw = -abs(raw)
    elif net > 0:
        raw = abs(raw)

    return max(-10.0, min(10.0, raw))


# ─────────────── MAIN SCORING FUNCTION ───────────────────────────────────────

def compute_bpa_rs_score(df: pd.DataFrame) -> dict:
    """
    Compute BPA relative strength score for a single ticker's 5-min data.
    Fixed bugs from v1: centered momentum, proper micro channel, etc.
    """
    if df is None or df.empty or len(df) < 10:
        return {
            "rs_score": 0, "day_type": "N/A", "market_cycle": "N/A",
            "cycle_position": 0, "ema_relationship": "N/A",
            "momentum_score": 0, "swing_score": 0, "trend_score": 0,
            "vol_score": 0, "num_setups": 0, "best_setup": "None",
            "best_conf": 0, "trend_direction": "N/A", "last_close": 0,
            "ema_20": 0, "always_in": 0, "micro_channel": 0,
            "climax_warning": 0, "ema_streak": 0, "overlap_score": 0,
            "error": "Insufficient data",
        }

    bars = bars_from_df(df)
    ema = compute_ema(bars)
    n = len(bars)

    # Inject EMA
    for i, b in enumerate(bars):
        b.ema_20 = ema[i]

    # 1. Day Type & Market Cycle
    day_type = classify_day_type(bars, ema)
    market_cycle = classify_market_cycle(bars, ema)
    day_score = DAY_TYPE_SCORE.get(day_type, 0)
    cycle_pos = CYCLE_POSITION.get(market_cycle, 1)

    # 2. EMA Relationship (-10 to +10)
    above_ema = sum(1 for i, b in enumerate(bars) if b.close > ema[i])
    ema_ratio = above_ema / n
    ema_score = (ema_ratio - 0.5) * 20

    # 3. EMA distance (-10 to +10)
    last_bar = bars[-1]
    last_ema = ema[-1]
    if last_ema > 0:
        ema_dist_pct = ((last_bar.close - last_ema) / last_ema) * 100
    else:
        ema_dist_pct = 0
    ema_dist_score = max(-10, min(10, ema_dist_pct * 3))  # Reduced multiplier (was 5)

    # 4. Momentum — FIXED: center body ratio around 0
    lookback = min(10, n)
    body_ratios = [b.body / b.range if b.range > 0 else 0 for b in bars[-lookback:]]
    avg_body_ratio = np.mean(body_ratios) if body_ratios else 0.5
    # Center: 0.5 is average, so (ratio - 0.5) * 20 gives -10 to +10
    body_quality = (avg_body_ratio - 0.5) * 20

    bull_count = sum(1 for b in bars[-lookback:] if b.is_bull)
    direction_strength = (bull_count - (lookback - bull_count)) / lookback  # -1 to +1

    momentum_score = body_quality * 0.4 + direction_strength * 8  # -12 to +12

    # 5. Swing Structure (-15 to +15)
    swing_lows = find_swing_lows(bars)
    swing_highs = find_swing_highs(bars)

    higher_lows = 0
    if len(swing_lows) >= 2:
        for i in range(1, len(swing_lows)):
            if bars[swing_lows[i]].low > bars[swing_lows[i - 1]].low:
                higher_lows += 1
            else:
                higher_lows -= 1

    higher_highs = 0
    if len(swing_highs) >= 2:
        for i in range(1, len(swing_highs)):
            if bars[swing_highs[i]].high > bars[swing_highs[i - 1]].high:
                higher_highs += 1
            else:
                higher_highs -= 1

    swing_score = max(-15, min(15, (higher_highs + higher_lows) * 3))

    # 6. Trend Ratio (-10 to +10)
    total_range = max(b.high for b in bars) - min(b.low for b in bars)
    net_move = bars[-1].close - bars[0].open
    trend_ratio = net_move / total_range if total_range > 0 else 0
    trend_score = trend_ratio * 10

    # 7. Cycle Position
    if 3 <= cycle_pos <= 8:
        cycle_score = 8
    elif 1 <= cycle_pos <= 2:
        cycle_score = 0  # Neutral, not biased
    elif 9 <= cycle_pos <= 11:
        cycle_score = 3
    else:
        cycle_score = -5

    # 8. Volume Trend
    if n >= 20:
        recent_vol = np.mean([b.volume for b in bars[-10:]])
        earlier_vol = np.mean([b.volume for b in bars[-20:-10]])
        if earlier_vol > 0:
            vol_ratio = recent_vol / earlier_vol
            vol_score = max(-5, min(5, (vol_ratio - 1) * 10))
        else:
            vol_score = 0
    else:
        vol_score = 0

    # 9. NEW: Always-In Direction (-20 to +20)
    ai_dir = compute_always_in_direction(bars)
    ai_score = ai_dir * 15  # Strong weight: -15 to +15

    # 10. NEW: Micro Channel (-15 to +15)
    mc_score = compute_micro_channel_score(bars)

    # 11. NEW: Climax Warning (-15 to 0)
    climax_penalty = compute_climax_warning(bars)

    # 12. NEW: EMA Streak (-12 to +12)
    streak_score = compute_ema_streak(bars, ema)

    # 13. NEW: Bar Overlap (-10 to +10)
    overlap_score = compute_bar_overlap(bars)

    # ─── COMPOSITE SCORE ───
    rs_score = (
        day_score * 2.5 +        # Day type: -25 to +25
        ai_score * 1.5 +         # Always-in: -22.5 to +22.5
        ema_score * 1.0 +        # EMA position: -10 to +10
        ema_dist_score * 0.8 +   # EMA distance: -8 to +8
        momentum_score * 0.8 +   # Momentum: -9.6 to +9.6
        swing_score * 0.8 +      # Swing: -12 to +12
        trend_score * 1.2 +      # Trend ratio: -12 to +12
        cycle_score * 0.5 +      # Cycle: -2.5 to +4
        vol_score * 0.3 +        # Volume: -1.5 to +1.5
        mc_score * 1.0 +         # Micro channel: -15 to +15
        climax_penalty * 1.0 +   # Climax: -15 to 0
        streak_score * 0.8 +     # EMA streak: -9.6 to +9.6
        overlap_score * 0.7      # Overlap: -7 to +7
    )

    rs_score = max(-100, min(100, rs_score))

    # Trend direction
    if rs_score > 15:
        trend_dir = "Bull"
    elif rs_score < -15:
        trend_dir = "Bear"
    else:
        trend_dir = "Neutral"

    ema_label = "Above" if last_bar.close > last_ema else "Below"

    # Setup detection (lightweight — skip for speed if needed)
    try:
        analysis = analyze_bars(df)
        setups = analysis.get("setups", [])
        num_setups = len(setups)
        best_setup = setups[0]["setup_name"] if setups else "None"
        best_conf = setups[0]["confidence"] if setups else 0
    except Exception:
        num_setups = 0
        best_setup = "N/A"
        best_conf = 0

    return {
        "rs_score": round(rs_score, 1),
        "day_type": day_type,
        "market_cycle": market_cycle,
        "cycle_position": cycle_pos,
        "ema_relationship": ema_label,
        "momentum_score": round(momentum_score, 1),
        "swing_score": round(swing_score, 1),
        "trend_score": round(trend_score, 1),
        "vol_score": round(vol_score, 1),
        "num_setups": num_setups,
        "best_setup": best_setup,
        "best_conf": round(best_conf, 2),
        "trend_direction": trend_dir,
        "last_close": round(last_bar.close, 2),
        "ema_20": round(last_ema, 2),
        "always_in": ai_dir,
        "micro_channel": round(mc_score, 1),
        "climax_warning": round(climax_penalty, 1),
        "ema_streak": round(streak_score, 1),
        "overlap_score": round(overlap_score, 1),
    }


def format_cycle_phase(cycle: str, pos: int) -> str:
    phase_map = {
        "Trading Range": "Accumulation / TR",
        "Breakout (Spike)": "Breakout Spike",
        "Micro Channel": "Micro Channel",
        "Tight Channel (Small PB Trend)": "Strong Trend",
        "Broad Bull Channel": "Maturing Channel",
        "Broad Bear Channel": "Maturing Channel",
    }
    return phase_map.get(cycle, cycle)


# ─────────────── CHART GENERATION ────────────────────────────────────────────

def generate_ticker_chart(df, ticker, score_data, ax=None, swing_highs=None, swing_lows=None):
    """Generate candlestick chart with EMA-20, swing markers, and BPA annotations."""
    if df is None or df.empty:
        return

    bars = bars_from_df(df)
    ema_vals = compute_ema(bars)
    ema_series = pd.Series(ema_vals, index=df.index[:len(ema_vals)])

    rs = score_data.get("rs_score", 0)
    if rs > 40:
        rs_color = "#00C853"
    elif rs > 15:
        rs_color = "#69F0AE"
    elif rs > -15:
        rs_color = "#FFA726"
    elif rs > -40:
        rs_color = "#FF8A65"
    else:
        rs_color = "#FF1744"

    mc = mpf.make_marketcolors(
        up="#26A69A", down="#EF5350",
        edge={"up": "#26A69A", "down": "#EF5350"},
        wick={"up": "#26A69A", "down": "#EF5350"},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", gridcolor="#333333",
        facecolor="#1a1a2e", figcolor="#1a1a2e",
        rc={"axes.labelcolor": "#cccccc", "xtick.color": "#999999", "ytick.color": "#999999"},
    )

    addplots = [mpf.make_addplot(ema_series, ax=ax, color="#FFA726", width=1.5, linestyle="--")]

    # Swing markers
    if swing_highs is None:
        swing_highs = find_swing_highs(bars)
    if swing_lows is None:
        swing_lows = find_swing_lows(bars)

    sh_data = pd.Series([np.nan] * len(df), index=df.index)
    for idx in swing_highs:
        if idx < len(df):
            sh_data.iloc[idx] = bars[idx].high * 1.001
    sl_data = pd.Series([np.nan] * len(df), index=df.index)
    for idx in swing_lows:
        if idx < len(df):
            sl_data.iloc[idx] = bars[idx].low * 0.999

    if sh_data.notna().any():
        addplots.append(mpf.make_addplot(sh_data, ax=ax, type="scatter", marker="v", markersize=20, color="#FF5252"))
    if sl_data.notna().any():
        addplots.append(mpf.make_addplot(sl_data, ax=ax, type="scatter", marker="^", markersize=20, color="#69F0AE"))

    mpf.plot(df, type="candle", style=style, ax=ax, addplot=addplots,
             volume=False, datetime_format="%H:%M", xrotation=0)

    # Annotations
    trend_arrow = "+" if rs > 15 else "-" if rs < -15 else "="
    ai = score_data.get("always_in", 0)
    ai_str = "AI-Long" if ai > 0 else "AI-Short" if ai < 0 else "AI-Flat"
    day_type = score_data.get("day_type", "N/A")
    cycle = format_cycle_phase(score_data.get("market_cycle", "N/A"), score_data.get("cycle_position", 0))
    best_setup = score_data.get("best_setup", "None")
    if best_setup == "None":
        best_setup = "-"
    elif len(best_setup) > 28:
        best_setup = best_setup[:28] + ".."

    ax.set_title(f"{ticker}  RS: {rs:+.1f}  [{ai_str}]",
                 fontsize=13, fontweight="bold", color=rs_color, pad=10, loc="left")

    ax.text(0.5, 1.02,
            f"{day_type}  |  {cycle}  |  EMA: {score_data.get('ema_relationship', '?')}  |  {best_setup}",
            transform=ax.transAxes, fontsize=8, color="#aaaaaa", ha="center", va="bottom")

    ax.text(0.98, 0.92, f"RS {rs:+.1f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold", color="white", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=rs_color, alpha=0.85, edgecolor="none"))

    last_close = score_data.get("last_close", 0)
    ema_20 = score_data.get("ema_20", 0)
    ax.text(0.98, 0.05, f"${last_close}  |  EMA ${ema_20}",
            transform=ax.transAxes, fontsize=8, color="#cccccc", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333", alpha=0.7, edgecolor="none"))

    # Climax warning
    cw = score_data.get("climax_warning", 0)
    if cw < -5:
        ax.text(0.02, 0.92, "CLIMAX WARNING",
                transform=ax.transAxes, fontsize=9, fontweight="bold", color="#FF1744",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", alpha=0.9, edgecolor="#FF1744"))

    mc = score_data.get("micro_channel", 0)
    if abs(mc) >= 8:
        mc_label = "MICRO CH BULL" if mc > 0 else "MICRO CH BEAR"
        ax.text(0.02, 0.80, mc_label,
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                color="#00C853" if mc > 0 else "#FF1744", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", alpha=0.9, edgecolor="none"))

    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)


def generate_rankings_table_page(pdf, ranked_results, sectors_data):
    """Generate a summary table page with full rankings and sector heatmap."""
    fig = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    n = len(ranked_results)
    ax.text(0.5, 0.98, f"FULL RANKINGS — {n} Tickers Scanned",
            transform=ax.transAxes, fontsize=16, fontweight="bold", color="white",
            ha="center", va="top")

    # Top 25 table
    ax.text(0.25, 0.93, "TOP 25 (Strongest)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#00C853", ha="center", va="top")

    header = f"{'#':<3} {'Ticker':<7} {'RS':>6}  {'AI':>8}  {'Day Type':<22}"
    ax.text(0.02, 0.89, header, transform=ax.transAxes, fontsize=7,
            color="#999999", ha="left", va="top", family="monospace")

    show_n = 25
    top_n = ranked_results[:show_n]
    for i, r in enumerate(top_n):
        ai_str = "Long" if r.get("always_in", 0) > 0 else "Short" if r.get("always_in", 0) < 0 else "Flat"
        line = f"{i+1:<3d} {r['ticker']:<7s} {r['rs_score']:>+6.1f}  {ai_str:>8s}  {r['day_type']:<22s}"
        y = 0.87 - i * 0.025
        color = "#00C853" if r["rs_score"] > 40 else "#69F0AE" if r["rs_score"] > 15 else "#cccccc"
        ax.text(0.02, y, line, transform=ax.transAxes, fontsize=6,
                color=color, ha="left", va="top", family="monospace")

    # Bottom 25
    ax.text(0.75, 0.93, "BOTTOM 25 (Weakest)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#FF1744", ha="center", va="top")

    ax.text(0.52, 0.89, header, transform=ax.transAxes, fontsize=7,
            color="#999999", ha="left", va="top", family="monospace")

    bot_n = ranked_results[-show_n:] if n >= show_n else ranked_results
    bot_n = list(reversed(bot_n))  # Weakest first
    for i, r in enumerate(bot_n[:show_n]):
        ai_str = "Long" if r.get("always_in", 0) > 0 else "Short" if r.get("always_in", 0) < 0 else "Flat"
        rank = n - i
        line = f"{rank:<3d} {r['ticker']:<7s} {r['rs_score']:>+6.1f}  {ai_str:>8s}  {r['day_type']:<22s}"
        y = 0.87 - i * 0.025
        color = "#FF1744" if r["rs_score"] < -40 else "#FF8A65" if r["rs_score"] < -15 else "#cccccc"
        ax.text(0.52, y, line, transform=ax.transAxes, fontsize=6,
                color=color, ha="left", va="top", family="monospace")

    # Sector heatmap at bottom
    if sectors_data:
        ax.text(0.5, 0.19, "SECTOR RELATIVE STRENGTH",
                transform=ax.transAxes, fontsize=11, fontweight="bold", color="#FFA726",
                ha="center", va="top")

        sectors_data.sort(key=lambda x: x[1], reverse=True)
        for i, (sector, avg, count) in enumerate(sectors_data):
            x = 0.05 + (i % 4) * 0.24
            y = 0.15 - (i // 4) * 0.04
            color = "#00C853" if avg > 10 else "#69F0AE" if avg > 0 else "#FF8A65" if avg > -10 else "#FF1744"
            ax.text(x, y, f"{sector:<18s} {avg:>+5.1f} ({count})",
                    transform=ax.transAxes, fontsize=7, color=color,
                    ha="left", va="top", family="monospace")

    # Breadth stats
    bulls = sum(1 for r in ranked_results if r["rs_score"] > 10)
    bears = sum(1 for r in ranked_results if r["rs_score"] < -10)
    avg_rs = np.mean([r["rs_score"] for r in ranked_results])
    ax.text(0.5, 0.03,
            f"Bullish: {bulls}/{n} ({bulls/n*100:.0f}%)  |  Bearish: {bears}/{n} ({bears/n*100:.0f}%)  |  Avg RS: {avg_rs:+.1f}",
            transform=ax.transAxes, fontsize=9, color="#cccccc", ha="center", va="bottom")

    pdf.savefig(fig, facecolor="#1a1a2e")
    plt.close(fig)


def generate_charts_pdf(ranked_results, ticker_dfs, output_path, sectors_data=None):
    """Generate PDF: cover + rankings table + top 20 charts + bottom 20 charts."""
    # Only chart top 20 and bottom 20 (not all 1500+)
    chart_top = 20
    chart_bot = 20
    n = len(ranked_results)

    top_results = ranked_results[:chart_top]
    bot_results = ranked_results[-chart_bot:] if n > chart_bot else []
    chart_results = top_results + bot_results
    charts_per_page = 4
    num_chart_pages = (len(chart_results) + charts_per_page - 1) // charts_per_page

    total_pages = 1 + 1 + num_chart_pages  # Cover + table + charts
    print(f"\nGenerating PDF ({n} tickers ranked, {len(chart_results)} charted, {total_pages} pages)...")

    with PdfPages(output_path) as pdf:
        # Cover page
        fig_cover = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
        ax = fig_cover.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

        ax.text(0.5, 0.72, "BPA DAILY RELATIVE STRENGTH", transform=ax.transAxes,
                fontsize=28, fontweight="bold", color="white", ha="center")
        ax.text(0.5, 0.62, "FULL US MARKET SCAN", transform=ax.transAxes,
                fontsize=28, fontweight="bold", color="#FFA726", ha="center")
        ax.text(0.5, 0.50, datetime.date.today().strftime("%A, %B %d, %Y"),
                transform=ax.transAxes, fontsize=16, color="#aaaaaa", ha="center")
        ax.text(0.5, 0.43, f"Al Brooks Price Action — {n} Tickers Analyzed",
                transform=ax.transAxes, fontsize=12, color="#777777", ha="center")

        bulls = sum(1 for r in ranked_results if r["rs_score"] > 10)
        bears = sum(1 for r in ranked_results if r["rs_score"] < -10)
        avg_rs = np.mean([r["rs_score"] for r in ranked_results])

        ax.text(0.5, 0.32,
                f"Bullish: {bulls}/{n} ({bulls/n*100:.0f}%)  |  Bearish: {bears}/{n} ({bears/n*100:.0f}%)  |  Avg RS: {avg_rs:+.1f}",
                transform=ax.transAxes, fontsize=11, color="#cccccc", ha="center")

        top5 = [f"{r['ticker']}({r['rs_score']:+.0f})" for r in ranked_results[:5]]
        bot5 = [f"{r['ticker']}({r['rs_score']:+.0f})" for r in ranked_results[-5:]]
        ax.text(0.5, 0.24, f"Strongest: {', '.join(top5)}", transform=ax.transAxes,
                fontsize=10, color="#00C853", ha="center")
        ax.text(0.5, 0.19, f"Weakest: {', '.join(bot5)}", transform=ax.transAxes,
                fontsize=10, color="#FF1744", ha="center")

        if avg_rs > 10:
            bias, bias_color = "BULLISH — Favor long setups, buy pullbacks", "#00C853"
        elif avg_rs < -10:
            bias, bias_color = "BEARISH — Favor short setups, sell rallies", "#FF1744"
        else:
            bias, bias_color = "MIXED — Trade both sides, be selective", "#FFA726"
        ax.text(0.5, 0.10, f"Market Bias: {bias}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=bias_color, ha="center")

        ax.text(0.5, 0.04, datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
                transform=ax.transAxes, fontsize=8, color="#555555", ha="center")

        plt.tight_layout()
        pdf.savefig(fig_cover, facecolor="#1a1a2e")
        plt.close(fig_cover)

        # Rankings table page
        generate_rankings_table_page(pdf, ranked_results, sectors_data)

        # Chart pages
        for page_idx in range(num_chart_pages):
            start_i = page_idx * charts_per_page
            end_i = min(start_i + charts_per_page, len(chart_results))
            page_items = chart_results[start_i:end_i]
            num_on_page = len(page_items)

            fig, axes = plt.subplots(num_on_page, 1, figsize=(11, 3.0 * num_on_page),
                                     facecolor="#1a1a2e")
            if num_on_page == 1:
                axes = [axes]

            for j, r in enumerate(page_items):
                ticker = r["ticker"]
                df = ticker_dfs.get(ticker)
                if df is not None and not df.empty:
                    generate_ticker_chart(df, ticker, r, ax=axes[j])
                else:
                    axes[j].set_facecolor("#1a1a2e")
                    axes[j].text(0.5, 0.5, f"{ticker} — No Data", transform=axes[j].transAxes,
                                 fontsize=14, color="#666666", ha="center", va="center")
                    axes[j].axis("off")

            section = "TOP" if start_i < chart_top else "BOTTOM"
            fig.text(0.5, 0.005,
                     f"BPA RS Rankings ({section}) — Page {page_idx + 3} — {datetime.date.today().strftime('%Y-%m-%d')}",
                     fontsize=8, color="#666666", ha="center")

            plt.tight_layout(rect=[0, 0.02, 1, 0.98])
            pdf.savefig(fig, facecolor="#1a1a2e")
            plt.close(fig)

            tickers_str = ", ".join(r["ticker"] for r in page_items)
            print(f"  Chart page {page_idx + 1}/{num_chart_pages} ({tickers_str})")

    print(f"\nPDF saved: {output_path}")


# ─────────────── MAIN ────────────────────────────────────────────────────────

def _resolve_api_key() -> str:
    """Resolve Databento API key from env or secrets files."""
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


def run_rs_rankings():
    """Main: fetch S&P 500 + sector ETFs via bulk, score, rank, generate PDF."""
    print("=" * 80)
    print("  BPA DAILY RELATIVE STRENGTH — FULL US MARKET SCAN")
    print(f"  Al Brooks Price Action & 17-Step Trend Cycle")
    print(f"  S&P 1500 Composite (500 + MidCap 400 + SmallCap 600)")
    print(f"  {datetime.date.today().strftime('%A, %B %d, %Y')}")
    print("=" * 80)

    # Get S&P 1500 tickers with sectors
    print("\nFetching S&P 1500 ticker list from Wikipedia...")
    sp_tickers, sector_map = get_sp1500_tickers_with_sectors()
    if not sp_tickers:
        print("WARNING: Could not fetch S&P 1500 list, using built-in universe")
        sp_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX",
            "MRK", "ABBV", "LLY", "KO", "PEP", "BAC", "COST", "AVGO", "TMO",
            "MCD", "CSCO", "ACN", "ABT", "DHR", "NEE", "LIN", "AMD", "TXN",
        ]

    # Combine with sector ETFs (dedup)
    all_tickers = list(dict.fromkeys(SECTOR_ETFS + sp_tickers))
    global STOCK_SECTOR_MAP
    STOCK_SECTOR_MAP = sector_map

    print(f"Universe: {len(all_tickers)} tickers ({len(sp_tickers)} S&P 1500 + {len(SECTOR_ETFS)} ETFs)")

    # Initialize data source
    api_key = _resolve_api_key()
    source = get_data_source("databento", api_key=api_key)
    print(f"Data source: {source.name()}")

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=5)).isoformat()
    end = today.isoformat()

    # ─── BULK FETCH ───
    print(f"\nBulk fetching {len(all_tickers)} tickers ({start} to {end})...")

    ticker_dfs = {}
    if isinstance(source, DatabentoSource):
        try:
            bulk_df = source.get_bulk_chart_data(all_tickers, start, end)
            if bulk_df is not None and not bulk_df.empty:
                # Split by symbol, keep last trading day
                for sym, group in bulk_df.groupby("symbol"):
                    group = group.drop(columns=["symbol", "BarNumber"], errors="ignore")
                    group["_date"] = group.index.date
                    last_day = group["_date"].max()
                    day_df = group[group["_date"] == last_day].drop(columns=["_date"])
                    if len(day_df) >= 10:
                        ticker_dfs[sym] = day_df
                print(f"  Bulk fetch returned data for {len(ticker_dfs)} tickers")
            else:
                print("  Bulk fetch returned empty — markets may be closed")
        except Exception as e:
            print(f"  Bulk fetch failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Fallback: sequential fetch (yfinance)
        print("  Using sequential fetch (non-Databento source)...")
        for i, ticker in enumerate(all_tickers[:50]):  # Limit for yfinance
            try:
                df = source.fetch_historical(ticker, start_date=start, end_date=end)
                if df is not None and not df.empty:
                    df["_date"] = df.index.date
                    last_day = df["_date"].max()
                    df = df[df["_date"] == last_day].drop(columns=["_date"])
                    if len(df) >= 10:
                        ticker_dfs[ticker] = df
            except Exception:
                pass
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{min(len(all_tickers), 50)}]...")

    if not ticker_dfs:
        print("\nNo data available. Markets may be closed.")
        return None

    # ─── SCORE ALL TICKERS ───
    print(f"\nScoring {len(ticker_dfs)} tickers...")
    results = []
    for i, (ticker, df) in enumerate(ticker_dfs.items()):
        try:
            score_data = compute_bpa_rs_score(df)
            score_data["ticker"] = ticker
            score_data["sector"] = STOCK_SECTOR_MAP.get(ticker, GICS_SECTORS.get(ticker, "Unknown"))
            results.append(score_data)
        except Exception as e:
            logger.warning(f"{ticker} scoring failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ticker_dfs)}] scored...")

    print(f"  Scored {len(results)} tickers successfully")

    if not results:
        print("No tickers scored.")
        return None

    # Sort by RS
    results.sort(key=lambda x: x["rs_score"], reverse=True)

    # ─── COMPUTE SECTOR AVERAGES ───
    sector_groups = {}
    for r in results:
        sec = r.get("sector", "Unknown")
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append(r["rs_score"])

    sectors_data = [(sec, np.mean(scores), len(scores)) for sec, scores in sector_groups.items()]
    sectors_data.sort(key=lambda x: x[1], reverse=True)

    # ─── PRINT RESULTS ───
    bulls = [r for r in results if r["rs_score"] > 10]
    bears = [r for r in results if r["rs_score"] < -10]
    avg_rs = np.mean([r["rs_score"] for r in results])

    print(f"\n{'=' * 90}")
    print(f"  RESULTS: {len(results)} tickers | Bullish: {len(bulls)} | Bearish: {len(bears)} | Avg RS: {avg_rs:+.1f}")
    print(f"{'=' * 90}")

    # Top 20
    print(f"\n  TOP 20 STRONGEST:")
    print(f"  {'#':<4} {'Ticker':<7} {'RS':>6} {'AI':>8} {'Day Type':<28} {'Sector':<20} {'Setup':<25}")
    print("  " + "-" * 100)
    for i, r in enumerate(results[:20]):
        ai_str = "Long" if r.get("always_in", 0) > 0 else "Short" if r.get("always_in", 0) < 0 else "Flat"
        setup = r["best_setup"][:24] if r["best_setup"] not in ("None", "N/A") else "-"
        print(f"  {i+1:<4d} {r['ticker']:<7s} {r['rs_score']:>+6.1f} {ai_str:>8s} {r['day_type']:<28s} {r.get('sector',''):<20s} {setup:<25s}")

    # Bottom 20
    print(f"\n  BOTTOM 20 WEAKEST:")
    print(f"  {'#':<4} {'Ticker':<7} {'RS':>6} {'AI':>8} {'Day Type':<28} {'Sector':<20} {'Setup':<25}")
    print("  " + "-" * 100)
    for i, r in enumerate(results[-20:]):
        rank = len(results) - 19 + i
        ai_str = "Long" if r.get("always_in", 0) > 0 else "Short" if r.get("always_in", 0) < 0 else "Flat"
        setup = r["best_setup"][:24] if r["best_setup"] not in ("None", "N/A") else "-"
        print(f"  {rank:<4d} {r['ticker']:<7s} {r['rs_score']:>+6.1f} {ai_str:>8s} {r['day_type']:<28s} {r.get('sector',''):<20s} {setup:<25s}")

    # Sector heatmap
    print(f"\n  SECTOR RELATIVE STRENGTH:")
    print(f"  {'Sector':<22s} {'Avg RS':>7} {'#':>4}  {'Bar':<30}")
    print("  " + "-" * 70)
    for sec, avg, count in sectors_data:
        bar_len = int(abs(avg) / 2)
        bar = ("+" * bar_len if avg > 0 else "-" * bar_len)[:30]
        print(f"  {sec:<22s} {avg:>+7.1f} {count:>4d}  {bar}")

    # Market breadth
    print(f"\n  MARKET BREADTH:")
    print(f"    Bullish (RS > +10):  {len(bulls)}/{len(results)} ({len(bulls)/len(results)*100:.0f}%)")
    print(f"    Bearish (RS < -10):  {len(bears)}/{len(results)} ({len(bears)/len(results)*100:.0f}%)")
    print(f"    Average RS:          {avg_rs:+.1f}")

    if avg_rs > 10:
        print(f"\n  Market: BULLISH — favor long setups, buy pullbacks")
    elif avg_rs < -10:
        print(f"\n  Market: BEARISH — favor short setups, sell rallies")
    else:
        print(f"\n  Market: MIXED — trade both sides, be selective")

    # ─── GENERATE PDF ───
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, f"BPA_RS_Rankings_{today.strftime('%Y%m%d')}.pdf")
    try:
        generate_charts_pdf(results, ticker_dfs, pdf_path, sectors_data)
    except Exception as e:
        print(f"\nPDF generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 90}")
    print(f"  Done — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 90}\n")

    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_rs_rankings()
