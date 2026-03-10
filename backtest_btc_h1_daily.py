#!/usr/bin/env python3
"""
backtest_btc_h1_daily.py — H1 (First Pullback) on BTC Daily Chart

Al Brooks says H1 is the highest probability pullback entry in a strong trend.
This backtest tests that claim on BTC's daily chart using IBIT (BlackRock Bitcoin ETF)
as the proxy.

Strategy:
  H1 Long: Price above 20-EMA → 1-bar pullback (bear bar) → bull bar breaks above
           Entry: buy stop above pullback bar high
           Stop: below pullback bar low
           Targets: 1R, 2R, 3R

  L1 Short: Price below 20-EMA → 1-bar rally (bull bar) → bear bar breaks below
            Entry: sell stop below rally bar low
            Stop: above rally bar high
            Targets: 1R, 2R, 3R

Hold limit: 20 bars (trading days)
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
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algo_engine import Bar, bars_from_df, compute_ema, find_swing_lows, find_swing_highs

logger = logging.getLogger(__name__)


# ─────────────── DATA FETCHING ───────────────────────────────────────────────

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


def fetch_daily_bars(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Databento.
    Uses 1-min data resampled to daily to avoid ohlcv-1d availability lag.
    """
    import databento as db

    client = db.Historical(api_key)
    dataset = "XNAS.ITCH"

    end_dt = datetime.datetime.fromisoformat(end)
    today = datetime.date.today()
    target_end = end_dt.date() + datetime.timedelta(days=1)
    if target_end > today:
        target_end = today
    db_end = f"{target_end.isoformat()}T00:00:00"
    db_start = f"{start}T00:00:00"

    print(f"  Fetching {ticker} 1-min data → resample to daily ({start} → {end})...")

    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=[ticker],
        schema="ohlcv-1m",
        start=db_start,
        end=db_end,
    )

    df = data.to_df()
    if df is None or df.empty:
        print(f"  No data returned for {ticker}")
        return pd.DataFrame()

    # Normalize columns
    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename_map)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Open", "High", "Low", "Close"])

    # Convert timezone
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    # Filter RTH only
    df = df.between_time("09:30", "15:59")

    # Resample to daily bars
    daily = df.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

    # Remove weekends/holidays (days with no data)
    daily = daily[daily["Volume"] > 0]

    print(f"  Got {len(daily)} daily bars for {ticker}")
    return daily


# ─────────────── H1/L1 DETECTION ────────────────────────────────────────────

def detect_h1_l1_setups(bars: list[Bar], ema: list[float]) -> list[dict]:
    """
    Detect H1 (first pullback long) and L1 (first pullback short) on daily bars.

    H1: Price above EMA → 1-2 bar pullback → bull bar breaks above pullback high
    L1: Price below EMA → 1-2 bar rally → bear bar breaks below rally low

    Only takes the FIRST pullback after a trend resumes (H1, not H2/H3).
    """
    setups = []
    n = len(bars)
    if n < 20:
        return setups

    avg_range = np.mean([b.range for b in bars if b.range > 0])

    # Track last setup bar to prevent clustering
    last_long_bar = -10
    last_short_bar = -10

    for i in range(5, n):
        b = bars[i]
        prev = bars[i - 1]
        prev2 = bars[i - 2] if i >= 2 else prev

        # ─── H1 LONG ───
        # Context: strong trend above EMA (majority of recent bars above)
        above_count = sum(1 for j in range(max(0, i - 10), i) if bars[j].close > ema[j])
        if above_count >= 7 and i - last_long_bar >= 5:
            # Pullback: previous bar is bear or makes lower low
            is_pullback = prev.is_bear or prev.low < prev2.low

            # 1-bar pullback only (H1, not deeper H2/H3)
            # Check that prior to the pullback, we had bull bars
            bars_before_pb = bars[max(0, i - 4):i - 1]
            bull_before = sum(1 for bb in bars_before_pb if bb.is_bull)

            if is_pullback and bull_before >= 2:
                # Current bar breaks above pullback high and is bullish
                if b.high > prev.high and b.is_bull:
                    # Verify this is a first pullback (not a choppy range)
                    recent_pullbacks = sum(
                        1 for j in range(max(0, i - 8), i - 1)
                        if bars[j].is_bear and bars[j].close > ema[j]
                    )
                    if recent_pullbacks <= 2:  # H1 = first/shallow pullback
                        entry = round(prev.high + 0.01, 2)
                        stop = round(min(prev.low, min(bars[j].low for j in range(max(0, i - 2), i + 1))) - 0.01, 2)
                        risk = entry - stop

                        if risk > 0 and risk < entry * 0.15:  # Max 15% risk (daily bars are bigger)
                            setups.append({
                                "bar_idx": i,
                                "direction": "Long",
                                "setup": "H1",
                                "entry": entry,
                                "stop": stop,
                                "risk": risk,
                                "date": getattr(b, '_date', None),
                            })
                            last_long_bar = i

        # ─── L1 SHORT ───
        below_count = sum(1 for j in range(max(0, i - 10), i) if bars[j].close < ema[j])
        if below_count >= 7 and i - last_short_bar >= 5:
            is_pullback = prev.is_bull or prev.high > prev2.high

            bars_before_pb = bars[max(0, i - 4):i - 1]
            bear_before = sum(1 for bb in bars_before_pb if bb.is_bear)

            if is_pullback and bear_before >= 2:
                if b.low < prev.low and b.is_bear:
                    recent_pullbacks = sum(
                        1 for j in range(max(0, i - 8), i - 1)
                        if bars[j].is_bull and bars[j].close < ema[j]
                    )
                    if recent_pullbacks <= 2:
                        entry = round(prev.low - 0.01, 2)
                        stop = round(max(prev.high, max(bars[j].high for j in range(max(0, i - 2), i + 1))) + 0.01, 2)
                        risk = stop - entry

                        if risk > 0 and risk < entry * 0.15:
                            setups.append({
                                "bar_idx": i,
                                "direction": "Short",
                                "setup": "L1",
                                "entry": entry,
                                "stop": stop,
                                "risk": risk,
                                "date": getattr(b, '_date', None),
                            })
                            last_short_bar = i

    return setups


# ─────────────── TRADE SIMULATION ────────────────────────────────────────────

@dataclass
class DailyTrade:
    bar_idx: int
    direction: str
    setup: str
    entry: float
    stop: float
    risk: float
    entry_date: str = ""

    # Outcomes at different R:R
    filled: bool = False
    fill_bar: int = 0

    # Results per target
    results: dict = field(default_factory=dict)

    # MAE/MFE
    mae_r: float = 0.0
    mfe_r: float = 0.0

    # For equity curve
    exit_bar: int = 0
    exit_date: str = ""


def simulate_daily_h1_trade(
    bars: list[Bar],
    setup: dict,
    timestamps: list[str],
    targets: list[float] = [1.0, 2.0, 3.0],
    hold_limit: int = 20,
) -> DailyTrade:
    """Simulate an H1/L1 trade on daily bars with hold limit."""

    trade = DailyTrade(
        bar_idx=setup["bar_idx"],
        direction=setup["direction"],
        setup=setup["setup"],
        entry=setup["entry"],
        stop=setup["stop"],
        risk=setup["risk"],
        entry_date=timestamps[setup["bar_idx"]] if setup["bar_idx"] < len(timestamps) else "",
    )

    signal_idx = setup["bar_idx"]
    entry = setup["entry"]
    stop = setup["stop"]
    risk = setup["risk"]
    direction = setup["direction"]

    # Check fill on next bar(s)
    filled = False
    fill_idx = None
    for look in range(signal_idx + 1, min(signal_idx + 4, len(bars))):
        b = bars[look]
        if direction == "Long" and b.high >= entry:
            filled = True
            fill_idx = look
            break
        elif direction == "Short" and b.low <= entry:
            filled = True
            fill_idx = look
            break

    if not filled:
        trade.filled = False
        return trade

    trade.filled = True
    trade.fill_bar = fill_idx

    # Track MAE/MFE and simulate for each target
    mae = 0.0
    mfe = 0.0

    for mult in targets:
        if direction == "Long":
            target_price = entry + risk * mult
        else:
            target_price = entry - risk * mult

        hit_target = False
        hit_stop = False
        exit_price = None
        exit_reason = "hold_limit"
        exit_idx = fill_idx

        for i in range(fill_idx, min(fill_idx + hold_limit, len(bars))):
            b = bars[i]

            # Track MAE/MFE (for largest target only)
            if mult == targets[-1]:
                if direction == "Long":
                    adv = (b.low - entry) / risk
                    fav = (b.high - entry) / risk
                else:
                    adv = (entry - b.high) / risk
                    fav = (entry - b.low) / risk
                mae = min(mae, adv)
                mfe = max(mfe, fav)

            # Check stop first
            if direction == "Long":
                if b.low <= stop:
                    hit_stop = True
                    exit_price = stop
                    exit_reason = "stop"
                    exit_idx = i
                    break
                if b.high >= target_price:
                    hit_target = True
                    exit_price = target_price
                    exit_reason = f"{mult}R_target"
                    exit_idx = i
                    break
            else:
                if b.high >= stop:
                    hit_stop = True
                    exit_price = stop
                    exit_reason = "stop"
                    exit_idx = i
                    break
                if b.low <= target_price:
                    hit_target = True
                    exit_price = target_price
                    exit_reason = f"{mult}R_target"
                    exit_idx = i
                    break

        # EOD fallback
        if not hit_target and not hit_stop:
            last_idx = min(fill_idx + hold_limit - 1, len(bars) - 1)
            exit_price = bars[last_idx].close
            exit_idx = last_idx

        if direction == "Long":
            pnl_r = (exit_price - entry) / risk
        else:
            pnl_r = (entry - exit_price) / risk

        trade.results[f"{mult}R"] = {
            "hit_target": hit_target,
            "hit_stop": hit_stop,
            "exit_reason": exit_reason,
            "pnl_r": round(pnl_r, 3),
            "is_winner": hit_target,
            "bars_held": exit_idx - fill_idx,
        }
        trade.exit_bar = exit_idx
        trade.exit_date = timestamps[exit_idx] if exit_idx < len(timestamps) else ""

    trade.mae_r = round(mae, 3)
    trade.mfe_r = round(mfe, 3)

    return trade


# ─────────────── MAIN BACKTEST ───────────────────────────────────────────────

def run_btc_h1_backtest(
    tickers: list[str] = None,
    start: str = "2024-01-15",
    end: str = None,
    hold_limit: int = 20,
    targets: list[float] = [1.0, 2.0, 3.0],
):
    """Run H1/L1 backtest on daily charts."""

    if tickers is None:
        tickers = ["IBIT"]  # BTC proxy (BlackRock Bitcoin ETF)
    if end is None:
        end = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    print("=" * 90)
    print("  BACKTEST: H1 (FIRST PULLBACK) ON DAILY CHART")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Period: {start} → {end}")
    print(f"  Hold limit: {hold_limit} days")
    print(f"  Targets: {', '.join(f'{t}:1' for t in targets)} R:R")
    print("=" * 90)

    api_key = _resolve_api_key()
    all_trades = []
    ticker_stats = {}

    for ticker in tickers:
        print(f"\n{'─' * 60}")
        print(f"  {ticker}")
        print(f"{'─' * 60}")

        try:
            df = fetch_daily_bars(ticker, start, end, api_key)
        except Exception as e:
            print(f"  Failed to fetch {ticker}: {e}")
            continue

        if df.empty or len(df) < 30:
            print(f"  Not enough data for {ticker} ({len(df)} bars)")
            continue

        bars = bars_from_df(df)
        ema = compute_ema(bars)
        timestamps = [str(idx) for idx in df.index]

        # Inject EMA
        for i, b in enumerate(bars):
            b.ema_20 = ema[i]

        # Detect H1/L1
        setups = detect_h1_l1_setups(bars, ema)
        print(f"  {len(setups)} H1/L1 setups detected")

        h1_count = sum(1 for s in setups if s["setup"] == "H1")
        l1_count = sum(1 for s in setups if s["setup"] == "L1")
        print(f"    H1 (Long): {h1_count}")
        print(f"    L1 (Short): {l1_count}")

        # Simulate
        trades = []
        last_exit = -999
        for setup in setups:
            # Prevent overlapping trades
            if setup["bar_idx"] <= last_exit + 2:
                continue

            trade = simulate_daily_h1_trade(bars, setup, timestamps, targets, hold_limit)
            if trade.filled:
                trade.entry_date = timestamps[trade.fill_bar] if trade.fill_bar < len(timestamps) else ""
                trades.append(trade)
                last_exit = trade.exit_bar

        filled_count = len(trades)
        print(f"  {filled_count} trades filled")

        # Per-ticker stats
        if trades:
            for mult in targets:
                key = f"{mult}R"
                wins = sum(1 for t in trades if t.results.get(key, {}).get("is_winner", False))
                stops = sum(1 for t in trades if t.results.get(key, {}).get("hit_stop", False))
                pnls = [t.results.get(key, {}).get("pnl_r", 0) for t in trades]
                total_r = sum(pnls)
                wr = wins / len(trades) * 100

                gp = sum(p for p in pnls if p > 0)
                gl = abs(sum(p for p in pnls if p < 0))
                pf = gp / gl if gl > 0 else float('inf') if gp > 0 else 0

                print(f"    {ticker} @ {mult}:1 → {len(trades)} trades, "
                      f"WR: {wr:.1f}%, PF: {pf:.2f}, Total: {total_r:+.1f}R")

            ticker_stats[ticker] = {
                "df": df, "bars": bars, "ema": ema, "setups": setups,
                "trades": trades, "timestamps": timestamps,
            }

        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades generated.")
        return

    # ─── PRINT RESULTS ───
    print_results(all_trades, targets, tickers)

    # ─── GENERATE PDF ───
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "BPA_BTC_H1_Daily_Backtest.pdf")
    generate_pdf(all_trades, targets, ticker_stats, pdf_path)

    return pdf_path


def print_results(trades, targets, tickers):
    """Print comprehensive results."""
    longs = [t for t in trades if t.direction == "Long"]
    shorts = [t for t in trades if t.direction == "Short"]

    print(f"\n{'=' * 90}")
    print(f"  RESULTS — {len(trades)} trades ({len(longs)} H1 long, {len(shorts)} L1 short)")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'=' * 90}")

    print(f"\n  {'Strategy':<28} {'N':>5} {'WR%':>7} {'PF':>6} {'AvgR':>8} {'TotalR':>8} {'AvgBars':>8}")
    print("  " + "-" * 75)

    for mult in targets:
        key = f"{mult}R"
        for label, subset in [("All", trades), ("H1 Long", longs), ("L1 Short", shorts)]:
            if not subset:
                continue
            wins = sum(1 for t in subset if t.results.get(key, {}).get("is_winner", False))
            pnls = [t.results.get(key, {}).get("pnl_r", 0) for t in subset]
            bars_held = [t.results.get(key, {}).get("bars_held", 0) for t in subset]
            n = len(subset)
            wr = wins / n * 100
            total = sum(pnls)
            avg = np.mean(pnls)
            avg_bars = np.mean(bars_held) if bars_held else 0

            gp = sum(p for p in pnls if p > 0)
            gl = abs(sum(p for p in pnls if p < 0))
            pf = gp / gl if gl > 0 else float('inf') if gp > 0 else 0

            name = f"{label} @ {mult}:1"
            print(f"  {name:<28} {n:>5} {wr:>6.1f}% {pf:>6.2f} {avg:>+8.3f} {total:>+8.1f} {avg_bars:>8.1f}")

    # MAE/MFE
    maes = [t.mae_r for t in trades]
    mfes = [t.mfe_r for t in trades]
    print(f"\n  MAE/MFE:")
    print(f"    Avg MAE: {np.mean(maes):+.2f}R  (worst drawdown)")
    print(f"    Avg MFE: {np.mean(mfes):+.2f}R  (best unrealized)")
    print(f"    MFE > 2R: {sum(1 for m in mfes if m > 2) / len(mfes) * 100:.0f}%")
    print(f"    MFE > 3R: {sum(1 for m in mfes if m > 3) / len(mfes) * 100:.0f}%")

    # Win rate by bars held
    if trades:
        print(f"\n  BARS HELD DISTRIBUTION (@ 1:1):")
        key = f"{targets[0]}R"
        for bucket_label, lo, hi in [("1-3 days", 1, 3), ("4-7 days", 4, 7),
                                      ("8-14 days", 8, 14), ("15-20 days", 15, 20)]:
            bucket = [t for t in trades
                      if lo <= t.results.get(key, {}).get("bars_held", 0) <= hi]
            if bucket:
                bw = sum(1 for t in bucket if t.results[key]["is_winner"])
                print(f"    {bucket_label:<12} {len(bucket):>4} trades  WR: {bw/len(bucket)*100:>5.1f}%")

    # Chronological trade log
    print(f"\n  TRADE LOG (last 20):")
    print(f"  {'Date':<12} {'Dir':<6} {'Setup':<5} {'Entry':>8} {'Stop':>8} {'Risk':>7} {'1R P&L':>7} {'2R P&L':>7} {'3R P&L':>7}")
    print("  " + "-" * 80)
    for t in trades[-20:]:
        date_str = t.entry_date[:10] if t.entry_date else "?"
        r1 = t.results.get("1.0R", {}).get("pnl_r", 0)
        r2 = t.results.get("2.0R", {}).get("pnl_r", 0)
        r3 = t.results.get("3.0R", {}).get("pnl_r", 0)
        print(f"  {date_str:<12} {t.direction:<6} {t.setup:<5} {t.entry:>8.2f} {t.stop:>8.2f} "
              f"{t.risk:>7.2f} {r1:>+7.2f} {r2:>+7.2f} {r3:>+7.2f}")


# ─────────────── PDF REPORT ──────────────────────────────────────────────────

def generate_pdf(trades, targets, ticker_stats, output_path):
    """Generate PDF with equity curves and chart annotations."""
    print(f"\nGenerating PDF...")

    longs = [t for t in trades if t.direction == "Long"]
    shorts = [t for t in trades if t.direction == "Short"]

    with PdfPages(output_path) as pdf:
        # Page 1: Summary + Equity curves
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor="#1a1a2e")

        # Summary header
        ax1 = axes[0]
        ax1.set_facecolor("#1a1a2e")

        # Equity curves
        for mult in targets:
            key = f"{mult}R"
            cum = []
            running = 0
            for t in trades:
                running += t.results.get(key, {}).get("pnl_r", 0)
                cum.append(running)
            if cum:
                color = {"1.0R": "#FFA726", "2.0R": "#00C853", "3.0R": "#42A5F5"}.get(key, "#ccc")
                ax1.plot(cum, color=color, linewidth=1.8, label=f"{mult}:1 R:R")

        ax1.axhline(y=0, color="#555555", linewidth=0.5)

        # Compute stats for title
        stats = {}
        for mult in targets:
            key = f"{mult}R"
            pnls = [t.results.get(key, {}).get("pnl_r", 0) for t in trades]
            wins = sum(1 for t in trades if t.results.get(key, {}).get("is_winner"))
            wr = wins / len(trades) * 100
            total = sum(pnls)
            stats[key] = {"wr": wr, "total": total}

        tickers_str = ", ".join(ticker_stats.keys())
        ax1.set_title(f"H1/L1 Daily Backtest — {tickers_str}\n"
                      f"{len(trades)} trades | "
                      f"1:1 WR:{stats['1.0R']['wr']:.0f}% Total:{stats['1.0R']['total']:+.0f}R | "
                      f"3:1 WR:{stats['3.0R']['wr']:.0f}% Total:{stats['3.0R']['total']:+.0f}R",
                      fontsize=11, color="white", fontweight="bold")
        ax1.set_xlabel("Trade #", fontsize=9, color="#aaa")
        ax1.set_ylabel("Cumulative R", fontsize=9, color="#aaa")
        ax1.legend(fontsize=9, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")
        ax1.tick_params(colors="#999")
        ax1.grid(True, alpha=0.2)

        # Long vs Short comparison
        ax2 = axes[1]
        ax2.set_facecolor("#1a1a2e")

        for direction, subset, color, style in [
            ("H1 Long", longs, "#00C853", "-"),
            ("L1 Short", shorts, "#FF1744", "--"),
        ]:
            if not subset:
                continue
            cum = []
            running = 0
            for t in subset:
                running += t.results.get("1.0R", {}).get("pnl_r", 0)
                cum.append(running)
            ax2.plot(cum, color=color, linewidth=1.8, linestyle=style, label=direction)

        ax2.axhline(y=0, color="#555555", linewidth=0.5)
        ax2.set_title("H1 Longs vs L1 Shorts (@ 1:1 R:R)", fontsize=11, color="white", fontweight="bold")
        ax2.set_xlabel("Trade #", fontsize=9, color="#aaa")
        ax2.set_ylabel("Cumulative R", fontsize=9, color="#aaa")
        ax2.legend(fontsize=9, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")
        ax2.tick_params(colors="#999")
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

        # Page 2: Daily chart with H1/L1 markers for each ticker
        for ticker, data in ticker_stats.items():
            df = data["df"]
            bars = data["bars"]
            ema_vals = data["ema"]
            ticker_trades = data["trades"]

            if len(df) < 20:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(14, 7), facecolor="#1a1a2e")
            ax.set_facecolor("#1a1a2e")

            # Plot candlesticks manually (daily chart)
            dates = list(range(len(bars)))
            for i, b in enumerate(bars):
                color = "#26A69A" if b.is_bull else "#EF5350"
                # Body
                ax.plot([i, i], [b.low, b.high], color=color, linewidth=0.5)
                body_bottom = min(b.open, b.close)
                body_top = max(b.open, b.close)
                ax.add_patch(plt.Rectangle(
                    (i - 0.3, body_bottom), 0.6, body_top - body_bottom,
                    facecolor=color, edgecolor=color, linewidth=0.5
                ))

            # EMA line
            ax.plot(dates[:len(ema_vals)], ema_vals, color="#FFA726", linewidth=1.5,
                    linestyle="--", alpha=0.8, label="EMA-20")

            # Mark H1/L1 entries
            for t in ticker_trades:
                idx = t.bar_idx
                if idx >= len(bars):
                    continue
                if t.direction == "Long":
                    ax.annotate("H1", (idx, bars[idx].low * 0.995),
                                fontsize=8, fontweight="bold", color="#00C853",
                                ha="center", va="top",
                                arrowprops=dict(arrowstyle="->", color="#00C853", lw=1.2))
                else:
                    ax.annotate("L1", (idx, bars[idx].high * 1.005),
                                fontsize=8, fontweight="bold", color="#FF1744",
                                ha="center", va="bottom",
                                arrowprops=dict(arrowstyle="->", color="#FF1744", lw=1.2))

            # Axis labels with dates
            if len(df) > 0:
                tick_positions = list(range(0, len(bars), max(1, len(bars) // 15)))
                tick_labels = []
                for pos in tick_positions:
                    if pos < len(df):
                        tick_labels.append(df.index[pos].strftime("%b %d"))
                    else:
                        tick_labels.append("")
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=7, color="#999", rotation=45)

            h1_n = sum(1 for t in ticker_trades if t.direction == "Long")
            l1_n = sum(1 for t in ticker_trades if t.direction == "Short")

            ax.set_title(f"{ticker} Daily Chart — H1/L1 Entries ({h1_n} long, {l1_n} short)",
                         fontsize=14, color="white", fontweight="bold")
            ax.set_ylabel("Price", fontsize=9, color="#aaa")
            ax.tick_params(colors="#999")
            ax.legend(fontsize=9, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")
            ax.grid(True, alpha=0.15)

            plt.tight_layout()
            pdf.savefig(fig, facecolor="#1a1a2e")
            plt.close(fig)

        # Page 3: MAE/MFE distribution
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#1a1a2e")

        ax_mae = axes[0]
        ax_mae.set_facecolor("#1a1a2e")
        maes = [t.mae_r for t in trades]
        ax_mae.hist(maes, bins=20, color="#FF8A65", edgecolor="none", alpha=0.8)
        ax_mae.axvline(x=np.mean(maes), color="#FF1744", linewidth=2, linestyle="--",
                       label=f"Avg: {np.mean(maes):.2f}R")
        ax_mae.set_title("MAE Distribution (Worst Drawdown per Trade)", fontsize=11, color="white")
        ax_mae.set_xlabel("MAE (R-multiples)", fontsize=9, color="#aaa")
        ax_mae.tick_params(colors="#999")
        ax_mae.legend(fontsize=9, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")

        ax_mfe = axes[1]
        ax_mfe.set_facecolor("#1a1a2e")
        mfes = [t.mfe_r for t in trades]
        ax_mfe.hist(mfes, bins=20, color="#69F0AE", edgecolor="none", alpha=0.8)
        ax_mfe.axvline(x=np.mean(mfes), color="#00C853", linewidth=2, linestyle="--",
                       label=f"Avg: {np.mean(mfes):.2f}R")
        ax_mfe.set_title("MFE Distribution (Best Unrealized per Trade)", fontsize=11, color="white")
        ax_mfe.set_xlabel("MFE (R-multiples)", fontsize=9, color="#aaa")
        ax_mfe.tick_params(colors="#999")
        ax_mfe.legend(fontsize=9, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")

        plt.tight_layout()
        pdf.savefig(fig, facecolor="#1a1a2e")
        plt.close(fig)

    print(f"PDF saved: {output_path}")


# ─────────────── MAIN ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # BTC (via IBIT) + SPY for comparison
    run_btc_h1_backtest(
        tickers=["IBIT", "SPY"],
        start="2024-01-15",  # IBIT inception
        hold_limit=20,
        targets=[1.0, 2.0, 3.0],
    )
