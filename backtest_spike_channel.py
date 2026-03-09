"""
backtest_spike_channel.py — Backtest with-trend Spike entries
against real Databento 1-min data.

Compares two entry styles:
  A) At-the-market: enter immediately after the spike ends
  B) Pullback: wait for a small pullback after the spike, then enter

Also shows H1/L1 results (existing detector) for comparison.

Spike strength scoring (S0-S5) based on Brooks' signs of strength:
  1. Body gaps (measuring gaps)
  2. Closes near extremes (small tails)
  3. Surprise size (> 1.5× avg prior range)
  4. Minimal overlap between bars
  5. Consecutive new extremes

Usage:
    python3 backtest_spike_channel.py [1m|5m] [max_tickers]
"""

import re
import sys
import os
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import run_backtest, run_multi_day_backtest


# ─────────────────────────── DATA LOADING ────────────────────────────────────

def load_databento_1m(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["ts_event"])
    df = df.rename(columns={
        "ts_event": "Datetime",
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df = df.set_index("Datetime")
    df = df.between_time("14:30", "20:59")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return df_1m.resample("5min").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()


def split_by_day(df: pd.DataFrame) -> dict:
    days = {}
    for date, group in df.groupby(df.index.date):
        if len(group) >= 10:
            days[str(date)] = group
    return days


# ─────────────────────────── HELPERS ─────────────────────────────────────────

def is_spike_trade(name: str) -> bool:
    return bool(re.match(r"(Bull Spike Buy|Bear Spike Sell|Bull Spike Pullback Buy|Bear Spike Pullback Sell) S\d", name)) \
        or name in {"H1 in Strong Bull Spike", "L1 in Strong Bear Spike"}

def is_at_market(name: str) -> bool:
    return name.startswith("Bull Spike Buy S") or name.startswith("Bear Spike Sell S")

def is_pullback(name: str) -> bool:
    return name.startswith("Bull Spike Pullback Buy S") or name.startswith("Bear Spike Pullback Sell S")

def is_h1l1(name: str) -> bool:
    return name in {"H1 in Strong Bull Spike", "L1 in Strong Bear Spike"}

def get_strength(name: str) -> int:
    m = re.search(r" S(\d)$", name)
    return int(m.group(1)) if m else -1


# ─────────────────────────── BACKTEST RUNNER ─────────────────────────────────

def run_spike_backtest(tickers: list[str], data_dir: str, timeframe: str = "1m") -> dict:
    all_trades = []
    total_count = 0
    ticker_info = {}

    for ticker in tickers:
        filepath = os.path.join(data_dir, f"data_{ticker}_1m.csv")
        if not os.path.exists(filepath):
            continue

        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df_1m = load_databento_1m(filepath)
            df = resample_to_5m(df_1m) if timeframe == "5m" else df_1m
            daily = split_by_day(df)
        except Exception as e:
            print(f"ERR: {e}")
            continue

        result = run_multi_day_backtest(
            daily, mode="scalp", min_bars_between_trades=3,
            slippage=0.01, commission=0.005, ticker=ticker,
            profitable_only=False, use_setup_config=False,
        )
        trades = result["trades"]
        total_count += len(trades)

        spike_trades = [t for t in trades if is_spike_trade(t.setup_name)]
        all_trades.extend(spike_trades)

        am = sum(1 for t in spike_trades if is_at_market(t.setup_name))
        pb = sum(1 for t in spike_trades if is_pullback(t.setup_name))
        h1 = sum(1 for t in spike_trades if is_h1l1(t.setup_name))
        print(f"{len(daily)}d | {am} mkt, {pb} pb, {h1} h1l1")
        ticker_info[ticker] = {"days": len(daily), "am": am, "pb": pb, "h1": h1}

    return {"trades": all_trades, "total": total_count, "tickers": ticker_info}


# ─────────────────────────── REPORTING ───────────────────────────────────────

def stats_block(trade_list, label):
    if not trade_list:
        print(f"\n  {label}: No trades")
        return

    w = [t for t in trade_list if t.is_winner]
    l = [t for t in trade_list if not t.is_winner]
    wr = len(w) / len(trade_list) * 100
    avg_r = sum(t.r_multiple for t in trade_list) / len(trade_list)
    avg_w = sum(t.r_multiple for t in w) / len(w) if w else 0
    avg_l = sum(t.r_multiple for t in l) / len(l) if l else 0
    gp = sum(t.pnl for t in w)
    gl = abs(sum(t.pnl for t in l))
    pf = gp / gl if gl > 0 else float('inf')
    avg_mae = sum(t.mae_r for t in trade_list) / len(trade_list)
    avg_mfe = sum(t.mfe_r for t in trade_list) / len(trade_list)
    avg_bars = sum(t.bars_held for t in trade_list) / len(trade_list)
    exits = {}
    for t in trade_list:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    print(f"\n  {label}")
    print(f"  {'─' * 55}")
    print(f"  Trades: {len(trade_list)}  |  Win: {len(w)} ({wr:.1f}%)  |  Lose: {len(l)} ({100-wr:.1f}%)")
    print(f"  Avg R:  {avg_r:+.3f}  |  Avg Win: {avg_w:+.3f}  |  Avg Loss: {avg_l:+.3f}")
    print(f"  PF:     {pf:.2f}    |  MAE: {avg_mae:.2f}R  |  MFE: {avg_mfe:.2f}R  |  Bars: {avg_bars:.1f}")
    print(f"  Exits:  {exits}")

    # R:R sweep
    print(f"  R:R sweep: ", end="")
    for rr in [0.5, 0.75, 1.0, 1.5, 2.0]:
        hits = sum(1 for t in trade_list if t.mfe_r >= rr)
        hr = hits / len(trade_list) * 100
        ev = (hr / 100) * rr - (1 - hr / 100)
        marker = " ***" if ev > 0 else ""
        print(f"{rr}:1={hr:.0f}%({ev:+.2f}R){marker}  ", end="")
    print()


def print_report(result: dict):
    trades = result["trades"]
    print("\n" + "=" * 70)
    print("  SPIKE WITH-TREND ENTRIES — BACKTEST REPORT")
    print("  (with Brooks Strength Scoring S0-S5)")
    print("=" * 70)

    if not trades:
        print("\n  No spike trades found.")
        return

    am_trades = [t for t in trades if is_at_market(t.setup_name)]
    pb_trades = [t for t in trades if is_pullback(t.setup_name)]
    h1_trades = [t for t in trades if is_h1l1(t.setup_name)]

    print(f"\n  Total spike trades: {len(trades)} across {len(result['tickers'])} tickers")

    stats_block(trades, "ALL SPIKE ENTRIES COMBINED")

    # ── Style comparison ──
    print(f"\n  {'=' * 55}")
    print(f"  STYLE COMPARISON")
    print(f"  {'=' * 55}")

    stats_block(am_trades, "STYLE A: AT-THE-MARKET (enter right after spike)")
    stats_block(pb_trades, "STYLE B: PULLBACK (wait for dip, then enter)")
    stats_block(h1_trades, "STYLE C: H1/L1 (first pullback bar confirmation)")

    # ── Strength score breakdown (the key analysis) ──
    scored_trades = [t for t in trades if get_strength(t.setup_name) >= 0]
    if scored_trades:
        print(f"\n  {'=' * 55}")
        print(f"  STRENGTH SCORE BREAKDOWN (Brooks Signs of Strength)")
        print(f"  {'=' * 55}")
        print(f"  Scoring: body gaps, closes near extremes, surprise size,")
        print(f"           minimal overlap, consecutive new extremes")

        for s in range(6):
            s_trades = [t for t in scored_trades if get_strength(t.setup_name) == s]
            stats_block(s_trades, f"STRENGTH S{s}/5")

        # Cumulative: S3+ (strong spikes only)
        strong = [t for t in scored_trades if get_strength(t.setup_name) >= 3]
        stats_block(strong, "STRONG SPIKES (S3+/5)")

        weak = [t for t in scored_trades if get_strength(t.setup_name) <= 1]
        stats_block(weak, "WEAK SPIKES (S0-S1/5)")

        # Summary table
        print(f"\n  {'=' * 55}")
        print(f"  STRENGTH SUMMARY TABLE")
        print(f"  {'=' * 55}")
        print(f"  {'Score':<8} {'Trades':>7} {'WR':>7} {'Avg R':>8} {'PF':>7} {'MFE':>7}")
        print(f"  {'─' * 48}")
        for s in range(6):
            s_trades = [t for t in scored_trades if get_strength(t.setup_name) == s]
            if not s_trades:
                print(f"  S{s}/5    {'0':>7}")
                continue
            w = [t for t in s_trades if t.is_winner]
            l = [t for t in s_trades if not t.is_winner]
            wr = len(w) / len(s_trades) * 100
            avg_r = sum(t.r_multiple for t in s_trades) / len(s_trades)
            gp = sum(t.pnl for t in w)
            gl = abs(sum(t.pnl for t in l))
            pf = gp / gl if gl > 0 else float('inf')
            avg_mfe = sum(t.mfe_r for t in s_trades) / len(s_trades)
            print(f"  S{s}/5    {len(s_trades):>7} {wr:>6.1f}% {avg_r:>+8.3f} {pf:>7.2f} {avg_mfe:>6.2f}R")

    # ── Direction breakdown ──
    print(f"\n  {'=' * 55}")
    print(f"  DIRECTION BREAKDOWN")
    print(f"  {'=' * 55}")

    bull_am = [t for t in am_trades if "Bull" in t.setup_name]
    bear_am = [t for t in am_trades if "Bear" in t.setup_name]
    bull_pb = [t for t in pb_trades if "Bull" in t.setup_name]
    bear_pb = [t for t in pb_trades if "Bear" in t.setup_name]

    stats_block(bull_am, "Bull Spike Buy (at-market)")
    stats_block(bear_am, "Bear Spike Sell (at-market)")
    stats_block(bull_pb, "Bull Spike Pullback Buy")
    stats_block(bear_pb, "Bear Spike Pullback Sell")

    # ── Trade log ──
    print(f"\n  {'=' * 55}")
    print(f"  TRADE LOG (first 40)")
    print(f"  {'=' * 55}")
    print(f"  {'Tkr':<5} {'Setup':<32} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'P&L':>8} {'R':>7} {'Reason':<12}")
    print(f"  {'─' * 105}")
    for t in trades[:40]:
        print(f"  {t.ticker:<5} {t.setup_name:<32} {t.direction:<6} {t.entry_price:>8.2f} {t.exit_price:>8.2f} {t.pnl:>+8.2f} {t.r_multiple:>+7.2f} {t.exit_reason:<12}")


# ─────────────────────────── MAIN ────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir
    if not glob.glob(os.path.join(data_dir, "data_*_1m.csv")):
        explicit = "/Users/williamkosloski/BPA-Bot-1"
        if glob.glob(os.path.join(explicit, "data_*_1m.csv")):
            data_dir = explicit

    files = glob.glob(os.path.join(data_dir, "data_*_1m.csv"))
    tickers = sorted([os.path.basename(f).replace("data_", "").replace("_1m.csv", "") for f in files])

    if not tickers:
        print("No Databento data files found.")
        sys.exit(1)

    timeframe = "1m"
    max_tickers = len(tickers)
    for arg in sys.argv[1:]:
        if arg in ("1m", "5m"):
            timeframe = arg
        elif arg.isdigit():
            max_tickers = int(arg)

    tickers = tickers[:max_tickers]
    print(f"Tickers: {', '.join(tickers)}  |  TF: {timeframe}\n")

    result = run_spike_backtest(tickers, data_dir, timeframe=timeframe)
    print_report(result)
