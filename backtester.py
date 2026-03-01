"""
backtester.py — Al Brooks Price Action Backtesting Engine

Runs the algo_engine against historical data and simulates trades using
Al Brooks risk management rules:
  - Stop loss: opposite side of the signal bar
  - Scalp target: 1:1 risk/reward
  - Swing target: 2:1 risk/reward

Outputs a full report: win rate, P&L, max drawdown, Sharpe ratio,
profit factor, and a complete trade log.

Usage:
    from backtester import run_backtest
    report = run_backtest(df)  # df = 5-min OHLCV DataFrame
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from algo_engine import analyze_bars, bars_from_df, compute_ema, Bar


# ─────────────────────────── TRADE MODEL ─────────────────────────────────────

@dataclass
class Trade:
    """A single simulated trade."""
    entry_bar: int
    entry_price: float
    entry_time: str
    setup_name: str
    direction: str           # "Long" or "Short"
    order_type: str          # "Stop" or "Limit"
    stop_loss: float
    scalp_target: float      # 1:1 R/R
    swing_target: float      # 2:1 R/R
    risk_per_share: float

    # Filled after trade resolves
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""    # "scalp_target", "swing_target", "stop_loss", "eod_close"
    pnl: float = 0.0
    r_multiple: float = 0.0  # P&L expressed as multiple of risk
    is_winner: bool = False
    bars_held: int = 0
    mae: float = 0.0         # Maximum Adverse Excursion (worst drawdown during trade)
    mfe: float = 0.0         # Maximum Favorable Excursion (best unrealized profit)
    mae_bar: int = 0         # Bar where MAE occurred
    mfe_bar: int = 0         # Bar where MFE occurred
    mae_r: float = 0.0       # MAE as R-multiple
    mfe_r: float = 0.0       # MFE as R-multiple


# ─────────────────────────── RISK MANAGEMENT ─────────────────────────────────

def calculate_al_brooks_levels(
    signal_bar: Bar,
    direction: str,
    tick_size: float = 0.01,
) -> dict:
    """
    Calculate stop loss and targets using Al Brooks rules.

    Long entry (buy stop above signal bar high):
      - Stop loss = signal bar low - 1 tick
      - Risk = entry - stop
      - Scalp target = entry + 1x risk (1:1)
      - Swing target = entry + 2x risk (2:1)

    Short entry (sell stop below signal bar low):
      - Stop loss = signal bar high + 1 tick
      - Risk = stop - entry
      - Scalp target = entry - 1x risk
      - Swing target = entry - 2x risk
    """
    if direction == "Long":
        entry = round(signal_bar.high + tick_size, 2)
        stop = round(signal_bar.low - tick_size, 2)
        risk = round(entry - stop, 2)
        if risk <= 0:
            risk = tick_size
        scalp = round(entry + risk, 2)
        swing = round(entry + 2 * risk, 2)
    else:  # Short
        entry = round(signal_bar.low - tick_size, 2)
        stop = round(signal_bar.high + tick_size, 2)
        risk = round(stop - entry, 2)
        if risk <= 0:
            risk = tick_size
        scalp = round(entry - risk, 2)
        swing = round(entry - 2 * risk, 2)

    return {
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "scalp_target": scalp,
        "swing_target": swing,
    }


# ─────────────────────────── TRADE SIMULATOR ─────────────────────────────────

def simulate_trade(
    trade: Trade,
    bars: list[Bar],
    timestamps: list[str],
    mode: str = "scalp",
) -> Trade:
    """
    Walk forward through bars after entry to determine trade outcome.
    Tracks MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion).

    mode: "scalp" uses 1:1 target, "swing" uses 2:1 target.
    """
    target = trade.scalp_target if mode == "scalp" else trade.swing_target
    start_idx = trade.entry_bar  # entry_bar is 1-indexed, bars list is 0-indexed

    # Track MAE/MFE through the life of the trade
    max_adverse = 0.0   # Worst unrealized loss (positive number = bad)
    max_favorable = 0.0 # Best unrealized profit
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    for i in range(start_idx, len(bars)):
        bar = bars[i]

        # Calculate unrealized P&L extremes for MAE/MFE
        if trade.direction == "Long":
            adverse = trade.entry_price - bar.low    # How far price dropped below entry
            favorable = bar.high - trade.entry_price  # How far price rose above entry
        else:
            adverse = bar.high - trade.entry_price   # How far price rose above entry (bad for short)
            favorable = trade.entry_price - bar.low   # How far price dropped below entry (good for short)

        if adverse > max_adverse:
            max_adverse = adverse
            mae_bar_idx = i
        if favorable > max_favorable:
            max_favorable = favorable
            mfe_bar_idx = i

        if trade.direction == "Long":
            # Check stop first (conservative — assumes adverse fill first)
            if bar.low <= trade.stop_loss:
                trade.exit_bar = bar.idx
                trade.exit_price = trade.stop_loss
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(trade.stop_loss - trade.entry_price, 2)
                break

            # Check target
            if bar.high >= target:
                trade.exit_bar = bar.idx
                trade.exit_price = target
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(target - trade.entry_price, 2)
                break

        else:  # Short
            # Check stop first
            if bar.high >= trade.stop_loss:
                trade.exit_bar = bar.idx
                trade.exit_price = trade.stop_loss
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(trade.entry_price - trade.stop_loss, 2)
                break

            # Check target
            if bar.low <= target:
                trade.exit_bar = bar.idx
                trade.exit_price = target
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(trade.entry_price - target, 2)
                break
    else:
        # End of day — close at last bar's close
        last = bars[-1]
        trade.exit_bar = last.idx
        trade.exit_price = last.close
        trade.exit_time = timestamps[-1] if timestamps else ""
        trade.exit_reason = "eod_close"
        if trade.direction == "Long":
            trade.pnl = round(last.close - trade.entry_price, 2)
        else:
            trade.pnl = round(trade.entry_price - last.close, 2)

    trade.is_winner = trade.pnl > 0
    trade.bars_held = trade.exit_bar - trade.entry_bar
    if trade.risk_per_share > 0:
        trade.r_multiple = round(trade.pnl / trade.risk_per_share, 2)
        trade.mae_r = round(max_adverse / trade.risk_per_share, 2)
        trade.mfe_r = round(max_favorable / trade.risk_per_share, 2)
    else:
        trade.r_multiple = 0.0
        trade.mae_r = 0.0
        trade.mfe_r = 0.0

    # Store MAE/MFE values
    trade.mae = round(max_adverse, 2)
    trade.mfe = round(max_favorable, 2)
    trade.mae_bar = bars[mae_bar_idx].idx if mae_bar_idx < len(bars) else 0
    trade.mfe_bar = bars[mfe_bar_idx].idx if mfe_bar_idx < len(bars) else 0

    return trade


# ─────────────────────────── BACKTEST RUNNER ─────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    mode: str = "scalp",
    min_bars_between_trades: int = 3,
) -> dict:
    """
    Run a full backtest on a single day of 5-min OHLCV data.

    Returns:
        dict with keys: trades, summary, equity_curve
    """
    bars = bars_from_df(df)
    if len(bars) < 10:
        return _empty_report()

    ema = compute_ema(bars)
    timestamps = [str(idx) for idx in df.index]

    # Run algo to get setups
    analysis = analyze_bars(df)
    setups = analysis.get("setups", [])

    if not setups:
        return _empty_report()

    # Convert setups to trades
    trades: list[Trade] = []
    last_entry_bar = -999

    for setup in setups:
        entry_bar_num = setup["entry_bar"]
        bar_idx = entry_bar_num - 1  # Convert to 0-indexed

        if bar_idx < 1 or bar_idx >= len(bars):
            continue

        # Enforce minimum spacing between trades
        if entry_bar_num - last_entry_bar < min_bars_between_trades:
            continue

        signal_bar = bars[bar_idx - 1]  # Signal bar is the bar BEFORE entry
        name = setup["setup_name"]

        # Determine direction from setup name
        # Count bull vs bear keyword matches to handle confluence names
        # that may contain both (e.g. "Bear Spike & Channel Bottom + Bull Flag")
        buy_keywords = ["Bull", "Bottom", "Buy", "High"]
        sell_keywords = ["Bear", "Top", "Sell", "Low"]
        bull_score = sum(name.count(kw) for kw in buy_keywords)
        bear_score = sum(name.count(kw) for kw in sell_keywords)

        if bull_score > bear_score:
            direction = "Long"
        elif bear_score > bull_score:
            direction = "Short"
        elif bull_score == bear_score and bull_score > 0:
            # Tie-breaker: check if price is above or below EMA
            direction = "Long" if bars[bar_idx].close > ema[bar_idx] else "Short"
        else:
            continue  # No directional keywords at all — skip

        # Calculate Al Brooks levels from signal bar
        levels = calculate_al_brooks_levels(signal_bar, direction)

        trade = Trade(
            entry_bar=entry_bar_num,
            entry_price=levels["entry"],
            entry_time=timestamps[bar_idx] if bar_idx < len(timestamps) else "",
            setup_name=name,
            direction=direction,
            order_type=setup.get("order_type", "Stop"),
            stop_loss=levels["stop"],
            scalp_target=levels["scalp_target"],
            swing_target=levels["swing_target"],
            risk_per_share=levels["risk"],
        )

        # Simulate the trade
        trade = simulate_trade(trade, bars, timestamps, mode=mode)
        trades.append(trade)
        last_entry_bar = entry_bar_num

    # Build report
    summary = _compute_summary(trades, mode)
    equity_curve = _build_equity_curve(trades)

    return {
        "trades": trades,
        "summary": summary,
        "equity_curve": equity_curve,
        "analysis": analysis,
    }


def run_multi_day_backtest(
    daily_dataframes: dict[str, pd.DataFrame],
    mode: str = "scalp",
    min_bars_between_trades: int = 3,
) -> dict:
    """
    Run backtest across multiple days of data.

    Args:
        daily_dataframes: dict of {date_str: DataFrame} for each trading day
        mode: "scalp" (1:1) or "swing" (2:1)

    Returns:
        Aggregated report across all days.
    """
    all_trades: list[Trade] = []
    daily_results: list[dict] = []

    for date_str, df in sorted(daily_dataframes.items()):
        result = run_backtest(df, mode, min_bars_between_trades)
        day_trades = result["trades"]
        all_trades.extend(day_trades)
        daily_results.append({
            "date": date_str,
            "trades": len(day_trades),
            "winners": sum(1 for t in day_trades if t.is_winner),
            "pnl": sum(t.pnl for t in day_trades),
            "setups_found": len(result["analysis"].get("setups", [])),
            "day_type": result["analysis"].get("day_type", "N/A"),
        })

    summary = _compute_summary(all_trades, mode)
    summary["total_days"] = len(daily_dataframes)
    summary["days_with_trades"] = sum(1 for d in daily_results if d["trades"] > 0)
    summary["daily_results"] = daily_results

    equity_curve = _build_equity_curve(all_trades)

    return {
        "trades": all_trades,
        "summary": summary,
        "equity_curve": equity_curve,
    }


# ─────────────────────────── DAILY CHART BACKTEST ─────────────────────────────

def run_daily_backtest(
    df: pd.DataFrame,
    mode: str = "swing",
    min_bars_between_trades: int = 2,
    hold_limit: int = 20,
) -> dict:
    """
    Run a backtest on DAILY bars (one bar per trading day).
    Unlike the intraday backtester, trades here span multiple days and
    there is no forced EOD close. Trades exit when they hit target, stop,
    or the hold limit (max bars to stay in a trade).

    Args:
        df: DataFrame with daily OHLCV, indexed by date
        mode: "scalp" (1:1 R/R) or "swing" (2:1 R/R)
        min_bars_between_trades: min days between entries
        hold_limit: max days to hold a single trade before forced exit

    Returns:
        dict with keys: trades, summary, equity_curve, analysis
    """
    bars = bars_from_df(df)
    if len(bars) < 20:
        return _empty_report()

    ema = compute_ema(bars)
    timestamps = [str(idx) for idx in df.index]

    # Run algo to get setups on the full daily series
    analysis = analyze_bars(df)
    setups = analysis.get("setups", [])

    if not setups:
        return _empty_report()

    # Convert setups to trades
    trades: list[Trade] = []
    last_exit_bar = -999  # Track last EXIT bar (not entry) to avoid overlapping trades

    for setup in setups:
        entry_bar_num = setup["entry_bar"]
        bar_idx = entry_bar_num - 1

        if bar_idx < 1 or bar_idx >= len(bars):
            continue

        # Don't enter while a previous trade is still open — wait until after last exit
        if entry_bar_num <= last_exit_bar:
            continue

        # Enforce minimum spacing
        if trades and entry_bar_num - trades[-1].entry_bar < min_bars_between_trades:
            continue

        signal_bar = bars[bar_idx - 1]
        name = setup["setup_name"]

        # Direction scoring (same as intraday)
        buy_keywords = ["Bull", "Bottom", "Buy", "High"]
        sell_keywords = ["Bear", "Top", "Sell", "Low"]
        bull_score = sum(name.count(kw) for kw in buy_keywords)
        bear_score = sum(name.count(kw) for kw in sell_keywords)

        if bull_score > bear_score:
            direction = "Long"
        elif bear_score > bull_score:
            direction = "Short"
        elif bull_score == bear_score and bull_score > 0:
            direction = "Long" if bars[bar_idx].close > ema[bar_idx] else "Short"
        else:
            continue

        levels = calculate_al_brooks_levels(signal_bar, direction)

        trade = Trade(
            entry_bar=entry_bar_num,
            entry_price=levels["entry"],
            entry_time=timestamps[bar_idx] if bar_idx < len(timestamps) else "",
            setup_name=name,
            direction=direction,
            order_type=setup.get("order_type", "Stop"),
            stop_loss=levels["stop"],
            scalp_target=levels["scalp_target"],
            swing_target=levels["swing_target"],
            risk_per_share=levels["risk"],
        )

        # Simulate — but with a hold limit instead of EOD close
        trade = _simulate_daily_trade(trade, bars, timestamps, mode=mode, hold_limit=hold_limit)
        trades.append(trade)
        last_exit_bar = trade.exit_bar

    summary = _compute_summary(trades, mode)
    summary["total_bars"] = len(bars)
    equity_curve = _build_equity_curve(trades)

    return {
        "trades": trades,
        "summary": summary,
        "equity_curve": equity_curve,
        "analysis": analysis,
    }


def _simulate_daily_trade(
    trade: Trade,
    bars: list[Bar],
    timestamps: list[str],
    mode: str = "swing",
    hold_limit: int = 20,
) -> Trade:
    """
    Walk forward through DAILY bars. Same logic as simulate_trade but
    instead of EOD close, we use a hold_limit (max bars in trade).
    """
    target = trade.scalp_target if mode == "scalp" else trade.swing_target
    start_idx = trade.entry_bar  # 1-indexed, bars list is 0-indexed

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    for i in range(start_idx, min(start_idx + hold_limit, len(bars))):
        bar = bars[i]

        if trade.direction == "Long":
            adverse = trade.entry_price - bar.low
            favorable = bar.high - trade.entry_price
        else:
            adverse = bar.high - trade.entry_price
            favorable = trade.entry_price - bar.low

        if adverse > max_adverse:
            max_adverse = adverse
            mae_bar_idx = i
        if favorable > max_favorable:
            max_favorable = favorable
            mfe_bar_idx = i

        if trade.direction == "Long":
            if bar.low <= trade.stop_loss:
                trade.exit_bar = bar.idx
                trade.exit_price = trade.stop_loss
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(trade.stop_loss - trade.entry_price, 2)
                break
            if bar.high >= target:
                trade.exit_bar = bar.idx
                trade.exit_price = target
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(target - trade.entry_price, 2)
                break
        else:
            if bar.high >= trade.stop_loss:
                trade.exit_bar = bar.idx
                trade.exit_price = trade.stop_loss
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(trade.entry_price - trade.stop_loss, 2)
                break
            if bar.low <= target:
                trade.exit_bar = bar.idx
                trade.exit_price = target
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(trade.entry_price - target, 2)
                break
    else:
        # Hit hold limit — close at last bar's close
        last_idx = min(start_idx + hold_limit - 1, len(bars) - 1)
        last = bars[last_idx]
        trade.exit_bar = last.idx
        trade.exit_price = last.close
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "hold_limit"
        if trade.direction == "Long":
            trade.pnl = round(last.close - trade.entry_price, 2)
        else:
            trade.pnl = round(trade.entry_price - last.close, 2)

    trade.is_winner = trade.pnl > 0
    trade.bars_held = trade.exit_bar - trade.entry_bar
    if trade.risk_per_share > 0:
        trade.r_multiple = round(trade.pnl / trade.risk_per_share, 2)
        trade.mae_r = round(max_adverse / trade.risk_per_share, 2)
        trade.mfe_r = round(max_favorable / trade.risk_per_share, 2)
    else:
        trade.r_multiple = 0.0
        trade.mae_r = 0.0
        trade.mfe_r = 0.0

    trade.mae = round(max_adverse, 2)
    trade.mfe = round(max_favorable, 2)
    trade.mae_bar = bars[mae_bar_idx].idx if mae_bar_idx < len(bars) else 0
    trade.mfe_bar = bars[mfe_bar_idx].idx if mfe_bar_idx < len(bars) else 0

    return trade


# ─────────────────────────── REPORT COMPUTATION ──────────────────────────────

def _compute_summary(trades: list[Trade], mode: str) -> dict:
    """Compute comprehensive backtest statistics."""
    if not trades:
        return _empty_summary()

    total = len(trades)
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]
    win_count = len(winners)
    loss_count = len(losers)

    win_rate = win_count / total if total > 0 else 0.0
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / total if total > 0 else 0.0

    avg_win = np.mean([t.pnl for t in winners]) if winners else 0.0
    avg_loss = np.mean([t.pnl for t in losers]) if losers else 0.0
    largest_win = max((t.pnl for t in winners), default=0.0)
    largest_loss = min((t.pnl for t in losers), default=0.0)

    # Profit factor: gross profit / gross loss
    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # R-multiple stats
    avg_r = np.mean([t.r_multiple for t in trades])
    total_r = sum(t.r_multiple for t in trades)

    # Max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (using per-trade returns)
    if len(trades) > 1:
        returns = [t.pnl for t in trades]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        # Annualize: assume ~252 trading days, ~3 trades/day avg
        trades_per_year = 252 * max(total / max(len(set(t.entry_time[:10] for t in trades if t.entry_time)), 1), 1)
        sharpe_annualized = sharpe * math.sqrt(min(trades_per_year, 756))
    else:
        sharpe = 0.0
        sharpe_annualized = 0.0

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    streak_type = None
    for t in trades:
        if t.is_winner:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    # Average bars held
    avg_bars_held = np.mean([t.bars_held for t in trades])

    # MAE / MFE statistics
    avg_mae = float(np.mean([t.mae for t in trades]))
    avg_mfe = float(np.mean([t.mfe for t in trades]))
    avg_mae_r = float(np.mean([t.mae_r for t in trades]))
    avg_mfe_r = float(np.mean([t.mfe_r for t in trades]))
    # MAE/MFE by outcome
    avg_mae_winners = float(np.mean([t.mae for t in winners])) if winners else 0.0
    avg_mae_losers = float(np.mean([t.mae for t in losers])) if losers else 0.0
    avg_mfe_winners = float(np.mean([t.mfe for t in winners])) if winners else 0.0
    avg_mfe_losers = float(np.mean([t.mfe for t in losers])) if losers else 0.0
    # Edge ratio: MFE/MAE — how much the market goes for you vs against you
    edge_ratio = avg_mfe / avg_mae if avg_mae > 0 else float("inf")

    # Expectancy per trade (avg_pnl already computed above)
    expectancy = avg_pnl
    # Kelly criterion: W - (1-W)/payoff_ratio
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    kelly = (win_rate - (1 - win_rate) / payoff_ratio) if payoff_ratio != float("inf") and payoff_ratio > 0 else 0.0
    # Recovery factor: total_pnl / max_drawdown
    recovery_factor = total_pnl / max_dd if max_dd > 0 else float("inf")
    # CPC ratio (Cost Per Contract / reward-to-risk): (win_rate * avg_win) / (loss_rate * abs(avg_loss))
    loss_rate = 1 - win_rate
    cpc_ratio = (win_rate * avg_win) / (loss_rate * abs(avg_loss)) if loss_rate > 0 and avg_loss != 0 else float("inf")

    # Direction breakdown
    longs = [t for t in trades if t.direction == "Long"]
    shorts = [t for t in trades if t.direction == "Short"]
    long_stats = {
        "count": len(longs),
        "wins": sum(1 for t in longs if t.is_winner),
        "pnl": round(sum(t.pnl for t in longs), 2),
        "win_rate": round(sum(1 for t in longs if t.is_winner) / len(longs), 3) if longs else 0.0,
    }
    short_stats = {
        "count": len(shorts),
        "wins": sum(1 for t in shorts if t.is_winner),
        "pnl": round(sum(t.pnl for t in shorts), 2),
        "win_rate": round(sum(1 for t in shorts if t.is_winner) / len(shorts), 3) if shorts else 0.0,
    }

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Setup breakdown
    setup_stats = {}
    for t in trades:
        name = t.setup_name
        if name not in setup_stats:
            setup_stats[name] = {
                "count": 0, "wins": 0, "losses": 0,
                "pnl": 0.0, "total_r": 0.0,
                "win_pnl": 0.0, "loss_pnl": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0,
            }
        s = setup_stats[name]
        s["count"] += 1
        if t.is_winner:
            s["wins"] += 1
            s["win_pnl"] = round(s["win_pnl"] + t.pnl, 2)
        else:
            s["losses"] += 1
            s["loss_pnl"] = round(s["loss_pnl"] + t.pnl, 2)
        s["pnl"] = round(s["pnl"] + t.pnl, 2)
        s["total_r"] = round(s["total_r"] + t.r_multiple, 2)
        s["best_trade"] = max(s["best_trade"], t.pnl)
        s["worst_trade"] = min(s["worst_trade"], t.pnl)

    # Add derived stats to each setup
    for name, s in setup_stats.items():
        s["win_rate"] = round(s["wins"] / s["count"], 3) if s["count"] > 0 else 0.0
        s["avg_pnl"] = round(s["pnl"] / s["count"], 2) if s["count"] > 0 else 0.0
        s["avg_r"] = round(s["total_r"] / s["count"], 2) if s["count"] > 0 else 0.0
        s["profit_factor"] = round(s["win_pnl"] / abs(s["loss_pnl"]), 2) if s["loss_pnl"] != 0 else float('inf') if s["win_pnl"] > 0 else 0.0

    return {
        "mode": mode,
        "total_trades": total,
        "winners": win_count,
        "losers": loss_count,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "avg_winner": round(float(avg_win), 2),
        "avg_loser": round(float(avg_loss), 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_r_multiple": round(float(avg_r), 2),
        "total_r": round(total_r, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sharpe_annualized": round(sharpe_annualized, 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_bars_held": round(float(avg_bars_held), 1),
        "avg_mae": round(avg_mae, 2),
        "avg_mfe": round(avg_mfe, 2),
        "avg_mae_r": round(avg_mae_r, 2),
        "avg_mfe_r": round(avg_mfe_r, 2),
        "avg_mae_winners": round(avg_mae_winners, 2),
        "avg_mae_losers": round(avg_mae_losers, 2),
        "avg_mfe_winners": round(avg_mfe_winners, 2),
        "avg_mfe_losers": round(avg_mfe_losers, 2),
        "edge_ratio": round(edge_ratio, 2),
        "expectancy": round(expectancy, 4),
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float("inf") else 999.0,
        "kelly_pct": round(kelly * 100, 1),
        "recovery_factor": round(recovery_factor, 2) if recovery_factor != float("inf") else 999.0,
        "cpc_ratio": round(cpc_ratio, 2) if cpc_ratio != float("inf") else 999.0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "long_stats": long_stats,
        "short_stats": short_stats,
        "exit_reasons": exit_reasons,
        "setup_stats": setup_stats,
    }


def _build_equity_curve(trades: list[Trade]) -> list[dict]:
    """Build an equity curve from trade sequence."""
    curve = [{"trade_num": 0, "equity": 0.0, "pnl": 0.0}]
    equity = 0.0
    for i, t in enumerate(trades):
        equity += t.pnl
        curve.append({
            "trade_num": i + 1,
            "equity": round(equity, 2),
            "pnl": round(t.pnl, 2),
            "setup": t.setup_name,
            "direction": t.direction,
            "r_multiple": t.r_multiple,
        })
    return curve


def _empty_report() -> dict:
    return {
        "trades": [],
        "summary": _empty_summary(),
        "equity_curve": [{"trade_num": 0, "equity": 0.0, "pnl": 0.0}],
        "analysis": {},
    }


def _empty_summary() -> dict:
    return {
        "mode": "N/A",
        "total_trades": 0,
        "winners": 0,
        "losers": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_pnl_per_trade": 0.0,
        "avg_winner": 0.0,
        "avg_loser": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "profit_factor": 0.0,
        "avg_r_multiple": 0.0,
        "total_r": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "sharpe_annualized": 0.0,
        "max_win_streak": 0,
        "max_loss_streak": 0,
        "avg_bars_held": 0.0,
        "avg_mae": 0.0,
        "avg_mfe": 0.0,
        "avg_mae_r": 0.0,
        "avg_mfe_r": 0.0,
        "avg_mae_winners": 0.0,
        "avg_mae_losers": 0.0,
        "avg_mfe_winners": 0.0,
        "avg_mfe_losers": 0.0,
        "edge_ratio": 0.0,
        "expectancy": 0.0,
        "payoff_ratio": 0.0,
        "kelly_pct": 0.0,
        "recovery_factor": 0.0,
        "cpc_ratio": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "long_stats": {"count": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0},
        "short_stats": {"count": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0},
        "exit_reasons": {},
        "setup_stats": {},
    }


# ─────────────────────────── TRADE LOG EXPORT ────────────────────────────────

def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert trade list to a DataFrame for export."""
    rows = []
    for t in trades:
        rows.append({
            "Setup": t.setup_name,
            "Direction": t.direction,
            "Entry Bar": t.entry_bar,
            "Entry Price": t.entry_price,
            "Entry Time": t.entry_time,
            "Stop Loss": t.stop_loss,
            "Scalp Target": t.scalp_target,
            "Swing Target": t.swing_target,
            "Risk/Share": t.risk_per_share,
            "Exit Bar": t.exit_bar,
            "Exit Price": t.exit_price,
            "Exit Time": t.exit_time,
            "Exit Reason": t.exit_reason,
            "P&L": t.pnl,
            "R Multiple": t.r_multiple,
            "Winner": t.is_winner,
            "Bars Held": t.bars_held,
            "MAE": t.mae,
            "MFE": t.mfe,
            "MAE (R)": t.mae_r,
            "MFE (R)": t.mfe_r,
            "MAE Bar": t.mae_bar,
            "MFE Bar": t.mfe_bar,
        })
    return pd.DataFrame(rows)


def export_trade_log_csv(trades: list[Trade], path: str = "backtest_trades.csv"):
    """Export trade log to CSV file."""
    df = trades_to_dataframe(trades)
    df.to_csv(path, index=False)
    return path


# ─────────────────────────── CLI TEST ────────────────────────────────────────

if __name__ == "__main__":
    import time

    # Quick test with sample data
    print("Backtester ready. Import run_backtest(df) to use.")
    print("Or run: python backtester.py <ticker>")

    import sys
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        try:
            import yfinance as yf
            print(f"Fetching 5-min data for {ticker}...")
            df = yf.download(ticker, period="1d", interval="5m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            start = time.perf_counter()
            report = run_backtest(df, mode="scalp")
            elapsed = (time.perf_counter() - start) * 1000

            s = report["summary"]
            print(f"\n⚡ Backtest completed in {elapsed:.1f}ms")
            print(f"Total Trades: {s['total_trades']}")
            print(f"Win Rate: {s['win_rate']:.1%}")
            print(f"Total P&L: ${s['total_pnl']:.2f}/share")
            print(f"Profit Factor: {s['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {s['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: ${s['max_drawdown']:.2f}/share")
            print(f"Avg R-Multiple: {s['avg_r_multiple']:.2f}R")
            print(f"\nSetup Breakdown:")
            for name, stats in s["setup_stats"].items():
                print(f"  {name}: {stats['count']} trades, {stats['win_rate']:.0%} win rate, ${stats['pnl']:.2f} P&L")
            print(f"\nTrades:")
            for t in report["trades"]:
                icon = "✅" if t.is_winner else "❌"
                print(f"  {icon} {t.setup_name} ({t.direction}) bar {t.entry_bar} → {t.exit_reason}: ${t.pnl:+.2f} ({t.r_multiple:+.1f}R)")
        except ImportError:
            print("yfinance not installed — provide a DataFrame directly.")
