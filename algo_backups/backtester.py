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
    exit_reason: str = ""    # "scalp_target", "swing_target", "stop_loss", "eod_close", "unfilled"
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

    # Context fields — filled during backtest for trade review
    ticker: str = ""
    day_type: str = ""           # e.g. "Trend Day", "Trading Range Day"
    market_cycle: str = ""       # e.g. "Bull Trend", "Bear Pullback"
    confidence: float = 0.0      # Setup confidence score (0-1)
    ema_position: str = ""       # "Above EMA" or "Below EMA" at entry
    with_trend: bool = False     # Was trade in direction of the trend?
    gap_day: bool = False        # Was there a significant opening gap?
    num_setups_on_bar: int = 1   # How many setups fired at same bar (confluence)


# ─────────────────────────── FADE SETUPS ─────────────────────────────────────
# Setups whose original direction is wrong 80%+ of the time at 1:1 R/R.
# The backtester flips these to trade the opposite direction and prefixes
# the name with "Fade " so they are clearly identifiable.
FADE_SETUPS = {
    "Lower Low Double Bottom",
    "Higher High Double Top",
    "Consecutive Buy Climaxes (Reversal)",
    "Consecutive Sell Climaxes (Reversal)",
    "Exhaustive Bear Climax at MM",
    "Exhaustive Bull Climax at MM",
    "Weak Bull Breakout Test",
    "Weak Bear Breakout Test",
    "Bull Breakout Pullback",
    "Quiet Bear Flag at MA",
    "Quiet Bull Flag at MA",
    "Bear Stairs Reversal (3rd/4th Push)",
}


# ─────────────────────────── SETUP CONFIG (DATA-DRIVEN R:R) ────────────────────
# Each profitable setup mapped to its optimal R:R ratio (net of commissions + spread).
# Derived from 60 days of 5-min data across 20 liquid tickers (~17k trades).
# Setups not listed here were net-negative after costs.
SETUP_CONFIG = {
    # Tier A fades — best at 1:1
    "Fade Lower Low Double Bottom":              {"rr": 1.0, "profitable": True},
    "Fade Higher High Double Top":               {"rr": 1.0, "profitable": True},
    "Fade Consecutive Sell Climaxes (Reversal)":  {"rr": 1.0, "profitable": True},
    "Fade Consecutive Buy Climaxes (Reversal)":   {"rr": 1.0, "profitable": True},
    "Fade Bull Breakout Pullback":               {"rr": 1.0, "profitable": True},
    "Fade Bear Stairs Reversal (3rd/4th Push)":   {"rr": 1.0, "profitable": True},
    # Tier B — standard setups at 1:1 or 0.75:1
    "Double Top":                                {"rr": 1.0, "profitable": True},
    "Double Bottom":                             {"rr": 0.75, "profitable": True},
    "Lower High Double Top":                     {"rr": 1.0, "profitable": True},
    "Fade Exhaustive Bull Climax at MM":          {"rr": 0.75, "profitable": True},
    # Tier C — scalps at 0.5:1
    "Higher Low Double Bottom":                  {"rr": 1.0, "profitable": True},
    "Wedge Bottom":                              {"rr": 0.5, "profitable": True},
    "Wedge Top":                                 {"rr": 0.5, "profitable": True},
    # Removed: Fade Weak Bull/Bear Breakout Test — negative EV in backtests
}


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

def _check_limit_order_fill(
    trade: Trade,
    bars: list[Bar],
    look_ahead: int = 3,
) -> tuple[bool, int]:
    """
    Check if a limit order gets filled within look_ahead bars.

    For Long: bar's LOW must touch or go below the limit entry price.
    For Short: bar's HIGH must touch or go above the limit entry price.

    Returns: (is_filled, bar_index)
    bar_index is the index where it was filled, or -1 if not filled.
    """
    start_idx = trade.entry_bar
    end_idx = min(start_idx + look_ahead, len(bars))

    for i in range(start_idx, end_idx):
        bar = bars[i]
        if trade.direction == "Long":
            # For long limit, price must come DOWN to the buy price
            if bar.low <= trade.entry_price:
                return (True, i)
        else:  # Short
            # For short limit, price must come UP to the sell price
            if bar.high >= trade.entry_price:
                return (True, i)

    return (False, -1)


def simulate_trade(
    trade: Trade,
    bars: list[Bar],
    timestamps: list[str],
    mode: str = "scalp",
    slippage: float = 0.0,
    commission: float = 0.0,
) -> Trade:
    """
    Walk forward through bars after entry to determine trade outcome.
    Tracks MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion).

    For limit orders: checks if price touches the entry within 3 bars.
    For stop orders: uses existing logic.

    Applies slippage and commission to entry and exit prices.

    mode: "scalp" uses 1:1 target, "swing" uses 2:1 target.
    slippage: dollars per share, applied to entry and exit (worse fill)
    commission: dollars per share per side (total = 2 * commission * shares)
    """
    target = trade.scalp_target if mode == "scalp" else trade.swing_target
    start_idx = trade.entry_bar  # entry_bar is 1-indexed, bars list is 0-indexed

    # ─── LIMIT ORDER FILL CHECK ───
    if trade.order_type == "Limit":
        filled, fill_idx = _check_limit_order_fill(trade, bars, look_ahead=3)
        if not filled:
            # Mark as unfilled
            trade.exit_reason = "unfilled"
            trade.pnl = 0.0
            trade.r_multiple = 0.0
            trade.is_winner = False
            trade.bars_held = 0
            trade.mae = 0.0
            trade.mfe = 0.0
            trade.mae_bar = 0
            trade.mfe_bar = 0
            trade.mae_r = 0.0
            trade.mfe_r = 0.0
            return trade
        start_idx = fill_idx  # Start tracking from fill bar

    # Track MAE/MFE through the life of the trade
    max_adverse = 0.0   # Worst unrealized loss (positive number = bad)
    max_favorable = 0.0 # Best unrealized profit
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    # Apply slippage to entry price
    adjusted_entry = trade.entry_price
    if trade.direction == "Long":
        adjusted_entry += slippage  # Long: slippage makes entry worse (higher)
    else:
        adjusted_entry -= slippage  # Short: slippage makes entry worse (lower)

    for i in range(start_idx, len(bars)):
        bar = bars[i]

        # Calculate unrealized P&L extremes for MAE/MFE
        if trade.direction == "Long":
            adverse = adjusted_entry - bar.low    # How far price dropped below entry
            favorable = bar.high - adjusted_entry  # How far price rose above entry
        else:
            adverse = bar.high - adjusted_entry   # How far price rose above entry (bad for short)
            favorable = adjusted_entry - bar.low   # How far price dropped below entry (good for short)

        if adverse > max_adverse:
            max_adverse = adverse
            mae_bar_idx = i
        if favorable > max_favorable:
            max_favorable = favorable
            mfe_bar_idx = i

        if trade.direction == "Long":
            # Check stop first (conservative — assumes adverse fill first)
            if bar.low <= trade.stop_loss:
                exit_price = trade.stop_loss - slippage  # Slippage worsens exit
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(exit_price - adjusted_entry, 2)
                break

            # Check target
            if bar.high >= target:
                exit_price = target - slippage  # Slippage worsens exit
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(exit_price - adjusted_entry, 2)
                break

        else:  # Short
            # Check stop first
            if bar.high >= trade.stop_loss:
                exit_price = trade.stop_loss + slippage  # Slippage worsens exit
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(adjusted_entry - exit_price, 2)
                break

            # Check target
            if bar.low <= target:
                exit_price = target + slippage  # Slippage worsens exit
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(adjusted_entry - exit_price, 2)
                break
    else:
        # End of day — close at last bar's close
        last = bars[-1]
        exit_price = last.close
        if trade.direction == "Long":
            exit_price -= slippage  # Slippage worsens exit
        else:
            exit_price += slippage  # Slippage worsens exit

        trade.exit_bar = last.idx
        trade.exit_price = round(exit_price, 2)
        trade.exit_time = timestamps[-1] if timestamps else ""
        trade.exit_reason = "eod_close"
        if trade.direction == "Long":
            trade.pnl = round(exit_price - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - exit_price, 2)

    # Apply commission (2 sides)
    commission_cost = 2 * commission
    trade.pnl = round(trade.pnl - commission_cost, 2)

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
    slippage: float = 0.0,
    commission: float = 0.0,
    ticker: str = "",
    profitable_only: bool = False,
    use_setup_config: bool = True,
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

    # Sort chronologically for sequential simulation
    setups.sort(key=lambda s: s["entry_bar"])

    # Convert setups to trades
    trades: list[Trade] = []
    last_entry_bar = -999

    for setup in setups:
        entry_bar_num = setup["entry_bar"]
        bar_idx = entry_bar_num - 1  # Convert to 0-indexed

        if bar_idx < 0 or bar_idx >= len(bars):
            continue

        # Enforce minimum spacing between trades
        if entry_bar_num - last_entry_bar < min_bars_between_trades:
            continue

        # The signal bar is the bar AT entry_bar (the bar that defines the setup).
        # The actual trade entry happens on the NEXT bar (entry_bar + 1).
        signal_bar = bars[bar_idx]
        actual_entry_bar = entry_bar_num + 1
        actual_bar_idx = bar_idx + 1

        if actual_bar_idx >= len(bars):
            continue

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
            direction = "Long" if bars[actual_bar_idx].close > ema[actual_bar_idx] else "Short"
        else:
            continue  # No directional keywords at all — skip

        # Calculate Al Brooks levels from signal bar in ORIGINAL direction
        levels = calculate_al_brooks_levels(signal_bar, direction)

        # Fade logic: keep same entry/stop/target prices, just take the other side.
        # Original long: entry above bar, stop below, target above entry
        # Fade: sell at same entry, target = original stop, stop = original target
        is_fade = name in FADE_SETUPS
        if is_fade:
            direction = "Short" if direction == "Long" else "Long"
            name = f"Fade {name}"
            # Swap stop and scalp target (the original stop becomes the fade target)
            old_stop = levels["stop"]
            old_scalp = levels["scalp_target"]
            old_swing = levels["swing_target"]
            levels["stop"] = old_scalp          # original target is now the stop
            levels["scalp_target"] = old_stop   # original stop is now the target
            levels["swing_target"] = old_stop   # same for swing (1:1 fade)
            # Risk is now distance from entry to new stop (the old target)
            levels["risk"] = abs(levels["stop"] - levels["entry"])

        # Filter: profitable_only mode skips setups not in SETUP_CONFIG
        if profitable_only and use_setup_config:
            if name not in SETUP_CONFIG or not SETUP_CONFIG[name].get("profitable", False):
                continue

        # Apply setup-specific R:R from config (overrides default 1:1 scalp target)
        if use_setup_config and name in SETUP_CONFIG:
            cfg_rr = SETUP_CONFIG[name]["rr"]
            risk = levels["risk"]
            if direction == "Long":
                levels["scalp_target"] = round(levels["entry"] + cfg_rr * risk, 2)
            else:
                levels["scalp_target"] = round(levels["entry"] - cfg_rr * risk, 2)
            # Swing target stays at 2:1 for optionality

        # Context: EMA position at entry
        entry_close = bars[actual_bar_idx].close
        entry_ema = ema[actual_bar_idx] if actual_bar_idx < len(ema) else 0
        ema_pos = "Above EMA" if entry_close > entry_ema else "Below EMA"

        # Context: is this trade with-trend?
        # Simple heuristic: Long above EMA = with trend, Short below EMA = with trend
        with_trend = (direction == "Long" and entry_close > entry_ema) or \
                     (direction == "Short" and entry_close < entry_ema)

        # Context: gap day detection (compare bar 0 open to prior close if available)
        gap_day = False
        if len(bars) > 1:
            gap = abs(bars[0].open - bars[0].close) / max(bars[0].close, 0.01)
            # Check if first bar gapped significantly from its own range
            first_bar_range = bars[0].high - bars[0].low
            avg_range = sum(b.high - b.low for b in bars[:min(10, len(bars))]) / min(10, len(bars))
            if first_bar_range > avg_range * 1.5:
                gap_day = True

        trade = Trade(
            entry_bar=actual_entry_bar,
            entry_price=levels["entry"],
            entry_time=timestamps[actual_bar_idx] if actual_bar_idx < len(timestamps) else "",
            setup_name=name,
            direction=direction,
            order_type=setup.get("order_type", "Stop"),
            stop_loss=levels["stop"],
            scalp_target=levels["scalp_target"],
            swing_target=levels["swing_target"],
            risk_per_share=levels["risk"],
            # Context
            day_type=analysis.get("day_type", ""),
            market_cycle=analysis.get("market_cycle", ""),
            confidence=setup.get("confidence", 0.0),
            ema_position=ema_pos,
            with_trend=with_trend,
            gap_day=gap_day,
            num_setups_on_bar=setup.get("num_setups_on_bar", 1),
            ticker=ticker,
        )

        # Simulate the trade
        trade = simulate_trade(trade, bars, timestamps, mode=mode, slippage=slippage, commission=commission)
        trades.append(trade)
        last_entry_bar = actual_entry_bar

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
    slippage: float = 0.0,
    commission: float = 0.0,
    ticker: str = "",
    profitable_only: bool = False,
    use_setup_config: bool = True,
) -> dict:
    """
    Run backtest across multiple days of data.

    Args:
        daily_dataframes: dict of {date_str: DataFrame} for each trading day
        mode: "scalp" (1:1) or "swing" (2:1)
        slippage: dollars per share applied to entries and exits
        commission: dollars per share per side

    Returns:
        Aggregated report across all days.
    """
    all_trades: list[Trade] = []
    daily_results: list[dict] = []

    for date_str, df in sorted(daily_dataframes.items()):
        result = run_backtest(df, mode, min_bars_between_trades, slippage, commission, ticker=ticker, profitable_only=profitable_only, use_setup_config=use_setup_config)
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
    slippage: float = 0.0,
    commission: float = 0.0,
    ticker: str = "",
    profitable_only: bool = False,
    use_setup_config: bool = True,
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
        slippage: dollars per share applied to entries and exits
        commission: dollars per share per side

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

    # CRITICAL: sort setups chronologically so the backtester walks forward
    # (analyze_bars returns them sorted by confidence, which breaks sequential simulation)
    setups.sort(key=lambda s: s["entry_bar"])

    # Convert setups to trades
    trades: list[Trade] = []
    last_exit_bar = -999  # Track last EXIT bar (not entry) to avoid overlapping trades

    for setup in setups:
        entry_bar_num = setup["entry_bar"]
        bar_idx = entry_bar_num - 1  # Convert to 0-indexed

        if bar_idx < 0 or bar_idx >= len(bars):
            continue

        # The signal bar is the bar AT entry_bar (the bar that defines the setup).
        # The actual trade entry happens on the NEXT bar (entry_bar + 1).
        signal_bar = bars[bar_idx]
        actual_entry_bar = entry_bar_num + 1
        actual_bar_idx = bar_idx + 1

        if actual_bar_idx >= len(bars):
            continue

        # Don't enter while a previous trade is still open — wait until after last exit
        if actual_entry_bar <= last_exit_bar:
            continue

        # Enforce minimum spacing
        if trades and actual_entry_bar - trades[-1].entry_bar < min_bars_between_trades:
            continue

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
            direction = "Long" if bars[actual_bar_idx].close > ema[actual_bar_idx] else "Short"
        else:
            continue

        # Calculate Al Brooks levels from signal bar in ORIGINAL direction
        levels = calculate_al_brooks_levels(signal_bar, direction)

        # Fade logic: keep same prices, take the other side
        is_fade = name in FADE_SETUPS
        if is_fade:
            direction = "Short" if direction == "Long" else "Long"
            name = f"Fade {name}"
            old_stop = levels["stop"]
            old_scalp = levels["scalp_target"]
            levels["stop"] = old_scalp
            levels["scalp_target"] = old_stop
            levels["swing_target"] = old_stop
            levels["risk"] = abs(levels["stop"] - levels["entry"])

        # Filter: profitable_only mode skips setups not in SETUP_CONFIG
        if profitable_only and use_setup_config:
            if name not in SETUP_CONFIG or not SETUP_CONFIG[name].get("profitable", False):
                continue

        # Apply setup-specific R:R from config (overrides default target)
        if use_setup_config and name in SETUP_CONFIG:
            cfg_rr = SETUP_CONFIG[name]["rr"]
            risk = levels["risk"]
            if direction == "Long":
                levels["scalp_target"] = round(levels["entry"] + cfg_rr * risk, 2)
            else:
                levels["scalp_target"] = round(levels["entry"] - cfg_rr * risk, 2)

        # Context: EMA position at entry
        entry_close = bars[actual_bar_idx].close
        entry_ema = ema[actual_bar_idx] if actual_bar_idx < len(ema) else 0
        ema_pos = "Above EMA" if entry_close > entry_ema else "Below EMA"
        with_trend = (direction == "Long" and entry_close > entry_ema) or \
                     (direction == "Short" and entry_close < entry_ema)

        trade = Trade(
            entry_bar=actual_entry_bar,
            entry_price=levels["entry"],
            entry_time=timestamps[actual_bar_idx] if actual_bar_idx < len(timestamps) else "",
            setup_name=name,
            direction=direction,
            order_type=setup.get("order_type", "Stop"),
            stop_loss=levels["stop"],
            scalp_target=levels["scalp_target"],
            swing_target=levels["swing_target"],
            risk_per_share=levels["risk"],
            # Context
            ticker=ticker,
            day_type=analysis.get("day_type", ""),
            market_cycle=analysis.get("market_cycle", ""),
            confidence=setup.get("confidence", 0.0),
            ema_position=ema_pos,
            with_trend=with_trend,
            num_setups_on_bar=setup.get("num_setups_on_bar", 1),
        )

        # Simulate — but with a hold limit instead of EOD close
        trade = _simulate_daily_trade(trade, bars, timestamps, mode=mode, hold_limit=hold_limit, slippage=slippage, commission=commission)
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
    slippage: float = 0.0,
    commission: float = 0.0,
) -> Trade:
    """
    Walk forward through DAILY bars. Same logic as simulate_trade but
    instead of EOD close, we use a hold_limit (max bars in trade).

    For limit orders: checks if price touches the entry within 3 bars.
    Applies slippage and commission.
    """
    target = trade.scalp_target if mode == "scalp" else trade.swing_target
    start_idx = trade.entry_bar  # 1-indexed, bars list is 0-indexed

    # ─── LIMIT ORDER FILL CHECK ───
    if trade.order_type == "Limit":
        filled, fill_idx = _check_limit_order_fill(trade, bars, look_ahead=3)
        if not filled:
            # Mark as unfilled
            trade.exit_reason = "unfilled"
            trade.pnl = 0.0
            trade.r_multiple = 0.0
            trade.is_winner = False
            trade.bars_held = 0
            trade.mae = 0.0
            trade.mfe = 0.0
            trade.mae_bar = 0
            trade.mfe_bar = 0
            trade.mae_r = 0.0
            trade.mfe_r = 0.0
            return trade
        start_idx = fill_idx  # Start tracking from fill bar

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    # Apply slippage to entry price
    adjusted_entry = trade.entry_price
    if trade.direction == "Long":
        adjusted_entry += slippage  # Long: slippage makes entry worse (higher)
    else:
        adjusted_entry -= slippage  # Short: slippage makes entry worse (lower)

    for i in range(start_idx, min(start_idx + hold_limit, len(bars))):
        bar = bars[i]

        if trade.direction == "Long":
            adverse = adjusted_entry - bar.low
            favorable = bar.high - adjusted_entry
        else:
            adverse = bar.high - adjusted_entry
            favorable = adjusted_entry - bar.low

        if adverse > max_adverse:
            max_adverse = adverse
            mae_bar_idx = i
        if favorable > max_favorable:
            max_favorable = favorable
            mfe_bar_idx = i

        if trade.direction == "Long":
            if bar.low <= trade.stop_loss:
                exit_price = trade.stop_loss - slippage
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(exit_price - adjusted_entry, 2)
                break
            if bar.high >= target:
                exit_price = target - slippage
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(exit_price - adjusted_entry, 2)
                break
        else:
            if bar.high >= trade.stop_loss:
                exit_price = trade.stop_loss + slippage
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "stop_loss"
                trade.pnl = round(adjusted_entry - exit_price, 2)
                break
            if bar.low <= target:
                exit_price = target + slippage
                trade.exit_bar = bar.idx
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = f"{mode}_target"
                trade.pnl = round(adjusted_entry - exit_price, 2)
                break
    else:
        # Hit hold limit — close at last bar's close
        last_idx = min(start_idx + hold_limit - 1, len(bars) - 1)
        last = bars[last_idx]
        exit_price = last.close
        if trade.direction == "Long":
            exit_price -= slippage
        else:
            exit_price += slippage

        trade.exit_bar = last.idx
        trade.exit_price = round(exit_price, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "hold_limit"
        if trade.direction == "Long":
            trade.pnl = round(exit_price - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - exit_price, 2)

    # Apply commission (2 sides)
    commission_cost = 2 * commission
    trade.pnl = round(trade.pnl - commission_cost, 2)

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


# ─────────────────────────── MONTE CARLO & RISK ANALYSIS ─────────────────────

def run_monte_carlo(
    trades: list[Trade],
    n_simulations: int = 1000,
    starting_capital: float = 10000.0,
) -> dict:
    """
    Run Monte Carlo simulation on trade sequence to estimate distribution of outcomes.

    Randomly shuffles trade order n_simulations times and computes equity curves.
    Tracks final equity, max drawdown, and drawdown duration for each simulation.

    Args:
        trades: list of completed Trade objects
        n_simulations: number of random shuffles to run
        starting_capital: starting account balance

    Returns:
        dict with percentiles, risk metrics, and all equity/drawdown values
    """
    if not trades:
        return {
            "median_final_equity": starting_capital,
            "p5_final_equity": starting_capital,
            "p95_final_equity": starting_capital,
            "median_max_dd": 0.0,
            "p95_max_dd": 0.0,
            "risk_of_ruin_pct": 0.0,
            "avg_max_dd_duration": 0,
            "all_final_equities": [starting_capital],
            "all_max_drawdowns": [0.0],
        }

    all_final_equities = []
    all_max_drawdowns = []
    all_dd_durations = []
    ruin_count = 0
    ruin_threshold = starting_capital * 0.5

    for _ in range(n_simulations):
        # Shuffle trades randomly
        shuffled_trades = trades.copy()
        np.random.shuffle(shuffled_trades)

        # Build equity curve for this shuffle
        equity_curve = _build_equity_curve(shuffled_trades, starting_capital)

        # Extract final equity
        final_equity = equity_curve[-1]["equity"] if equity_curve else starting_capital
        all_final_equities.append(final_equity)

        # Track if ruin occurred (dropped below 50% of starting capital)
        if final_equity < ruin_threshold:
            ruin_count += 1

        # Calculate max drawdown and duration
        max_dd = 0.0
        peak_equity = starting_capital
        dd_duration = 0
        max_dd_duration = 0
        current_dd_duration = 0

        for point in equity_curve[1:]:  # Skip initial point
            equity = point["equity"]
            if equity > peak_equity:
                peak_equity = equity
                # Reset duration counters if we broke the drawdown
                if current_dd_duration > 0:
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                    current_dd_duration = 0

            dd = peak_equity - equity
            if dd > 0:
                current_dd_duration += 1

            if dd > max_dd:
                max_dd = dd

        max_dd_duration = max(max_dd_duration, current_dd_duration)
        all_max_drawdowns.append(max_dd)
        all_dd_durations.append(max_dd_duration)

    # Compute percentiles
    final_equities_sorted = sorted(all_final_equities)
    drawdowns_sorted = sorted(all_max_drawdowns)

    p5_idx = max(0, int(len(final_equities_sorted) * 0.05))
    p95_idx = min(len(final_equities_sorted) - 1, int(len(final_equities_sorted) * 0.95))

    p5_final = final_equities_sorted[p5_idx]
    p95_final = final_equities_sorted[p95_idx]
    median_final = final_equities_sorted[len(final_equities_sorted) // 2]

    p95_dd_idx = min(len(drawdowns_sorted) - 1, int(len(drawdowns_sorted) * 0.95))
    median_max_dd = drawdowns_sorted[len(drawdowns_sorted) // 2]
    p95_max_dd = drawdowns_sorted[p95_dd_idx]

    avg_dd_duration = int(np.mean(all_dd_durations)) if all_dd_durations else 0
    risk_of_ruin_pct = (ruin_count / n_simulations) * 100 if n_simulations > 0 else 0.0

    return {
        "median_final_equity": round(median_final, 2),
        "p5_final_equity": round(p5_final, 2),
        "p95_final_equity": round(p95_final, 2),
        "median_max_dd": round(median_max_dd, 2),
        "p95_max_dd": round(p95_max_dd, 2),
        "risk_of_ruin_pct": round(risk_of_ruin_pct, 1),
        "avg_max_dd_duration": avg_dd_duration,
        "all_final_equities": [round(x, 2) for x in all_final_equities],
        "all_max_drawdowns": [round(x, 2) for x in all_max_drawdowns],
    }


# ─────────────────────────── WALK-FORWARD TESTING ──────────────────────────────

def run_walk_forward(
    trades: list[Trade],
    n_folds: int = 5,
    mode: str = "swing",
) -> dict:
    """
    Run walk-forward analysis to test robustness out-of-sample.

    Splits trades chronologically into n_folds. For each fold:
      - Uses all OTHER folds as "in-sample"
      - Uses this fold as "out-of-sample"

    Compares performance metrics IS vs OOS to detect overfitting.

    Args:
        trades: list of completed Trade objects (should be in chronological order)
        n_folds: number of folds for cross-validation
        mode: "scalp" or "swing" for target mode

    Returns:
        dict with fold results and robustness metrics
    """
    if not trades or n_folds < 2:
        return {
            "folds": [],
            "avg_is_win_rate": 0.0,
            "avg_oos_win_rate": 0.0,
            "degradation_pct": 0.0,
            "avg_is_pf": 1.0,
            "avg_oos_pf": 1.0,
            "is_robust": False,
        }

    # Split trades into folds chronologically
    fold_size = len(trades) // n_folds
    folds = []
    for i in range(n_folds):
        if i == n_folds - 1:
            # Last fold gets remaining trades
            fold = trades[i * fold_size:]
        else:
            fold = trades[i * fold_size:(i + 1) * fold_size]
        folds.append(fold)

    fold_results = []
    all_is_win_rates = []
    all_oos_win_rates = []
    all_is_pf = []
    all_oos_pf = []

    for oos_idx in range(n_folds):
        # In-sample: all folds except oos_idx
        is_trades = []
        for i in range(n_folds):
            if i != oos_idx:
                is_trades.extend(folds[i])

        # Out-of-sample: fold at oos_idx
        oos_trades = folds[oos_idx]

        # Compute summaries
        is_summary = _compute_summary(is_trades, mode)
        oos_summary = _compute_summary(oos_trades, mode)

        is_win_rate = is_summary["win_rate"]
        oos_win_rate = oos_summary["win_rate"]
        is_pf = is_summary["profit_factor"]
        oos_pf = oos_summary["profit_factor"]

        all_is_win_rates.append(is_win_rate)
        all_oos_win_rates.append(oos_win_rate)
        all_is_pf.append(is_pf)
        all_oos_pf.append(oos_pf)

        fold_results.append({
            "fold_num": oos_idx + 1,
            "in_sample": {
                "trade_count": len(is_trades),
                "win_rate": round(is_win_rate, 4),
                "total_pnl": round(is_summary["total_pnl"], 2),
                "profit_factor": round(is_pf, 2),
            },
            "out_of_sample": {
                "trade_count": len(oos_trades),
                "win_rate": round(oos_win_rate, 4),
                "total_pnl": round(oos_summary["total_pnl"], 2),
                "profit_factor": round(oos_pf, 2),
            },
        })

    # Compute aggregate metrics
    avg_is_wr = np.mean(all_is_win_rates) if all_is_win_rates else 0.0
    avg_oos_wr = np.mean(all_oos_win_rates) if all_oos_win_rates else 0.0

    # Degradation: how much OOS win rate drops vs IS
    degradation = (avg_is_wr - avg_oos_wr) / avg_is_wr * 100 if avg_is_wr > 0 else 0.0

    avg_is_pf_val = np.mean(all_is_pf) if all_is_pf else 1.0
    avg_oos_pf_val = np.mean(all_oos_pf) if all_oos_pf else 1.0

    # Strategy is robust if OOS win rate > 45% AND OOS profit factor > 1.0
    is_robust = (avg_oos_wr > 0.45) and (avg_oos_pf_val > 1.0)

    return {
        "folds": fold_results,
        "avg_is_win_rate": round(avg_is_wr, 4),
        "avg_oos_win_rate": round(avg_oos_wr, 4),
        "degradation_pct": round(degradation, 1),
        "avg_is_pf": round(avg_is_pf_val, 2),
        "avg_oos_pf": round(avg_oos_pf_val, 2),
        "is_robust": is_robust,
    }


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


def _build_equity_curve(trades: list[Trade], starting_capital: float = 10000.0) -> list[dict]:
    """Build an equity curve from trade sequence.

    Position sizing: for each trade, risk 1% of current account balance.
    Shares = (account * 0.01) / risk_per_share.
    Dollar P&L = shares * per-share P&L.
    """
    account = starting_capital
    curve = [{"trade_num": 0, "equity": round(account, 2), "pnl": 0.0,
              "pnl_pct": 0.0, "shares": 0, "account": round(account, 2)}]
    for i, t in enumerate(trades):
        risk_dollar = account * 0.01  # risk 1% of account
        if t.risk_per_share > 0:
            shares = int(risk_dollar / t.risk_per_share)
        else:
            shares = int(account / t.entry_price) if t.entry_price > 0 else 0
        shares = max(shares, 1)  # always at least 1 share
        dollar_pnl = round(t.pnl * shares, 2)
        account = round(account + dollar_pnl, 2)
        pnl_pct = round(dollar_pnl / (account - dollar_pnl) * 100, 2) if (account - dollar_pnl) > 0 else 0.0
        curve.append({
            "trade_num": i + 1,
            "equity": account,
            "pnl": dollar_pnl,
            "pnl_pct": pnl_pct,
            "pnl_per_share": round(t.pnl, 2),
            "shares": shares,
            "account": account,
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
        row = {}
        if hasattr(t, "ticker") and t.ticker:
            row["Ticker"] = t.ticker
        row.update({
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
        rows.append(row)
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
