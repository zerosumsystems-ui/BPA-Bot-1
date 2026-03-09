"""
compare_strategies.py — Comprehensive Trade Management Strategy Comparison

Tests first pullback setups across multiple exit strategies to find the
optimal trade management approach. Runs all strategies on the SAME trade
signals so results are directly comparable.

Strategies tested:
  1. Swing 3:1  — Fixed 3:1 R:R target
  2. Swing 2:1  — Fixed 2:1 R:R target
  3. Swing 1.5:1 — Fixed 1.5:1 R:R target
  4. Scalp 1:1  — Fixed 1:1 R:R target
  5. Runner     — 1R target, move stop to BE, let rest run to EOD/hold limit
  6. Scale Out  — Exit half at 1R, trail rest to 2R or EOD
  7. BE Trail   — Move stop to breakeven after 1R MFE, target 3R
  8. Aggr Trail — Trail stop 1 bar behind (tightest trailing stop)
  9. Time Stop  — Exit after N bars if not at target (with R:R target)
 10. Fixed $    — Fixed dollar stop ($0.50) instead of signal bar risk
 11. 2R + Trail — Take nothing at 2R, trail stop to 1R, let it run
 12. Chandelier — Trail stop at entry + (MFE - 1R), locks in gains dynamically
"""

from __future__ import annotations

import os
import sys
import copy
import math
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algo_engine import analyze_bars, bars_from_df, compute_ema, Bar
from backtester import Trade, calculate_al_brooks_levels, _check_limit_order_fill, FADE_SETUPS, SETUP_CONFIG
from data_source import get_data_source


# ─────────────────────── STRATEGY SIMULATORS ──────────────────────────────────

def _base_setup(trade, bars, timestamps, slippage=0.0):
    """Common setup for all strategies: handle limit fill, apply slippage."""
    start_idx = trade.entry_bar

    if trade.order_type == "Limit":
        filled, fill_idx = _check_limit_order_fill(trade, bars, look_ahead=3)
        if not filled:
            trade.exit_reason = "unfilled"
            trade.pnl = 0.0
            trade.r_multiple = 0.0
            trade.is_winner = False
            trade.bars_held = 0
            return trade, -1, 0.0
        start_idx = fill_idx

    adjusted_entry = trade.entry_price
    if trade.direction == "Long":
        adjusted_entry += slippage
    else:
        adjusted_entry -= slippage

    return trade, start_idx, adjusted_entry


def _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx):
    """Common finalization for all strategies."""
    trade.is_winner = trade.pnl > 0
    trade.bars_held = trade.exit_bar - trade.entry_bar
    if trade.risk_per_share > 0:
        trade.r_multiple = round(trade.pnl / trade.risk_per_share, 2)
        trade.mae_r = round(max_adverse / trade.risk_per_share, 2)
        trade.mfe_r = round(max_favorable / trade.risk_per_share, 2)
    trade.mae = round(max_adverse, 2)
    trade.mfe = round(max_favorable, 2)
    trade.mae_bar = bars[mae_bar_idx].idx if mae_bar_idx < len(bars) else 0
    trade.mfe_bar = bars[mfe_bar_idx].idx if mfe_bar_idx < len(bars) else 0
    return trade


def _exit_trade(trade, bar, adjusted_entry, slippage, reason, timestamps, i, is_long):
    """Helper to set exit fields."""
    if is_long:
        if reason == "stop_loss":
            exit_price = trade.stop_loss - slippage
        elif "target" in reason:
            exit_price = trade._target_price - slippage
        else:
            exit_price = bar.close - slippage
        trade.pnl = round(exit_price - adjusted_entry, 2)
    else:
        if reason == "stop_loss":
            exit_price = trade.stop_loss + slippage
        elif "target" in reason:
            exit_price = trade._target_price + slippage
        else:
            exit_price = bar.close + slippage
        trade.pnl = round(adjusted_entry - exit_price, 2)

    trade.exit_bar = bar.idx
    trade.exit_price = round(exit_price, 2)
    trade.exit_time = timestamps[i] if i < len(timestamps) else ""
    trade.exit_reason = reason


def sim_fixed_rr(trade, bars, timestamps, rr_mult, slippage=0.0, hold_limit=999):
    """Simulate with a fixed R:R target."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    risk = trade.risk_per_share
    is_long = trade.direction == "Long"
    target = adjusted_entry + rr_mult * risk if is_long else adjusted_entry - rr_mult * risk
    trade._target_price = target

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
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

        # Check stop first
        if is_long and bar.low <= trade.stop_loss:
            _exit_trade(trade, bar, adjusted_entry, slippage, "stop_loss", timestamps, i, is_long)
            break
        elif not is_long and bar.high >= trade.stop_loss:
            _exit_trade(trade, bar, adjusted_entry, slippage, "stop_loss", timestamps, i, is_long)
            break

        # Check target
        if is_long and bar.high >= target:
            _exit_trade(trade, bar, adjusted_entry, slippage, f"target_{rr_mult}R", timestamps, i, is_long)
            break
        elif not is_long and bar.low <= target:
            _exit_trade(trade, bar, adjusted_entry, slippage, f"target_{rr_mult}R", timestamps, i, is_long)
            break
    else:
        # EOD / hold limit
        last_idx = end_idx - 1
        bar = bars[last_idx]
        trade._target_price = bar.close
        _exit_trade(trade, bar, adjusted_entry, slippage, "eod_close", timestamps, last_idx, is_long)

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_runner(trade, bars, timestamps, slippage=0.0, hold_limit=999):
    """Take half at 1R, move stop to BE, let the rest run to EOD."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    risk = trade.risk_per_share
    is_long = trade.direction == "Long"
    target_1r = adjusted_entry + risk if is_long else adjusted_entry - risk
    stop = trade.stop_loss
    half_exited = False
    half_pnl = 0.0

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
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

        # Check stop
        if is_long and bar.low <= stop:
            if half_exited:
                trade.pnl = round(half_pnl + (stop - slippage - adjusted_entry) * 0.5, 2)
            else:
                trade.pnl = round(stop - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "stop_loss"
            break
        elif not is_long and bar.high >= stop:
            if half_exited:
                trade.pnl = round(half_pnl + (adjusted_entry - stop - slippage) * 0.5, 2)
            else:
                trade.pnl = round(adjusted_entry - stop - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "stop_loss"
            break

        # Check 1R target for first half
        if not half_exited:
            if is_long and bar.high >= target_1r:
                half_pnl = (target_1r - slippage - adjusted_entry) * 0.5
                half_exited = True
                stop = adjusted_entry  # Move stop to breakeven
            elif not is_long and bar.low <= target_1r:
                half_pnl = (adjusted_entry - target_1r - slippage) * 0.5
                half_exited = True
                stop = adjusted_entry
    else:
        last_idx = end_idx - 1
        bar = bars[last_idx]
        if half_exited:
            if is_long:
                trade.pnl = round(half_pnl + (bar.close - slippage - adjusted_entry) * 0.5, 2)
            else:
                trade.pnl = round(half_pnl + (adjusted_entry - bar.close - slippage) * 0.5, 2)
        else:
            if is_long:
                trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
            else:
                trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "eod_close"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_scale_out(trade, bars, timestamps, slippage=0.0, hold_limit=999):
    """Exit half at 1R, trail the rest with stop at 1R, target 2R."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    risk = trade.risk_per_share
    is_long = trade.direction == "Long"
    target_1r = adjusted_entry + risk if is_long else adjusted_entry - risk
    target_2r = adjusted_entry + 2 * risk if is_long else adjusted_entry - 2 * risk
    stop = trade.stop_loss
    half_exited = False
    half_pnl = 0.0

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
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

        # Check stop
        if is_long and bar.low <= stop:
            if half_exited:
                trade.pnl = round(half_pnl + (stop - slippage - adjusted_entry) * 0.5, 2)
            else:
                trade.pnl = round(stop - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "stop_loss"
            break
        elif not is_long and bar.high >= stop:
            if half_exited:
                trade.pnl = round(half_pnl + (adjusted_entry - stop - slippage) * 0.5, 2)
            else:
                trade.pnl = round(adjusted_entry - stop - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "stop_loss"
            break

        # First half: exit at 1R
        if not half_exited:
            if is_long and bar.high >= target_1r:
                half_pnl = (target_1r - slippage - adjusted_entry) * 0.5
                half_exited = True
                stop = adjusted_entry + risk  # Trail stop to 1R profit
                # Actually trail to BE for scale out
                stop = adjusted_entry
            elif not is_long and bar.low <= target_1r:
                half_pnl = (adjusted_entry - target_1r - slippage) * 0.5
                half_exited = True
                stop = adjusted_entry

        # Second half: check 2R target
        if half_exited:
            if is_long and bar.high >= target_2r:
                trade.pnl = round(half_pnl + (target_2r - slippage - adjusted_entry) * 0.5, 2)
                trade.exit_bar = bar.idx
                trade.exit_price = round(target_2r, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "scale_2r"
                break
            elif not is_long and bar.low <= target_2r:
                trade.pnl = round(half_pnl + (adjusted_entry - target_2r - slippage) * 0.5, 2)
                trade.exit_bar = bar.idx
                trade.exit_price = round(target_2r, 2)
                trade.exit_time = timestamps[i] if i < len(timestamps) else ""
                trade.exit_reason = "scale_2r"
                break
    else:
        last_idx = end_idx - 1
        bar = bars[last_idx]
        if half_exited:
            if is_long:
                trade.pnl = round(half_pnl + (bar.close - slippage - adjusted_entry) * 0.5, 2)
            else:
                trade.pnl = round(half_pnl + (adjusted_entry - bar.close - slippage) * 0.5, 2)
        else:
            if is_long:
                trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
            else:
                trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "eod_close"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_be_trail(trade, bars, timestamps, slippage=0.0, hold_limit=999):
    """Move stop to BE after price hits 1R MFE, target 3R."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    risk = trade.risk_per_share
    is_long = trade.direction == "Long"
    target_3r = adjusted_entry + 3 * risk if is_long else adjusted_entry - 3 * risk
    stop = trade.stop_loss
    be_activated = False

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
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

        # Move stop to BE once we've seen 1R favorable
        if not be_activated and favorable >= risk:
            be_activated = True
            stop = adjusted_entry

        # Check stop
        if is_long and bar.low <= stop:
            trade.pnl = round(stop - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "be_stop" if be_activated else "stop_loss"
            break
        elif not is_long and bar.high >= stop:
            trade.pnl = round(adjusted_entry - stop - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "be_stop" if be_activated else "stop_loss"
            break

        # Check 3R target
        if is_long and bar.high >= target_3r:
            trade.pnl = round(target_3r - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(target_3r, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "target_3R"
            break
        elif not is_long and bar.low <= target_3r:
            trade.pnl = round(adjusted_entry - target_3r - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(target_3r, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "target_3R"
            break
    else:
        last_idx = end_idx - 1
        bar = bars[last_idx]
        if is_long:
            trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "eod_close"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_aggr_trail(trade, bars, timestamps, slippage=0.0, hold_limit=999):
    """Aggressive trailing stop: trail behind each bar's extreme."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    is_long = trade.direction == "Long"
    stop = trade.stop_loss

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
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

        # Check stop
        if is_long and bar.low <= stop:
            trade.pnl = round(stop - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "trail_stop"
            break
        elif not is_long and bar.high >= stop:
            trade.pnl = round(adjusted_entry - stop - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "trail_stop"
            break

        # Trail stop: move to prior bar's low/high
        if i > start_idx:
            prev = bars[i - 1]
            if is_long:
                new_stop = prev.low
                if new_stop > stop:
                    stop = new_stop
            else:
                new_stop = prev.high
                if new_stop < stop:
                    stop = new_stop
    else:
        last_idx = end_idx - 1
        bar = bars[last_idx]
        if is_long:
            trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "eod_close"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_time_stop(trade, bars, timestamps, rr_mult=2.0, max_bars=10, slippage=0.0, hold_limit=999):
    """Exit at R:R target OR after max_bars, whichever comes first."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    risk = trade.risk_per_share
    is_long = trade.direction == "Long"
    target = adjusted_entry + rr_mult * risk if is_long else adjusted_entry - rr_mult * risk
    trade._target_price = target

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    time_limit = min(start_idx + max_bars, start_idx + hold_limit, len(bars))
    for i in range(start_idx, time_limit):
        bar = bars[i]

        if is_long:
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

        # Check stop
        if is_long and bar.low <= trade.stop_loss:
            _exit_trade(trade, bar, adjusted_entry, slippage, "stop_loss", timestamps, i, is_long)
            break
        elif not is_long and bar.high >= trade.stop_loss:
            _exit_trade(trade, bar, adjusted_entry, slippage, "stop_loss", timestamps, i, is_long)
            break

        # Check target
        if is_long and bar.high >= target:
            _exit_trade(trade, bar, adjusted_entry, slippage, f"target_{rr_mult}R", timestamps, i, is_long)
            break
        elif not is_long and bar.low <= target:
            _exit_trade(trade, bar, adjusted_entry, slippage, f"target_{rr_mult}R", timestamps, i, is_long)
            break
    else:
        # Time stop — close at market
        last_idx = time_limit - 1
        bar = bars[last_idx]
        if is_long:
            trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "time_stop"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_fixed_dollar_stop(trade, bars, timestamps, fixed_stop=0.50, rr_mult=2.0, slippage=0.0, hold_limit=999):
    """Use a fixed dollar stop instead of signal bar risk, with R:R target on top."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    is_long = trade.direction == "Long"
    # Override stop with fixed dollar amount
    if is_long:
        stop = adjusted_entry - fixed_stop
        target = adjusted_entry + fixed_stop * rr_mult
    else:
        stop = adjusted_entry + fixed_stop
        target = adjusted_entry - fixed_stop * rr_mult

    # Override risk for R-multiple calculation
    trade.risk_per_share = fixed_stop
    trade._target_price = target

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
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

        # Check stop
        if is_long and bar.low <= stop:
            trade.pnl = round(stop - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "stop_loss"
            break
        elif not is_long and bar.high >= stop:
            trade.pnl = round(adjusted_entry - stop - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "stop_loss"
            break

        # Check target
        if is_long and bar.high >= target:
            trade.pnl = round(target - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(target, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = f"target_{rr_mult}R"
            break
        elif not is_long and bar.low <= target:
            trade.pnl = round(adjusted_entry - target - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(target, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = f"target_{rr_mult}R"
            break
    else:
        last_idx = end_idx - 1
        bar = bars[last_idx]
        if is_long:
            trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "eod_close"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_2r_trail(trade, bars, timestamps, slippage=0.0, hold_limit=999):
    """No partial exit. After 2R MFE, trail stop to lock in 1R. Target: let it run."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    risk = trade.risk_per_share
    is_long = trade.direction == "Long"
    stop = trade.stop_loss
    trailing_activated = False

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
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

        # Activate trailing after 2R favorable
        if not trailing_activated and favorable >= 2 * risk:
            trailing_activated = True
            if is_long:
                stop = adjusted_entry + risk  # Lock in 1R
            else:
                stop = adjusted_entry - risk

        # Ratchet trail: once activated, keep moving stop up
        if trailing_activated:
            if is_long:
                new_stop = bar.high - 1.5 * risk  # Trail 1.5R behind the high
                if new_stop > stop:
                    stop = new_stop
            else:
                new_stop = bar.low + 1.5 * risk
                if new_stop < stop:
                    stop = new_stop

        # Check stop
        if is_long and bar.low <= stop:
            trade.pnl = round(stop - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "trail_stop" if trailing_activated else "stop_loss"
            break
        elif not is_long and bar.high >= stop:
            trade.pnl = round(adjusted_entry - stop - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "trail_stop" if trailing_activated else "stop_loss"
            break
    else:
        last_idx = end_idx - 1
        bar = bars[last_idx]
        if is_long:
            trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "eod_close"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


def sim_chandelier(trade, bars, timestamps, slippage=0.0, hold_limit=999):
    """Chandelier exit: trail stop at (highest high since entry) - 2R for longs."""
    trade, start_idx, adjusted_entry = _base_setup(trade, bars, timestamps, slippage)
    if start_idx < 0:
        return trade

    risk = trade.risk_per_share
    is_long = trade.direction == "Long"
    stop = trade.stop_loss
    best_price = adjusted_entry  # Track highest high / lowest low

    max_adverse = 0.0
    max_favorable = 0.0
    mae_bar_idx = start_idx
    mfe_bar_idx = start_idx

    end_idx = min(start_idx + hold_limit, len(bars))
    for i in range(start_idx, end_idx):
        bar = bars[i]

        if is_long:
            adverse = adjusted_entry - bar.low
            favorable = bar.high - adjusted_entry
            if bar.high > best_price:
                best_price = bar.high
            # Chandelier stop: highest high - 2R
            chandelier_stop = best_price - 2 * risk
            if chandelier_stop > stop:
                stop = chandelier_stop
        else:
            adverse = bar.high - adjusted_entry
            favorable = adjusted_entry - bar.low
            if bar.low < best_price:
                best_price = bar.low
            chandelier_stop = best_price + 2 * risk
            if chandelier_stop < stop:
                stop = chandelier_stop

        if adverse > max_adverse:
            max_adverse = adverse
            mae_bar_idx = i
        if favorable > max_favorable:
            max_favorable = favorable
            mfe_bar_idx = i

        # Check stop
        if is_long and bar.low <= stop:
            trade.pnl = round(stop - slippage - adjusted_entry, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "chandelier_stop"
            break
        elif not is_long and bar.high >= stop:
            trade.pnl = round(adjusted_entry - stop - slippage, 2)
            trade.exit_bar = bar.idx
            trade.exit_price = round(stop, 2)
            trade.exit_time = timestamps[i] if i < len(timestamps) else ""
            trade.exit_reason = "chandelier_stop"
            break
    else:
        last_idx = end_idx - 1
        bar = bars[last_idx]
        if is_long:
            trade.pnl = round(bar.close - slippage - adjusted_entry, 2)
        else:
            trade.pnl = round(adjusted_entry - bar.close - slippage, 2)
        trade.exit_bar = bar.idx
        trade.exit_price = round(bar.close, 2)
        trade.exit_time = timestamps[last_idx] if last_idx < len(timestamps) else ""
        trade.exit_reason = "eod_close"

    return _finalize(trade, bars, timestamps, adjusted_entry, slippage, max_adverse, max_favorable, mae_bar_idx, mfe_bar_idx)


# ─────────────────────── STRATEGY REGISTRY ────────────────────────────────────

STRATEGIES = {
    "Swing 3:1":    lambda t, b, ts, sl: sim_fixed_rr(t, b, ts, 3.0, sl),
    "Swing 2:1":    lambda t, b, ts, sl: sim_fixed_rr(t, b, ts, 2.0, sl),
    "Swing 1.5:1":  lambda t, b, ts, sl: sim_fixed_rr(t, b, ts, 1.5, sl),
    "Scalp 1:1":    lambda t, b, ts, sl: sim_fixed_rr(t, b, ts, 1.0, sl),
    "Runner":       lambda t, b, ts, sl: sim_runner(t, b, ts, sl),
    "Scale Out":    lambda t, b, ts, sl: sim_scale_out(t, b, ts, sl),
    "BE Trail":     lambda t, b, ts, sl: sim_be_trail(t, b, ts, sl),
    "Aggr Trail":   lambda t, b, ts, sl: sim_aggr_trail(t, b, ts, sl),
    "Time 10bar":   lambda t, b, ts, sl: sim_time_stop(t, b, ts, 2.0, 10, sl),
    "Time 5bar":    lambda t, b, ts, sl: sim_time_stop(t, b, ts, 2.0, 5, sl),
    "Fixed $0.50":  lambda t, b, ts, sl: sim_fixed_dollar_stop(t, b, ts, 0.50, 2.0, sl),
    "Fixed $1.00":  lambda t, b, ts, sl: sim_fixed_dollar_stop(t, b, ts, 1.00, 2.0, sl),
    "2R + Trail":   lambda t, b, ts, sl: sim_2r_trail(t, b, ts, sl),
    "Chandelier":   lambda t, b, ts, sl: sim_chandelier(t, b, ts, sl),
}


# ─────────────────────── TRADE GENERATION (SHARED SIGNALS) ────────────────────

def generate_base_trades(df: pd.DataFrame, ticker: str = "") -> list[dict]:
    """
    Generate trade signals from a single day of 5-min data.
    Returns raw trade info dicts (not yet simulated) so each strategy
    can simulate the same entries independently.
    """
    bars = bars_from_df(df)
    if len(bars) < 10:
        return []

    ema = compute_ema(bars)
    timestamps = [str(idx) for idx in df.index]
    analysis = analyze_bars(df)
    setups = analysis.get("setups", [])
    if not setups:
        return []

    setups.sort(key=lambda s: s["entry_bar"])

    trades_info = []
    last_entry_bar = -999
    min_bars_between = 3

    for setup in setups:
        entry_bar_num = setup["entry_bar"]
        bar_idx = entry_bar_num - 1

        if bar_idx < 0 or bar_idx >= len(bars):
            continue
        if entry_bar_num - last_entry_bar < min_bars_between:
            continue

        signal_bar = bars[bar_idx]
        actual_entry_bar = entry_bar_num + 1
        actual_bar_idx = bar_idx + 1

        if actual_bar_idx >= len(bars):
            continue

        name = setup["setup_name"]

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

        levels = calculate_al_brooks_levels(signal_bar, direction)

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

        entry_close = bars[actual_bar_idx].close
        entry_ema = ema[actual_bar_idx] if actual_bar_idx < len(ema) else 0
        ema_pos = "Above EMA" if entry_close > entry_ema else "Below EMA"
        with_trend = (direction == "Long" and entry_close > entry_ema) or \
                     (direction == "Short" and entry_close < entry_ema)

        trades_info.append({
            "entry_bar": actual_entry_bar,
            "entry_price": levels["entry"],
            "entry_time": timestamps[actual_bar_idx] if actual_bar_idx < len(timestamps) else "",
            "setup_name": name,
            "direction": direction,
            "order_type": setup.get("order_type", "Stop"),
            "stop_loss": levels["stop"],
            "scalp_target": levels["scalp_target"],
            "swing_target": levels["swing_target"],
            "risk_per_share": levels["risk"],
            "ticker": ticker,
            "ema_position": ema_pos,
            "with_trend": with_trend,
            "confidence": setup.get("confidence", 0.0),
        })

        last_entry_bar = actual_entry_bar

    return trades_info, bars, timestamps


def make_trade(info: dict) -> Trade:
    """Create a fresh Trade object from info dict."""
    return Trade(
        entry_bar=info["entry_bar"],
        entry_price=info["entry_price"],
        entry_time=info["entry_time"],
        setup_name=info["setup_name"],
        direction=info["direction"],
        order_type=info["order_type"],
        stop_loss=info["stop_loss"],
        scalp_target=info["scalp_target"],
        swing_target=info["swing_target"],
        risk_per_share=info["risk_per_share"],
        ticker=info.get("ticker", ""),
        ema_position=info.get("ema_position", ""),
        with_trend=info.get("with_trend", False),
        confidence=info.get("confidence", 0.0),
    )


# ─────────────────────── METRICS COMPUTATION ──────────────────────────────────

def compute_metrics(trades: list[Trade]) -> dict:
    """Compute comprehensive metrics for a list of trades."""
    if not trades:
        return {
            "trades": 0, "wr": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "rr": 0.0, "min_wr": 0.0, "edge": 0.0, "ev_trade": 0.0,
            "ev_r": 0.0, "pf": 0.0, "total_pnl": 0.0, "sharpe": 0.0,
            "max_dd": 0.0, "avg_bars": 0.0,
        }

    total = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    win_count = len(winners)

    wr = win_count / total
    avg_win = np.mean([t.pnl for t in winners]) if winners else 0.0
    avg_loss = np.mean([abs(t.pnl) for t in losers]) if losers else 0.001
    rr = avg_win / avg_loss if avg_loss > 0 else 999.0

    # Minimum WR needed to break even at this R:R
    min_wr = 1 / (1 + rr) if rr > 0 else 1.0
    edge = wr - min_wr

    total_pnl = sum(t.pnl for t in trades)
    ev_trade = total_pnl / total

    avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0.0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Sharpe
    returns = [t.pnl for t in trades]
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0

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

    avg_bars = np.mean([t.bars_held for t in trades])

    return {
        "trades": total,
        "wr": round(wr, 3),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "rr": round(rr, 2),
        "min_wr": round(min_wr, 3),
        "edge": round(edge, 3),
        "ev_trade": round(ev_trade, 3),
        "ev_r": round(float(avg_r), 3),
        "pf": round(pf, 2),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "avg_bars": round(float(avg_bars), 1),
    }


# ─────────────────────── MAIN ─────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  TRADE MANAGEMENT STRATEGY COMPARISON — First Pullback Setups")
    print("=" * 90)

    # Load Databento key
    key_path = Path("/etc/secrets/DATABENTO_API_KEY")
    if key_path.exists():
        api_key = key_path.read_text().strip()
    else:
        api_key = os.environ.get("DATABENTO_API_KEY", "")

    if not api_key:
        print("ERROR: No DATABENTO_API_KEY found")
        sys.exit(1)

    source = get_data_source(api_key=api_key)

    # Ticker universe — use Most Liquid for broad coverage
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "AMD", "GOOG"]

    # Date range: ~6 months of data
    end = datetime.date.today()
    days = 120  # trading days
    calendar_days = int(days * 1.45) + 1
    start = end - datetime.timedelta(days=calendar_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"\nTickers: {', '.join(tickers)}")
    print(f"Period: {start_str} to {end_str} ({days} trading days)")
    print(f"Strategies: {len(STRATEGIES)}")
    print()

    # ── Phase 1: Fetch data ──
    print("Fetching data from Databento...")
    try:
        bulk_df = source.get_bulk_chart_data(tickers, start_str, end_str)
    except Exception as e:
        print(f"ERROR fetching data: {e}")
        sys.exit(1)

    if bulk_df.empty:
        print("ERROR: No data returned")
        sys.exit(1)

    # Split by symbol
    fetched = {}
    if "symbol" in bulk_df.columns:
        for sym, group in bulk_df.groupby("symbol"):
            fetched[sym] = group.drop(columns=["symbol", "BarNumber"], errors="ignore")

    print(f"Got data for {len(fetched)} tickers")

    # ── Phase 2: Generate all trade signals ──
    print("Generating trade signals...")
    all_trade_infos = []
    all_bars_map = {}  # (ticker, date) -> (bars, timestamps)
    signal_count = 0

    for ticker in tickers:
        full_df = fetched.get(ticker)
        if full_df is None or full_df.empty:
            print(f"  {ticker}: no data")
            continue

        if isinstance(full_df.columns, pd.MultiIndex):
            full_df.columns = full_df.columns.get_level_values(0)

        full_df.index = pd.to_datetime(full_df.index)

        day_count = 0
        for date, group in full_df.groupby(full_df.index.date):
            if len(group) < 10:
                continue

            try:
                result = generate_base_trades(group, ticker=ticker)
                if not result or not result[0]:
                    continue
                trade_infos, bars, timestamps = result
                key = f"{ticker}|{date}"
                all_bars_map[key] = (bars, timestamps)
                for info in trade_infos:
                    info["_day_key"] = key
                all_trade_infos.extend(trade_infos)
                signal_count += len(trade_infos)
                day_count += 1
            except Exception as e:
                continue

        print(f"  {ticker}: {day_count} days processed")

    print(f"\nTotal signals: {signal_count}")

    # ── Phase 3: Run all strategies on same signals ──
    print("\nRunning strategies...")
    strategy_trades = {name: [] for name in STRATEGIES}

    for info in all_trade_infos:
        day_key = info["_day_key"]
        bars, timestamps = all_bars_map[day_key]

        for strat_name, sim_fn in STRATEGIES.items():
            trade = make_trade(info)
            try:
                trade = sim_fn(trade, bars, timestamps, 0.0)
                if trade.exit_reason != "unfilled":
                    strategy_trades[strat_name].append(trade)
            except Exception:
                continue

    # ── Phase 4: Compute metrics and display ──
    print("\n")
    print("=" * 90)
    print("  RESULTS")
    print("=" * 90)

    results = []
    for name in STRATEGIES:
        trades = strategy_trades[name]
        m = compute_metrics(trades)
        m["name"] = name
        results.append(m)

    # Sort by EV/trade descending
    results.sort(key=lambda x: x["ev_trade"], reverse=True)

    # Print main comparison table
    header = f"{'Strategy':<14} {'WR':>6} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} {'MinWR':>6} {'Edge':>7} {'EV/trd':>8} {'EV(R)':>7} {'PF':>6} {'P&L':>10} {'Sharpe':>7} {'MaxDD':>8} {'Bars':>5}"
    print(header)
    print("-" * len(header))

    for r in results:
        edge_str = f"{r['edge']:+.1%}" if r['edge'] != 0 else "0.0%"
        ev_str = f"${r['ev_trade']:+.3f}" if r['ev_trade'] != 0 else "$0.000"
        pnl_str = f"${r['total_pnl']:+,.0f}"
        print(
            f"{r['name']:<14} {r['wr']:>5.1%} {r['avg_win']:>7.2f} {r['avg_loss']:>7.2f} "
            f"{r['rr']:>5.2f} {r['min_wr']:>5.1%} {edge_str:>7} {ev_str:>8} "
            f"{r['ev_r']:>+6.3f} {r['pf']:>5.2f} {pnl_str:>10} {r['sharpe']:>6.3f} "
            f"${r['max_dd']:>7.2f} {r['avg_bars']:>5.1f}"
        )

    # Highlight the winner
    best = results[0]
    print(f"\n{'='*90}")
    print(f"  WINNER: {best['name']}")
    print(f"{'='*90}")
    print(f"  Best EV/trade: ${best['ev_trade']:+.3f}")
    print(f"  Best EV(R):    {best['ev_r']:+.3f}R per trade")
    print(f"  Best Edge:     {best['edge']:+.1%} above breakeven")
    print(f"  Profit Factor: {best['pf']:.2f}")
    print(f"  Total P&L:     ${best['total_pnl']:+,.2f}")
    print(f"  Sharpe:        {best['sharpe']:.3f}")

    # Per-ticker breakdown for top 3 strategies
    top3 = [r["name"] for r in results[:3]]
    print(f"\n{'='*90}")
    print(f"  TOP 3 STRATEGIES — PER-TICKER BREAKDOWN")
    print(f"{'='*90}")

    ticker_header = f"{'Ticker':<8}"
    for sn in top3:
        ticker_header += f" | {sn:<14} {'WR':>5} {'EV(R)':>6} {'P&L':>8}"
    print(ticker_header)
    print("-" * len(ticker_header))

    for ticker in tickers:
        row = f"{ticker:<8}"
        for sn in top3:
            ticker_trades = [t for t in strategy_trades[sn] if t.ticker == ticker]
            if not ticker_trades:
                row += f" | {'--':<14} {'--':>5} {'--':>6} {'--':>8}"
            else:
                m = compute_metrics(ticker_trades)
                row += f" | {m['trades']:<14} {m['wr']:>4.0%} {m['ev_r']:>+5.3f} ${m['total_pnl']:>+7.0f}"
        print(row)

    # Exit reason breakdown for top 3
    print(f"\n{'='*90}")
    print(f"  EXIT REASON BREAKDOWN — TOP 3")
    print(f"{'='*90}")
    for sn in top3:
        trades = strategy_trades[sn]
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        total = len(trades)
        print(f"\n  {sn}:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            # PnL for this exit reason
            reason_pnl = sum(t.pnl for t in trades if t.exit_reason == reason)
            print(f"    {reason:<20} {count:>5} ({pct:>5.1f}%)  P&L: ${reason_pnl:>+8.2f}")


if __name__ == "__main__":
    main()
