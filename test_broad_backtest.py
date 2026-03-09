"""
Backtest comparison: Broad Channel filter fix (RULE 3).
Generates synthetic broad channel price data and compares
performance with the bug vs the fix.
"""
import numpy as np
import pandas as pd
from algo_engine import analyze_bars, filter_by_context, Setup


def make_broad_bull_channel_day(bars=78, seed=42):
    """Generate a synthetic broad bull channel day (78 five-min bars ~ full session)."""
    rng = np.random.RandomState(seed)
    base = 450.0
    prices = []
    for i in range(bars):
        drift = 0.15  # slow upward drift
        noise = rng.normal(0, 0.4)
        o = base + i * drift + noise
        c = o + rng.normal(0.05, 0.3)
        h = max(o, c) + abs(rng.normal(0, 0.2))
        l = min(o, c) - abs(rng.normal(0, 0.2))
        prices.append((round(o,2), round(h,2), round(l,2), round(c,2)))
        base_adj = c - (base + i * drift)  # carry forward
    df = pd.DataFrame(prices, columns=["Open","High","Low","Close"])
    df["Volume"] = 100000
    return df


def make_broad_bear_channel_day(bars=78, seed=99):
    """Generate a synthetic broad bear channel day."""
    rng = np.random.RandomState(seed)
    base = 450.0
    prices = []
    for i in range(bars):
        drift = -0.15
        noise = rng.normal(0, 0.4)
        o = base + i * drift + noise
        c = o + rng.normal(-0.05, 0.3)
        h = max(o, c) + abs(rng.normal(0, 0.2))
        l = min(o, c) - abs(rng.normal(0, 0.2))
        prices.append((round(o,2), round(h,2), round(l,2), round(c,2)))
    df = pd.DataFrame(prices, columns=["Open","High","Low","Close"])
    df["Volume"] = 100000
    return df


def _buggy_filter(setups, day_type, market_cycle):
    """The old broken filter — checks 'Broad Range' which never matches."""
    filtered = []
    fade_keywords = ["Fade", "Top", "Bottom", "Reversal", "Exhaustion", "Test"]
    trend_keywords = ["Flag", "Breakout", "Stairs", "H1", "L1", "Pullback"]
    for s in setups:
        setup_name = s.setup_type if hasattr(s, 'setup_type') else getattr(s, 'setup_name', '')
        if "Spike" in market_cycle or "Trend" in day_type:
            if any(k in setup_name for k in fade_keywords):
                continue
        if "Tight" in market_cycle or "Trading Range" in day_type:
            if any(k in setup_name for k in trend_keywords) and "Major" not in setup_name:
                continue
        # BUG: "Broad Range" never matches "Broad Bull/Bear Channel"
        if "Broad Range" in market_cycle:
            if "Breakout" in setup_name and "Test" not in setup_name:
                continue
        filtered.append(s)
    return filtered


def run_with_filter(df, use_fix=True):
    """Run backtest with the bug or the fix by monkey-patching algo_engine."""
    from backtester import run_backtest
    import algo_engine

    original_filter = algo_engine.filter_by_context

    if not use_fix:
        algo_engine.filter_by_context = _buggy_filter

    try:
        result = run_backtest(df, mode="scalp", slippage=0.01, commission=0.005)
    finally:
        algo_engine.filter_by_context = original_filter

    return result


def print_summary(label, summary, trades):
    """Print key backtest metrics."""
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Total trades:    {summary.get('total_trades', 0)}")
    print(f"  Win rate:        {summary.get('win_rate', 0)*100:.1f}%")
    print(f"  Avg R:R:         {summary.get('avg_r_multiple', 0):.2f}R")
    print(f"  Total R:         {summary.get('total_r', 0):.2f}R")
    print(f"  Total P&L:       ${summary.get('total_pnl', 0):.2f}")
    print(f"  Profit factor:   {summary.get('profit_factor', 0):.2f}")
    print(f"  Avg win:         ${summary.get('avg_win', 0):.2f}")
    print(f"  Avg loss:        ${summary.get('avg_loss', 0):.2f}")
    print(f"  Expectancy:      ${summary.get('expectancy', 0):.4f}/trade")
    print(f"  Max drawdown:    ${summary.get('max_drawdown', 0):.2f}")
    print(f"  Edge ratio:      {summary.get('edge_ratio', 0):.2f}")

    # Show which setups were taken
    setup_stats = summary.get("setup_breakdown", {})
    if setup_stats:
        print(f"\n  Setup breakdown:")
        for name, stats in sorted(setup_stats.items(), key=lambda x: -x[1].get('count',0)):
            w = stats.get('wins', 0)
            c = stats.get('count', 0)
            wr = w/c*100 if c else 0
            pnl = stats.get('pnl', 0)
            print(f"    {name:40s}  {c} trades  {wr:5.1f}% win  ${pnl:+.2f}")

    # Show broad channel trades specifically
    broad_trades = [t for t in trades if "Broad" in t.market_cycle]
    if broad_trades:
        bw = sum(1 for t in broad_trades if t.is_winner)
        print(f"\n  Trades in Broad Channels: {len(broad_trades)} ({bw} wins, {len(broad_trades)-bw} losses)")

    breakout_trades = [t for t in trades if "Breakout" in t.setup_name or "BO" in t.setup_name]
    if breakout_trades:
        bw = sum(1 for t in breakout_trades if t.is_winner)
        pnl = sum(t.pnl for t in breakout_trades)
        print(f"  Breakout trades: {len(breakout_trades)} ({bw} wins) P&L: ${pnl:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("BROAD CHANNEL BACKTEST: BEFORE vs AFTER RULE 3 FIX")
    print("=" * 60)

    # Generate multiple days of synthetic data
    seeds_bull = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    seeds_bear = [99, 150, 250, 350, 450, 550, 650, 750, 850, 950]

    all_trades_before = []
    all_trades_after = []

    print("\nRunning backtests on 20 synthetic broad channel days...")

    for i, seed in enumerate(seeds_bull):
        df = make_broad_bull_channel_day(seed=seed)

        res_before = run_with_filter(df, use_fix=False)
        res_after = run_with_filter(df, use_fix=True)

        all_trades_before.extend(res_before["trades"])
        all_trades_after.extend(res_after["trades"])

    for i, seed in enumerate(seeds_bear):
        df = make_broad_bear_channel_day(seed=seed)

        res_before = run_with_filter(df, use_fix=False)
        res_after = run_with_filter(df, use_fix=True)

        all_trades_before.extend(res_before["trades"])
        all_trades_after.extend(res_after["trades"])

    # Compute aggregate summaries
    from backtester import _compute_summary

    summary_before = _compute_summary(all_trades_before, "scalp")
    summary_after = _compute_summary(all_trades_after, "scalp")

    print_summary("BEFORE FIX (Broad Range bug — breakouts NOT filtered)", summary_before, all_trades_before)
    print_summary("AFTER FIX (Broad channels correctly filter breakouts)", summary_after, all_trades_after)

    # Delta
    print(f"\n{'='*60}")
    print("COMPARISON (After - Before)")
    print(f"{'='*60}")
    dt = summary_after.get('total_trades',0) - summary_before.get('total_trades',0)
    dw = (summary_after.get('win_rate',0) - summary_before.get('win_rate',0)) * 100
    dp = summary_after.get('total_pnl',0) - summary_before.get('total_pnl',0)
    dr = summary_after.get('avg_r_multiple',0) - summary_before.get('avg_r_multiple',0)
    dpf = summary_after.get('profit_factor',0) - summary_before.get('profit_factor',0)
    print(f"  Trades:        {dt:+d}")
    print(f"  Win rate:      {dw:+.1f}%")
    print(f"  Avg R:         {dr:+.2f}R")
    print(f"  Total P&L:     ${dp:+.2f}")
    print(f"  Profit factor: {dpf:+.2f}")
    print(f"{'='*60}")
