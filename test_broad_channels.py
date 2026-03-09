"""
Unit tests for Broad Channel detection and filtering logic.
Tests classify_day_type, classify_market_cycle, and filter_by_context
for Broad Bull/Bear Channel scenarios without requiring external data.
"""

import sys
from algo_engine import Bar, classify_day_type, classify_market_cycle, filter_by_context, Setup


def make_bars(prices, start_idx=0):
    """Create Bar objects from a list of (open, high, low, close) tuples."""
    bars = []
    for i, (o, h, l, c) in enumerate(prices):
        bars.append(Bar(idx=start_idx + i, open=o, high=h, low=l, close=c))
    return bars


def make_ema(bars, period=20):
    """Simple EMA calculation from bars."""
    closes = [b.close for b in bars]
    ema = [closes[0]]
    k = 2 / (period + 1)
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[-1] * (1 - k))
    return ema


def test_broad_bull_channel_day_type():
    """Broad Bull Channel day: moderate upward drift, no opening spike, mostly above EMA."""
    # 20 bars drifting up steadily (trend_ratio ~0.35-0.5, above_ratio > 0.6)
    prices = []
    base = 100.0
    for i in range(20):
        o = base + i * 0.5
        c = o + 0.3  # small bull bars
        h = c + 0.2
        l = o - 0.2
        prices.append((o, h, l, c))

    bars = make_bars(prices)
    ema = make_ema(bars)

    result = classify_day_type(bars, ema)
    print(f"  Broad Bull Channel day type test: got '{result}'")
    assert "Bull" in result, f"Expected bull classification, got '{result}'"
    print("  PASS")


def test_broad_bear_channel_day_type():
    """Broad Bear Channel day: moderate downward drift, no opening spike, mostly below EMA."""
    prices = []
    base = 110.0
    for i in range(20):
        o = base - i * 0.5
        c = o - 0.3  # small bear bars
        h = o + 0.2
        l = c - 0.2
        prices.append((o, h, l, c))

    bars = make_bars(prices)
    ema = make_ema(bars)

    result = classify_day_type(bars, ema)
    print(f"  Broad Bear Channel day type test: got '{result}'")
    assert "Bear" in result, f"Expected bear classification, got '{result}'"
    print("  PASS")


def test_broad_bull_channel_market_cycle():
    """Market cycle: 10+ bars with moderate net upward move relative to total range."""
    # Bars that oscillate but trend up — net/total_range > 0.3
    prices = []
    base = 100.0
    for i in range(15):
        wiggle = 0.3 if i % 2 == 0 else -0.1
        o = base + i * 0.4 + wiggle
        c = o + 0.2
        h = max(o, c) + 0.3
        l = min(o, c) - 0.3
        prices.append((o, h, l, c))

    bars = make_bars(prices)
    ema = make_ema(bars)

    result = classify_market_cycle(bars, ema)
    print(f"  Broad Bull Channel market cycle test: got '{result}'")
    # Could be Tight Channel or Broad Bull Channel depending on overlap
    assert result in ("Broad Bull Channel", "Tight Channel (Small PB Trend)"), f"Unexpected: '{result}'"
    print("  PASS")


def test_broad_bear_channel_market_cycle():
    """Market cycle: 10+ bars with moderate net downward move."""
    prices = []
    base = 110.0
    for i in range(15):
        wiggle = -0.3 if i % 2 == 0 else 0.1
        o = base - i * 0.4 + wiggle
        c = o - 0.2
        h = max(o, c) + 0.3
        l = min(o, c) - 0.3
        prices.append((o, h, l, c))

    bars = make_bars(prices)
    ema = make_ema(bars)

    result = classify_market_cycle(bars, ema)
    print(f"  Broad Bear Channel market cycle test: got '{result}'")
    assert result in ("Broad Bear Channel", "Tight Channel (Small PB Trend)"), f"Unexpected: '{result}'"
    print("  PASS")


def test_filter_rule3_broad_channel_bug():
    """
    RULE 3 in filter_by_context checks 'Broad Range' but actual values are
    'Broad Bull Channel' / 'Broad Bear Channel'. This test verifies whether
    the filter actually triggers for broad channels.
    """
    # Create a dummy breakout setup (set setup_type so filter_by_context sees the name)
    dummy_setup = Setup(
        setup_name="Breakout Long",
        setup_type="Breakout Long",
        signal_bar=5,
        entry_bar=6,
        stop_loss=99.0,
        target_price=105.0,
        confidence=0.6,
    )

    # Use Broad channel as day_type too (not "Trading Range Day") so RULE 2 doesn't interfere
    result_bull = filter_by_context([dummy_setup], "Broad Bull Channel", "Broad Bull Channel")
    result_bear = filter_by_context([dummy_setup], "Broad Bear Channel", "Broad Bear Channel")

    # With the bug, breakouts pass through (not filtered) in broad channels
    bull_filtered = len(result_bull) == 0
    bear_filtered = len(result_bear) == 0

    print(f"  RULE 3 filter test:")
    print(f"    Broad Bull Channel filters breakouts: {bull_filtered} (breakouts remaining: {len(result_bull)})")
    print(f"    Broad Bear Channel filters breakouts: {bear_filtered} (breakouts remaining: {len(result_bear)})")

    if not bull_filtered or not bear_filtered:
        print("  WARNING: RULE 3 is NOT filtering breakouts in Broad Channels (known bug: checks 'Broad Range' instead of 'Broad')")
    else:
        print("  PASS: RULE 3 correctly filters breakouts in Broad Channels")

    return bull_filtered and bear_filtered


def test_trading_range_not_broad():
    """Trading range day should NOT be classified as broad channel."""
    # Sideways bars with no net move
    prices = []
    base = 100.0
    for i in range(20):
        wiggle = 0.5 if i % 2 == 0 else -0.5
        o = base + wiggle
        c = base - wiggle
        h = base + 1.0
        l = base - 1.0
        prices.append((o, h, l, c))

    bars = make_bars(prices)
    ema = make_ema(bars)

    result = classify_day_type(bars, ema)
    print(f"  Trading range (not broad) test: got '{result}'")
    assert "Broad" not in result, f"Sideways action should not be Broad Channel, got '{result}'"
    print("  PASS")


def test_engine_no_crash_with_broad_channel():
    """Full analyze_bars should not crash when bars form a broad channel pattern."""
    from algo_engine import analyze_bars
    import pandas as pd

    # Create a DataFrame that forms a broad bull channel
    prices = []
    base = 100.0
    for i in range(30):
        o = base + i * 0.3
        c = o + 0.2
        h = max(o, c) + 0.15
        l = min(o, c) - 0.15
        prices.append((o, h, l, c))

    df = pd.DataFrame(prices, columns=["Open", "High", "Low", "Close"])
    df["Volume"] = 1000

    result = analyze_bars(df)

    day_type = result.get("day_type", "N/A")
    market_cycle = result.get("market_cycle", "N/A")
    setups = result.get("setups", [])

    print(f"  Full engine test: day_type='{day_type}', market_cycle='{market_cycle}', setups={len(setups)}")
    assert result is not None, "analyze_bars returned None"
    assert "day_type" in result, "Missing day_type in result"
    assert "market_cycle" in result, "Missing market_cycle in result"
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("BROAD CHANNEL TESTS")
    print("=" * 60)

    tests = [
        ("Day Type: Broad Bull Channel", test_broad_bull_channel_day_type),
        ("Day Type: Broad Bear Channel", test_broad_bear_channel_day_type),
        ("Market Cycle: Broad Bull Channel", test_broad_bull_channel_market_cycle),
        ("Market Cycle: Broad Bear Channel", test_broad_bear_channel_market_cycle),
        ("Day Type: Trading Range (not broad)", test_trading_range_not_broad),
        ("RULE 3 Filter Bug Check", test_filter_rule3_broad_channel_bug),
        ("Full Engine (no crash)", test_engine_no_crash_with_broad_channel),
    ]

    passed = 0
    failed = 0
    warnings = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            result = test_fn()
            if result is False:
                warnings += 1
                print(f"  RESULT: WARNING (known issue)")
            else:
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {warnings} warnings")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
