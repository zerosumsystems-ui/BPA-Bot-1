<style>
  body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1E293B; line-height: 1.6; margin: 40px; background-color: #F8FAFC; }
  .header-box { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 40px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
  h1 { margin: 0; font-size: 26pt; font-weight: 800; }
  .subtitle { font-size: 13pt; opacity: 0.9; margin-top: 10px; font-weight: 300; }
  
  h2 { color: #0F172A; margin-top: 50px; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; font-size: 20pt; font-weight: 700; }
  .card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
  h3 { color: #334155; font-size: 15pt; margin-top: 0; }
  
  .badge { display: inline-block; padding: 5px 12px; border-radius: 9999px; font-size: 0.8em; font-weight: 700; margin-bottom: 15px; margin-right: 10px; letter-spacing: 0.5px; text-transform: uppercase; }
  .bull { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
  .bear { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
  .neutral { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
  .source { background: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; float: right; }
  
  .math { background: #F1F5F9; padding: 12px 18px; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 10.5pt; color: #475569; margin: 15px 0; border: 1px solid #E2E8F0; }
  .math b { color: #0F172A; }
  
  pre { background: #0F172A; color: #F8FAFC; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'JetBrains Mono', 'Menlo', monospace; font-size: 9pt; line-height: 1.4; }
  code { color: #EF4444; background: #FEF2F2; padding: 2px 6px; border-radius: 4px; font-family: 'Menlo', monospace; font-size: 0.95em; }
  pre code { color: inherit; background: none; padding: 0; }
</style>

<div class="header-box">
  <h1>🤖 Exhaustive Algorithm Logic Manual</h1>
  <div class="subtitle">Complete technical documentation mapping every single Al Brooks setup modeled in the AI Engine to its exact Python formula.</div>
</div>

<h2>Part I: The Core Engine (<code>algo_engine.py</code>)</h2>
<p>These algorithms run natively on the central parsing engine, detecting primary structure.</p>

<div class="card">
  <span class="badge source">algo_engine.py</span>
  <h3>1. Opening Spikes & Reversal Traps</h3>
  <span class="badge bull">Bull Spike</span> <span class="badge bear">Bear Spike</span> <span class="badge neutral">Reversal Trap</span>
  
  <p><b>Logic:</b> Scans the first 12 bars (first hour). Drops strict relative size requirements on the first 4 bars (9:30 AM drop) to track absolute momentum. If a strong trend bar is immediately swallowed by a massive opposite trend bar, it's a Trap.</p>
  <div class="math">
    <b>Baseline:</b> <code>avg_range = mean([last 20 bars range])</code><br>
    <b>Threshold:</b> <code>1.0x avg_range</code> (Bars 1-4) or <code>1.25x avg_range</code> (Bars 5+)<br>
    <b>Trigger:</b> Bar is a Strong Trend Bar AND <code>range >= (avg_range * threshold)</code> or <code>(i < 5 and bar.range >= avg_range * 0.5)</code>
  </div>
</div>

<div class="card">
  <span class="badge source">algo_engine.py</span>
  <h3>2. Standard Breakouts</h3>
  <span class="badge bull">Bull Breakout</span> <span class="badge bear">Bear Breakout</span>
  
  <p><b>Logic:</b> Triggers when the market forcibly breaks a localized 10-bar resistance/support wall with immense momentum, jumping the Moving Average.</p>
  <div class="math">
    <b>Extreme Check:</b> <code>bar.close > max(past 10 highs)</code><br>
    <b>Momentum Check:</b> <code>bar.is_strong_bull(avg_range)</code> AND crosses EMA.<br>
    <b>Continuation Check:</b> Re-triggered if the <i>previous</i> bar was strong and the current bar is a massive anomaly: <code>bar.range >= avg_range * 1.5</code>
  </div>
</div>

<div class="card">
  <span class="badge source">algo_engine.py</span>
  <h3>3. Trend Pullback Flags (H1/H2 / L1/L2)</h3>
  <span class="badge bull">H1/H2 Flag</span> <span class="badge bear">L1/L2 Flag</span>
  
  <p><b>Logic:</b> A rolling 5-bar counter that mathematically sums the number of pullback attempts. When the market breaks back in the trend direction, the counter issues the grade (H1, H2, or H3).</p>
  <div class="math">
    <b>Pullback Definition:</b> <code>b.is_bear OR b.low < prev.low OR b.low < prev2.low</code><br>
    <b>Trigger:</b> <code>bar.high > prev.high</code> AND the previous sequence contained pullbacks.
  </div>
</div>

<div class="card">
  <span class="badge source">algo_engine.py</span>
  <h3>4. Moving Average Gap Bars</h3>
  <span class="badge neutral">Exhaustion Scalp</span>
  
  <p><b>Logic:</b> Fades deep exhaustion pullbacks occurring after a trend was so strong that it stayed entirely clear of the 20-EMA for at least 20 consecutive candles.</p>
  <div class="math">
    <b>Constraint Check:</b> Iterate last 20 bars: <code>all(b.low > b.ema_20)</code><br>
    <b>Trigger:</b> The current bar finally touches: <code>bar.low <= bar.ema_20</code>
  </div>
</div>

<div class="card">
  <span class="badge source">algo_engine.py</span>
  <h3>5. Wedges & Double Bottoms/Tops</h3>
  <span class="badge neutral">Swing Reversal</span>
  
  <p><b>Logic:</b> Evaluates localized extremums derived from <code>find_swing_lows()</code>. A Wedge requires three monotonically ascending/descending swing points. A Double Top/Bottom requires two swing points whose absolute heights are within 0.3% mathematical variance.</p>
  <div class="math">
    <b>Wedge Bottom:</b> <code>lo3 < lo2 < lo1</code> (Three consecutive lower lows in swing array)<br>
    <b>Double Bottom:</b> <code>abs(lo1 - lo2) / ((lo1+lo2)/2) < 0.003</code>
  </div>
</div>

<div class="card">
  <span class="badge source">algo_engine.py</span>
  <h3>6. Tight Flags (ii / ioi)</h3>
  <span class="badge neutral">Equilibrium Breakout</span>
  
  <p><b>Logic:</b> Inside bars indicating extreme volatility contraction.</p>
  <div class="math">
    <b>Inside Bar:</b> <code>b.high <= prev.high AND b.low >= prev.low</code><br>
    <b>ii Trigger:</b> Current is Inside, Previous is Inside.<br>
    <b>ioi Trigger:</b> Current is Inside, Previous is Outside, Prev-Prev is Inside.
  </div>
</div>

<h2>Part II: Custom Strategy Plugins (<code>user_algos/</code>)</h2>
<p>Advanced, specific setups parsed dynamically through the sandbox directory.</p>

<div class="card">
  <span class="badge source">user_algos/best_setups.py</span>
  <h3>7. Major Trend Reversal (MTR)</h3>
  <span class="badge neutral">Trend Change</span>
  
  <p><b>Logic:</b> Detects the end of a dominant sequence. Requires an 80% trend adherence over 15 bars, a brutal MA violation, and a failed re-test of the absolute high/low.</p>
  <div class="math">
    <b>Prior Trend:</b> <code>sum(1 for b in past_15 if b.close < b.ema_20) >= 12</code><br>
    <b>MA Violator:</b> <code>any(b.close > b.ema_20 in recent_5)</code><br>
    <b>Failed Test:</b> <code>current.low >= trend_low * 0.995</code> AND it forms a strong Bull signal bar.
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/best_setups.py</span>
  <h3>8. Final Flag Reversals</h3>
  <span class="badge neutral">Exhaustion Failure</span>
  
  <p><b>Logic:</b> A tight sideways flag in a mature trend that achieves a breakout, but instantly fails the breakout.</p>
  <div class="math">
    <b>Tight Flag Check:</b> <code>flag_range < (bar.close * 0.015)</code> over a 5-bar sequence.<br>
    <b>Trigger:</b> <code>bar.high > flag_high</code> (Upside breakout) AND <code>bar.close < flag_low</code> (Immediate catastrophic failure).
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/best_setups.py</span>
  <h3>9. Breakout Pullbacks</h3>
  <span class="badge bull">BOPB</span>
  
  <p><b>Logic:</b> Validates that a massive breakout candle has successfully re-tested its origin wall and survived.</p>
  <div class="math">
    <b>Breakout Candle:</b> <code>range > base_range * 0.5</code> crossing the base wall.<br>
    <b>Trigger:</b> Current bar pulls down exactly to the wall <code>curr.low <= base_top AND curr.low >= (base_top * 0.998)</code> and closes above its midpoint.
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/limit_setups.py</span>
  <h3>10. Fading Weak H1/L1 Pullbacks</h3>
  <span class="badge neutral">Limit Order Scalp</span>
  
  <p><b>Logic:</b> Identifies a steep new trend and places limit orders to trap weak counter-trenders taking sub-par entries.</p>
  <div class="math">
    <b>Steep Trend:</b> <code>ema_now > ema_10_ago * 1.002</code><br>
    <b>Weak Pullback:</b> A bear bar whose <code>range() < ema * 0.002</code> (Microscopic push).<br>
    <b>Trigger:</b> Issues a Limit Order to BUY below the low of the weak bear bar, capturing their stop-loss failures.
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/limit_setups.py</span>
  <h3>11. Weak Breakout Tests</h3>
  <span class="badge neutral">Limit Order Trap</span>
  
  <p><b>Logic:</b> Fading breakouts that close poorly.</p>
  <div class="math">
    <b>Trigger:</b> Current bar breaks a local 5-bar extreme (e.g., <code>low < local_low</code>) but closes badly (<code>close > midpoint</code>). Issues limit order directly on the close.
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/limit_setups.py</span>
  <h3>12. Quiet Flag MA Entries</h3>
  <span class="badge neutral">Limit Order Baseline</span>
  
  <p><b>Logic:</b> Buying the Moving Average in dead-quiet horizontal periods if the macro trend is intact.</p>
  <div class="math">
    <b>Quiet Check:</b> 4-bar average range is <code>< ema * 0.0015</code><br>
    <b>Trigger:</b> <code>abs(close - ema) < avg_range</code>. Limit order issued at prior bar's extreme.
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/range_setups.py</span>
  <h3>13. Bear Stairs Exhaustion (3rd/4th Push)</h3>
  <span class="badge bull">Reversal Setup</span>
  
  <p><b>Logic:</b> Counting local minima in a stair-step decline. The 3rd or 4th push usually traps late bears.</p>
  <div class="math">
    <b>Counter:</b> Increments if a bar's low is surrounded by 2 higher lows on both sides.<br>
    <b>Trigger:</b> <code>minima >= 3</code> AND current bar is a strong bull reversal <code>body > range * 0.5</code>
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/range_setups.py</span>
  <h3>14. Spike & Channel Exhaustion</h3>
  <span class="badge neutral">Momentum Decay</span>
  
  <p><b>Logic:</b> Compares the average range of the early phase (the spike) to the average range of the late phase (the channel) over a 20-bar lookback.</p>
  <div class="math">
    <b>Trigger:</b> If <code>late_channel_average_range < spike_average_range * 0.5</code>, momentum has decayed by 50%. A strong opposite close triggers the fade.
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/range_setups.py</span>
  <h3>15. H1/L1 in Strong Fresh Spikes</h3>
  <span class="badge bull">Continuation</span>
  
  <p><b>Logic:</b> Entering on the very first pause of a monstrous breakout.</p>
  <div class="math">
    <b>Sequence:</b> Requires 3 consecutive strong trend bars with no overlap.<br>
    <b>Trigger:</b> Submits an H1 at the very first bar that prints a lower low than its predecessor but closes firmly up.
  </div>
</div>

<div class="card">
  <span class="badge source">user_algos/range_setups.py</span>
  <h3>16. Range Boundary Fades (2nd Entries)</h3>
  <span class="badge neutral">Range Reversal</span>
  
  <p><b>Logic:</b> Waiting for the edge of a verified 20-bar flat box to be touched and rejected.</p>
  <div class="math">
    <b>Flat Box Check:</b> <code>abs(ema_start - ema_end) < box_height * 0.2</code> (EMA is horizontal).<br>
    <b>Trigger:</b> Price pushes to <code>range_high * 0.998</code> but closes below its midpoint.
  </div>
</div>
