<style>
  body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1E293B; line-height: 1.6; margin: 40px; background-color: #F8FAFC; }
  .header-box { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 40px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
  h1 { margin: 0; font-size: 28pt; font-weight: 800; letter-spacing: -0.5px; }
  .subtitle { font-size: 14pt; opacity: 0.9; margin-top: 10px; font-weight: 300; }
  
  .phase-header { display: flex; align-items: center; margin-top: 40px; margin-bottom: 20px; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; }
  .phase-num { background: #3B82F6; color: white; border-radius: 50%; width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; font-size: 14pt; font-weight: bold; margin-right: 15px; }
  h2 { color: #0F172A; margin: 0; font-size: 22pt; font-weight: 700; }
  
  .theory-box { background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 15px 20px; font-style: italic; color: #1E40AF; margin-bottom: 25px; border-radius: 0 8px 8px 0; }
  
  .card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); }
  h3 { color: #334155; font-size: 16pt; margin-top: 0; display: flex; align-items: center; }
  
  .badge { display: inline-block; padding: 6px 14px; border-radius: 9999px; font-size: 0.85em; font-weight: 700; margin-bottom: 15px; margin-right: 10px; letter-spacing: 0.5px; text-transform: uppercase; }
  .bull { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
  .bear { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
  .neutral { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
  
  .math { background: #F1F5F9; padding: 12px 18px; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 11pt; color: #475569; margin: 15px 0; border: 1px solid #E2E8F0; }
  .math b { color: #0F172A; }
  
  pre { background: #0F172A; color: #F8FAFC; padding: 20px; border-radius: 10px; overflow-x: auto; font-family: 'JetBrains Mono', 'Menlo', monospace; font-size: 9.5pt; line-height: 1.5; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); }
  code { color: #EF4444; background: #FEF2F2; padding: 2px 6px; border-radius: 4px; font-family: 'Menlo', monospace; font-size: 0.95em; }
  pre code { color: inherit; background: none; padding: 0; }
</style>

<div class="header-box">
  <h1>🤖 AI Setup Logic: The Progression of a Trend</h1>
  <div class="subtitle">Mapping Al Brooks' Market Cycle directly to Python computational modeling.</div>
</div>

<div class="theory-box">
  "The market is always trying to break out, then trying to make every breakout fail. A breakout is a period of CERTAINTY — both sides agree current prices are wrong; the market moves quickly to find new uncertainty... The most important recurring pattern is the Spike and Channel, which eventually devolves into a Trading Range before reversing." — Al Brooks
</div>

<!-- PHASE 1 -->
<div class="phase-header">
  <div class="phase-num">1</div>
  <h2>The Breakout (The Spike)</h2>
</div>

<div class="card">
  <h3>Opening Drive Spikes & High-Momentum Breakouts</h3>
  <span class="badge bull">Bull Spike</span> <span class="badge bear">Bear Spike</span>
  
  <p><b>The Cycle State:</b> A period of absolute certainty. The market erupts from an equilibrium state (opening range or prior trading range). The goal here is to enter at the market or on a Stop order, as pullbacks are rare and shallow.</p>

  <div class="math">
    <b>Baseline Volatility:</b> <code>avg_range = mean([last 20 bars range])</code><br>
    <b>Momentum Threshold:</b> <code>1.0x avg_range</code> (Bars 1-4) or <code>1.25x avg_range</code> (Bars 5+)<br>
    <b>Algorithmic Trigger:</b> Bar is a Strong Trend Bar AND <code>range >= threshold</code>
  </div>
  <p><i>Note: To prevent the chaotic 9:30 AM print from blinding the moving average, the first 4 bars dynamically drop the strict scalar and accept any bar providing >50% of the opening volatility.</i></p>

<pre><code class="language-python"># algo_engine.py -> detect_opening_reversals_and_spikes()
threshold = 1.0 if i < 5 else 1.25
is_spike = bar.is_strong_bull(avg_range) and \
          (bar.range >= (avg_range * threshold) or (i < 5 and bar.range >= avg_range * 0.5))

if is_spike:
    setups.append(Setup(
        setup_name="Spike / Opening Breakout",
        order_type="Stop/Market", # Buy the close of the strong spike
        confidence=0.60
    ))</code></pre>
</div>


<!-- PHASE 2 -->
<div class="phase-header">
  <div class="phase-num">2</div>
  <h2>The First Pullback (Transition to Channel)</h2>
</div>

<div class="card">
  <h3>High 1 / Low 1 Flags & Breakout Pullbacks</h3>
  <span class="badge bull">Bull Flag (H1/H2)</span> <span class="badge bear">Bear Flag (L1/L2)</span>

  <p><b>The Cycle State:</b> The initial Spike is exhausted, and early profit-takers step in, creating the first pullback. This first pullback forms the start of the "Channel" phase. The algorithm natively tracks localized pullbacks to fade weak counter-trenders.</p>

  <div class="math">
    <b>Pullback Definition:</b> A bar with `close < open` (Bear Bar) OR `low < low[last 2 bars]`<br>
    <b>Resumption Trigger:</b> The exact bar that breaks `high > prev.high`<br>
    <b>Grading Lookup:</b> Count occurrences of the "Pullback Definition" over a rolling 5-bar window to assign H1 or H2.
  </div>

<pre><code class="language-python"># algo_engine.py -> detect_high_low_flags()
def is_pullback(b, prev, prev2):
    return b.is_bear or b.low < prev.low or b.low < prev2.low

if bar.high > prev.high and is_pullback(prev, bars[i-2], bars[i-3]):
    # Look back 5 bars to count total recent pullbacks
    num_pb = sum(1 for j in range(max(0, i-5), i) 
                 if is_pullback(bars[j], bars[max(0, j-1)], bars[max(0, j-2)]))
                 
    flag_num = min(3, max(1, num_pb))
    setups.append(Setup(name=f"High {flag_num} Bull Flag", order_type="Stop"))
</code></pre>
</div>


<!-- PHASE 3 -->
<div class="phase-header">
  <div class="phase-num">3</div>
  <h2>The Trading Range (Equilibrium)</h2>
</div>

<div class="card">
  <h3>Fading Opening Reversals & Limit Order Traps</h3>
  <span class="badge neutral">Limit Order Scalp</span> <span class="badge bear">High-Probability Trap</span>

  <p><b>The Cycle State:</b> The Channel has lost its steepness and devolved into a Trading Range (uncertainty). In a trading range, <i>most breakout attempts fail</i>. The algorithm switches to fading weak breakouts using Limit Orders.</p>

  <div class="math">
    <b>Trap Logic:</b> Sequence of <code>[Massive Trend Bar] -> [Opposite Massive Trend Bar]</code><br>
    <b>Weak Pullback Logic:</b> <code>range < ema * 0.002</code>
  </div>

<pre><code class="language-python"># user_algos/limit_setups.py
if is_spike_bull and prev_is_spike_bear:
     # The Bears spiked down, but Bulls immediately spiked it back up! The Bears are trapped.
     setups.append(Setup(
         setup_name="Opening Reversal (Bull Trap)",
         order_type="Stop/Limit", # Fade the weak bears
         confidence=0.70 # Exceedingly high probability layout
     ))
</code></pre>
</div>


<!-- PHASE 4 -->
<div class="phase-header">
  <div class="phase-num">4</div>
  <h2>The Reversal</h2>
</div>

<div class="card">
  <h3>Major Trend Reversals (MTR)</h3>
  <span class="badge neutral">Trend Change</span>

  <p><b>The Cycle State:</b> The Trading Range has exhausted the dominant side. A Major Trend Reversal begins with a strong break of the Moving Average, followed by a failed re-test of the absolute extreme. A new Spike is about to begin in the opposite direction.</p>

  <div class="math">
    <b>Trend Validation:</b> A multi-leg push generating an absolute extreme (Swing High or Low).<br>
    <b>MA Break:</b> A strong close clearly crossing the 20-bar EMA.<br>
    <b>Test & Failure:</b> A subsequent push that tests the absolute extreme but fails to maintain the trend.
  </div>

<pre><code class="language-python"># user_algos/best_setups.py
abs_high = max(b.high for b in bars[max(0, i-30):i])
# Was the highest point at least 5 bars ago? (Significant pullback required)
if recent_high.idx <= bar.idx - 5:
    # Did we cross the EMA during the pullback?
    if min(b.low for b in bars[recent_high.idx:i]) < ema[i-2]:
        if bar.is_strong_bear(avg_range): # The re-test failed with a strong bear bar
            setups.append(Setup(name="Major Trend Reversal", order_type="Stop", conf=0.70))
</code></pre>
</div>
