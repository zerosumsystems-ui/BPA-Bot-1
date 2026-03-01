# Quantitative Alpha Strategies: Algorithmic Implementation of Institutional Price Action Frameworks

**CONFIDENTIAL INTERNAL MEMORANDUM**
**Distribution: Quantitative Trading Desk & Portfolio Management**

## Executive Summary
This memorandum outlines the successful mathematical codification of three distinct, high-probability price action architectures into automated execution logic. Derived from verified methodologies authored by Al Brooks, these algorithms isolate deep market inefficiencies through real-time statistical analysis of price momentum, mean reversion tendencies, and complex multi-leg corrections.

The resultant models minimize emotional interference, enforcing strict probabilistic entry criteria aligned with institutional execution standards.

---

## 1. Multi-Leg Trend Continuation ("High 2 / Low 2" Matrix)
**Module:** `template_algo.py` | **Target Environment:** Established Directional Momentum

**Architectural Thesis:** 
Efficient markets rarely re-rate linearly. During persistent directional trends, minor counter-trend flow momentarily depresses asset prices before institutional participants aggressively re-enter. A single reversal attempt ("High 1") carries lower statistical edge due to potential deeper retracements. A secondary, successive failure of counter-trend liquidity to break structure ("High 2") confirms the dominant trend remains highly intact.

**Algorithmic Execution Logic:**
The engine establishes a primary directional bias by evaluating short-term flow against longer-period exponential moving averages (EMA). It simultaneously maintains stateful arrays of local swing extrema. Upon detecting two distinct counter-trend legs inside the broader directional move, followed immediately by a strong momentum candle closing near its extreme in the direction of the primary trend, the algorithm mathematically validates the "High 2 / Low 2" setup and executes a continuation entry. 

**Statistical Edge:** This model avoids premature entries and forces the market to prove secondary exhaustion of counter-trend participants before committing capital.

---

## 2. Exhaustion Extrema Detection ("Wedge Formation")
**Module:** `advanced_setups.py` | **Target Environment:** Trend Termination / Reversal 

**Architectural Thesis:**
Prolonged directional moves often terminate in parabolic exhaustion rather than smooth transitions. This exhaustion frequently manifests as three distinct, diminishing pushes into an extreme price level. Each successive push covers less distance, heavily indicating dying momentum and an impending, aggressive mean-reversion event as late participants are trapped at the extrema.

**Algorithmic Execution Logic:**
The `detect_wedge_patterns` construct continuously scans historical swing extrema (both local valleys and peaks). The algorithm strictly requires:
1. Three consecutive, distinct extensions against an established pullback structure.
2. Diminishing angular velocity between the pushes.
3. A confirmed, robust reversal "Signal Bar" rejecting the third push and closing heavily in the opposite direction.

**Statistical Edge:** By enforcing a strict 3-push variable constraint paired with an immediate contrarian signal candle, the algorithm isolates high-variance, asymmetric R:R (Risk-to-Reward) reversal setups, capitalizing on trapped retail liquidity.

---

## 3. Dynamic Mean Reversion ("20-EMA Rubber Band")
**Module:** `advanced_setups.py` | **Target Environment:** Accelerated Momentum

**Architectural Thesis:**
The 20-period Exponential Moving Average (EMA) operates as a dynamic measure of short-term institutional fair value. When an asset experiences severe momentum divergence from this median, statistical gravity eventually forces a reversion to the mean. Pullbacks that touch the 20-EMA during strong trends represent the highest-probability zones for institutional rebuying.

**Algorithmic Execution Logic:**
The `detect_two_legged_pullbacks` engine implements a strict 15-to-20 intra-bar lookback array to confirm unilateral price acceptance either completely above or below the EMA. Once trend strength is mathematically verified, the algorithm idles until a solitary candle retraces to physically intersect the EMA boundary (enforced by a microscopic 0.2% padding tolerance). If the ensuing candle violently rejects the EMA and closes favorably, the engine triggers a continuation signal.

**Statistical Edge:** This model provides maximum structural safety by utilizing standard deviations of dynamic support/resistance, minimizing stop-loss distances while maximizing potential directional yield.
