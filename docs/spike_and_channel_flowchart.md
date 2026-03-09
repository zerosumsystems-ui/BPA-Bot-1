# Spike and Channel — BPA Flowchart

> Al Brooks considers Spike and Channel one of the most important trend patterns.
> Every trend day is a variation of Spike and Channel.

---

## Overview

**Spike and Channel** is a two-phase trend pattern:
1. **Spike Phase** — Strong breakout with consecutive trend bars (the "surprise")
2. **Channel Phase** — Weaker continuation trend after the spike loses momentum

The channel eventually exhausts (usually via a wedge/3-push pattern) and evolves into a Trading Range or reversal.

---

## Flowchart: Spike and Channel Detection & Trading

```
┌─────────────────────────────────────────────────┐
│              START: NEW BAR ARRIVES              │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: DETECT SPIKE                           │
│                                                 │
│  Is there a cluster of 3+ consecutive bars      │
│  closing in the same direction with bodies      │
│  > 50% of range?                                │
│                                                 │
│  Bull Spike: 3+ bull bars closing near highs    │
│  Bear Spike: 3+ bear bars closing near lows     │
└──────────┬──────────────────┬───────────────────┘
           │ YES              │ NO
           ▼                  ▼
┌──────────────────┐   ┌──────────────┐
│  SPIKE DETECTED  │   │  No pattern  │
│                  │   │  — STOP —    │
│  Record:         │   └──────────────┘
│  • Spike High    │
│  • Spike Low     │
│  • Spike Size    │
│  • Direction     │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: CHECK FOR BODY GAPS                    │
│                                                 │
│  Do the spike bars have body gaps?              │
│  (close[n] > open[n+1] for bull,                │
│   close[n] < open[n+1] for bear)                │
│                                                 │
│  Body gaps = Measuring gaps → VERY strong trend  │
│  (Small Pullback Trend, not just a channel)     │
└──────────┬──────────────────┬───────────────────┘
           │ YES              │ NO
           ▼                  ▼
┌──────────────────┐   ┌──────────────────────────┐
│ STRONG SPIKE     │   │ NORMAL SPIKE             │
│ (Small PB Trend) │   │ (Standard S&C)           │
│                  │   │                          │
│ Only trade WITH  │   │ Channel phase will be    │
│ the trend.       │   │ weaker — both sides can  │
│ Never fade.      │   │ profit in the channel.   │
└────────┬─────────┘   └────────────┬─────────────┘
         │                          │
         └──────────┬───────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: WAIT FOR CHANNEL TRANSITION            │
│                                                 │
│  After the spike, monitor for momentum decay:   │
│                                                 │
│  • Bar ranges shrink (< 50% of spike avg)       │
│  • Bars start overlapping more                  │
│  • Pullbacks to EMA begin occurring             │
│  • Bull/bear bars start alternating             │
│                                                 │
│  Channel starts when avg bar range drops below  │
│  50% of the spike's average bar range.          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 4: CLASSIFY CHANNEL TYPE                  │
│                                                 │
│  ┌─────────────────┐  ┌──────────────────────┐  │
│  │  TIGHT CHANNEL  │  │   BROAD CHANNEL      │  │
│  │                 │  │                      │  │
│  │ • No close past │  │ • Bars close past    │  │
│  │   EMA on wrong  │  │   EMA occasionally   │  │
│  │   side          │  │ • Deeper pullbacks   │  │
│  │ • Small PBs     │  │ • Both sides profit  │  │
│  │ • ONLY trade    │  │ • Trade PBs to EMA   │  │
│  │   with trend    │  │   (with trend) AND   │  │
│  │                 │  │   fades at channel   │  │
│  │                 │  │   extremes           │  │
│  └─────────────────┘  └──────────────────────┘  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 5: CALCULATE MEASURED MOVE TARGET         │
│                                                 │
│  Bull S&C:                                      │
│    MM Target = Spike High + Spike Size          │
│    (Double the initial spike from its high)     │
│                                                 │
│  Bear S&C:                                      │
│    MM Target = Spike Low - Spike Size           │
│    (Double the initial spike from its low)      │
│                                                 │
│  Plot MM target as reference line on chart.     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 6: MONITOR FOR CHANNEL EXHAUSTION         │
│                                                 │
│  Count pushes in the channel direction:         │
│                                                 │
│  • Push = swing high (bull) / swing low (bear)  │
│    that extends the channel                     │
│  • 3 pushes = WEDGE pattern = exhaustion likely │
│                                                 │
│  Also watch for:                                │
│  • Climax bar at MM target                      │
│  • Bars with large tails against trend          │
│  • Shrinking momentum (each push smaller)       │
│  • Body gaps closing (if they existed)          │
└──────────┬──────────────────┬───────────────────┘
           │                  │
           │ 3+ Pushes        │ < 3 Pushes
           │ OR MM Hit        │
           ▼                  ▼
┌────────────────────┐  ┌─────────────────────────┐
│  EXHAUSTION SIGNAL │  │  CHANNEL CONTINUES      │
│                    │  │                         │
│  Look for reversal │  │  Keep trading with      │
│  bar:              │  │  trend. Buy PBs to EMA  │
│  • Strong bar      │  │  (bull) or sell rallies  │
│    against trend   │  │  to EMA (bear).         │
│  • Closes past     │  │                         │
│    midpoint        │  │  Go back to Step 6.     │
│  • At/near MM      │  └─────────────────────────┘
│    target          │
└────────┬───────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  STEP 7: TRADE THE EXHAUSTION                   │
│                                                 │
│  Bull S&C Exhaustion (SELL):                    │
│  • Entry: Sell stop 1 tick below signal bar low │
│  • Stop: 1 tick above signal bar high           │
│  • Target: EMA or start of channel              │
│                                                 │
│  Bear S&C Exhaustion (BUY):                     │
│  • Entry: Buy stop 1 tick above signal bar high │
│  • Stop: 1 tick below signal bar low            │
│  • Target: EMA or start of channel              │
│                                                 │
│  Expect: At least 2 legs sideways/down (bull)   │
│          or 2 legs sideways/up (bear)           │
│                                                 │
│  After S&C exhaustion → Trading Range likely    │
│  (not immediate opposite trend)                 │
└─────────────────────────────────────────────────┘
```

---

## Key Rules Summary

| # | Rule |
|---|------|
| 1 | Two phases: Strong spike → weaker channel |
| 2 | Body gaps in spike = measuring gaps → very strong trend |
| 3 | Body gaps closing = weaker (channel, not Small PB Trend) |
| 4 | Channel typically ends with wedge (3 pushes) |
| 5 | MM target = spike size doubled from spike extreme |
| 6 | Only buy in tight bull channel (never sell) |
| 7 | Both sides can profit in broader channels |
| 8 | S&C patterns are fractal — they nest within each other |
| 9 | 40%+ chance of trend day after major surprise bar |
| 10 | After S&C exhaustion, expect at least 2 legs of correction |

---

## Decision Tree: How to Trade Each Phase

```
                    SPIKE DETECTED
                         │
              ┌──────────┴──────────┐
              │                     │
         WITH TREND             AGAINST TREND
              │                     │
              ▼                     ▼
     ┌────────────────┐    ┌────────────────┐
     │ Buy H1/L1      │    │ DO NOT TRADE   │
     │ (1st pullback   │    │ against fresh  │
     │  in spike)      │    │ spike. Wait    │
     │                 │    │ for channel.   │
     │ Very high prob. │    └────────────────┘
     │ 85%+ WR         │
     └────────────────┘

              CHANNEL PHASE
                   │
        ┌──────────┴──────────┐
        │                     │
   TIGHT CHANNEL        BROAD CHANNEL
        │                     │
        ▼                     ▼
  ┌──────────────┐    ┌───────────────────┐
  │ ONLY WITH    │    │ WITH TREND:       │
  │ TREND        │    │  Buy PB to EMA    │
  │              │    │                   │
  │ Buy PB to   │    │ COUNTER TREND:    │
  │ EMA (bull)   │    │  Fade at channel  │
  │              │    │  boundary (limit  │
  │ Sell rally   │    │  order at prior   │
  │ to EMA       │    │  high/low)        │
  │ (bear)       │    └───────────────────┘
  └──────────────┘

         CHANNEL EXHAUSTION (3 PUSHES / MM HIT)
                         │
                         ▼
              ┌──────────────────────┐
              │  FADE THE CHANNEL    │
              │                      │
              │  Expect:             │
              │  • 2 legs correction │
              │  • Trading Range     │
              │  • NOT immediate     │
              │    opposite trend    │
              └──────────────────────┘
```

---

## Indicator Companion

See `tradingview_indicators/11_spike_and_channel.pine` for the TradingView implementation that:
- Detects bull and bear spikes (3+ consecutive trend bars)
- Marks the spike-to-channel transition
- Plots measured move targets
- Counts channel pushes (wedge detection)
- Alerts on channel exhaustion at 3rd push or MM target
