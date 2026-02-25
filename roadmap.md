# Algorithmic Trading Bot Transition Roadmap

This roadmap outlines the journey from your current **Streamlit Supervised Learning Trainer** to a **Live Algorithmic Execution Bot**. You are currently building the foundation in Phase 1.

---

## Phase 1: Data Gathering & Supervised Learning (Current Phase)

Your current Streamlit app (`app.py`) is the engine for this phase. You must train the AI to see the market exactly as you do using Al Brooks Price Action.

> [!IMPORTANT]
> The quality of a machine learning model is strictly bound by the quality of its training data ("Garbage in, garbage out"). Your primary job right now is human-in-the-loop validation.

### Action Items
1. **Accumulate Data:** Use the "Training Lab" tab to review, correct, and validate as many 5-minute daily charts as possible. Aim for a minimum of 500-1000 highly accurate days of data spanning different market conditions (bull markets, bear markets, high VIX days, low VIX days).
2. **Refine Edge Cases:** Continue adding notes to your `brooks_encyclopedia_learnings.md` whenever the bot completely fails to recognize a specific setup.
3. **Expand the Feature Set:** If you find yourself consistently looking at another metric (e.g., volume spikes, an RSI divergence, or the 200-SMA) to make decisions, we need to add that indicator to the Streamlit chart and the Bot's JSON prompt so it can "see" it too.

---

## Phase 2: Feature Engineering (Translating Pictures to Math)

Machine learning models (like XGBoost or Neural Networks) do not understand pictures of candlesticks. They require structured, numerical data arrays. 

In this phase, we will write a Python script that takes your tagged `training_data.csv` and converts every single 5-minute bar into a rich mathematical "Feature Vector."

### Action Items
1. **Calculate Core Features:** For every bar in your historical data, calculate localized metrics.
   - Example Features: `distance_from_EMA20`, `body_size_pct`, `upper_wick_pct`, `lower_wick_pct`, `prior_3_bars_trend`, `volume_rolling_zscore`.
2. **Context Windowing:** Give the model "memory." Instead of just feeding it the current bar, we feed it a "rolling window" of the last 10, 20, or 30 bars so it understands the *context* leading up to the current moment.
3. **Target Label Encoding:** We take your human-validated "Teacher Overrides" (e.g., "The Day Type was a Trading Range Day" or "Setup 1 was a High 2 Bull Flag at Bar 14") and convert them into binary `1` or `0` classifications that the AI can predict.

---

## Phase 3: Model Training & Backtesting (The Offline Sandbox)

Once we have a master CSV of numerical features mapped to your approved labels, we build the actual predictive model. This happens entirely offline on your local machine or a fast GPU cloud instance.

### Action Items
1. **Train an Ensemble Model:** We will likely use **XGBoost** (Extreme Gradient Boosting), which is currently the gold standard in Quantitative Finance for tabular data, as it is incredibly fast and highly immune to overfitting. We might also experiment with **LSTMs** (Long Short-Term Memory networks) specifically for analyzing the temporal sequence of the bars.
2. **Train a Multi-Head Architecture:** Instead of one massive brain, we train highly specialized models:
   - **Model A (The Classifier):** "What Day Type and Market Cycle are we currently in?"
   - **Model B (The Sniper):** "Given that we are in a Broad Bull Channel, is the *current* bar the exact triggering bar for a High 1 Setup?"
3. **Rigorous Backtesting:** We split your 1000 days of data: Train the model on 800 days, and strictly "test" it on the remaining 200 unseen days to see if its mathematical predictions match what you *would* have tagged visually.

---

## Phase 4: Paper Trading Infrastructure (The Live Simulation)

Once the model proves it can mathematically predict Al Brooks setups with >70% accuracy on historical data, we move to live, simulated markets. We completely bypass Streamlit for execution.

### Action Items
1. **Build the Execution Engine:** Write a lightweight, headless Python script (no user interface) that runs 24/7 on a cloud server.
2. **Connect to a Brokerage API:** Use a developer-friendly brokerage like **Alpaca** or **Interactive Brokers (IBKR)**.
3. **Real-Time Data Streaming:** Connect the bot to a live WebSockets feed. Every time a new 5-minute bar closes, the bot instantly calculates the Phase 2 mathematical features, feeds them to the Phase 3 XGBoost model, and gets a prediction in milliseconds.
4. **Simulate Orders:** If the model predicts a >85% probability of a valid setup, it sends a *simulated* Paper Trade to the API (e.g., a Limit Order at the 20-EMA, with a fixed Stop Loss 2 ticks below the signal bar).

> [!TIP]
> You will let this Paper Trading bot run for **at least 1-3 months** without touching it, strictly analyzing its win/loss ratio and drawdown metrics.

---

## Phase 5: Live Algorithmic Execution (Wall Street Deployment)

The final summit. If the paper trading bot is consistently profitable across different market conditions for months, you authorize it to trade real capital.

### Action Items
1. **Switch API Keys:** Flip the toggle from your Alpaca Paper Trading keys to your Live Production keys.
2. **Implement Hard Risk Controls:** Algorithmic trading is dangerous. A glitch can drain an account in seconds. We code strictly enforced "Circuit Breakers" directly into the bot:
   - *Example:* "If daily drawdown exceeds $500, immediately cancel all pending orders and shut down the Python script until tomorrow."
3. **Latency Optimization:** If you decide to trade on the 1-minute chart instead of the 5-minute chart, you move the Python bot off Render and onto a dedicated AWS server located as geographically close to the New York exchange servers as possible to reduce fiber-optic ping times.

---

## What to do next?
For now? **Keep correcting charts.** The smartest XGBoost model in the world is utterly useless without your deep, human-validated Al Brooks context in `training_data.csv`. 

Once you hit around **200 to 500 validated days** of `training_data.csv`, let me know, and we will begin coding Phase 2!
