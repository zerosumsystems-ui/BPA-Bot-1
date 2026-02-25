# BPA-Bot-1 — Trading Bot Trainer

A human-in-the-loop dashboard for training a Gemini-powered trading bot on **Al Brooks' Price Action** methodology.

## How It Works

1. The app loads a random S&P 500 stock's 5-minute chart
2. Gemini analyzes the chart and predicts 5 trade setups using Al Brooks' rules
3. You (the teacher) review, correct, or approve the bot's guesses
4. Your corrections are saved to `training_data.csv` and fed back into a living encyclopedia of rules

Over time, the bot learns your trading style grounded in price action theory.

## Stack

- **Frontend:** Streamlit with custom "Lion King Sunset" theme
- **Charts:** Plotly candlestick + EMA20 overlay
- **AI:** Google Gemini 3.1 Pro (vision — analyzes chart images)
- **Data:** yFinance for live S&P 500 5-minute OHLCV

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"
# Or add it to .streamlit/secrets.toml:
#   GEMINI_API_KEY = "your-key-here"

# Run
streamlit run app.py
```

## Deploying to Render

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com) connected to this repo
3. Render auto-detects the `render.yaml` Blueprint
4. Set `GEMINI_API_KEY` in the Render Environment tab
5. Deploy

See `render.yaml` for the full configuration.

## Project Structure

```
app.py                          # Main Streamlit application
render.yaml                     # Render.com deployment config
requirements.txt                # Python dependencies
brooks_encyclopedia_learnings.md # Living encyclopedia of Al Brooks rules (auto-updated)
training_data.csv               # Saved training rows (gitignored, generated at runtime)
.streamlit/
  config.toml                   # Streamlit theme (Lion King Sunset)
  secrets.toml                  # API keys (gitignored, local only)
```

## Built By

[Zero Sum Systems](https://github.com/zerosumsystems-ui)
