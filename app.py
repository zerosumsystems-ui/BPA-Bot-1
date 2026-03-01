"""
Human-in-the-Loop Trading Bot Trainer
Built with Streamlit, Databento, Plotly, and Gemini.
"""
from __future__ import annotations

import os
import io
import json
import logging
import random
import datetime
import pathlib
from pathlib import Path
import time
import concurrent.futures
import threading
import asyncio

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from algo_engine import analyze_bars
from data_source import get_data_source
from google import genai

# ─────────────────────────── CONFIG ──────────────────────────────────────────

st.set_page_config(
    page_title="Trading Bot Trainer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use DATA_DIR env var for persistent storage (Render Disk mount)
# Locally defaults to "." so nothing changes for local dev
DATA_DIR = pathlib.Path(os.environ.get("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = pathlib.Path(__file__).parent
ENCYCLOPEDIA_PATH = BASE_DIR / "brooks_encyclopedia_learnings.md"

TRAINING_CSV = DATA_DIR / "training_data.csv"
DO_NOT_TRADE_JSON = DATA_DIR / "do_not_trade.json"

CSV_COLUMNS = [
    "timestamp", "ticker",
    "bot_day_type", "bot_market_cycle",
    "bot_setup_1", "bot_setup_1_bar", "bot_setup_1_price", "bot_setup_1_order_type",
    "bot_setup_2", "bot_setup_2_bar", "bot_setup_2_price", "bot_setup_2_order_type",
    "bot_setup_3", "bot_setup_3_bar", "bot_setup_3_price", "bot_setup_3_order_type",
    "bot_setup_4", "bot_setup_4_bar", "bot_setup_4_price", "bot_setup_4_order_type",
    "bot_setup_5", "bot_setup_5_bar", "bot_setup_5_price", "bot_setup_5_order_type",
    "bot_setup_6", "bot_setup_6_bar", "bot_setup_6_price", "bot_setup_6_order_type",
    "bot_setup_7", "bot_setup_7_bar", "bot_setup_7_price", "bot_setup_7_order_type",
    "bot_setup_8", "bot_setup_8_bar", "bot_setup_8_price", "bot_setup_8_order_type",
    "bot_setup_9", "bot_setup_9_bar", "bot_setup_9_price", "bot_setup_9_order_type",
    "bot_setup_10", "bot_setup_10_bar", "bot_setup_10_price", "bot_setup_10_order_type",
    "bot_setup_11", "bot_setup_11_bar", "bot_setup_11_price", "bot_setup_11_order_type",
    "bot_setup_12", "bot_setup_12_bar", "bot_setup_12_price", "bot_setup_12_order_type",
    "bot_setup_13", "bot_setup_13_bar", "bot_setup_13_price", "bot_setup_13_order_type",
    "bot_setup_14", "bot_setup_14_bar", "bot_setup_14_price", "bot_setup_14_order_type",
    "bot_setup_15", "bot_setup_15_bar", "bot_setup_15_price", "bot_setup_15_order_type",
    "bot_action", "bot_confidence",
    "override_day_type", "override_market_cycle",
    "override_setup_1", "override_setup_1_bar", "override_setup_1_price", "override_setup_1_order_type",
    "override_setup_2", "override_setup_2_bar", "override_setup_2_price", "override_setup_2_order_type",
    "override_setup_3", "override_setup_3_bar", "override_setup_3_price", "override_setup_3_order_type",
    "override_setup_4", "override_setup_4_bar", "override_setup_4_price", "override_setup_4_order_type",
    "override_setup_5", "override_setup_5_bar", "override_setup_5_price", "override_setup_5_order_type",
    "override_setup_6", "override_setup_6_bar", "override_setup_6_price", "override_setup_6_order_type",
    "override_setup_7", "override_setup_7_bar", "override_setup_7_price", "override_setup_7_order_type",
    "override_setup_8", "override_setup_8_bar", "override_setup_8_price", "override_setup_8_order_type",
    "override_setup_9", "override_setup_9_bar", "override_setup_9_price", "override_setup_9_order_type",
    "override_setup_10", "override_setup_10_bar", "override_setup_10_price", "override_setup_10_order_type",
    "override_setup_11", "override_setup_11_bar", "override_setup_11_price", "override_setup_11_order_type",
    "override_setup_12", "override_setup_12_bar", "override_setup_12_price", "override_setup_12_order_type",
    "override_setup_13", "override_setup_13_bar", "override_setup_13_price", "override_setup_13_order_type",
    "override_setup_14", "override_setup_14_bar", "override_setup_14_price", "override_setup_14_order_type",
    "override_setup_15", "override_setup_15_bar", "override_setup_15_price", "override_setup_15_order_type",
    "override_action", "teacher_notes",
]

# ─────────────────────────── FORM OPTIONS ────────────────────────────────────

DAY_TYPE_OPTIONS = [
    "Approve Bot's Guess",
    "Trading Range Day",
    "Triangle Trading Range Day",
    "Bull Trend From The Open",
    "Bear Trend From The Open",
    "Trending Trading Range Day (Bull)",
    "Trending Trading Range Day (Bear)",
    "Small Pullback Bull Trend",
    "Small Pullback Bear Trend",
    "Spike and Channel Bull Trend",
    "Spike and Channel Bear Trend",
    "Broad Bull Channel",
    "Broad Bear Channel",
    "Shrinking Stairs",
    "Reversal Day (Bull)",
    "Reversal Day (Bear)",
    "Crash Day",
    "Climax Day",
]

MARKET_CYCLE_OPTIONS = [
    "Approve Bot's Guess",
    "Breakout (Spike)",
    "Micro Channel",
    "Tight Channel (Small PB Trend)",
    "Broad Bull Channel",
    "Broad Bear Channel",
    "Trading Range",
]

SETUP_OPTIONS = [
    "Approve Bot's Guess",
    "High 1 Bull Flag",
    "High 2 Bull Flag",
    "High 3 Bull Flag",
    "High 4 Bull Flag",
    "Low 1 Bear Flag",
    "Low 2 Bear Flag",
    "Low 3 Bear Flag",
    "Low 4 Bear Flag",
    "Double Bottom",
    "Double Top",
    "Higher Low Double Bottom",
    "Lower Low Double Bottom",
    "Lower High Double Top",
    "Higher High Double Top",
    "Major Trend Reversal (Bull)",
    "Major Trend Reversal (Bear)",
    "Wedge Bottom",
    "Wedge Top",
    "Parabolic Wedge Bottom",
    "Parabolic Wedge Top",
    "Spike and Channel Bull",
    "Spike and Channel Bear",
    "Head & Shoulders Bottom",
    "Head & Shoulders Top",
    "Final Bull Flag",
    "Final Bear Flag",
    "Breakout (BO)",
    "Breakout Test",
    "Failed Breakout (Bull Trap)",
    "Failed Breakout (Bear Trap)",
    "Measuring Gap / Exhaustion Gap",
    "Buy Climax",
    "Sell Climax",
    "Ledge Bottom",
    "Ledge Top",
    "ii Pattern",
    "ioi Pattern",
    "OO Pattern",
    "Opening Reversal (Bull)",
    "Opening Reversal (Bear)",
    "20-Gap Bar Buy",
    "20-Gap Bar Sell",
    "Cup and Handle",
]

ACTION_OPTIONS = [
    "Approve Bot's Guess",
    "Buy",
    "Sell",
    "Wait / No Trade",
]

ORDER_OPTIONS = [
    "Approve Bot's Guess",
    "Stop",
    "Limit",
]

# ─────────────────────────── API KEYS ────────────────────────────────────────

def get_api_key() -> str:
    """Load GEMINI_API_KEY from the environment, Streamlit secrets, or Render secret files."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    
    # Check Render secret file
    render_secret = Path("/etc/secrets/GEMINI_API_KEY")
    if render_secret.exists():
        return render_secret.read_text().strip()
        
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return key
    except (FileNotFoundError, KeyError):
        pass
    return ""


def get_databento_key() -> str:
    """Load DATABENTO_API_KEY from the environment, Streamlit secrets, or Render secret files."""
    key = os.environ.get("DATABENTO_API_KEY")
    if key:
        return key
        
    # Check Render secret file
    render_secret = Path("/etc/secrets/DATABENTO_API_KEY")
    if render_secret.exists():
        return render_secret.read_text().strip()
        
    try:
        key = st.secrets["DATABENTO_API_KEY"]
        if key:
            return key
    except (FileNotFoundError, KeyError):
        pass
    return ""


@st.cache_resource
def _init_data_source_v2():
    """
    Initialize the data source once per app session.
    (Cache invalidated to force Databento loading over yfinance fallback)
    """
    source_type = os.environ.get("DATA_SOURCE", "auto")
    db_key = get_databento_key()
    return get_data_source(source_type, api_key=db_key)


# ─────────────────────────── S&P 500 LIST ────────────────────────────────────

@st.cache_data(ttl=86400)
def get_sp500_tickers() -> list[str]:
    """Scrape the S&P 500 ticker list from Wikipedia using pandas."""
    import requests as _req
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = _req.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    dnt = load_do_not_trade()
    return [t for t in tickers if t not in dnt]

def load_do_not_trade() -> set[str]:
    """Load the list of tickers to exclude from training."""
    if DO_NOT_TRADE_JSON.exists():
        try:
            return set(json.loads(DO_NOT_TRADE_JSON.read_text()))
        except Exception:
            pass
    return set()

def add_to_do_not_trade(ticker: str):
    """Add a ticker to the exclusion list and force a cache reset."""
    dnt = load_do_not_trade()
    dnt.add(ticker)
    DO_NOT_TRADE_JSON.write_text(json.dumps(list(dnt)))
    get_sp500_tickers.clear()

# ─────────────────────────── DATA FETCHING ───────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def fetch_chart_data_v2(ticker: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame | None:
    """Fetch 5-minute OHLCV data for *ticker* using the configured data source (Databento → yFinance fallback)."""
    try:
        source = _init_data_source_v2()
        df = source.fetch_historical(ticker, start_date, end_date)

        if df is None or df.empty:
            return None
        # Flatten multi-level columns if present (yFinance fallback may produce these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        # Create a sequential Bar Number for the day, starting at 1
        df["BarNumber"] = range(1, len(df) + 1)
        return df
    except Exception:
        return None


def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Build a responsive candlestick + EMA20 Plotly chart."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["BarNumber"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#4ade80", # Brighter green
        decreasing_line_color="#f87171", # Brighter red
        increasing_fillcolor="#4ade80",
        decreasing_fillcolor="#f87171",
    ))

    fig.add_trace(go.Scatter(
        x=df["BarNumber"],
        y=df["EMA20"],
        mode="lines",
        name="EMA 20",
        line=dict(color="#60a5fa", width=2.5), # Thicker brighter blue
    ))

    fig.update_layout(
        title=go.layout.Title(text=f"{ticker} · 5-Min Chart (1 Day)", font=dict(size=20, color="#4a2311")),
        paper_bgcolor="#ffebd2", # Lion King Peach
        plot_bgcolor="#ffebd2",
        xaxis=go.layout.XAxis(
            rangeslider=dict(visible=False),
            type="linear",
            title=go.layout.xaxis.Title(text="Bar Number (5-Min)", font=dict(size=14, color="#4a2311")),
            gridcolor="#ffcc99", # Soft amber gridlines
            dtick=5, # Show a large number every 5 bars
            minor=dict(
                dtick=1, # Draw a faint gridline for every single bar (1-78)
                gridcolor="rgba(255, 204, 153, 0.4)",
            ),
            tick0=1,
            tickfont=dict(size=12, color="#7c4a2a"), # Rich lighter brown
        ),
        yaxis=go.layout.YAxis(
            gridcolor="#ffcc99", 
            title=go.layout.yaxis.Title(text="Price", font=dict(size=14, color="#4a2311")),
            tickfont=dict(size=12, color="#7c4a2a"),
        ),
        legend=go.layout.Legend(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=14, color="#4a2311")),
        margin=go.layout.Margin(l=10, r=10, t=60, b=10),
        height=600, # Taller chart
        autosize=True,
    )

    return fig


def build_trade_chart(df: pd.DataFrame, trade, ticker: str, is_daily: bool = False) -> go.Figure:
    """Build a candlestick chart highlighting a single trade's entry, exit, stop, and target."""
    # For daily charts, use the DataFrame index directly; for 5m, use BarNumber
    if is_daily:
        # Slice a window around the trade: 10 bars before entry to 10 bars after exit
        bar_indices = list(range(len(df)))
        pad_before = 10
        pad_after = 10
        start_idx = max(0, trade.entry_bar - 1 - pad_before)
        end_idx = min(len(df), trade.exit_bar + pad_after)
        window_df = df.iloc[start_idx:end_idx].copy()
        window_df["_x"] = range(len(window_df))
        x_vals = window_df["_x"]
        # Relative bar positions for the trade within the window
        entry_x = trade.entry_bar - 1 - start_idx
        exit_x = trade.exit_bar - 1 - start_idx if trade.exit_bar > 0 else entry_x
        x_title = "Bar"
        chart_title = f"{ticker} Daily -- {trade.setup_name} ({trade.direction})"
        # Build date labels for hover
        date_labels = [str(d.date()) if hasattr(d, 'date') else str(d) for d in window_df.index]
    else:
        window_df = df.copy()
        if "BarNumber" not in window_df.columns:
            window_df["BarNumber"] = range(1, len(window_df) + 1)
        x_vals = window_df["BarNumber"]
        entry_x = trade.entry_bar
        exit_x = trade.exit_bar if trade.exit_bar > 0 else entry_x
        x_title = "Bar Number (5-Min)"
        chart_title = f"{ticker} -- {trade.setup_name} ({trade.direction})"
        date_labels = None

    fig = go.Figure()

    # Highlight trade range with a shaded rectangle
    fig.add_vrect(
        x0=entry_x - 0.5, x1=exit_x + 0.5,
        fillcolor="rgba(100, 149, 237, 0.15)",
        line_width=0,
        layer="below",
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=x_vals,
        open=window_df["Open"],
        high=window_df["High"],
        low=window_df["Low"],
        close=window_df["Close"],
        name="OHLC",
        increasing_line_color="#4ade80",
        decreasing_line_color="#f87171",
        increasing_fillcolor="#4ade80",
        decreasing_fillcolor="#f87171",
    ))

    # EMA if available
    if "EMA20" in window_df.columns:
        fig.add_trace(go.Scatter(
            x=x_vals, y=window_df["EMA20"],
            mode="lines", name="EMA 20",
            line=dict(color="#60a5fa", width=2),
        ))

    # Entry marker
    fig.add_trace(go.Scatter(
        x=[entry_x], y=[trade.entry_price],
        mode="markers",
        marker=dict(symbol="triangle-up" if trade.direction == "Long" else "triangle-down",
                    size=14, color="#2196F3"),
        name="Entry",
    ))

    # Exit marker
    if trade.exit_bar > 0:
        exit_color = "#00C853" if trade.is_winner else "#FF1744"
        fig.add_trace(go.Scatter(
            x=[exit_x], y=[trade.exit_price],
            mode="markers",
            marker=dict(symbol="x", size=12, color=exit_color),
            name=f"Exit (${trade.pnl:+.2f})",
        ))

    # Stop loss line
    price_range = window_df["High"].max() - window_df["Low"].min()
    x_min = x_vals.min()
    x_max = x_vals.max()
    fig.add_trace(go.Scatter(
        x=[entry_x - 0.5, exit_x + 0.5],
        y=[trade.stop_loss, trade.stop_loss],
        mode="lines", line=dict(color="#FF1744", width=1.5, dash="dash"),
        name="Stop Loss", showlegend=True,
    ))

    # Target line
    target = trade.scalp_target if trade.scalp_target != trade.swing_target else trade.swing_target
    if hasattr(trade, 'swing_target') and trade.swing_target > 0:
        target = trade.swing_target
    fig.add_trace(go.Scatter(
        x=[entry_x - 0.5, exit_x + 0.5],
        y=[target, target],
        mode="lines", line=dict(color="#00C853", width=1.5, dash="dash"),
        name="Target", showlegend=True,
    ))

    fig.update_layout(
        title=chart_title,
        paper_bgcolor="#ffebd2",
        plot_bgcolor="#ffebd2",
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="linear",
            title=x_title,
            gridcolor="#ffcc99",
            tickfont=dict(size=11, color="#7c4a2a"),
        ),
        yaxis=dict(
            gridcolor="#ffcc99",
            title="Price",
            tickfont=dict(size=11, color="#7c4a2a"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=400,
        autosize=True,
    )

    return fig


# ─────────────────────────── ENCYCLOPEDIA CACHE ──────────────────────────────

@st.cache_data(ttl=None)
def load_encyclopedia() -> str:
    """Read the Brooks encyclopedia Markdown file once and cache it."""
    if ENCYCLOPEDIA_PATH.exists():
        return ENCYCLOPEDIA_PATH.read_text(encoding="utf-8")
    return ""

# ─────────────────────────── GEMINI BRAIN ────────────────────────────────────

SYSTEM_PROMPT = """You are an expert day trader trained strictly on Al Brooks' Price Action methodology.
Analyze the provided 5-minute chart image using your deep knowledge of Al Brooks' concepts.

You MUST return a strict JSON object with exactly these keys:
  day_type,        If unclear, guess the most likely context. Follow Al Brooks' terminology exactly:
        - Day Types: "Trading Range Day", "Triangle Trading Range Day", "Bull Trend From The Open", "Bear Trend From The Open", "Trending Trading Range Day (Bull)", "Trending Trading Range Day (Bear)", "Small Pullback Bull Trend", "Small Pullback Bear Trend", "Spike and Channel Bull Trend", "Spike and Channel Bear Trend", "Broad Bull Channel", "Broad Bear Channel", "Shrinking Stairs", "Reversal Day (Bull)", "Reversal Day (Bear)", "Crash Day", "Climax Day".]
- market_cycle: one of ["Breakout (Spike)","Micro Channel","Tight Channel (Small PB Trend)","Broad Bull Channel","Broad Bear Channel","Trading Range"]
- reasoning: A brief 1-sentence explanation of your overall analysis of the chart.
- setups: A list of up to 15 of the BEST setups of the day. Each item in the list MUST be a JSON object with exactly these keys: `{{"setup_name": "...", "entry_bar": 1, "entry_price": 0.00, "order_type": "...", "reason_1": "...", "reason_2": "..."}}`. `setup_name` MUST STRICTLY be one of ["High 1 Bull Flag","High 2 Bull Flag","High 3 Bull Flag","High 4 Bull Flag","Low 1 Bear Flag","Low 2 Bear Flag","Low 3 Bear Flag","Low 4 Bear Flag","Double Bottom","Double Top","Higher Low Double Bottom","Lower Low Double Bottom","Lower High Double Top","Higher High Double Top","Major Trend Reversal (Bull)","Major Trend Reversal (Bear)","Wedge Bottom","Wedge Top","Parabolic Wedge Bottom","Parabolic Wedge Top","Spike and Channel Bull","Spike and Channel Bear","Head & Shoulders Bottom","Head & Shoulders Top","Final Bull Flag","Final Bear Flag","Breakout (BO)","Breakout Test","Failed Breakout (Bull Trap)","Failed Breakout (Bear Trap)","Measuring Gap / Exhaustion Gap","Buy Climax","Sell Climax","Ledge Bottom","Ledge Top","ii Pattern","ioi Pattern","OO Pattern","Opening Reversal (Bull)","Opening Reversal (Bear)","20-Gap Bar Buy","20-Gap Bar Sell","Cup and Handle"]. DO NOT invent or use any other setup names. `entry_bar` is the integer Bar Number of the EXACT bar where the trade triggers and enters the market (NOT the signal bar that setup the trade). Note: The X-axis gridlines are printed in increments of 5 (1, 6, 11, 16, etc), you must count carefully. `order_type` MUST be exactly "Stop" or "Limit", denoting how the entry is executed mechanically. `reason_1` and `reason_2` are two distinct technical reasons justifying why this setup is valid (as Al Brooks requires 2 reasons to take any trade).
- action: one of ["Buy","Sell","Wait / No Trade"]
- confidence: a float between 0.0 and 1.0

Return ONLY the JSON object. No markdown, no explanation, no code fences."""

@st.cache_data(show_spinner=False, max_entries=50)
def _call_gemini_vision(ticker: str, img_bytes: bytes, system_text: str, api_key: str) -> dict:
    from google.genai import types
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=system_text,
        temperature=0.2,
        response_mime_type="application/json",
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text="Analyze this 5-minute price action chart:"),
                            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        ]
                    )
                ],
                config=config,
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()
            return json.loads(raw)

        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def analyze_chart(fig: go.Figure, ticker: str) -> dict:
    """Send chart image to Gemini and get a structured analysis."""
    api_key = get_api_key()
    if not api_key:
        return {
            "day_type": "N/A",
            "market_cycle": "N/A",
            "reasoning": "No API Key configured.",
            "setups": [
                {"setup_name": "N/A", "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A", "reason_1": "N/A", "reason_2": "N/A"}
            ] * 15,
            "action": "N/A",
            "confidence": 0.0,
            "_error": "GEMINI_API_KEY not set. Set it in the environment or .streamlit/secrets.toml",
        }

    # Convert figure to PNG bytes with smaller dimensions for latency optimization
    img_bytes = fig.to_image(format="png", width=800, height=400, scale=1)

    # No encyclopedia in the vision call — Gemini already knows Al Brooks.
    # This keeps the prompt small (~500 tokens) for fast responses.
    # The full encyclopedia is only used when updating rules from teacher notes.
    system_text = SYSTEM_PROMPT

    try:
        return _call_gemini_vision(ticker, img_bytes, system_text, api_key)
    except Exception as e:
        return {
            "day_type": "Error",
            "market_cycle": "Error",
            "reasoning": "API Request Failed.",
            "setups": [
                {"setup_name": "Error", "entry_bar": 0, "entry_price": 0.0, "order_type": "Error", "reason_1": "Error", "reason_2": "Error"}
            ] * 15,
            "action": "Error",
            "confidence": 0.0,
            "_error": str(e),
        }

def update_encyclopedia(notes: str):
    """Ask Gemini to integrate new Teacher Notes into the encyclopedia and save it."""
    api_key = get_api_key()
    if not api_key:
        return

    encyclopedia = load_encyclopedia()
    
    prompt = f"""You are maintaining a strict markdown encyclopedia of Al Brooks' Price Action rules.
Here is the current encyclopedia:
```markdown
{encyclopedia}
```

The human Teacher just corrected a chart and provided these new learning notes:
"{notes}"

TASK: Integrate these notes into the encyclopedia. If it introduces a new rule, add it in the appropriate section. If it corrects an existing rule, modify it. Keep it concise, organized, and strictly focused on Al Brooks terminology.
Return ONLY the raw updated markdown file content. Do not include ```markdown formatting fences at the start or end. Just the raw text."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )

            raw = response.text.strip()
            if raw.startswith("```markdown"):
                raw = raw.split("\n", 1)[-1]
            elif raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            # Save to disk
            ENCYCLOPEDIA_PATH.write_text(raw, encoding="utf-8")

            # Clear Streamlit cache so next run loads fresh rules
            load_encyclopedia.clear()

            break  # Success — exit retry loop

        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = 2 ** (attempt + 1)  # 2s, 4s, 8s
                st.warning(f"Rate limited by Gemini API. Retrying in {wait_seconds}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_seconds)
                continue
            st.error(f"Failed to auto-update encyclopedia: {e}")

def ask_bot_question(question: str, fig: go.Figure, analysis_json: dict) -> str:
    """Send a specific user question about the current chart to the bot."""
    api_key = get_api_key()
    if not api_key:
        return "API Key missing. Cannot answer questions."

    try:
        from google.genai import types
        client = genai.Client(api_key=api_key)
        
        # Convert figure to PNG bytes
        img_bytes = fig.to_image(format="png", width=1000, height=500, scale=1.5)
        
        # Build strict context
        encyclopedia = load_encyclopedia()
        prompt_context = f"""You are answering a question from your human Teacher about a specific 5-minute stock chart.
You recently analyzed this chart and outputted the following structured JSON response:
{json.dumps(analysis_json, indent=2)}

You must answer the Teacher's following question honestly and concisely based on your analysis and the Al Brooks rules. Do not hallucinate or guess. If you made a mistake in your JSON analysis, admit it.

Al Brooks Rules Context:
{encyclopedia}

Teacher's Question: {question}"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt_context),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ]
                )
            ],
            config=types.GenerateContentConfig(temperature=0.4)
        )
        return response.text.strip()
    except Exception as e:
        return f"Error connecting to bot: {str(e)}"

# ─────────────────────────── CSV HELPERS ─────────────────────────────────────

def load_training_csv() -> pd.DataFrame:
    """Load or create the training data CSV."""
    if TRAINING_CSV.exists():
        return pd.read_csv(TRAINING_CSV)
    return pd.DataFrame(columns=CSV_COLUMNS)


def save_row(row: dict):
    """Append one training row to the CSV."""
    df = load_training_csv()
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(TRAINING_CSV, index=False)


def delete_row(idx: int):
    """Delete a specific row from the CSV by integer index."""
    df = load_training_csv()
    if 0 <= idx < len(df):
        df = df.drop(index=idx).reset_index(drop=True)
        df.to_csv(TRAINING_CSV, index=False)

# ─────────────────────────── LOAD / INIT SESSION ─────────────────────────────

def load_new_chart():
    """Pick a random S&P 500 ticker and fetch data. Uses prefetch if available."""
    # Check if we have a prefetched chart ready
    if "prefetch_ready" in st.session_state and st.session_state["prefetch_ready"]:
        st.session_state["ticker"] = st.session_state.pop("prefetch_ticker")
        st.session_state["chart_df"] = st.session_state.pop("prefetch_df")
        st.session_state["chart_fig"] = st.session_state.pop("prefetch_fig")
        st.session_state["bot_analysis"] = st.session_state.pop("prefetch_analysis")
        st.session_state.pop("prefetch_ready", None)
        return

    # Fail fast if no API key is set on Render
    if not get_databento_key():
        st.error("DATABENTO_API_KEY is not set. Please add it to your Render Environment Variables. The app cannot function without data.")
        return

    tickers = get_sp500_tickers()
    random.shuffle(tickers)
    for i, t in enumerate(tickers):
        if i >= 5:  # Prevent hanging the server on API failures/rate limits
            st.error("Databento API failed to return data after 5 attempts. You may be out of credits, or the API is down. Check Render logs.")
            return

        df = fetch_chart_data_v2(t)
        if df is not None and len(df) > 30:
            st.session_state["ticker"] = t
            st.session_state["chart_df"] = df
            return
    st.error("Could not fetch data for any ticker. Please try again.")


def _add_annotations(fig, df, analysis, best_only=False):
    """Add setup marker annotations to a chart figure."""
    bot_setups = analysis.get("setups", [])
    if best_only:
        bot_setups = bot_setups[:1]  # Only show the #1 setup
    price_min = df["Low"].min()
    price_max = df["High"].max()
    price_range = price_max - price_min
    annot_offset = price_range * 0.06

    for i in range(15):
        obj = bot_setups[i] if i < len(bot_setups) else {}
        if isinstance(obj, str):
            obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0}

        b_name = obj.get("setup_name", "")
        b_bar = obj.get("entry_bar", 0)
        b_signal = obj.get("signal_bar", 0)  # Try to explicitly get signal bar if provided
        if not b_signal: b_signal = b_bar - 1 if b_bar > 0 else 0
        b_price = obj.get("entry_price", 0.0)
        b_order_type = obj.get("order_type", "") # Extract order type
        b_conf = obj.get("confidence", 0.0)      # Extract confidence

        if b_bar and b_name and b_name != "N/A" and b_name != "Error":
            bar_row = df[df["BarNumber"] == int(b_bar)]
            if not bar_row.empty:
                bar_low = bar_row["Low"].values[0]
                bar_close = bar_row["Close"].values[0]
            else:
                bar_low = price_min
                bar_close = price_min

            if b_price and (price_min - price_range) <= float(b_price) <= (price_max + price_range):
                validated_price = float(b_price)
            else:
                validated_price = bar_close

            action_dir = analysis.get("action", "")
            color = "#4ade80" if action_dir == "Buy" else "#f87171" if action_dir == "Sell" else "#fbbf24"
            
            # Color Signal Bar (Yellow)
            sig_bar = int(b_signal)
            sig_row = df[df["BarNumber"] == sig_bar]
            if not sig_row.empty:
                fig.add_trace(go.Candlestick(
                    x=[sig_row["BarNumber"].values[0]],
                    open=[sig_row["Open"].values[0]],
                    high=[sig_row["High"].values[0]],
                    low=[sig_row["Low"].values[0]],
                    close=[sig_row["Close"].values[0]],
                    name="Signal Bar",
                    increasing_line_color="#f59e0b", # Lion King Gold
                    decreasing_line_color="#f59e0b",
                    increasing_fillcolor="#f59e0b",
                    decreasing_fillcolor="#f59e0b",
                    showlegend=False,
                ))
            
            # Color Entry Bar (Purple)
            ent_bar = int(b_bar)
            ent_row = df[df["BarNumber"] == ent_bar]
            if not ent_row.empty:
                fig.add_trace(go.Candlestick(
                    x=[ent_row["BarNumber"].values[0]],
                    open=[ent_row["Open"].values[0]],
                    high=[ent_row["High"].values[0]],
                    low=[ent_row["Low"].values[0]],
                    close=[ent_row["Close"].values[0]],
                    name="Entry Bar",
                    increasing_line_color="#c084fc", # Light purple
                    decreasing_line_color="#c084fc",
                    increasing_fillcolor="#c084fc",
                    decreasing_fillcolor="#c084fc",
                    showlegend=False,
                ))
            
            fig.add_shape(
                type="line",
                x0=int(b_signal) - 0.45,
                x1=int(b_signal) + 0.45,
                y0=validated_price,
                y1=validated_price,
                line=dict(color="#fbbf24", width=3, dash="dot"),
                layer="above"
            )
            
            # Remove verbose wording from chart labels per user request
            label_text = f"#{i+1}"
            
            fig.add_annotation(
                x=int(b_bar),
                y=bar_low - annot_offset * (1 + i * 0.5),
                text=label_text,
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor=color,
                ay=40, # Pushed down further so it doesn't overlap the candlestick body
                ax=0,
                yanchor="top",
                font=dict(color=color, size=10, family="Arial"),
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor=color,
                borderpad=2,
                opacity=0.9
            )

    return fig

def _do_prefetch(use_algo: bool = False):
    """Background: fetch a random ticker, build chart, run analysis."""
    try:
        tickers = get_sp500_tickers()
        random.shuffle(tickers)
        for i, t in enumerate(tickers):
            if i >= 5:
                # Prevent hanging the server on API failures/rate limits.
                # Streamlit waits for non-daemon threads to finish before rendering!
                return {}
            df = fetch_chart_data_v2(t)
            if df is not None and len(df) > 30:
                fig = build_chart(df, t)
                if use_algo:
                    analysis = analyze_bars(df)
                else:
                    analysis = analyze_chart(fig, t)
                return {"ticker": t, "df": df, "fig": fig, "analysis": analysis}
        return {}
    except Exception:
        return {}


def start_prefetch():
    """Start background prefetch of next chart."""
    if "prefetch_future" not in st.session_state:
        use_algo = st.session_state.get("analysis_mode", "Algo (Instant)") == "Algo (Instant)"
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_prefetch, use_algo)
        st.session_state["prefetch_future"] = future
        st.session_state["prefetch_executor"] = executor


def check_prefetch():
    """Check if background prefetch is done and store results."""
    if "prefetch_future" in st.session_state:
        future = st.session_state["prefetch_future"]
        if future.done():
            result = future.result()
            if result:
                st.session_state["prefetch_ticker"] = result["ticker"]
                st.session_state["prefetch_df"] = result["df"]
                st.session_state["prefetch_fig"] = result["fig"]
                st.session_state["prefetch_analysis"] = result["analysis"]
                st.session_state["prefetch_ready"] = True
            executor = st.session_state.pop("prefetch_executor", None)
            if executor:
                executor.shutdown(wait=False)
            st.session_state.pop("prefetch_future", None)


# ─────────────────────────── SIDEBAR ─────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## Trading Bot Trainer")
        st.markdown("---")
        count = len(load_training_csv())
        st.metric("Charts Corrected", f"{count} / 100")
        st.progress(min(count / 100, 1.0))
        st.markdown("---")

        # Analysis mode toggle
        analysis_mode = st.radio(
            "Analysis Engine",
            ["Algo (Instant)", "Gemini (LLM)"],
            index=0,
            help="Algo runs locally in ~50ms. Gemini uses the LLM API (5-15s).",
        )
        st.session_state["analysis_mode"] = analysis_mode
        st.markdown("---")

        st.markdown(
            "Train a Gemini-powered bot on **Al Brooks' Price Action** by correcting its guesses."
        )
        st.markdown("---")
        source = _init_data_source_v2()
        st.caption(f"Built with Streamlit · Gemini · {source.name()}")
        
        # ── Ask the Bot (Teacher Workflow) ──
        st.markdown("---")
        st.subheader("Ask the Bot")

        # Initialize chat history if missing
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Render existing chat in a smaller container
        chat_container = st.container(height=250)
        with chat_container:
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Accept user input
        user_question = st.chat_input("Ask about this setup...")
        if user_question:
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            with st.spinner("Thinking..."):
                # Use the current chart and analysis if available
                if "chart_df" in st.session_state and "bot_analysis" in st.session_state:
                    fig = build_chart(st.session_state["chart_df"], st.session_state.get("ticker", ""))
                    analysis = st.session_state["bot_analysis"]
                    response = ask_bot_question(user_question, fig, analysis)
                else:
                    response = "Load a chart first (go to Training Lab and start a session), then I can answer questions about it."
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            st.rerun()

# ─────────────────────────── TRAINING LAB TAB ────────────────────────────────

def render_training_lab():

    # Ensure we have chart data in session state
    if "ticker" not in st.session_state:
        st.info("Welcome to the Training Lab. Ready to analyze Al Brooks setups?")
        if st.button("Start Training Session", type="primary", use_container_width=True):
            with st.spinner("Fetching a random S&P 500 chart..."):
                load_new_chart()
                st.rerun()
        return

    ticker = st.session_state["ticker"]
    df = st.session_state["chart_df"]
    
    # ── Setup Header Defaults from Bot Analysis if Available ──
    analysis = st.session_state.get("bot_analysis", {})
    
    bot_day_type = analysis.get("day_type", "?")
    bot_market_cycle = analysis.get("market_cycle", "?")
    
    dyn_day_opts = [f"Approve Bot's Guess: {bot_day_type}"] + DAY_TYPE_OPTIONS[1:]
    dyn_cycle_opts = [f"Approve Bot's Guess: {bot_market_cycle}"] + MARKET_CYCLE_OPTIONS[1:]

    # Bot's Reasoning
    bot_reasoning = analysis.get("reasoning", "")
    if bot_reasoning:
        with st.expander("View Bot's Setup Ranking and Reasoning", expanded=False):
            st.markdown(bot_reasoning)

    # Chart View toggle
    chart_view = st.radio(
        "Chart View",
        ["Best Setup Only", "All Setups", "Hidden"],
        horizontal=True,
        index=0,
        key=f"chart_view_{ticker}",
    )
    show_annotations = chart_view != "Hidden"

    # Select Best Trade of the Day
    bot_setups_list = analysis.get("setups", [])
    setup_labels = []
    for i, s in enumerate(bot_setups_list):
        if isinstance(s, str):
            setup_labels.append(f"{i+1}: {s}")
        else:
            sname = s.get("setup_name", "Unknown")
            sbar = s.get("entry_bar", "?")
            setup_labels.append(f"{i+1}: {sname} (Bar {sbar})")
    if not setup_labels:
        setup_labels = ["No setups detected"]

    best_col, name_col = st.columns(2)
    with best_col:
        st.markdown("**Select Best Trade of the Day**")
        best_trade = st.selectbox("Select Best Trade of the Day", setup_labels, index=0, key=f"best_trade_{ticker}", label_visibility="collapsed")
    with name_col:
        st.markdown("**Type Best Setup Name (if bot was wrong)**")
        override_setup_name = st.text_input("Type Best Setup Name (if bot was wrong)", placeholder="e.g., High 2 Bull Flag", key=f"override_name_{ticker}", label_visibility="collapsed")

    # ── Header Dropdowns ──
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.markdown("<h3 style='text-align: center; color: #4a2311; margin-bottom: 0px;'>Day Type</h3>", unsafe_allow_html=True)
        day_type = st.selectbox("Day Type", dyn_day_opts, index=0, key=f"day_type_{ticker}", label_visibility="collapsed")
    with top_col2:
        st.markdown("<h3 style='text-align: center; color: #4a2311; margin-bottom: 0px;'>Market Cycle</h3>", unsafe_allow_html=True)
        market_cycle = st.selectbox("Market Cycle", dyn_cycle_opts, index=0, key=f"market_cycle_{ticker}", label_visibility="collapsed")

    # PHASE 1: Rebuild base chart clean every time to support toggles
    fig = build_chart(df, ticker)

    if "bot_analysis" not in st.session_state:
        use_algo = st.session_state.get("analysis_mode", "Algo (Instant)") == "Algo (Instant)"
        if use_algo:
            # Fast algo engine — runs on raw OHLC data, no API call
            analysis = analyze_bars(df)
            st.session_state["bot_analysis"] = analysis
            st.rerun()
        else:
            # Gemini LLM vision — slower but more nuanced
            st.plotly_chart(fig, use_container_width=True, key="main_chart_loading")
            with st.spinner("Bot is analyzing the chart with Gemini..."):
                analysis = analyze_chart(fig, ticker)
            st.session_state["bot_analysis"] = analysis
            st.rerun()

    analysis = st.session_state["bot_analysis"]

    if show_annotations:
        _add_annotations(fig, df, analysis, best_only=(chart_view == "Best Setup Only"))

    # Check/start prefetch for next chart
    check_prefetch()
    if "prefetch_future" not in st.session_state and "prefetch_ready" not in st.session_state:
        start_prefetch()

    st.plotly_chart(fig, use_container_width=True, key="main_chart")


    # ── Action Buttons: Approve / Skip / Illiquid ──
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        approve_btn = st.button("Approve Day", width="stretch", type="primary")
    with btn_col2:
        skip_btn = st.button("Skip", width="stretch")
    with btn_col3:
        illiquid_btn = st.button("Illiquid", width="stretch")

    # ── Teacher Override & JSON Analysis ──
    col_json, col_teacher = st.columns(2, gap="large")
    
    with col_json:
        st.subheader("Bot's JSON Analysis")
        if "_error" in analysis:
            st.warning(analysis["_error"])
        with st.expander("Raw API JSON Output", expanded=False):
            st.json(analysis)

    with col_teacher:
        st.subheader("Teacher's Workflow")
        bot_action = analysis.get("action", "?")
        dyn_action = [f"Approve Bot's Guess: {bot_action}"] + ACTION_OPTIONS[1:]
        action = st.selectbox("Action", dyn_action, key=f"action_{ticker}")
        notes = st.text_area("Teacher's Notes", placeholder="Why did you override?", key=f"notes_{ticker}", height=100)

    # ── Handle Buttons ──
    if approve_btn:
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "ticker": ticker,
            "bot_day_type": analysis.get("day_type", ""),
            "bot_market_cycle": analysis.get("market_cycle", ""),
            "bot_action": analysis.get("action", ""),
            "bot_confidence": analysis.get("confidence", ""),
        }

        bot_setups = analysis.get("setups", [])
        for i in range(15):
            obj = bot_setups[i] if i < len(bot_setups) else {}
            if isinstance(obj, str):
                obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A"}
            row[f"bot_setup_{i+1}"] = obj.get("setup_name", "")
            row[f"bot_setup_{i+1}_bar"] = obj.get("entry_bar", 0)
            row[f"bot_setup_{i+1}_price"] = obj.get("entry_price", 0.0)
            row[f"bot_setup_{i+1}_order_type"] = obj.get("order_type", "")

        # Use overrides if teacher changed them, otherwise use bot's guess
        row["override_day_type"] = day_type if not str(day_type).startswith("Approve Bot") else analysis.get("day_type", "")
        row["override_market_cycle"] = market_cycle if not str(market_cycle).startswith("Approve Bot") else analysis.get("market_cycle", "")

        # If teacher typed a custom setup name, use that for override_setup_1
        if override_setup_name.strip():
            row["override_setup_1"] = override_setup_name.strip()
        else:
            for i in range(15):
                obj = bot_setups[i] if i < len(bot_setups) else {}
                if isinstance(obj, str):
                    obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A"}
                row[f"override_setup_{i+1}"] = obj.get("setup_name", "")
                row[f"override_setup_{i+1}_bar"] = obj.get("entry_bar", 0)
                row[f"override_setup_{i+1}_price"] = obj.get("entry_price", 0.0)
                row[f"override_setup_{i+1}_order_type"] = obj.get("order_type", "")

        row["override_action"] = action if not str(action).startswith("Approve Bot") else analysis.get("action", "")
        row["teacher_notes"] = notes

        # Update Encyclopedia if notes exist
        if notes.strip():
            with st.spinner("Teaching Bot... Updating Encyclopedia with your notes."):
                update_encyclopedia(notes.strip())
            st.toast("Encyclopedia permanently updated!")

        save_row(row)
        for key in ["ticker", "chart_df", "chart_fig", "bot_analysis", "chat_history"]:
            st.session_state.pop(key, None)
        st.rerun()

    if skip_btn or illiquid_btn:
        if illiquid_btn:
            add_to_do_not_trade(ticker)
            st.toast(f"{ticker} has been permanently excluded from training.")
        for key in ["ticker", "chart_df", "chart_fig", "bot_analysis", "chat_history"]:
            st.session_state.pop(key, None)
        st.rerun()

# ─────────────────────────── LIBRARY TAB (merged History + Examples) ──────────

def render_library():
    """Merged History + Examples tab: browse, filter, and manage training data."""
    df = load_training_csv()
    if df.empty:
        st.info("No training data yet. Start correcting charts in the Training Lab!")
        return

    # ── Filter Controls ──
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

    with fcol1:
        day_types = sorted(df["override_day_type"].dropna().unique().tolist())
        sel_day = st.multiselect("Day Type", day_types, placeholder="All day types")

    with fcol2:
        cycles = sorted(df["override_market_cycle"].dropna().unique().tolist())
        sel_cycle = st.multiselect("Market Cycle", cycles, placeholder="All cycles")

    with fcol3:
        setup_cols = [f"override_setup_{i}" for i in range(1, 16)]
        all_setups = set()
        for col in setup_cols:
            if col in df.columns:
                all_setups.update(df[col].dropna().unique().tolist())
        all_setups -= {"", "N/A", "Error"}
        sel_setup = st.multiselect("Setup", sorted(all_setups), placeholder="All setups")

    with fcol4:
        search_text = st.text_input("Search Notes", placeholder="e.g. wedge, reversal...")

    # ── Apply Filters ──
    mask = pd.Series([True] * len(df), index=df.index)
    if sel_day:
        mask &= df["override_day_type"].isin(sel_day)
    if sel_cycle:
        mask &= df["override_market_cycle"].isin(sel_cycle)
    if sel_setup:
        setup_mask = pd.Series([False] * len(df), index=df.index)
        for col in setup_cols:
            if col in df.columns:
                setup_mask |= df[col].isin(sel_setup)
        mask &= setup_mask
    if search_text.strip():
        mask &= df["teacher_notes"].fillna("").str.contains(search_text.strip(), case=False, na=False)

    filtered = df[mask].reset_index(drop=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} corrections")

    if filtered.empty:
        st.warning("No examples match your filters.")
        return

    # ── Display Results ──
    for idx, row in filtered.iterrows():
        ticker = row.get("ticker", "?")
        ts = row.get("timestamp", "")
        o_day = row.get("override_day_type", "")
        o_cycle = row.get("override_market_cycle", "")
        label = f"**{ticker}** — {o_day} | {o_cycle} — _{ts}_"

        with st.expander(label, expanded=False):
            ecol1, ecol2 = st.columns(2)

            with ecol1:
                st.markdown("**Your Correction:**")
                st.markdown(f"- Day Type: {o_day}")
                st.markdown(f"- Market Cycle: {o_cycle}")
                st.markdown(f"- Action: {row.get('override_action', '')}")
                for i in range(1, 16):
                    s_name = row.get(f"override_setup_{i}", "")
                    if s_name and s_name not in ("", "N/A", "Error"):
                        s_bar = row.get(f"override_setup_{i}_bar", "")
                        s_price = row.get(f"override_setup_{i}_price", "")
                        s_order = row.get(f"override_setup_{i}_order_type", "")
                        st.markdown(f"  {i}. **{s_name}** [{s_order}] @ Bar {s_bar}, ${s_price}")

            with ecol2:
                b_day = row.get("bot_day_type", "")
                b_cycle = row.get("bot_market_cycle", "")
                b_action = row.get("bot_action", "")
                st.markdown("**Bot's Guess:**")
                st.markdown(f"- Day Type: {b_day} {'(match)' if b_day == o_day else '(override)'}")
                st.markdown(f"- Market Cycle: {b_cycle} {'(match)' if b_cycle == o_cycle else '(override)'}")
                st.markdown(f"- Action: {b_action}")
                for i in range(1, 16):
                    s_name = row.get(f"bot_setup_{i}", "")
                    if s_name and s_name not in ("", "N/A", "Error"):
                        s_bar = row.get(f"bot_setup_{i}_bar", "")
                        s_price = row.get(f"bot_setup_{i}_price", "")
                        s_order = row.get(f"bot_setup_{i}_order_type", "")
                        st.markdown(f"  {i}. {s_name} [{s_order}] @ Bar {s_bar}, ${s_price}")

            notes = row.get("teacher_notes", "")
            if notes and str(notes).strip():
                st.markdown(f"**Notes:** {notes}")

            # Load chart button (lazy — only fetches when clicked)
            if st.button(f"Load Chart", key=f"load_chart_{idx}"):
                try:
                    dt = datetime.datetime.fromisoformat(ts)
                    start_str = dt.strftime("%Y-%m-%d")
                    end_str = (dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                    hist_df = fetch_chart_data_v2(ticker, start_date=start_str, end_date=end_str)
                    if hist_df is not None and not hist_df.empty:
                        hist_fig = build_chart(hist_df, ticker)
                        reconstructed = {"action": b_action, "setups": []}
                        for i in range(1, 16):
                            s_name = row.get(f"bot_setup_{i}", "")
                            if s_name and s_name not in ("N/A", "Error", ""):
                                reconstructed["setups"].append({
                                    "setup_name": s_name,
                                    "entry_bar": int(row.get(f"bot_setup_{i}_bar", 0) or 0),
                                    "entry_price": float(row.get(f"bot_setup_{i}_price", 0.0) or 0.0),
                                    "order_type": row.get(f"bot_setup_{i}_order_type", ""),
                                })
                        _add_annotations(hist_fig, hist_df, reconstructed)
                        st.plotly_chart(hist_fig, use_container_width=True, key=f"lib_chart_{idx}")
                    else:
                        st.warning("Chart data no longer available (may have expired).")
                except Exception as e:
                    st.error(f"Failed to load chart: {e}")

    # ── Delete Row ──
    st.markdown("---")
    del_col1, del_col2 = st.columns([3, 1])
    with del_col1:
        row_idx = st.number_input("Row index to delete", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
    with del_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Delete Row", type="primary"):
            delete_row(int(row_idx))
            st.rerun()

    # ── Raw Data Export ──
    with st.expander("Raw CSV Data", expanded=False):
        st.dataframe(df, width="stretch", height=300)
        csv_data = df.to_csv(index=False)
        st.download_button("Download CSV", csv_data, "training_data.csv", "text/csv")


# ─────────────────────────── SCANNER TAB (merged Scanner + Setups) ────────────

def render_scanner():
    """Merged Scanner + Setups Guide tab."""

    # ── Setup Definitions Reference ──
    with st.expander("Setup Definitions Reference", expanded=False):
        encyclopedia = load_encyclopedia()
        setup_name = st.selectbox("Select Setup to View", SETUP_OPTIONS[1:], key="ref_setup")
        lines = encyclopedia.splitlines()
        capture = False
        definition = []
        for line in lines:
            if line.startswith('### ') or line.startswith('## ') or line.startswith('#### '):
                clean_line = line.replace("#", "").replace("*", "").strip().lower()
                if setup_name.lower().strip() in clean_line:
                    capture = True
                    continue
                elif capture:
                    break
            if capture:
                definition.append(line)
        def_text = "\n".join(definition).strip()
        if def_text:
            st.info(def_text)
        else:
            st.caption("Definition not found in encyclopedia. Refer to Al Brooks' books.")

    st.markdown("---")

    # ── Scanner Controls ──
    col1, col2 = st.columns([3, 1])
    with col1:
        default_tickers = "AAPL, QQQ, TSLA, MSFT, NVDA, SPY"
        ticker_input = st.text_input("Tickers (comma-separated):", value=default_tickers)
    with col2:
        scanner_days = st.number_input("Days Back", min_value=1, max_value=1825, value=5, key="scanner_days")

    if st.button("Run Scanner", type="primary"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Enter at least one ticker.")
            return

        import datetime as _dt
        end = _dt.date.today()
        calendar_days = int(scanner_days * 1.45) + 1
        start = end - _dt.timedelta(days=calendar_days)

        st.session_state["scanner_start"] = start.strftime("%Y-%m-%d")
        st.session_state["scanner_end"] = end.strftime("%Y-%m-%d")

        results = []
        progress_bar = st.progress(0)

        for i, ticker in enumerate(tickers):
            df = fetch_chart_data_v2(ticker, start_date=st.session_state["scanner_start"], end_date=st.session_state["scanner_end"])
            if df is not None and not df.empty:
                analysis = analyze_bars(df)
                setups = analysis.get("setups", [])

                best_setup_name = "None"
                best_setup_bar = "-"
                best_setup_price = "-"

                if setups:
                    best = setups[0]
                    if isinstance(best, dict):
                        best_setup_name = best.get("setup_name", "Unknown")
                        e_bar = best.get("entry_bar", 0)
                        best_setup_bar = e_bar - 1 if e_bar > 0 else "-"
                        best_setup_price = best.get("entry_price", "-")
                    else:
                        best_setup_name = str(best)

                results.append({
                    "Ticker": ticker,
                    "Setups": len(setups),
                    "Best Setup": best_setup_name,
                    "Signal Bar": best_setup_bar,
                    "Entry Price": best_setup_price,
                    "Action": analysis.get("action", "-"),
                    "Day Type": analysis.get("day_type", "-"),
                })
            else:
                results.append({
                    "Ticker": ticker, "Setups": 0, "Best Setup": "No data",
                    "Signal Bar": "-", "Entry Price": "-", "Action": "-", "Day Type": "-",
                })

            progress_bar.progress((i + 1) / len(tickers))

        if results:
            st.session_state["scanner_results"] = pd.DataFrame(results)
            st.success("Scan complete!")

    # ── Display Results ──
    if "scanner_results" in st.session_state:
        st.markdown("---")
        res_df = st.session_state["scanner_results"]

        event = st.dataframe(
            res_df, width="stretch", hide_index=True,
            selection_mode="single-row", on_select="rerun",
        )

        if event.selection.rows:
            selected_idx = event.selection.rows[0]
            selected_ticker = res_df.iloc[selected_idx]["Ticker"]

            with st.spinner(f"Loading {selected_ticker} chart..."):
                s_start = st.session_state.get("scanner_start")
                s_end = st.session_state.get("scanner_end")
                df = fetch_chart_data_v2(selected_ticker, start_date=s_start, end_date=s_end)
                if df is not None and not df.empty:
                    analysis = analyze_bars(df)
                    fig = build_chart(df, selected_ticker)
                    if analysis and analysis.get("setups"):
                        fig = _add_annotations(fig, df, analysis, best_only=True)
                    st.plotly_chart(fig, use_container_width=True)

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Day Type", analysis.get("day_type", "—"))
                    with col_b:
                        st.metric("Action", analysis.get("action", "—"))
                    with col_c:
                        conf = analysis.get("confidence", 0)
                        st.metric("Confidence", f"{conf:.0%}" if isinstance(conf, (int, float)) else conf)

                    # Show reasoning
                    reasoning = analysis.get("reasoning", "")
                    if reasoning:
                        with st.expander("Bot's Reasoning", expanded=False):
                            st.markdown(reasoning)
                else:
                    st.error("Could not fetch chart data.")

    # ── S&P 500 Setup Finder ──
    st.markdown("---")
    st.markdown("#### Find Setup Examples in S&P 500")
    import datetime as _dt

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        find_setup = st.selectbox("Setup to Find", SETUP_OPTIONS[1:], key="find_setup")
    with fc2:
        scan_date = st.date_input("Date", value=_dt.date.today(), max_value=_dt.date.today(), key="find_date")
        if isinstance(scan_date, (tuple, list)):
            scan_date = scan_date[0] if scan_date else _dt.date.today()
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        find_btn = st.button("Find Examples", type="primary", key="find_btn")

    if find_btn:
        db_key = get_databento_key()
        if not db_key:
            st.warning("Setup finder requires Databento API key for bulk S&P 500 scanning. Without it, use the ticker scanner above.")
            return

        with st.spinner(f"Scanning S&P 500 for {find_setup} on {scan_date}..."):
            from data_source import get_data_source as _get_ds
            tickers = get_sp500_tickers()
            start_date = (scan_date - _dt.timedelta(days=4)).strftime("%Y-%m-%d")
            end_date = scan_date.strftime("%Y-%m-%d")

            ds = _get_ds(api_key=db_key)
            if not hasattr(ds, 'get_bulk_chart_data'):
                st.warning("Bulk scanning requires Databento. yFinance fallback doesn't support bulk fetching.")
                return

            bulk_df = ds.get_bulk_chart_data(tickers, start_date, end_date)
            if bulk_df.empty:
                st.error("No data returned. Check API credits or try a different date.")
                return

            found_charts = []
            grouped = bulk_df.groupby("symbol")
            progress_bar = st.progress(0, text="Analyzing...")
            total = len(grouped)

            for i, (sym, sym_df) in enumerate(grouped):
                progress_bar.progress((i + 1) / total, text=f"Analyzing {sym}...")
                try:
                    result = analyze_bars(sym_df)
                    for s in result.get("setups", []):
                        name = s.get("setup_name", "") if isinstance(s, dict) else getattr(s, "setup_name", "")
                        if name == find_setup:
                            fig = build_chart(sym_df, sym)
                            fig = _add_annotations(fig, sym_df, {"action": result.get("action", ""), "setups": [s]}, best_only=False)
                            found_charts.append((sym, fig))
                            break
                except Exception:
                    pass

            progress_bar.empty()
            if found_charts:
                st.success(f"Found {len(found_charts)} stocks with **{find_setup}**!")
                for sym, fig in found_charts:
                    with st.expander(f"{sym}", expanded=False):
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No examples of **{find_setup}** found on {scan_date}.")


# ─────────────────────────── BACKTEST TAB ─────────────────────────────────────

def render_backtest():
    """Backtesting tab with full report, equity curve, and trade log."""
    from backtester import run_backtest, run_multi_day_backtest, trades_to_dataframe

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        bt_ticker = st.text_input("Ticker", value="SPY", key="bt_ticker").upper().strip()
    with col2:
        bt_mode = st.selectbox("Mode", ["scalp", "swing"], key="bt_mode",
                                help="Scalp = 1:1 R/R target. Swing = 2:1 R/R target.")
    with col3:
        bt_days = st.number_input("Trading Days", min_value=1, max_value=60, value=30, key="bt_days",
                                   help="Trading days to backtest. 5-min data limited to ~40 days via yFinance.")
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run Backtest", key="bt_run", type="primary")

    if run_btn:
        with st.spinner(f"Backtesting {bt_ticker} over {bt_days} days ({bt_mode} mode)..."):
            source = _init_data_source_v2()
            import datetime as _dt
            end = _dt.date.today()
            calendar_days = int(bt_days * 1.45) + 1
            start = end - _dt.timedelta(days=calendar_days)

            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")

            # Warn about yFinance 60-day limit for 5m data
            if source.name() == "yFinance" and bt_days > 40:
                st.warning("yFinance only provides ~60 calendar days of 5-min data. Results may cover fewer days than requested.")

            full_df = None
            used_source = source.name()

            try:
                full_df = source.fetch_historical(bt_ticker, start_str, end_str)
            except Exception as e:
                st.warning(f"{source.name()} failed: {e}")

            # Count how many trading days we got
            def _count_days(df):
                if df is None or df.empty:
                    return 0
                idx = pd.to_datetime(df.index)
                return len(set(idx.date))

            got_days = _count_days(full_df)

            # If primary source returned too few days for a multi-day backtest, try yFinance fallback
            if got_days < max(2, bt_days // 2) and source.name() != "yFinance" and bt_days > 1:
                st.caption(f"{source.name()} returned only {got_days} day(s). Trying yFinance fallback...")
                try:
                    from data_source import YFinanceSource
                    yf_source = YFinanceSource()
                    yf_df = yf_source.fetch_historical(bt_ticker, start_str, end_str)
                    yf_days = _count_days(yf_df)
                    if yf_days > got_days:
                        full_df = yf_df
                        used_source = "yFinance"
                        got_days = yf_days
                except Exception as yf_err:
                    st.caption(f"yFinance fallback also failed: {yf_err}")

            st.caption(f"Data: **{used_source}** | {start} → {end}")

            if full_df is None or full_df.empty:
                st.warning(f"No data for {bt_ticker}. Check your API key or try a different ticker/range.")
                return

            if isinstance(full_df.columns, pd.MultiIndex):
                full_df.columns = full_df.columns.get_level_values(0)

            full_df.index = pd.to_datetime(full_df.index)
            daily_dfs = {}
            for date, group in full_df.groupby(full_df.index.date):
                if len(group) >= 10:
                    daily_dfs[str(date)] = group

            if not daily_dfs:
                st.warning("Not enough intraday data. Try a more recent date range.")
                return

            if len(daily_dfs) < bt_days:
                st.caption(f"Got **{len(daily_dfs)}** trading days of data (requested {bt_days})")

            report = run_multi_day_backtest(daily_dfs, mode=bt_mode)
            st.session_state["bt_report"] = report
            st.session_state["bt_daily_dfs"] = daily_dfs
            st.session_state["bt_ticker_used"] = bt_ticker

    report = st.session_state.get("bt_report")
    if not report:
        st.info("Configure settings and press **Run Backtest**. Uses 5-min intraday bars with EOD forced close.")
        return

    s = report["summary"]
    trades = report["trades"]

    if s["total_trades"] == 0:
        st.warning("No trades generated. The algo didn't find setups in this data.")
        return

    # ── Summary metrics ──
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", s["total_trades"])
    m2.metric("Win Rate", f"{s['win_rate']:.1%}")
    m3.metric("P&L", f"${s['total_pnl']:.2f}/sh")
    m4.metric("Profit Factor", f"{s['profit_factor']:.2f}")
    m5.metric("Sharpe", f"{s['sharpe_annualized']:.2f}")

    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric("Avg Win", f"${s['avg_winner']:.2f}")
    m7.metric("Avg Loss", f"${s['avg_loser']:.2f}")
    m8.metric("Max DD", f"${s['max_drawdown']:.2f}")
    m9.metric("Avg R", f"{s['avg_r_multiple']:.2f}R")
    m10.metric("Bars Held", f"{s['avg_bars_held']:.0f}")

    # ── Equity curve ──
    st.markdown("---")
    curve_df = pd.DataFrame(report["equity_curve"])
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=curve_df["trade_num"], y=curve_df["equity"],
        mode="lines+markers", line=dict(color="#00C853", width=2), marker=dict(size=6), name="Equity",
    ))
    fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_eq.update_layout(xaxis_title="Trade #", yaxis_title="P&L ($/share)", height=300, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── MAE / MFE ──
    with st.expander("MAE / MFE Analysis", expanded=False):
        mae_col1, mae_col2, mae_col3, mae_col4, mae_col5 = st.columns(5)
        mae_col1.metric("Avg MAE", f"${s['avg_mae']:.2f}")
        mae_col2.metric("Avg MFE", f"${s['avg_mfe']:.2f}")
        mae_col3.metric("MAE (R)", f"{s['avg_mae_r']:.2f}R")
        mae_col4.metric("MFE (R)", f"{s['avg_mfe_r']:.2f}R")
        mae_col5.metric("Edge Ratio", f"{s['edge_ratio']:.2f}")

        if trades:
            fig_mae = go.Figure()
            winners = [t for t in trades if t.is_winner]
            losers = [t for t in trades if not t.is_winner]
            if winners:
                fig_mae.add_trace(go.Scatter(
                    x=[t.mae_r for t in winners], y=[t.mfe_r for t in winners],
                    mode="markers", marker=dict(color="#00C853", size=10, opacity=0.7), name="Winners",
                    text=[f"{t.setup_name}<br>${t.pnl:.2f}" for t in winners],
                    hovertemplate="%{text}<br>MAE: %{x:.1f}R<br>MFE: %{y:.1f}R<extra></extra>",
                ))
            if losers:
                fig_mae.add_trace(go.Scatter(
                    x=[t.mae_r for t in losers], y=[t.mfe_r for t in losers],
                    mode="markers", marker=dict(color="#FF1744", size=10, opacity=0.7), name="Losers",
                    text=[f"{t.setup_name}<br>${t.pnl:.2f}" for t in losers],
                    hovertemplate="%{text}<br>MAE: %{x:.1f}R<br>MFE: %{y:.1f}R<extra></extra>",
                ))
            max_val = max(max((t.mae_r for t in trades), default=1), max((t.mfe_r for t in trades), default=1)) * 1.1
            fig_mae.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(dash="dash", color="gray", width=1), showlegend=False))
            fig_mae.update_layout(xaxis_title="MAE (R)", yaxis_title="MFE (R)", height=350, margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_mae, use_container_width=True)

    # ── Streaks & Exit Reasons ──
    col_exit, col_streak = st.columns(2)
    with col_exit:
        st.markdown("**Exit Reasons**")
        for reason, count in s["exit_reasons"].items():
            st.markdown(f"- {reason.replace('_', ' ').title()}: {count}")
    with col_streak:
        st.markdown("**Streaks**")
        st.markdown(f"- Best Win Streak: {s['max_win_streak']}")
        st.markdown(f"- Worst Loss Streak: {s['max_loss_streak']}")

    # ── Daily breakdown ──
    if "daily_results" in s:
        with st.expander("Daily Breakdown", expanded=False):
            daily_rows = [{
                "Date": d["date"], "Day Type": d["day_type"],
                "Setups": d["setups_found"], "Trades": d["trades"],
                "Winners": d["winners"], "P&L": f"${d['pnl']:.2f}",
            } for d in s["daily_results"]]
            if daily_rows:
                st.dataframe(pd.DataFrame(daily_rows), width="stretch", hide_index=True)

    # ── Trade log (clickable) ──
    st.markdown("**Trade Log** -- select a row to view the chart")
    trade_df = trades_to_dataframe(trades)
    selection = st.dataframe(
        trade_df, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key="bt_trade_select",
    )

    # Show chart for selected trade
    sel_rows = selection.selection.rows if selection and selection.selection else []
    if sel_rows:
        idx = sel_rows[0]
        if idx < len(trades):
            sel_trade = trades[idx]
            daily_dfs = st.session_state.get("bt_daily_dfs", {})
            used_ticker = st.session_state.get("bt_ticker_used", bt_ticker)
            # Find which day this trade belongs to by matching entry_time date
            trade_date = sel_trade.entry_time[:10] if sel_trade.entry_time else ""
            day_df = daily_dfs.get(trade_date)
            if day_df is not None and not day_df.empty:
                if "BarNumber" not in day_df.columns:
                    day_df = day_df.copy()
                    day_df["BarNumber"] = range(1, len(day_df) + 1)
                if "EMA20" not in day_df.columns:
                    day_df = day_df.copy()
                    day_df["EMA20"] = day_df["Close"].ewm(span=20, adjust=False).mean()
                fig_trade = build_trade_chart(day_df, sel_trade, used_ticker, is_daily=False)
                st.plotly_chart(fig_trade, use_container_width=True, key="bt_trade_chart")
            else:
                st.caption("Chart data not available for this trade's date.")

    csv_data = trade_df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, f"backtest_{bt_ticker}_{bt_mode}.csv", "text/csv")


# ─────────────────────────── DAILY BACKTEST TAB ──────────────────────────────

def render_backtest_daily():
    """Daily-chart backtesting — uses daily bars so yFinance can go back years."""
    from backtester import run_daily_backtest, trades_to_dataframe

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        dt_ticker = st.text_input("Ticker", value="SPY", key="dt_ticker").upper().strip()
    with col2:
        dt_mode = st.selectbox("Mode", ["swing", "scalp"], key="dt_mode",
                                help="Swing = 2:1 R/R target (default for daily). Scalp = 1:1 R/R.")
    with col3:
        dt_years = st.selectbox("Period", ["2y", "5y", "10y", "1y", "max"], key="dt_period",
                                 help="How far back to test. Daily bars from yFinance.")

    col4, col5, col6, col7 = st.columns([1, 1, 1, 1])
    with col4:
        dt_hold = st.number_input("Max Hold (days)", min_value=2, max_value=120, value=15, key="dt_hold",
                                   help="Max trading days to hold before forced exit")
    with col5:
        dt_gap = st.number_input("Min Gap Between Trades", min_value=0, max_value=20, value=3, key="dt_gap",
                                  help="Min bars after exit before entering next trade")
    with col6:
        dt_max_trades = st.number_input("Max Trades", min_value=10, max_value=500, value=200, key="dt_max_trades",
                                         help="Maximum number of trades to simulate")
    with col7:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run Backtest", key="dt_run", type="primary")

    if run_btn:
        with st.spinner(f"Backtesting {dt_ticker} daily chart ({dt_years}, {dt_mode} mode)..."):
            try:
                import yfinance as yf
            except ImportError:
                st.error("yfinance is required for daily backtesting. Install it with: pip install yfinance")
                return

            try:
                df = yf.download(dt_ticker, period=dt_years, interval="1d", progress=False)
            except Exception as e:
                st.error(f"Failed to fetch daily data: {e}")
                return

            if df is None or df.empty:
                st.warning(f"No daily data for {dt_ticker}.")
                return

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = ["Open", "High", "Low", "Close"]
            if not all(c in df.columns for c in required):
                st.warning("Missing OHLC columns in data.")
                return

            df = df.dropna(subset=required)

            if len(df) < 20:
                st.warning(f"Only {len(df)} daily bars — need at least 20.")
                return

            st.caption(f"Data: **yFinance daily** | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} bars)")

            report = run_daily_backtest(df, mode=dt_mode, hold_limit=dt_hold,
                                          max_trades=dt_max_trades,
                                          min_bars_between_trades=dt_gap)
            st.session_state["dt_report"] = report
            st.session_state["dt_source_df"] = df
            st.session_state["dt_ticker_used"] = dt_ticker

    report = st.session_state.get("dt_report")
    if not report:
        st.info("Configure settings and press **Run Backtest**. Uses daily bars from yFinance -- can test years of data.")
        return

    s = report["summary"]
    trades = report["trades"]

    if s["total_trades"] == 0:
        st.warning("No trades generated. The algo didn't find setups in this data.")
        return

    # ── Summary metrics ──
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", s["total_trades"])
    m2.metric("Win Rate", f"{s['win_rate']:.1%}")
    m3.metric("P&L", f"${s['total_pnl']:.2f}/sh")
    m4.metric("Profit Factor", f"{s['profit_factor']:.2f}")
    m5.metric("Sharpe", f"{s['sharpe_annualized']:.2f}")

    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric("Avg Win", f"${s['avg_winner']:.2f}")
    m7.metric("Avg Loss", f"${s['avg_loser']:.2f}")
    m8.metric("Max DD", f"${s['max_drawdown']:.2f}")
    m9.metric("Avg R", f"{s['avg_r_multiple']:.2f}R")
    m10.metric("Days Held", f"{s['avg_bars_held']:.1f}")

    # ── Equity curve ──
    st.markdown("---")
    curve_df = pd.DataFrame(report["equity_curve"])
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=curve_df["trade_num"], y=curve_df["equity"],
        mode="lines+markers", line=dict(color="#00C853", width=2), marker=dict(size=4), name="Equity",
    ))
    fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_eq.update_layout(xaxis_title="Trade #", yaxis_title="P&L ($/share)", height=300, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── MAE / MFE ──
    with st.expander("MAE / MFE Analysis", expanded=False):
        mae_col1, mae_col2, mae_col3, mae_col4, mae_col5 = st.columns(5)
        mae_col1.metric("Avg MAE", f"${s['avg_mae']:.2f}")
        mae_col2.metric("Avg MFE", f"${s['avg_mfe']:.2f}")
        mae_col3.metric("MAE (R)", f"{s['avg_mae_r']:.2f}R")
        mae_col4.metric("MFE (R)", f"{s['avg_mfe_r']:.2f}R")
        mae_col5.metric("Edge Ratio", f"{s['edge_ratio']:.2f}")

        if trades:
            fig_mae = go.Figure()
            winners = [t for t in trades if t.is_winner]
            losers = [t for t in trades if not t.is_winner]
            if winners:
                fig_mae.add_trace(go.Scatter(
                    x=[t.mae_r for t in winners], y=[t.mfe_r for t in winners],
                    mode="markers", marker=dict(color="#00C853", size=10, opacity=0.7), name="Winners",
                    text=[f"{t.setup_name}<br>${t.pnl:.2f}" for t in winners],
                    hovertemplate="%{text}<br>MAE: %{x:.1f}R<br>MFE: %{y:.1f}R<extra></extra>",
                ))
            if losers:
                fig_mae.add_trace(go.Scatter(
                    x=[t.mae_r for t in losers], y=[t.mfe_r for t in losers],
                    mode="markers", marker=dict(color="#FF1744", size=10, opacity=0.7), name="Losers",
                    text=[f"{t.setup_name}<br>${t.pnl:.2f}" for t in losers],
                    hovertemplate="%{text}<br>MAE: %{x:.1f}R<br>MFE: %{y:.1f}R<extra></extra>",
                ))
            max_val = max(max((t.mae_r for t in trades), default=1), max((t.mfe_r for t in trades), default=1)) * 1.1
            fig_mae.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(dash="dash", color="gray", width=1), showlegend=False))
            fig_mae.update_layout(xaxis_title="MAE (R)", yaxis_title="MFE (R)", height=350, margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_mae, use_container_width=True)

    # ── Streaks & Exit Reasons ──
    col_exit, col_streak = st.columns(2)
    with col_exit:
        st.markdown("**Exit Reasons**")
        for reason, count in s["exit_reasons"].items():
            st.markdown(f"- {reason.replace('_', ' ').title()}: {count}")
    with col_streak:
        st.markdown("**Streaks**")
        st.markdown(f"- Best Win Streak: {s['max_win_streak']}")
        st.markdown(f"- Worst Loss Streak: {s['max_loss_streak']}")

    # ── Trade log (clickable) ──
    st.markdown("**Trade Log** -- select a row to view the chart")
    trade_df = trades_to_dataframe(trades)
    dt_selection = st.dataframe(
        trade_df, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key="dt_trade_select",
    )

    # Show chart for selected trade
    dt_sel_rows = dt_selection.selection.rows if dt_selection and dt_selection.selection else []
    if dt_sel_rows:
        idx = dt_sel_rows[0]
        if idx < len(trades):
            sel_trade = trades[idx]
            source_df = st.session_state.get("dt_source_df")
            used_ticker = st.session_state.get("dt_ticker_used", dt_ticker)
            if source_df is not None and not source_df.empty:
                fig_trade = build_trade_chart(source_df, sel_trade, used_ticker, is_daily=True)
                st.plotly_chart(fig_trade, use_container_width=True, key="dt_trade_chart")
            else:
                st.caption("Chart data not available.")

    csv_data = trade_df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, f"daily_backtest_{dt_ticker}_{dt_mode}.csv", "text/csv", key="dt_csv")


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    render_sidebar()

    tab_train, tab_backtest, tab_daily, tab_scanner, tab_library = st.tabs(
        ["Training Lab", "Backtest 5m", "Backtest Daily", "Scanner", "Library"]
    )

    with tab_train:
        render_training_lab()

    with tab_backtest:
        render_backtest()

    with tab_daily:
        render_backtest_daily()

    with tab_scanner:
        render_scanner()

    with tab_library:
        render_library()


if __name__ == "__main__":
    main()
