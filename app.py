"""
Human-in-the-Loop Trading Bot Trainer
Built with Streamlit, yFinance, Plotly, and Gemini.
"""
from __future__ import annotations

import os
import io
import json
import random
import datetime
import pathlib
import concurrent.futures
import threading

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from google import genai

# ─────────────────────────── CONFIG ──────────────────────────────────────────

st.set_page_config(
    page_title="Trading Bot Trainer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use DATA_DIR env var for persistent storage (Render Disk mount)
# Locally defaults to "." so nothing changes for local dev
DATA_DIR = pathlib.Path(os.environ.get("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_CSV = DATA_DIR / "training_data.csv"
ENCYCLOPEDIA_PATH = DATA_DIR / "brooks_encyclopedia_learnings.md"

CSV_COLUMNS = [
    "timestamp", "ticker",
    "bot_day_type", "bot_market_cycle",
    "bot_setup_1", "bot_setup_1_bar", "bot_setup_1_price", "bot_setup_1_order_type",
    "bot_setup_2", "bot_setup_2_bar", "bot_setup_2_price", "bot_setup_2_order_type",
    "bot_setup_3", "bot_setup_3_bar", "bot_setup_3_price", "bot_setup_3_order_type",
    "bot_setup_4", "bot_setup_4_bar", "bot_setup_4_price", "bot_setup_4_order_type",
    "bot_setup_5", "bot_setup_5_bar", "bot_setup_5_price", "bot_setup_5_order_type",
    "bot_action", "bot_confidence",
    "override_day_type", "override_market_cycle",
    "override_setup_1", "override_setup_1_bar", "override_setup_1_price", "override_setup_1_order_type",
    "override_setup_2", "override_setup_2_bar", "override_setup_2_price", "override_setup_2_order_type",
    "override_setup_3", "override_setup_3_bar", "override_setup_3_price", "override_setup_3_order_type",
    "override_setup_4", "override_setup_4_bar", "override_setup_4_price", "override_setup_4_order_type",
    "override_setup_5", "override_setup_5_bar", "override_setup_5_price", "override_setup_5_order_type",
    "override_action", "teacher_notes",
]

# ─────────────────────────── FORM OPTIONS ────────────────────────────────────

DAY_TYPE_OPTIONS = [
    "Approve Bot's Guess",
    "Trading Range Day",
    "Bull Trend From The Open",
    "Bear Trend From The Open",
    "Trending Trading Range Day",
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
    "Broad Channel",
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

# ─────────────────────────── API KEY ─────────────────────────────────────────

def get_api_key() -> str:
    """Load GEMINI_API_KEY from the environment or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return key
    except (FileNotFoundError, KeyError):
        pass
    return ""

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
    return tickers

# ─────────────────────────── DATA FETCHING ───────────────────────────────────

def fetch_chart_data(ticker: str) -> pd.DataFrame | None:
    """Fetch 3 days of 5-minute OHLCV data for *ticker*."""
    try:
        df = yf.download(ticker, period="1d", interval="5m", progress=False)
        if df is None or df.empty:
            return None
        # Flatten multi-level columns if present
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
            dtick=5, # Show a tick every 5 bars
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

# ─────────────────────────── ENCYCLOPEDIA CACHE ──────────────────────────────

@st.cache_data(ttl=None)
def load_encyclopedia() -> str:
    """Read the Brooks encyclopedia Markdown file once and cache it."""
    if ENCYCLOPEDIA_PATH.exists():
        return ENCYCLOPEDIA_PATH.read_text(encoding="utf-8")
    return ""

# ─────────────────────────── GEMINI BRAIN ────────────────────────────────────

SYSTEM_PROMPT = """You are an expert day trader trained strictly on Al Brooks' Price Action methodology.
Analyze the provided 5-minute chart image using ONLY the provided encyclopedia rules.
Do not use outside trading knowledge.

Encyclopedia rules:
{encyclopedia}

You MUST return a strict JSON object with exactly these keys:
  day_type, market_cycle, reasoning, setups, action, confidence

- day_type: one of ["Trading Range Day","Bull Trend From The Open","Bear Trend From The Open","Trending Trading Range Day","Small Pullback Bull Trend","Small Pullback Bear Trend","Spike and Channel Bull Trend","Spike and Channel Bear Trend","Broad Bull Channel","Broad Bear Channel","Shrinking Stairs","Reversal Day (Bull)","Reversal Day (Bear)","Crash Day","Climax Day"]
- market_cycle: one of ["Breakout (Spike)","Micro Channel","Tight Channel (Small PB Trend)","Broad Channel","Trading Range"]
- reasoning: A brief 1-sentence explanation of your overall analysis of the chart.
- setups: A list of the top 5 BEST setups of the day. Each item in the list MUST be a JSON object with exactly these keys: `{{"setup_name": "...", "entry_bar": 1, "entry_price": 0.00, "order_type": "...", "reason_1": "...", "reason_2": "..."}}`. `setup_name` MUST STRICTLY be one of ["High 1 Bull Flag","High 2 Bull Flag","High 3 Bull Flag","High 4 Bull Flag","Low 1 Bear Flag","Low 2 Bear Flag","Low 3 Bear Flag","Low 4 Bear Flag","Double Bottom","Double Top","Higher Low Double Bottom","Lower Low Double Bottom","Lower High Double Top","Higher High Double Top","Major Trend Reversal (Bull)","Major Trend Reversal (Bear)","Wedge Bottom","Wedge Top","Parabolic Wedge Bottom","Parabolic Wedge Top","Spike and Channel Bull","Spike and Channel Bear","Head & Shoulders Bottom","Head & Shoulders Top","Final Bull Flag","Final Bear Flag","Breakout (BO)","Breakout Test","Failed Breakout (Bull Trap)","Failed Breakout (Bear Trap)","Measuring Gap / Exhaustion Gap","Buy Climax","Sell Climax","Ledge Bottom","Ledge Top","ii Pattern","ioi Pattern","OO Pattern","Opening Reversal (Bull)","Opening Reversal (Bear)","20-Gap Bar Buy","20-Gap Bar Sell","Cup and Handle"]. DO NOT invent or use any other setup names. `entry_bar` is the integer Bar Number of the entry, as shown on great chart's x-axis (e.g., 18). `order_type` MUST be exactly "Stop" or "Limit", denoting how the entry is executed mechanically. `reason_1` and `reason_2` are two distinct technical reasons justifying why this setup is valid (as Al Brooks requires 2 reasons to take any trade).
- action: one of ["Buy","Sell","Wait / No Trade"]
- confidence: a float between 0.0 and 1.0

Return ONLY the JSON object. No markdown, no explanation, no code fences."""

def _get_or_create_cache(client, system_text: str) -> str:
    """Retrieve existing GenAI context cache or create a new one."""
    from google.genai import types
    cache_name = st.session_state.get("gemini_cache_name")
    
    if cache_name:
        try:
            client.caches.get(name=cache_name)
            return cache_name
        except Exception:
            pass # Invalid or expired cache
            
    try:
        cache = client.caches.create(
            model="models/gemini-2.5-pro",
            config=types.CreateCachedContentConfig(
                system_instruction=system_text,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text="Respond to charts using the provided encyclopedia rules.")])],
                ttl="3600s",
            )
        )
        st.session_state["gemini_cache_name"] = cache.name
        return cache.name
    except Exception as e:
        print(f"GenAI Cache creation failed: {e}")
        return None


def analyze_chart(fig: go.Figure) -> dict:
    """Send chart image to Gemini and get a structured analysis."""
    api_key = get_api_key()
    if not api_key:
        return {
            "day_type": "N/A",
            "market_cycle": "N/A",
            "reasoning": "No API Key configured.",
            "setups": [
                {"setup_name": "N/A", "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A", "reason_1": "N/A", "reason_2": "N/A"}
            ] * 5,
            "action": "N/A",
            "confidence": 0.0,
            "_error": "GEMINI_API_KEY not set. Set it in the environment or .streamlit/secrets.toml",
        }

    # Convert figure to PNG bytes with smaller dimensions for latency optimization
    img_bytes = fig.to_image(format="png", width=1000, height=500, scale=1.5)

    encyclopedia = load_encyclopedia()
    system_text = SYSTEM_PROMPT.format(encyclopedia=encyclopedia if encyclopedia else "(no rules loaded yet)")

    try:
        from google.genai import types
        client = genai.Client(api_key=api_key)
        cache_name = _get_or_create_cache(client, system_text)
        
        if cache_name:
            config = types.GenerateContentConfig(
                cached_content=cache_name,
                temperature=0.2,
                response_mime_type="application/json",
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_text,
                temperature=0.2,
                response_mime_type="application/json",
            )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
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
        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        return json.loads(raw)

    except Exception as e:
        return {
            "day_type": "Error",
            "market_cycle": "Error",
            "reasoning": "API Request Failed.",
            "setups": [
                {"setup_name": "Error", "entry_bar": 0, "entry_price": 0.0, "order_type": "Error", "reason_1": "Error", "reason_2": "Error"}
            ] * 5,
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

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
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
        
        # Purge the GenAI context cache so it rebuilds with new rules!
        if "gemini_cache_name" in st.session_state:
            try:
                client.caches.delete(name=st.session_state["gemini_cache_name"])
            except Exception:
                pass
            del st.session_state["gemini_cache_name"]

    except Exception as e:
        st.error(f"Failed to auto-update encyclopedia: {e}")

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

    tickers = get_sp500_tickers()
    random.shuffle(tickers)
    for t in tickers:
        df = fetch_chart_data(t)
        if df is not None and len(df) > 30:
            st.session_state["ticker"] = t
            st.session_state["chart_df"] = df
            return
    st.error("Could not fetch data for any ticker. Please try again.")


def _add_annotations(fig, df, analysis):
    """Add setup marker annotations to a chart figure."""
    bot_setups = analysis.get("setups", [])
    price_min = df["Low"].min()
    price_max = df["High"].max()
    price_range = price_max - price_min
    annot_offset = price_range * 0.06

    for i in range(5):
        obj = bot_setups[i] if i < len(bot_setups) else {}
        if isinstance(obj, str):
            obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0}

        b_name = obj.get("setup_name", "")
        b_bar = obj.get("entry_bar", 0)
        b_price = obj.get("entry_price", 0.0)

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
            fig.add_shape(
                type="line",
                x0=int(b_bar) - 0.45,
                x1=int(b_bar) + 0.45,
                y0=validated_price,
                y1=validated_price,
                line=dict(color="#fbbf24", width=3, dash="dot"),
                layer="above"
            )
            fig.add_annotation(
                x=int(b_bar),
                y=bar_low - annot_offset * (1 + i * 0.5),
                text=f"{b_name}<br>(Bar {b_bar})",
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor=color,
                ay=20,
                ax=0,
                yanchor="top",
                font=dict(color=color, size=10, family="Arial"),
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor=color,
                borderwidth=1,
                borderpad=2,
                opacity=0.9
            )


def _do_prefetch():
    """Background: fetch a random ticker, build chart, query Gemini."""
    try:
        tickers = get_sp500_tickers()
        random.shuffle(tickers)
        for t in tickers:
            df = fetch_chart_data(t)
            if df is not None and len(df) > 30:
                fig = build_chart(df, t)
                analysis = analyze_chart(fig)
                _add_annotations(fig, df, analysis)
                return {"ticker": t, "df": df, "fig": fig, "analysis": analysis}
        return {}
    except Exception:
        return {}


def start_prefetch():
    """Start background prefetch of next chart."""
    if "prefetch_future" not in st.session_state:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_prefetch)
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
        st.markdown("## 📈 Trading Bot Trainer")
        st.markdown("---")
        count = len(load_training_csv())
        st.metric("📊 Charts Corrected", f"{count} / 100")
        st.progress(min(count / 100, 1.0))
        st.markdown("---")
        st.markdown(
            "Train a Gemini-powered bot on **Al Brooks' Price Action** by correcting its guesses."
        )
        st.markdown("---")
        st.caption("Built with Streamlit · Gemini · yFinance")

# ─────────────────────────── TRAINING LAB TAB ────────────────────────────────

def render_training_lab():
    # Inject Custom CSS to make the form more compact
    st.markdown(
        """
        <style>
        /* Reduce font sizes and paddings in inputs */
        div[data-testid="stForm"] {
            font-size: 0.85rem;
        }
        div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label, div[data-testid="stNumberInput"] label {
            font-size: 0.85rem !important;
            min-height: 0px !important;
            padding-bottom: 2px !important;
        }
        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div {
            min-height: 32px;
            font-size: 0.85rem;
        }
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
            padding: 4px 8px;
            font-size: 0.85rem;
        }
        /* Tighten margin between form rows */
        div[data-testid="stVerticalBlock"] > div {
            padding-bottom: 0.2rem;
        }
        /* Make markdown subheaders smaller */
        div[data-testid="stForm"] p {
            font-size: 0.9rem;
            margin-bottom: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Ensure we have chart data in session state
    if "ticker" not in st.session_state:
        load_new_chart()
    if "ticker" not in st.session_state:
        return  # error was already shown

    ticker = st.session_state["ticker"]
    df = st.session_state["chart_df"]

    # PHASE 1: Build and show the chart IMMEDIATELY (no waiting for Gemini)
    if "chart_fig" not in st.session_state:
        fig = build_chart(df, ticker)
        st.session_state["chart_fig"] = fig

    fig = st.session_state["chart_fig"]

    if "bot_analysis" not in st.session_state:
        # Show the clean chart right away with a spinner below
        st.plotly_chart(fig, use_container_width=True, key="main_chart_loading")
        with st.spinner("\U0001f916 Bot is analyzing the chart with Gemini..."):
            analysis = analyze_chart(fig)
        st.session_state["bot_analysis"] = analysis
        # Add annotations now that we have analysis
        _add_annotations(fig, df, analysis)
        st.session_state["chart_fig"] = fig
        # Start prefetching the NEXT chart in background
        start_prefetch()
        st.rerun()

    analysis = st.session_state["bot_analysis"]

    # Check/start prefetch for next chart
    check_prefetch()
    if "prefetch_future" not in st.session_state and "prefetch_ready" not in st.session_state:
        start_prefetch()

    with st.form("override_form", clear_on_submit=True):
        st.plotly_chart(fig, use_container_width=True, key="main_chart")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            approve = st.form_submit_button("✅ Approve All", use_container_width=True)
        with btn_col2:
            submit = st.form_submit_button("📝 Submit Corrections", use_container_width=True)
        with btn_col3:
            skip = st.form_submit_button("⏭️ Skip Chart", use_container_width=True)
            
        st.markdown("---")

        col_bot, col_form = st.columns([1, 1], gap="large")

        # ── Bot Analysis Column ──
        with col_bot:
            st.subheader("🤖 Bot's Analysis")
            if "_error" in analysis:
                st.warning(analysis["_error"])
            st.json(analysis)

        # ── Teacher Override Column ──
        with col_form:
            st.subheader("🎓 Teacher's Override")
            day_type = st.selectbox("Day Type", DAY_TYPE_OPTIONS)
            market_cycle = st.selectbox("Market Cycle", MARKET_CYCLE_OPTIONS)
            st.markdown("**Top 5 Setups (Name, Bar #, Price, Order Type):**")
            override_setups = []
            override_bars = []
            override_prices = []
            override_orders = []

            bot_setups = analysis.get("setups", [])
            for i in range(5):
                obj = bot_setups[i] if i < len(bot_setups) else {}
                if isinstance(obj, str): # Fallback if bot hallucinates structure
                    obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A"}
                
                b_name = obj.get("setup_name", "N/A")
                b_bar = obj.get("entry_bar", 0)
                b_price = obj.get("entry_price", 0.0)
                b_order = obj.get("order_type", "N/A")

                st.markdown(f"**Setup {i+1}** *(Bot: {b_name} [{b_order}] @ Bar {b_bar}, ${b_price})*")
                scol1, scol2, scol3, scol4 = st.columns([3, 1, 1, 2])
                with scol1:
                    sel = st.selectbox("Name", SETUP_OPTIONS, key=f"setup_name_{i}")
                    override_setups.append(sel)
                with scol2:
                    br = st.number_input("Bar #", value=int(b_bar) if b_bar else 0, step=1, key=f"setup_bar_{i}")
                    override_bars.append(br)
                with scol3:
                    pr = st.number_input("Price", value=float(b_price) if b_price else 0.0, step=0.1, format="%.2f", key=f"setup_price_{i}")
                    override_prices.append(pr)
                with scol4:
                    ort = st.selectbox("Order Type", ORDER_OPTIONS, key=f"setup_order_{i}")
                    override_orders.append(ort)

            action = st.selectbox("Action", ACTION_OPTIONS)
            notes = st.text_area("Teacher's Notes", placeholder="Why did you override?")

    # ── Handle Buttons ──
    if approve or submit:
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "ticker": ticker,
            "bot_day_type": analysis.get("day_type", ""),
            "bot_market_cycle": analysis.get("market_cycle", ""),
            "bot_action": analysis.get("action", ""),
            "bot_confidence": analysis.get("confidence", ""),
        }
        
        bot_setups = analysis.get("setups", [])
        for i in range(5):
            obj = bot_setups[i] if i < len(bot_setups) else {}
            if isinstance(obj, str):
                obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A"}
            row[f"bot_setup_{i+1}"] = obj.get("setup_name", "")
            row[f"bot_setup_{i+1}_bar"] = obj.get("entry_bar", 0)
            row[f"bot_setup_{i+1}_price"] = obj.get("entry_price", 0.0)
            row[f"bot_setup_{i+1}_order_type"] = obj.get("order_type", "")

        if approve:
            row["override_day_type"] = analysis.get("day_type", "")
            row["override_market_cycle"] = analysis.get("market_cycle", "")
            for i in range(5):
                obj = bot_setups[i] if i < len(bot_setups) else {}
                if isinstance(obj, str):
                    obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A"}
                row[f"override_setup_{i+1}"] = obj.get("setup_name", "")
                row[f"override_setup_{i+1}_bar"] = obj.get("entry_bar", 0)
                row[f"override_setup_{i+1}_price"] = obj.get("entry_price", 0.0)
                row[f"override_setup_{i+1}_order_type"] = obj.get("order_type", "")
            row["override_action"] = analysis.get("action", "")
            row["teacher_notes"] = notes
        else:  # submit corrections
            row["override_day_type"] = day_type if day_type != "Approve Bot's Guess" else analysis.get("day_type", "")
            row["override_market_cycle"] = market_cycle if market_cycle != "Approve Bot's Guess" else analysis.get("market_cycle", "")
            for i in range(5):
                obj = bot_setups[i] if i < len(bot_setups) else {}
                if isinstance(obj, str):
                    obj = {"setup_name": obj, "entry_bar": 0, "entry_price": 0.0, "order_type": "N/A"}
                
                # If teacher says "Approve", use the bot's guessed name
                if override_setups[i] == "Approve Bot's Guess":
                    row[f"override_setup_{i+1}"] = obj.get("setup_name", "")
                else:
                    row[f"override_setup_{i+1}"] = override_setups[i]

                row[f"override_setup_{i+1}_bar"] = override_bars[i]
                row[f"override_setup_{i+1}_price"] = override_prices[i]
                
                if override_orders[i] == "Approve Bot's Guess":
                    row[f"override_setup_{i+1}_order_type"] = obj.get("order_type", "")
                else:
                    row[f"override_setup_{i+1}_order_type"] = override_orders[i]

            row["override_action"] = action if action != "Approve Bot's Guess" else analysis.get("action", "")
            row["teacher_notes"] = notes

        # Update Encyclopedia if notes exist
        if notes.strip():
            with st.spinner("🧠 Teaching Bot... Updating Encyclopedia with your notes."):
                update_encyclopedia(notes.strip())
            st.toast("Encyclopedia permanently updated!", icon="✅")

        save_row(row)
        # Clear session state to force new chart
        for key in ["ticker", "chart_df", "chart_fig", "bot_analysis"]:
            st.session_state.pop(key, None)
        st.rerun()

    if skip:
        for key in ["ticker", "chart_df", "chart_fig", "bot_analysis"]:
            st.session_state.pop(key, None)
        st.rerun()

# ─────────────────────────── HISTORY TAB ─────────────────────────────────────

def render_history():
    df = load_training_csv()
    if df.empty:
        st.info("No training data yet. Start correcting charts in the Training Lab!")
        return

    st.dataframe(df, use_container_width=True, height=500)

    st.markdown("---")
    st.subheader("🗑️ Delete a Row")
    col_del, col_btn = st.columns([3, 1])
    with col_del:
        row_idx = st.number_input(
            "Row index to delete",
            min_value=0,
            max_value=max(len(df) - 1, 0),
            value=0,
            step=1,
        )
    with col_btn:
        if st.button("🗑️ Delete Row", use_container_width=True, type="primary"):
            delete_row(int(row_idx))
            st.rerun()


# ─────────────────────────── EXAMPLES TAB ────────────────────────────────────

def render_examples():
    """Browse training examples filtered by day type, market cycle, setup, or notes."""
    df = load_training_csv()
    if df.empty:
        st.info("No training data yet. Start correcting charts in the Training Lab!")
        return

    st.markdown("#### Filter Your Training Examples")

    # ── Filter Controls ──
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

    with fcol1:
        day_types = sorted(df["override_day_type"].dropna().unique().tolist())
        sel_day = st.multiselect("Day Type", day_types, placeholder="All day types")

    with fcol2:
        cycles = sorted(df["override_market_cycle"].dropna().unique().tolist())
        sel_cycle = st.multiselect("Market Cycle", cycles, placeholder="All cycles")

    with fcol3:
        # Gather all unique setup names across all 5 setup columns
        setup_cols = [f"override_setup_{i}" for i in range(1, 6)]
        all_setups = set()
        for col in setup_cols:
            if col in df.columns:
                all_setups.update(df[col].dropna().unique().tolist())
        all_setups.discard("")
        all_setups.discard("N/A")
        all_setups.discard("Error")
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

    # ── Summary ──
    st.markdown(f"**Showing {len(filtered)} of {len(df)} examples**")
    st.markdown("---")

    if filtered.empty:
        st.warning("No examples match your filters. Try broadening your search.")
        return

    # ── Display Results as Expanders ──
    for idx, row in filtered.iterrows():
        ticker = row.get("ticker", "?")
        ts = row.get("timestamp", "")
        o_day = row.get("override_day_type", "")
        o_cycle = row.get("override_market_cycle", "")
        label = f"**{ticker}** — {o_day} | {o_cycle} — _{ts}_"

        with st.expander(label, expanded=False):
            ecol1, ecol2 = st.columns(2)

            with ecol1:
                st.markdown("**Teacher\'s Correction:**")
                st.markdown(f"- **Day Type:** {o_day}")
                st.markdown(f"- **Market Cycle:** {o_cycle}")
                st.markdown(f"- **Action:** {row.get('override_action', '')}")
                st.markdown("")
                st.markdown("**Setups:**")
                for i in range(1, 6):
                    s_name = row.get(f"override_setup_{i}", "")
                    s_bar = row.get(f"override_setup_{i}_bar", "")
                    s_price = row.get(f"override_setup_{i}_price", "")
                    s_order = row.get(f"override_setup_{i}_order_type", "")
                    if s_name and s_name not in ("", "N/A", "Error"):
                        st.markdown(f"  {i}. **{s_name}** [{s_order}] @ Bar {s_bar}, ${s_price}")

            with ecol2:
                st.markdown("**Bot\'s Original Guess:**")
                b_day = row.get("bot_day_type", "")
                b_cycle = row.get("bot_market_cycle", "")
                b_action = row.get("bot_action", "")
                day_diff = " ✏️" if b_day != o_day else " ✅"
                cycle_diff = " ✏️" if b_cycle != o_cycle else " ✅"
                st.markdown(f"- **Day Type:** {b_day}{day_diff}")
                st.markdown(f"- **Market Cycle:** {b_cycle}{cycle_diff}")
                st.markdown(f"- **Action:** {b_action}")
                st.markdown("")
                st.markdown("**Bot\'s Setups:**")
                for i in range(1, 6):
                    s_name = row.get(f"bot_setup_{i}", "")
                    s_bar = row.get(f"bot_setup_{i}_bar", "")
                    s_price = row.get(f"bot_setup_{i}_price", "")
                    s_order = row.get(f"bot_setup_{i}_order_type", "")
                    if s_name and s_name not in ("", "N/A", "Error"):
                        st.markdown(f"  {i}. {s_name} [{s_order}] @ Bar {s_bar}, ${s_price}")

            # Teacher notes
            notes = row.get("teacher_notes", "")
            if notes and str(notes).strip():
                st.markdown("---")
                st.markdown(f"**Teacher\'s Notes:** {notes}")


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    render_sidebar()

    tab_train, tab_history, tab_examples = st.tabs(["🧪 Training Lab", "📜 History", "📚 Examples"])

    with tab_train:
        render_training_lab()

    with tab_history:
        render_history()

    with tab_examples:
        render_examples()


if __name__ == "__main__":
    main()
