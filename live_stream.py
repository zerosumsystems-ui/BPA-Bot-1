"""
live_stream.py — Databento Live Streaming for Real-Time Trading Signals

Subscribes to Databento's live 5-minute OHLCV bars and runs the algo engine
on each new bar for real-time pattern detection.

Usage (within Streamlit):
    from live_stream import LiveBarStream
    stream = LiveBarStream(api_key, ticker, on_bar_callback)
    stream.start()
    ...
    stream.stop()
"""

import os
import time
import logging
import threading
import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiveBar:
    """A single live 5-min bar."""
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class LiveBarStream:
    """
    Manages a Databento live subscription for 5-minute OHLCV bars.

    Collects bars into a rolling DataFrame and calls back with updated
    analysis on each new bar.
    """

    def __init__(
        self,
        api_key: str,
        ticker: str,
        dataset: str = "XNAS.ITCH",
        on_bar: Optional[Callable[[pd.DataFrame, dict], None]] = None,
        max_bars: int = 78,  # One full trading day of 5-min bars
    ):
        self._api_key = api_key
        self._ticker = ticker
        self._dataset = dataset
        self._on_bar = on_bar
        self._max_bars = max_bars

        self._bars: deque[LiveBar] = deque(maxlen=max_bars)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_analysis: dict = {}
        self._error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_analysis(self) -> dict:
        return self._last_analysis

    @property
    def error(self) -> Optional[str]:
        return self._error

    def get_dataframe(self) -> pd.DataFrame:
        """Return current bars as a DataFrame matching app expectations."""
        if not self._bars:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        rows = [
            {
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": b.volume,
            }
            for b in self._bars
        ]
        df = pd.DataFrame(rows)
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["BarNumber"] = range(1, len(df) + 1)
        return df

    def start(self):
        """Start the live streaming thread."""
        if self._running:
            logger.warning("Stream already running")
            return

        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True
        logger.info(f"Live stream started for {self._ticker}")

    def stop(self):
        """Stop the live streaming thread."""
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(f"Live stream stopped for {self._ticker}")

    def _run(self):
        """Main streaming loop (runs in background thread)."""
        try:
            import databento as db

            client = db.Live(key=self._api_key)
            client.subscribe(
                dataset=self._dataset,
                schema="ohlcv-5m",
                symbols=[self._ticker],
            )

            for record in client:
                if self._stop_event.is_set():
                    break

                # Extract bar data from Databento record
                try:
                    bar = LiveBar(
                        timestamp=datetime.datetime.now(),
                        open=float(record.open) / 1e9,   # Databento uses fixed-point
                        high=float(record.high) / 1e9,
                        low=float(record.low) / 1e9,
                        close=float(record.close) / 1e9,
                        volume=float(record.volume),
                    )
                    self._bars.append(bar)

                    # Run algo analysis on updated bars
                    df = self.get_dataframe()
                    if len(df) >= 5:
                        from algo_engine import analyze_bars
                        self._last_analysis = analyze_bars(df)

                    # Callback with updated data
                    if self._on_bar:
                        self._on_bar(df, self._last_analysis)

                except Exception as e:
                    logger.error(f"Error processing bar: {e}")
                    continue

        except Exception as e:
            self._error = str(e)
            logger.error(f"Live stream error: {e}")
        finally:
            self._running = False

    def _run_simulated(self):
        """
        Simulated streaming for testing outside market hours.
        Generates fake bars every 5 seconds.
        """
        import random

        price = 150.0  # Starting price
        while not self._stop_event.is_set():
            change = random.gauss(0, 0.5)
            o = price
            h = o + abs(random.gauss(0, 0.3))
            l = o - abs(random.gauss(0, 0.3))
            c = o + change
            v = random.randint(10000, 100000)

            bar = LiveBar(
                timestamp=datetime.datetime.now(),
                open=o, high=max(o, h, c), low=min(o, l, c),
                close=c, volume=v,
            )
            self._bars.append(bar)
            price = c

            df = self.get_dataframe()
            if len(df) >= 5:
                from algo_engine import analyze_bars
                self._last_analysis = analyze_bars(df)

            if self._on_bar:
                self._on_bar(df, self._last_analysis)

            # Wait 5 seconds between simulated bars
            for _ in range(50):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)
