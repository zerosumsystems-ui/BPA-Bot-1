"""
data_source.py — Databento Data Source

Fetches 5-minute and daily OHLCV bars for US equities via Databento.

Usage:
    from data_source import get_data_source
    source = get_data_source(api_key="db-xxx")
    df = source.fetch_historical("AAPL")
"""

import os
import re
import logging
import datetime
import time
import socket
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────── RETRY UTILITIES ─────────────────────────────────

def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient error worth retrying with backoff."""
    err_str = str(exc).lower()
    if "502" in err_str or "503" in err_str or "bad gateway" in err_str or "service unavailable" in err_str:
        return True
    if "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str or "too many requests" in err_str:
        return True
    if "timeout" in err_str or "timed out" in err_str:
        return True
    if isinstance(exc, (socket.timeout, ConnectionError, TimeoutError, OSError)):
        return True
    if "connection" in err_str and ("error" in err_str or "reset" in err_str or "refused" in err_str):
        return True
    return False


def _is_rate_limit(exc: Exception) -> bool:
    """Check if error is specifically a rate limit (needs longer backoff)."""
    err_str = str(exc).lower()
    return "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str or "too many requests" in err_str

# ─────────────────────────── EXCHANGE MAPPING ────────────────────────────────

DATABENTO_DATASET = "XNAS.ITCH"

# TTL for the cached available_end date (seconds).
_AVAILABLE_END_TTL = 300  # 5 minutes


# ─────────────────────────── DATABENTO SOURCE ────────────────────────────────

class DatabentoSource:
    """Fetch OHLCV data from Databento with retry logic for transient errors."""

    def __init__(self, api_key: str):
        import databento as db
        self._api_key = api_key
        self._client = db.Historical(api_key)
        self._max_retries = 3
        self._base_wait = 1  # seconds; exponential backoff: 1s, 2s, 4s
        self._last_error_message = None
        self._available_end: Optional[str] = None
        self._available_end_ts: float = 0.0

    def name(self) -> str:
        return "Databento"

    # ── Internal helpers ──

    def _clamp_end(self, db_end: str) -> str:
        """Use cached available_end if fresh and the request exceeds it."""
        if (self._available_end
                and db_end > self._available_end
                and (time.time() - self._available_end_ts) < _AVAILABLE_END_TTL):
            return self._available_end
        return db_end

    def _cache_available_end(self, end_str: str):
        """Store the server-reported available end with a timestamp."""
        self._available_end = end_str
        self._available_end_ts = time.time()

    @staticmethod
    def _make_date_range(start: str, end: str):
        """Convert ISO date strings to Databento start/end with +1 day clamping."""
        start_dt = datetime.datetime.fromisoformat(start)
        end_dt = datetime.datetime.fromisoformat(end)
        s_str = start_dt.strftime("%Y-%m-%d")
        target_e = end_dt.date() + datetime.timedelta(days=1)
        max_end = datetime.date.today() + datetime.timedelta(days=1)
        if target_e > max_end:
            target_e = max_end
        e_str = target_e.strftime("%Y-%m-%d")
        return f"{s_str}T00:00:00", f"{e_str}T00:00:00"

    def _fetch_with_retry(self, dataset: str, ticker: str, schema: str,
                          db_start: str, db_end: str):
        """
        Single timeseries.get_range() call with retry + exponential backoff.
        Handles data_end_after_available_end corrections.
        Returns the raw data object on success, None on failure.
        """
        db_end = self._clamp_end(db_end)

        for attempt in range(self._max_retries):
            try:
                data = self._client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[ticker],
                    stype_in="raw_symbol",
                    schema=schema,
                    start=db_start,
                    end=db_end,
                )
                return data

            except Exception as ex:
                err_str = str(ex)

                if "data_end_after_available_end" in err_str:
                    match = re.search(r"available up to '([^']+)'", err_str)
                    if match:
                        new_end_str = match.group(1).replace(" ", "T")
                        self._cache_available_end(new_end_str)
                        logger.info(f"Databento: caching available_end={new_end_str}, retrying")
                        try:
                            return self._client.timeseries.get_range(
                                dataset=dataset,
                                symbols=[ticker],
                                stype_in="raw_symbol",
                                schema=schema,
                                start=db_start,
                                end=new_end_str,
                            )
                        except Exception as retry_ex:
                            logger.warning(f"Databento {schema} retry with corrected end also failed: {retry_ex}")
                            return None
                    else:
                        logger.warning(f"Could not parse available_end from: {err_str[:200]}")
                        return None

                if _is_transient_error(ex):
                    if attempt < self._max_retries - 1:
                        wait = self._base_wait * (2 ** attempt)
                        if _is_rate_limit(ex):
                            wait = min(wait * 2, 16)
                        logger.warning(
                            f"Databento {schema} transient error (attempt {attempt + 1}/{self._max_retries}), "
                            f"retrying in {wait}s: {err_str[:120]}"
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.error(f"Databento {schema} failed after {self._max_retries} attempts: {err_str[:200]}")
                        self._last_error_message = f"{schema}: {err_str[:200]}"
                        return None
                else:
                    logger.warning(f"Databento {schema} non-transient error: {err_str[:200]}")
                    self._last_error_message = f"{schema}: {err_str[:200]}"
                    return None

        return None

    def _fetch_batch_with_retry(self, dataset: str, batch: list, schema: str,
                                db_start: str, db_end: str, client=None):
        """Fetch a batch of tickers with retry. Returns DataFrame or None."""
        _client = client or self._client
        db_end = self._clamp_end(db_end)

        for attempt in range(self._max_retries):
            try:
                data = _client.timeseries.get_range(
                    dataset=dataset,
                    symbols=batch,
                    stype_in="raw_symbol",
                    schema=schema,
                    start=db_start,
                    end=db_end,
                )
                return data.to_df() if data else None

            except Exception as ex:
                err_str = str(ex)

                if "data_end_after_available_end" in err_str:
                    match = re.search(r"available up to '([^']+)'", err_str)
                    if match:
                        new_end_str = match.group(1).replace(" ", "T")
                        self._cache_available_end(new_end_str)
                        logger.info(f"Databento bulk: caching available_end={new_end_str}, retrying")
                        try:
                            data = _client.timeseries.get_range(
                                dataset=dataset,
                                symbols=batch,
                                stype_in="raw_symbol",
                                schema=schema,
                                start=db_start,
                                end=new_end_str,
                            )
                            return data.to_df() if data else None
                        except Exception as retry_ex:
                            logger.warning(f"Databento bulk retry with corrected end also failed: {retry_ex}")
                            self._last_error_message = f"bulk_batch: {str(retry_ex)[:200]}"
                            return None
                    else:
                        logger.warning(f"Could not parse available_end from: {err_str[:200]}")
                        self._last_error_message = f"bulk_batch: {err_str[:200]}"
                        return None

                if _is_transient_error(ex):
                    if attempt < self._max_retries - 1:
                        wait = self._base_wait * (2 ** attempt)
                        if _is_rate_limit(ex):
                            wait = min(wait * 2, 16)
                        logger.warning(
                            f"Databento bulk transient error (attempt {attempt + 1}/{self._max_retries}), "
                            f"retrying in {wait}s: {err_str[:120]}"
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.error(f"Databento bulk failed after {self._max_retries} attempts: {err_str[:200]}")
                        self._last_error_message = f"bulk_batch: {err_str[:200]}"
                        return None
                else:
                    logger.warning(f"Databento bulk non-transient error: {err_str[:200]}")
                    self._last_error_message = f"bulk_batch: {err_str[:200]}"
                    return None

        return None

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename Databento lowercase columns to capitalized, keep OHLCV."""
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns=rename_map)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep].dropna(subset=["Open", "High", "Low", "Close"])

    @staticmethod
    def _to_eastern(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure index is US/Eastern timezone."""
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        return df

    @staticmethod
    def _resample_5min_rth(df: pd.DataFrame) -> pd.DataFrame:
        """Resample to 5-min bars and filter to Regular Trading Hours."""
        df = df.resample("5min").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna()
        if not df.empty:
            df = df.between_time("09:30", "15:59")
        return df

    # ── Public API ──

    def fetch_historical(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch 5-minute OHLCV bars for a single ticker."""
        if start_date and end_date:
            start, end = start_date, end_date
        else:
            today = datetime.date.today()
            start = (today - datetime.timedelta(days=4)).isoformat()
            end = today.isoformat()

        db_start, db_end = self._make_date_range(start, end)

        logger.info(f"Databento: ohlcv-1m for {ticker} ({start} -> {end})")
        data = self._fetch_with_retry(DATABENTO_DATASET, ticker, "ohlcv-1m", db_start, db_end)

        if data is None:
            logger.warning(f"Databento ohlcv-1m returned None for {ticker}")
            return None

        df = data.to_df()
        if df is None or df.empty:
            logger.warning(f"Databento ohlcv-1m returned empty for {ticker}")
            return None

        df = self._normalize_columns(df)
        if df.empty:
            return None

        df = self._to_eastern(df)
        df = self._resample_5min_rth(df)

        # Only keep most recent day when no explicit date range requested
        if not start_date and not end_date and not df.empty:
            df["_date"] = df.index.date
            last_day = df["_date"].max()
            df = df[df["_date"] == last_day].drop(columns=["_date"])

        logger.info(f"Databento: got {len(df)} bars for {ticker}")
        self._last_error_message = None
        return df

    def fetch_daily(
        self,
        ticker: str,
        period: str = "2y",
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars for long-horizon backtesting."""
        today = datetime.date.today()
        period_map = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "20y": 7300, "max": 7300}
        days_back = period_map.get(period, 730)
        start = (today - datetime.timedelta(days=days_back)).isoformat()
        end = today.isoformat()

        db_start, db_end = self._make_date_range(start, end)

        logger.info(f"Databento: ohlcv-1d for {ticker} ({start} -> {end})")
        data = self._fetch_with_retry(DATABENTO_DATASET, ticker, "ohlcv-1d", db_start, db_end)

        if data is None:
            return None

        df = data.to_df()
        if df is None or df.empty:
            return None

        df = self._normalize_columns(df)
        if df.empty:
            return None

        df = self._to_eastern(df)

        logger.info(f"Databento: got {len(df)} daily bars for {ticker}")
        return df

    def get_bulk_chart_data(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch data for multiple tickers in batches.
        Returns a single DataFrame with a 'symbol' column, resampled to 5-min RTH.
        """
        if not tickers:
            return pd.DataFrame()

        db_start, db_end = self._make_date_range(start, end)

        batch_size = 50
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        logger.info(f"Databento bulk: fetching {len(tickers)} tickers in {len(batches)} batches")

        import concurrent.futures
        import databento as db

        def fetch_batch(batch):
            thread_client = db.Historical(self._api_key)
            return self._fetch_batch_with_retry(DATABENTO_DATASET, batch, "ohlcv-1m", db_start, db_end, client=thread_client)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batches), 10)) as executor:
            results = list(executor.map(fetch_batch, batches))

        all_dfs = []
        succeeded = 0
        failed = 0
        for df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                succeeded += 1
            else:
                failed += 1

        if failed > 0:
            logger.warning(
                f"Databento bulk: {failed}/{len(batches)} batches failed "
                f"({succeeded} succeeded, covering ~{succeeded * batch_size} of {len(tickers)} tickers)"
            )

        if not all_dfs:
            logger.error("Databento bulk: no data from any batch")
            return pd.DataFrame()

        big_df = pd.concat(all_dfs)
        if big_df.empty or "symbol" not in big_df.columns:
            return pd.DataFrame()

        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        big_df = big_df.rename(columns=rename_map)
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "symbol"] if c in big_df.columns]
        big_df = big_df[keep_cols].dropna(subset=["Open", "High", "Low", "Close"])

        if big_df.empty:
            return pd.DataFrame()

        big_df = self._to_eastern(big_df)

        processed_dfs = []
        for sym, group in big_df.groupby("symbol"):
            resampled = group.drop(columns=["symbol"]).resample("5min").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()

            resampled = resampled.between_time("09:30", "15:59")
            if resampled.empty:
                continue

            resampled["symbol"] = sym
            resampled["BarNumber"] = range(1, len(resampled) + 1)
            processed_dfs.append(resampled)

        if not processed_dfs:
            return pd.DataFrame()

        return pd.concat(processed_dfs)


# ─────────────────────────── FACTORY ─────────────────────────────────────────

def get_data_source(api_key: Optional[str] = None) -> DatabentoSource:
    """Create a Databento data source. Raises if no key."""
    databento_key = api_key or os.environ.get("DATABENTO_API_KEY", "")
    if not databento_key:
        raise RuntimeError("DATABENTO_API_KEY is not set")
    return DatabentoSource(databento_key)
