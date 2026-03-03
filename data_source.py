"""
data_source.py — Data Source Abstraction Layer

Supports Databento (primary) and yFinance (fallback) for fetching
5-minute OHLCV bars for US equities.

Usage:
    from data_source import get_data_source
    source = get_data_source("databento", api_key="db-xxx")
    df = source.fetch_historical("AAPL")
"""

import os
import logging
import datetime
import time
import socket
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────── RETRY UTILITIES ─────────────────────────────────

def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient error worth retrying with backoff."""
    err_str = str(exc).lower()
    # HTTP 502 Bad Gateway, 503 Service Unavailable
    if "502" in err_str or "503" in err_str or "bad gateway" in err_str or "service unavailable" in err_str:
        return True
    # HTTP 429 Rate Limit
    if "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str or "too many requests" in err_str:
        return True
    # Timeouts
    if "timeout" in err_str or "timed out" in err_str:
        return True
    # Connection errors
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

# Dataset: XNAS.ITCH (Nasdaq TotalView-ITCH) covers ALL NMS securities,
# including NYSE-listed stocks that trade on Nasdaq. OHLCV price data is
# accurate for all symbols. Volume reflects only Nasdaq's share (~16-20%
# of consolidated volume for NYSE primary-listed stocks) — acceptable for
# price-action analysis but not for volume-based strategies.
#
# If consolidated volume is needed, migrate to DBEQ.BASIC / DBEQ.MAX
# or supplement with XNYS.PILLAR for NYSE-primary stocks.

# NYSE-primary S&P 500 stocks (retained for potential future dataset routing)
NYSE_TICKERS = {
    "A", "AAL", "AAP", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP",
    "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "ALB", "ALK", "ALL",
    "ALLE", "AME", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA",
    "APD", "APH", "ARE", "ATO", "AVB", "AVY", "AWK", "AXP", "BA", "BAC",
    "BAX", "BBY", "BDX", "BEN", "BF.B", "BG", "BIIB", "BK", "BKNG", "BKR",
    "BLK", "BMY", "BR", "BRK.B", "BRO", "BSX", "BWA", "BXP", "C", "CAG",
    "CAH", "CARR", "CAT", "CB", "CBRE", "CCI", "CCL", "CDAY", "CF", "CHD",
    "CI", "CINF", "CL", "CLX", "CMA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COO", "COP", "COR", "COST", "CPB", "CPRT", "CRL", "CRM",
    "CSCO", "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS",
    "CVX", "CZR", "D", "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI",
    "DHR", "DIS", "DISH", "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE",
    "DUK", "DVA", "DVN", "DXC", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX",
    "EIX", "EL", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT",
    "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS",
    "FISV", "FITB", "FLT", "FMC", "FOX", "FOXA", "FRC", "FRT", "FTNT",
    "GD", "GE", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL",
    "GPC", "GPN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HOLX",
    "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUM", "HWM", "IBM",
    "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH",
    "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J",
    "JBHT", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS",
    "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "L", "LDOS",
    "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW",
    "LRCX", "LUMN", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR",
    "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META",
    "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH",
    "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT",
    "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE",
    "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP",
    "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS", "NWSA", "O", "ODFL",
    "OGN", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PARA",
    "PAYC", "PAYX", "PCAR", "PCG", "PEAK", "PEG", "PEP", "PFE", "PFG",
    "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD", "PM", "PNC", "PNR",
    "PNW", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PVH", "PWR",
    "PXD", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", "REGN", "RF",
    "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX",
    "SBAC", "SBNY", "SBUX", "SCHW", "SEE", "SHW", "SIVB", "SJM", "SLB",
    "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STT", "STX", "STZ",
    "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH",
    "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
    "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS",
    "URI", "USB", "V", "VFC", "VICI", "VLO", "VMC", "VNO", "VRSK", "VRSN",
    "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC",
    "WELL", "WFC", "WHR", "WM", "WMB", "WMT", "WRB", "WRK", "WST", "WTW",
    "WY", "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA",
    "ZION", "ZTS",
}


def get_dataset_for_ticker(ticker: str) -> str:
    """Return the Databento dataset for a given ticker.

    Currently returns XNAS.ITCH for all tickers. NYSE-listed stocks trade
    on Nasdaq under NMS rules, so OHLCV price data is available (volume
    reflects only Nasdaq's portion of trading).
    """
    return "XNAS.ITCH"


# ─────────────────────────── BASE CLASS ──────────────────────────────────────

class DataSource(ABC):
    """Abstract base class for OHLCV data sources."""

    @abstractmethod
    def fetch_historical(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 5-minute OHLCV bars.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        or None on failure.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this data source."""
        ...


# ─────────────────────────── DATABENTO SOURCE ────────────────────────────────

class DatabentoSource(DataSource):
    """Fetch OHLCV data from Databento with retry logic for transient errors."""

    def __init__(self, api_key: str):
        import databento as db
        self._api_key = api_key
        self._client = db.Historical(api_key)
        # Retry configuration
        self._max_retries = 3
        self._base_wait = 1  # seconds; exponential backoff: 1s, 2s, 4s
        self._last_error_message = None

    def name(self) -> str:
        return "Databento"

    # ── Retry helpers ──

    def _fetch_with_retry(self, dataset: str, ticker: str, schema: str,
                          db_start: str, db_end: str):
        """
        Single timeseries.get_range() call with retry + exponential backoff.
        Retries on transient HTTP errors (502, 503, 429, timeouts).
        Preserves the data_end_after_available_end correction logic.
        Returns the raw data object on success, None on failure.
        """
        import re

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

                # Special case: data_end_after_available_end — correct end date and retry once
                if "data_end_after_available_end" in err_str:
                    match = re.search(r"available up to '([^']+)'", err_str)
                    if match:
                        new_end_str = match.group(1).replace(" ", "T")
                        logger.info(f"Databento: data_end_after_available_end, retrying with {new_end_str}")
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

                # Transient error — retry with backoff
                if _is_transient_error(ex):
                    if attempt < self._max_retries - 1:
                        wait = self._base_wait * (2 ** attempt)
                        if _is_rate_limit(ex):
                            wait = min(wait * 2, 16)  # Longer backoff for rate limits
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
                    # Non-transient error — fail fast, don't waste retries
                    logger.warning(f"Databento {schema} non-transient error: {err_str[:200]}")
                    self._last_error_message = f"{schema}: {err_str[:200]}"
                    return None

        return None

    def _fetch_batch_with_retry(self, dataset: str, batch: list, schema: str,
                                db_start: str, db_end: str, client=None):
        """
        Fetch a batch of tickers with retry + exponential backoff.
        Accepts an optional client for thread-safe bulk operations.
        Returns DataFrame on success, None on failure.
        """
        import re
        _client = client or self._client

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

                # Handle data_end_after_available_end: correct end date and retry once
                if "data_end_after_available_end" in err_str:
                    match = re.search(r"available up to '([^']+)'", err_str)
                    if match:
                        new_end_str = match.group(1).replace(" ", "T")
                        logger.info(f"Databento bulk batch: data_end_after_available_end, retrying with {new_end_str}")
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
                            logger.warning(f"Databento bulk batch retry with corrected end also failed: {retry_ex}")
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
                            f"Databento bulk batch transient error (attempt {attempt + 1}/{self._max_retries}), "
                            f"retrying in {wait}s: {err_str[:120]}"
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.error(f"Databento bulk batch failed after {self._max_retries} attempts: {err_str[:200]}")
                        self._last_error_message = f"bulk_batch: {err_str[:200]}"
                        return None
                else:
                    logger.warning(f"Databento bulk batch non-transient error: {err_str[:200]}")
                    self._last_error_message = f"bulk_batch: {err_str[:200]}"
                    return None

        return None

    def fetch_historical(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:

        dataset = get_dataset_for_ticker(ticker)

        # Default: trailing 3 days to account for weekends/holidays/delayed free tier
        if start_date and end_date:
            start = start_date
            end = end_date
        else:
            today = datetime.date.today()
            start = (today - datetime.timedelta(days=4)).isoformat()
            end = today.isoformat()

        # Compute Databento-format date strings once
        start_dt = datetime.datetime.fromisoformat(start)
        end_dt = datetime.datetime.fromisoformat(end)
        s_str = start_dt.strftime("%Y-%m-%d")
        # Databento's end date is strictly exclusive — add 1 day.
        # Do NOT clamp to today: the data_end_after_available_end handler
        # in _fetch_with_retry will correct overshoot automatically.
        target_e = end_dt.date() + datetime.timedelta(days=1)
        e_str = target_e.strftime("%Y-%m-%d")
        db_start = f"{s_str}T00:00:00"
        db_end = f"{e_str}T00:00:00"

        # Try schemas in order of preference.
        # Each schema is retried internally (with backoff) before falling through.
        schemas_to_try = ["ohlcv-1m", "ohlcv-1s", "ohlcv-1d"]

        for schema in schemas_to_try:
            try:
                logger.info(f"Databento: trying {dataset}/{schema} for {ticker} ({start} → {end})")

                # Retry-enabled fetch
                data = self._fetch_with_retry(dataset, ticker, schema, db_start, db_end)

                if data is None:
                    logger.warning(f"Databento {schema} returned None for {ticker}, trying next schema")
                    continue

                df = data.to_df()

                if df is None or df.empty:
                    logger.warning(f"Databento {schema} returned empty for {ticker}")
                    continue

                # Normalize columns: lowercase → uppercase
                rename_map = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
                df = df.rename(columns=rename_map)

                # Keep only the columns we need
                keep_cols = ["Open", "High", "Low", "Close", "Volume"]
                available = [c for c in keep_cols if c in df.columns]
                df = df[available]

                # Drop any rows with NaN prices
                df = df.dropna(subset=["Open", "High", "Low", "Close"])

                if df.empty:
                    continue

                # Resample to 5-minute bars if not already daily
                if schema != "ohlcv-1d":
                    df = df.resample("5min").agg({
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }).dropna()

                    if not df.empty:
                        # Ensure timezone is US/Eastern for RTH filtering
                        if df.index.tzinfo is None:
                            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
                        else:
                            df.index = df.index.tz_convert("US/Eastern")

                        # Filter to Regular Trading Hours (RTH)
                        df = df.between_time("09:30", "15:59")

                        # Only keep the most recent day when no date range was explicitly requested
                        if not start_date and not df.empty:
                            df["_date"] = df.index.date
                            last_day = df["_date"].max()
                            df = df[df["_date"] == last_day].drop(columns=["_date"])

                logger.info(f"Databento: got {len(df)} bars via {schema}")
                self._last_error_message = None  # Clear on success
                return df

            except Exception as e:
                logger.warning(f"Databento {schema} failed for {ticker}: {e}")
                continue

        logger.error(f"Databento: all schemas failed for {ticker}")
        return None

    def get_bulk_chart_data(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch data for multiple tickers concurrently in batches using Databento.
        Failed batches are retried with exponential backoff before being skipped.
        Returns a single DataFrame with a 'symbol' column.
        """
        if not self._client or not tickers:
            return pd.DataFrame()

        dataset = get_dataset_for_ticker(tickers[0])
        schema = "ohlcv-1m"
        all_dfs = []

        # Convert simple ISO 'YYYY-MM-DD' to exact datetimes needed by XNAS
        s_dt = datetime.datetime.fromisoformat(start)
        e_dt = datetime.datetime.fromisoformat(end)
        db_start = s_dt.strftime("%Y-%m-%dT00:00:00")

        # Databento's end date is exclusive — add 1 day for full coverage.
        # Do NOT clamp to today: the data_end_after_available_end handler
        # in _fetch_batch_with_retry will correct overshoot automatically.
        target_e = e_dt.date() + datetime.timedelta(days=1)
        db_end = f"{target_e.strftime('%Y-%m-%d')}T00:00:00"

        # Split tickers into batches of 50 to avoid any API limits
        batch_size = 50
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

        logger.info(f"Databento bulk: fetching {len(tickers)} tickers in {len(batches)} batches")

        import concurrent.futures
        import databento as db

        def fetch_batch(batch):
            # Per-thread client to avoid sharing HTTP sessions across threads
            thread_client = db.Historical(self._api_key)
            return self._fetch_batch_with_retry(dataset, batch, schema, db_start, db_end, client=thread_client)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_batch, batches))

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

        # Normalize Databento columns
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        big_df = big_df.rename(columns=rename_map)

        keep_cols = ["Open", "High", "Low", "Close", "Volume", "symbol"]
        available = [c for c in keep_cols if c in big_df.columns]
        big_df = big_df[available].dropna(subset=["Open", "High", "Low", "Close"])

        if big_df.empty:
            return pd.DataFrame()

        # Databento returns index as ts_event (UTC)
        if big_df.index.tzinfo is None:
            big_df.index = big_df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            big_df.index = big_df.index.tz_convert("US/Eastern")

        # Process each symbol: resample to 5min, filter RTH
        processed_dfs = []
        for sym, group in big_df.groupby("symbol"):
            resampled = group.resample("5min").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()

            # RTH filter
            resampled = resampled.between_time("09:30", "15:59")
            if resampled.empty:
                continue

            # Keep all available days for algorithm seeding
            resampled["symbol"] = sym
            resampled["BarNumber"] = range(1, len(resampled) + 1)
            processed_dfs.append(resampled)

        if not processed_dfs:
            return pd.DataFrame()

        return pd.concat(processed_dfs)

# ─────────────────────────── YFINANCE SOURCE ─────────────────────────────────

class YFinanceSource(DataSource):
    """Fallback data source using yFinance (free, no API key needed, but limited to ~60 days of 5-min data)."""

    def name(self) -> str:
        return "yFinance"

    def fetch_historical(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — cannot use YFinanceSource")
            return None

        try:
            # yfinance 5m data is limited to last 60 days
            if start_date and end_date:
                df = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
            else:
                df = yf.download(ticker, period="1d", interval="5m", progress=False)

            if df is None or df.empty:
                return None

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalize column names to capitalized (yfinance 0.2.31+ returns lowercase)
            col_map = {c.lower(): c.capitalize() for c in df.columns if isinstance(c, str)}
            # Handle 'adj close' -> 'Adj Close' specially
            if "adj close" in col_map:
                col_map["adj close"] = "Adj Close"
            df.columns = [col_map.get(c.lower(), c) if isinstance(c, str) else c for c in df.columns]

            # Ensure required columns exist
            required = ["Open", "High", "Low", "Close"]
            if not all(c in df.columns for c in required):
                logger.warning(f"yFinance: missing columns for {ticker}: got {list(df.columns)}")
                return None

            df = df.dropna(subset=required)

            if df.empty:
                return None

            # Filter to RTH if timezone info is available
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert("US/Eastern")
                df = df.between_time("09:30", "15:59")

            # Keep only the most recent trading day for single-day view
            if not start_date and not df.empty:
                df["_date"] = df.index.date
                last_day = df["_date"].max()
                df = df[df["_date"] == last_day].drop(columns=["_date"])

            if df.empty:
                return None

            logger.info(f"yFinance: got {len(df)} bars for {ticker}")
            return df

        except Exception as e:
            logger.warning(f"yFinance failed for {ticker}: {e}")
            return None


# ─────────────────────────── FACTORY ─────────────────────────────────────────

def get_data_source(
    source_type: str = "databento",
    api_key: Optional[str] = None,
) -> DataSource:
    """
    Create a data source.
    Tries Databento first (better data, longer history). Falls back to yFinance if no key.
    """
    databento_key = api_key or os.environ.get("DATABENTO_API_KEY", "")

    if databento_key:
        try:
            return DatabentoSource(databento_key)
        except Exception as e:
            logger.warning(f"Databento init failed ({e}), falling back to yFinance")

    logger.info("Using yFinance as data source (no Databento key or Databento failed)")
    return YFinanceSource()
