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
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────── EXCHANGE MAPPING ────────────────────────────────

# Major NYSE-listed S&P 500 stocks (everything else defaults to Nasdaq)
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
    """Return the Databento dataset for a given ticker."""
    # The $199/month Equities plan uses the XNAS.ITCH dataset.
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
    """Fetch OHLCV data from Databento."""

    def __init__(self, api_key: str):
        import databento as db
        self._api_key = api_key
        self._client = db.Historical(api_key)

    def name(self) -> str:
        return "Databento"

    def fetch_historical(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        import streamlit as st

        dataset = get_dataset_for_ticker(ticker)

        # Default: trailing 3 days to account for weekends/holidays/delayed free tier
        if start_date and end_date:
            start = start_date
            end = end_date
        else:
            today = datetime.date.today()
            start = (today - datetime.timedelta(days=4)).isoformat()
            end = today.isoformat() # Queries up to exactly midnight UTC today (yesterday's data complete)

        # Try schemas in order of preference
        schemas_to_try = ["ohlcv-1m", "ohlcv-1s", "ohlcv-1d"]

        for schema in schemas_to_try:
            try:
                start_dt = datetime.datetime.fromisoformat(start)
                end_dt = datetime.datetime.fromisoformat(end)
                
                s_str = start_dt.strftime("%Y-%m-%d")
                e_str = end_dt.strftime("%Y-%m-%d")
                if s_str == e_str:
                    e_str = (end_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                
                logger.info(f"Databento: trying {dataset}/{schema} for {ticker} ({start} → {end})")
                
                try:
                    # Provide strict ISO-8601 strings with T separator that Databento wants
                    db_start = f"{s_str}T00:00:00"
                    db_end = f"{e_str}T00:00:00"
                    
                    data = self._client.timeseries.get_range(
                        dataset=dataset,
                        symbols=[ticker],
                        stype_in="raw_symbol",
                        schema=schema,
                        start=db_start,
                        end=db_end,
                    )
                except Exception as ex:
                    err_str = str(ex)
                    if "data_end_after_available_end" in err_str:
                        import re
                        match = re.search(r"available up to '([^']+)'", err_str)
                        if match:
                            # Replace space with T to create a perfect ISO string
                            new_end_str = match.group(1).replace(" ", "T")
                            logger.info(f"Databento: retrying with max available end {new_end_str}")
                            data = self._client.timeseries.get_range(
                                dataset=dataset,
                                symbols=[ticker],
                                stype_in="raw_symbol",
                                schema=schema,
                                start=db_start,
                                end=new_end_str,
                            )
                        else:
                            raise
                    else:
                        raise
                        
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
                    
                    # Keep only the most recent trading day to prevent chart distortion
                    if not df.empty:
                        # Ensure timezone is US/Eastern for RTH filtering
                        if df.index.tzinfo is None:
                            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
                        else:
                            df.index = df.index.tz_convert("US/Eastern")
                            
                        # Filter to Regular Trading Hours (RTH)
                        df = df.between_time("09:30", "15:59")
                        
                        # Extract the most recent single date
                        if not df.empty:
                            df["_date"] = df.index.date
                            last_day = df["_date"].max()
                            df = df[df["_date"] == last_day].drop(columns=["_date"])

                logger.info(f"Databento: got {len(df)} bars via {schema}")
                return df

            except Exception as e:
                logger.warning(f"Databento {schema} failed for {ticker}: {e}")
                continue

        # No fallback to yfinance; Databento is required for full range.
        logger.error(f"Databento: all schemas failed for {ticker}")
        return None

    def get_bulk_chart_data(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch data for multiple tickers concurrently in batches using Databento.
        Returns a single DataFrame with a 'symbol' column.
        """
        if not self._client:
            return pd.DataFrame()

        dataset = "XNAS.ITCH"
        schema = "ohlcv-1m"
        all_dfs = []

        # Convert simple ISO 'YYYY-MM-DD' to exact datetimes needed by XNAS
        s_dt = datetime.datetime.fromisoformat(start)
        e_dt = datetime.datetime.fromisoformat(end)
        db_start = s_dt.strftime("%Y-%m-%dT00:00:00")
        
        # Databento needs the end date to be exclusive/next day for full day coverage
        e_str = e_dt.strftime("%Y-%m-%d")
        if s_dt.strftime("%Y-%m-%d") == e_str:
            e_str = (e_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        db_end = f"{e_str}T00:00:00"

        # Split tickers into batches of 50 to avoid any API limits
        batch_size = 50
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

        import concurrent.futures
        def fetch_batch(batch):
            try:
                data = self._client.timeseries.get_range(
                    dataset=dataset,
                    symbols=batch,
                    stype_in="raw_symbol",
                    schema=schema,
                    start=db_start,
                    end=db_end,
                )
                df = data.to_df()
                return df
            except Exception as e:
                logger.error(f"Databento bulk error on batch: {e}")
                return pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_batch, batches))

        for df in results:
            if df is not None and not df.empty:
                all_dfs.append(df)

        if not all_dfs:
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
            # Resample
            resampled = group.resample("5min").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            
            # RTH filter
            resampled = resampled.between_time("09:30", "15:59")
            if resampled.empty:
                continue
                
            # Keep only the newest day's data
            resampled["_date"] = resampled.index.date
            last_day = resampled["_date"].max()
            resampled = resampled[resampled["_date"] == last_day].drop(columns=["_date"])
            
            if resampled.empty:
                continue
                
            # Format output schema
            resampled["symbol"] = sym
            resampled["BarNumber"] = range(1, len(resampled) + 1)
            processed_dfs.append(resampled)

        if not processed_dfs:
            return pd.DataFrame()

        return pd.concat(processed_dfs)

# ─────────────────────────── FACTORY ─────────────────────────────────────────

def get_data_source(
    source_type: str = "databento",
    api_key: Optional[str] = None,
) -> DataSource:
    """
    Create a data source.
    Forces Databento to avoid 60-day limits from yfinance.
    """
    databento_key = api_key or os.environ.get("DATABENTO_API_KEY", "")
    
    if not databento_key:
        logger.warning("No Databento API key provided. Add to .streamlit/secrets.toml!")
        
    return DatabentoSource(databento_key)
