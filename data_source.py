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
    # Most S&P 500 are on both exchanges; use XNAS.ITCH as default
    # since it covers Nasdaq-listed + UTP-eligible NYSE stocks
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

class DabentoSource(DataSource):
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
        try:
            import databento as db

            dataset = get_dataset_for_ticker(ticker)

            # Default: today's trading day
            if start_date and end_date:
                start = start_date
                end = end_date
            else:
                today = datetime.date.today()
                start = today.isoformat()
                end = (today + datetime.timedelta(days=1)).isoformat()

            data = self._client.timeseries.get_range(
                dataset=dataset,
                symbols=[ticker],
                schema="ohlcv-5m",
                start=start,
                end=end,
            )

            df = data.to_df()
            if df is None or df.empty:
                logger.warning(f"Databento returned no data for {ticker}")
                return None

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
                return None

            return df

        except Exception as e:
            logger.error(f"Databento fetch failed for {ticker}: {e}")
            return None


# ─────────────────────────── YFINANCE SOURCE ─────────────────────────────────

class YFinanceSource(DataSource):
    """Fetch OHLCV data from Yahoo Finance (fallback)."""

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

            if start_date and end_date:
                df = yf.download(ticker, start=start_date, end=end_date,
                                 interval="5m", progress=False)
            else:
                df = yf.download(ticker, period="1d", interval="5m",
                                 progress=False)

            if df is None or df.empty:
                return None

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            return df

        except Exception as e:
            logger.error(f"yFinance fetch failed for {ticker}: {e}")
            return None


# ─────────────────────────── FALLBACK SOURCE ─────────────────────────────────

class FallbackSource(DataSource):
    """Try primary source, fall back to secondary on failure."""

    def __init__(self, primary: DataSource, fallback: DataSource):
        self._primary = primary
        self._fallback = fallback

    def name(self) -> str:
        return self._primary.name()

    def fetch_historical(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        df = self._primary.fetch_historical(ticker, start_date, end_date)
        if df is not None and not df.empty:
            return df

        logger.warning(
            f"{self._primary.name()} failed for {ticker}, "
            f"falling back to {self._fallback.name()}"
        )
        return self._fallback.fetch_historical(ticker, start_date, end_date)


# ─────────────────────────── FACTORY ─────────────────────────────────────────

def get_data_source(
    source_type: str = "databento",
    api_key: Optional[str] = None,
) -> DataSource:
    """
    Create a data source.

    Args:
        source_type: "databento", "yfinance", or "auto" (databento with yfinance fallback)
        api_key: Databento API key. If None, reads from DATABENTO_API_KEY env var.
    """
    if source_type == "yfinance":
        return YFinanceSource()

    databento_key = api_key or os.environ.get("DATABENTO_API_KEY", "")

    if source_type == "databento":
        if not databento_key:
            logger.warning("No DATABENTO_API_KEY found, using yFinance")
            return YFinanceSource()
        return DabentoSource(databento_key)

    # "auto" mode: Databento primary with yFinance fallback
    if databento_key:
        return FallbackSource(
            primary=DabentoSource(databento_key),
            fallback=YFinanceSource(),
        )
    return YFinanceSource()
