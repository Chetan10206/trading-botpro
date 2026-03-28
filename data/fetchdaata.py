"""
fetch_data.py
─────────────────────────────────────────────────────────────────────────────
Fetches 1-minute OHLCV data for backtesting the NY Open FVG Strategy.

Supported sources
  • yfinance  – free, delayed (~15 min), great for equities / ETFs / crypto
  • Alpaca    – free paper-trading API, real-time + historical, requires keys

Alpaca setup
  1. Copy .env.example → .env
  2. Paste your keys from https://app.alpaca.markets → API Keys
  3. Run with --source alpaca

Usage
  from data.fetch_data import DataFetcher
  df = DataFetcher().get("SPY", "2024-01-01", "2024-06-30")
  df = DataFetcher(source="alpaca").get("SPY", "2024-01-01", "2024-06-30")
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# ── Load .env automatically (works even if python-dotenv is not installed) ──
try:
    from dotenv import load_dotenv
    # Walk up from this file's directory to find .env in the project root
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)  # don't overwrite real env vars
        logging.getLogger(__name__).debug(".env loaded from %s", _env_path)
    else:
        logging.getLogger(__name__).debug(
            "No .env file found at %s — copy .env.example to .env to add Alpaca keys", _env_path
        )
except ImportError:
    pass  # python-dotenv not installed; keys must be set in the shell environment

logger = logging.getLogger(__name__)

# ── Timezone constants ──────────────────────────────────────────────────────
NY_TZ  = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# ── Local cache directory ────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """
    Unified 1-minute OHLCV provider.

    Parameters
    ----------
    source : str
        "yfinance" (default) or "alpaca"
    use_cache : bool
        Cache fetched data to disk (Parquet) to avoid repeated downloads.
    """

    def __init__(self, source: str = "yfinance", use_cache: bool = True):
        self.source    = source.lower()
        self.use_cache = use_cache

        if self.source == "alpaca":
            self._init_alpaca()

    # ── Public API ─────────────────────────────────────────────────────────

    def get(
        self,
        symbol: str,
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return a clean 1-minute OHLCV DataFrame indexed by UTC timestamps.

        Columns: open, high, low, close, volume
        Index  : DatetimeIndex (UTC, timezone-aware)
        """
        cache_file = CACHE_DIR / f"{symbol}_{start}_{end}_1min.parquet"

        if self.use_cache and cache_file.exists() and not force_refresh:
            logger.info("Loading %s from cache …", symbol)
            df = pd.read_parquet(cache_file)
            return self._post_process(df, symbol)

        logger.info("Fetching %s (%s → %s) via %s …", symbol, start, end, self.source)

        if self.source == "alpaca":
            df = self._fetch_alpaca(symbol, start, end)
        else:
            df = self._fetch_yfinance(symbol, start, end)

        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol} ({start} → {end})")

        df = self._post_process(df, symbol)

        if self.use_cache:
            df.to_parquet(cache_file)
            logger.info("Cached %d rows → %s", len(df), cache_file)

        return df

    # ── yfinance backend ──────────────────────────────────────────────────

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        yfinance returns UTC for most symbols.  We standardise to UTC here.

        NOTE: yfinance limits 1-min data to the last 30 days per request.
        For longer date ranges, we chunk into 7-day windows.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
        delta    = (end_dt - start_dt).days

        if delta <= 7:
            return self._yf_single(symbol, start, end)

        # Chunk into 7-day windows to stay within yfinance limits
        chunks   = []
        cursor   = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=7), end_dt)
            logger.info("  chunk %s → %s", cursor.date(), chunk_end.date())
            chunk = self._yf_single(
                symbol,
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
            cursor = chunk_end
            time.sleep(0.5)   # be polite to the API

        if not chunks:
            return pd.DataFrame()

        return pd.concat(chunks).drop_duplicates()

    def _yf_single(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval="1m", start=start, end=end, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.tz_convert(UTC_TZ)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]

    # ── Alpaca backend ─────────────────────────────────────────────────────

    def _init_alpaca(self):
        # ── Check alpaca-py is installed ──────────────────────────────────
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests  import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            self._AlpacaClient  = StockHistoricalDataClient
            self._AlpacaRequest = StockBarsRequest
            self._AlpacaTF      = TimeFrame
        except ImportError:
            raise ImportError(
                "alpaca-py is not installed.\n"
                "  Fix: pip install alpaca-py\n"
                "  Then add your keys to .env (copy from .env.example)"
            )

        # ── Read keys (dotenv was already loaded at module import time) ───
        key    = os.getenv("ALPACA_API_KEY",    "").strip()
        secret = os.getenv("ALPACA_SECRET_KEY", "").strip()

        # Friendly error with actionable steps
        if not key or not secret:
            env_path = Path(__file__).parent.parent / ".env"
            raise EnvironmentError(
                "\n"
                "  ✗  Alpaca API keys not found.\n\n"
                "  Steps to fix:\n"
                "    1. Go to https://app.alpaca.markets → API Keys → Generate\n"
                "    2. Copy .env.example → .env  (project root)\n"
                f"       ({env_path})\n"
                "    3. Fill in ALPACA_API_KEY and ALPACA_SECRET_KEY\n"
                "    4. Re-run your command\n\n"
                "  Alternatively, export them in your shell:\n"
                "    export ALPACA_API_KEY=PKxxxxxxxx\n"
                "    export ALPACA_SECRET_KEY=xxxxxxxx\n"
            )

        # Validate key format (Alpaca paper keys start with PK, live with AK)
        if not (key.startswith("PK") or key.startswith("AK")):
            logger.warning(
                "ALPACA_API_KEY doesn't look right (expected to start with PK or AK). "
                "Double-check your .env file."
            )

        # Detect paper vs live from key prefix
        is_paper = key.startswith("PK")
        if is_paper:
            logger.info("Alpaca: using PAPER trading keys")
        else:
            logger.info("Alpaca: using LIVE trading keys")

        self._alpaca = self._AlpacaClient(key, secret)

    def _fetch_alpaca(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        bars = self._alpaca.get_stock_bars(req).df

        if bars.empty:
            return pd.DataFrame()

        # Alpaca returns MultiIndex (symbol, timestamp) — flatten
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")

        bars.index  = pd.DatetimeIndex(bars.index).tz_convert(UTC_TZ)
        bars.columns = [c.lower() for c in bars.columns]
        return bars[["open", "high", "low", "close", "volume"]]

    # ── Post-processing (shared) ───────────────────────────────────────────

    def _post_process(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy()

        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC_TZ)
        else:
            df.index = df.index.tz_convert(UTC_TZ)

        # Drop incomplete / bad candles
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[df["volume"] > 0]

        # Sanity checks
        df = df[df["high"] >= df["low"]]
        df = df[df["close"] > 0]

        # Sort chronologically
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Add NY-time helper columns (useful in strategy logic)
        df["ny_time"]  = df.index.tz_convert(NY_TZ)
        df["ny_hour"]  = df["ny_time"].dt.hour
        df["ny_minute"] = df["ny_time"].dt.minute
        df["ny_hhmm"]  = df["ny_hour"] * 100 + df["ny_minute"]
        df["date"]     = df["ny_time"].dt.date

        logger.info(
            "%s: %d rows  |  %s → %s",
            symbol,
            len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )
        return df


# ── Convenience wrapper ────────────────────────────────────────────────────

def get_data(
    symbol: str,
    start: str,
    end: str,
    source: str = "yfinance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Module-level shortcut.

    Example
    -------
    >>> from data.fetch_data import get_data
    >>> df = get_data("SPY", "2024-01-01", "2024-03-31")
    """
    return DataFetcher(source=source, use_cache=use_cache).get(symbol, start, end)


# ── CLI quick-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    symbol = "SPY"
    end_dt = datetime.now(NY_TZ)
    # yfinance 1-min limit: last 30 days
    start_dt = end_dt - timedelta(days=7)

    df = get_data(
        symbol,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )

    print(f"\n{'─'*60}")
    print(f"  {symbol}  |  {len(df):,} rows fetched")
    print(f"{'─'*60}")
    print(df.tail(10).to_string())
    print(f"\nColumns : {list(df.columns)}")
    print(f"NY range: {df['ny_hhmm'].min()}  →  {df['ny_hhmm'].max()}")
