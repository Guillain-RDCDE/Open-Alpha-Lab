"""Data layer — Yahoo! Finance with a local parquet cache.

The adjustment mode is a *decision, not a detail*: dividends and splits move
return between the night and day legs (a stock trades ex-dividend at the open),
so the mode you pick changes the very thing Knuteson measures. Document your
choice in any figure you publish.

Modes
-----
'split_only'   (default) prices adjusted for splits but NOT dividends. Keeps
               the genuine ex-dividend gap in the overnight leg, which is what
               a price-taker actually experiences. Closest to "real life".
'total_return' fully adjusted (splits + dividends). Removes the ex-div gap;
               cleaner academic series but hides a real overnight cost/benefit.
'raw'          unadjusted as reported. Maximises artefacts — useful to SHOW the
               artefact problem, never to trade on.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.environ.get("OVERNIGHT_CACHE", "_cache"))

# A small, liquid, geographically spread set of index ETFs / indices.
# ETFs (SPY/QQQ/EWJ...) have cleaner open auctions than spot indices for a
# would-be trader; spot ^-tickers are there for the "academic" reproduction.
WORLD_INDICES = {
    "SPY": "USA (S&P 500 ETF)",
    "QQQ": "USA (Nasdaq-100 ETF)",
    "EWU": "UK (MSCI UK ETF)",
    "EWG": "Germany (MSCI Germany ETF)",
    "EWQ": "France (MSCI France ETF)",
    "EWJ": "Japan (MSCI Japan ETF)",
    "EWH": "Hong Kong (MSCI HK ETF)",
    "EWZ": "Brazil (MSCI Brazil ETF)",
    "INDA": "India (MSCI India ETF)",
    "FXI": "China (large-cap ETF)",
}

_MODES = ("split_only", "total_return", "raw")


def fetch(
    ticker: str,
    start: str = "1993-01-01",
    end: str | None = None,
    mode: str = "split_only",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLC for ``ticker`` and return a clean DataFrame.

    Caches to ``_cache/<ticker>_<mode>.parquet``. Requires network on a cache
    miss (yfinance). Columns returned: Open, High, Low, Close (and Volume).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_{mode}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = _download(ticker, start, end, mode)
    if use_cache:
        df.to_parquet(cache_path)
    return df


def _download(ticker: str, start: str, end: str | None, mode: str) -> pd.DataFrame:
    """Pull from yfinance and apply the chosen adjustment mode."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "yfinance is required for live data; `pip install yfinance` "
            "or use overnight.diagnostics.synthetic_ohlc for offline work."
        ) from exc

    # auto_adjust=False gives us raw OHLC + 'Adj Close' so we can choose the mode.
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker!r} (check ticker / network).")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    out = _apply_mode(raw, mode)
    out.index.name = "Date"
    return out[["Open", "High", "Low", "Close", "Volume"]]


def _apply_mode(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Translate raw yfinance OHLC + 'Adj Close' into the requested series."""
    df = raw.copy()
    if mode == "raw":
        return df

    if mode == "total_return":
        # Scale every OHLC by the same Adj-Close/Close factor -> fully adjusted.
        if "Adj Close" not in df.columns:
            return df
        factor = df["Adj Close"] / df["Close"]
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * factor
        return df

    # split_only: adjust for splits but NOT dividends. Reconstruct the split
    # factor from yfinance's 'Stock Splits' column (0 where no split).
    splits = df.get("Stock Splits")
    if splits is None or (splits == 0).all():
        return df  # nothing to do; raw == split-adjusted
    ratio = splits.replace(0, 1.0)
    # Cumulative split factor applied backwards (older prices divided down).
    cum = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] / cum
    df["Volume"] = df["Volume"] * cum
    return df
