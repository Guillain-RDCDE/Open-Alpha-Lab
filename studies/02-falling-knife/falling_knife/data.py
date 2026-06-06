"""Data access: Nasdaq-100 daily OHLC, cached locally.

We deliberately support BOTH faces of the same index:

    * ``^NDX``  — the spot index, history back to 1985. Long sample (1987 crash,
      the full 2000 dot-com bust), great for statistics, but NOT tradeable:
      there is no real spread/slippage on an index level.
    * ``QQQ``   — the Invesco ETF that tracks NDX, history from 1999. Shorter,
      but this is what you could actually *execute*, with real auction prints.

Crossing the two is the whole point: ``^NDX`` tells you whether the effect is
*real and stable*, ``QQQ`` tells you whether it is *capturable*.

Adjustment mode matters and is a *decision*, not a detail:
    * ``'split_only'`` (default) — adjust for splits only. Dividend drops stay in
      the price, so a dividend looks like a small overnight gap. Closest to what
      a price-taker sees on the tape.
    * ``'total_return'`` — Yahoo's fully adjusted close (splits + dividends).
      Cleaner for long-horizon compounding, but it silently moves return between
      sessions.
    * ``'raw'`` — no adjustment at all (splits visible). For diagnostics only.

All downloads are cached as parquet under ``_cache/`` so reruns are offline.
"""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd

AdjustMode = Literal["split_only", "total_return", "raw"]

# The indices we cross-check. Each has a "spot" face (deep history, great stats,
# not tradeable) and an ETF face (shorter, but real execution).
UNDERLYINGS = {
    "ndx": "^NDX",    # Nasdaq-100 spot, deep history (since 1985)
    "qqq": "QQQ",     # Nasdaq-100 ETF (since 1999)
    "spx": "^GSPC",   # S&P 500 spot, very deep history (since 1927 on Yahoo)
    "spy": "SPY",     # S&P 500 ETF (since 1993)
}

# Convenience pairings (spot for stats, ETF for execution) used by the
# cross-index example.
INDEX_PAIRS = {
    "Nasdaq-100": ("^NDX", "QQQ"),
    "S&P 500": ("^GSPC", "SPY"),
}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_cache")

OHLC_COLS = ["Open", "High", "Low", "Close"]


def _cache_path(ticker: str, mode: AdjustMode) -> str:
    safe = ticker.replace("^", "_").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}__{mode}.parquet")


def fetch(
    ticker: str,
    *,
    mode: AdjustMode = "split_only",
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a daily OHLC ``DataFrame`` indexed by date.

    Columns: ``Open, High, Low, Close`` (always), plus ``Volume`` when the
    provider supplies it. Adjustment follows ``mode`` (see module docstring).

    ``ticker`` may be a Yahoo symbol (``'^NDX'``, ``'QQQ'``) or one of the short
    keys in :data:`UNDERLYINGS` (``'ndx'``, ``'qqq'``).
    """
    ticker = UNDERLYINGS.get(ticker.lower(), ticker)
    path = _cache_path(ticker, mode)

    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
        return _slice(df, start, end)

    df = _download(ticker, mode)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(path)
    return _slice(df, start, end)


def _download(ticker: str, mode: AdjustMode) -> pd.DataFrame:
    """Pull from Yahoo! Finance and normalise to clean OHLC columns."""
    import yfinance as yf  # deferred import: keep the package importable offline

    # auto_adjust=True gives split+dividend adjusted OHLC. We then reconstruct
    # the requested mode from the two raw faces Yahoo exposes (Close vs Adj).
    raw = yf.download(
        ticker,
        period="max",          # full history; ^NDX starts 1985-10, QQQ 1999-03.
        auto_adjust=False,
        actions=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker!r}. Check the symbol and your "
            f"network connection (Yahoo! Finance)."
        )

    # yfinance can return a MultiIndex (column, ticker) for single symbols.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    out = _apply_mode(raw, mode)
    out.index.name = "Date"
    out = out.dropna(subset=["Open", "Close"]).sort_index()
    return out


def _apply_mode(raw: pd.DataFrame, mode: AdjustMode) -> pd.DataFrame:
    """Build OHLC under the requested adjustment convention.

    Yahoo gives us raw OHLC plus an ``Adj Close`` (splits + dividends). We derive
    a per-day adjustment factor and apply it consistently across O/H/L/C so the
    intraday geometry (high>=close>=low etc.) is preserved.
    """
    cols = {c: c for c in OHLC_COLS}
    base = raw[[cols[c] for c in OHLC_COLS]].copy()
    base.columns = OHLC_COLS

    if mode == "raw":
        out = base
    elif mode == "total_return":
        if "Adj Close" not in raw.columns:
            raise RuntimeError("Adj Close missing; cannot build total_return mode.")
        factor = raw["Adj Close"] / raw["Close"]
        out = base.mul(factor, axis=0)
    elif mode == "split_only":
        # Splits move every price by the same ratio; dividends do not. We back
        # out the split-only factor from Yahoo's split history when present,
        # otherwise fall back to raw (no splits in the window).
        split = raw["Stock Splits"].replace(0.0, 1.0) if "Stock Splits" in raw else None
        if split is None or (split == 1.0).all():
            out = base
        else:
            # Cumulative split factor, applied retroactively (older bars shrink).
            cum = split.replace(1.0, 1.0)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
            out = base.div(cum, axis=0)
    else:  # pragma: no cover - guarded by Literal typing
        raise ValueError(f"Unknown adjustment mode: {mode!r}")

    if "Volume" in raw.columns:
        out["Volume"] = raw["Volume"]
    return out


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df.copy()


def daily_returns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Attach the return series the rest of the package consumes.

    Adds:
        * ``r_cc``       close-to-close return (the headline daily return)
        * ``r_oc``       open-to-close (intraday) return
        * ``r_co``       previous-close-to-open (overnight gap) return
        * ``r_intraday_low``  open-to-low return (worst intraday drawdown of the day)
    """
    out = ohlc.copy()
    prev_close = out["Close"].shift(1)
    out["r_cc"] = out["Close"] / prev_close - 1.0
    out["r_oc"] = out["Close"] / out["Open"] - 1.0
    out["r_co"] = out["Open"] / prev_close - 1.0
    out["r_intraday_low"] = out["Low"] / out["Open"] - 1.0
    return out
