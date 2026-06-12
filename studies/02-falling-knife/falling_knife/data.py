"""Data access: Nasdaq-100 daily OHLC, cached locally.

We deliberately support BOTH faces of the same index:

    * ``^NDX``  — the spot index, history back to 1985. Long sample (1987 crash,
      the full 2000 dot-com bust), great for statistics, but NOT tradeable:
      there is no real spread/slippage on an index level.
    * ``QQQ``   — the Invesco ETF that tracks NDX, history from 1999. Shorter,
      but this is what you could actually *execute*, with real auction prints.

Crossing the two is the whole point: ``^NDX`` tells you whether the effect is
*real and stable*, ``QQQ`` tells you whether it is *capturable*.

Adjustment mode matters and is a *decision*, not a detail. Yahoo's convention,
pinned empirically in ``quantlab/data.py`` and its live test: ``auto_adjust=False``
OHLC arrives **already split-adjusted**; only ``Adj Close`` adds the dividends.

    * ``'split_only'`` (default) — adjust for splits only. Dividend drops stay in
      the price, so a dividend looks like a small overnight gap. Closest to what
      a price-taker sees on the tape. Given Yahoo's convention this is a NO-OP on
      the downloaded data — kept as a named mode so the choice stays explicit.
      (An earlier revision divided by a reconstructed split factor a *second*
      time, manufacturing a fake ~+100% overnight gap at QQQ's 2000-03-20 split —
      in the middle of the dot-com sample the crash events live in.)
    * ``'total_return'`` — fully adjusted (splits + dividends). Cleaner for
      long-horizon compounding, but it silently moves return between sessions.
    * ``'raw'`` — as-traded prices: the split adjustment is multiplied BACK OUT
      using Yahoo's split events. Maximises artefacts across split dates; for
      diagnostics only, never to trade on.

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

    Yahoo's ``auto_adjust=False`` OHLC is ALREADY split-adjusted (convention
    pinned in ``quantlab/data.py`` and its live test); ``Adj Close`` carries the
    dividend adjustment on top. Whatever the mode, the per-day factor is applied
    consistently across O/H/L/C so the intraday geometry (high>=close>=low etc.)
    is preserved.

      split_only   -> no-op: the prices are already what this mode means. (The
                      bug this replaces: rebuilding a cumulative split factor
                      from 'Stock Splits' and dividing AGAIN — a double
                      adjustment that grew a fabricated overnight gap of roughly
                      the split ratio at every split date.)
      total_return -> scale OHLC by Adj Close / Close (the dividend factor).
      raw          -> as-traded: multiply OHLC back up by the cumulative product
                      of all strictly-later split ratios (the split applies from
                      the split date's open, so that day is already in new units).
    """
    cols = {c: c for c in OHLC_COLS}
    base = raw[[cols[c] for c in OHLC_COLS]].copy()
    base.columns = OHLC_COLS

    if mode == "split_only":
        out = base
    elif mode == "total_return":
        if "Adj Close" not in raw.columns:
            raise RuntimeError("Adj Close missing; cannot build total_return mode.")
        factor = raw["Adj Close"] / raw["Close"]
        out = base.mul(factor, axis=0)
    elif mode == "raw":
        # Undo Yahoo's split adjustment to recover the as-traded tape.
        split = raw["Stock Splits"].replace(0.0, 1.0) if "Stock Splits" in raw else None
        if split is None or (split == 1.0).all():
            out = base  # never split: split-adjusted == as-traded
        else:
            # Cumulative factor of strictly-future splits: days BEFORE a k:1
            # split are multiplied back up by k.
            cum = split[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
            out = base.mul(cum, axis=0)
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
