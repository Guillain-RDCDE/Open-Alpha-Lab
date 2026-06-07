"""Data access: the S&P 500 (the thing you buy) and the VIX (the thing you watch).

This study has two data objects, not one, and keeping them straight is half the
discipline:

    * the **market** — what the rule actually buys. Two faces, exactly as in
      Study 02: ``^GSPC`` (spot, deep history since 1927 on Yahoo, great stats,
      NOT tradeable) and ``SPY`` (the ETF, since 1993, real prints, what you'd
      execute).
    * the **gauge** — ``^VIX``, the signal. A *level*, not a price: no splits, no
      dividends, nothing to adjust. You cannot buy spot VIX, so it never appears
      on the buy side — only as the trigger.

The two trade on the same US session calendar, so a VIX signal lines up 1:1 with
the next S&P bar. :func:`aligned` intersects them and is the canonical way to get
a (market, gauge) pair the rest of the package consumes.

Adjustment mode (for the *market* only) is a decision, not a detail — same three
modes and same rationale as Study 02. The VIX is always taken raw.

All downloads cache as parquet under ``_cache/`` so reruns are offline.
"""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd

AdjustMode = Literal["split_only", "total_return", "raw"]

# The market we buy (spot for stats, ETF for execution) and the gauge we watch.
MARKETS = {
    "spx": "^GSPC",   # S&P 500 spot, very deep history (since 1927 on Yahoo)
    "spy": "SPY",     # S&P 500 ETF (since 1993) — the tradeable face
    "ndx": "^NDX",    # Nasdaq-100 spot, for the cross-index robustness check
    "qqq": "QQQ",     # Nasdaq-100 ETF
}
GAUGE = "^VIX"        # CBOE Volatility Index, daily close (since 1990)

# Spot for stats, ETF for execution — the pair the headline analysis runs on.
MARKET_PAIRS = {
    "S&P 500": ("^GSPC", "SPY"),
    "Nasdaq-100": ("^NDX", "QQQ"),
}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_cache")

OHLC_COLS = ["Open", "High", "Low", "Close"]


def _cache_path(ticker: str, mode: AdjustMode) -> str:
    safe = ticker.replace("^", "_").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}__{mode}.parquet")


# --------------------------------------------------------------------------- #
# Download + cache (the market side reuses Study 02's adjustment machinery)
# --------------------------------------------------------------------------- #

def fetch(
    ticker: str,
    *,
    mode: AdjustMode = "split_only",
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a daily OHLC ``DataFrame`` indexed by date.

    Columns: ``Open, High, Low, Close`` (always), plus ``Volume`` when supplied.
    Adjustment follows ``mode``. ``ticker`` may be a Yahoo symbol (``'^GSPC'``,
    ``'SPY'``, ``'^VIX'``) or a short key in :data:`MARKETS`.
    """
    ticker = MARKETS.get(ticker.lower(), ticker)
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

    raw = yf.download(
        ticker,
        period="max",
        auto_adjust=False,
        actions=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker!r}. Check the symbol and your "
            f"network connection (Yahoo! Finance)."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    out = _apply_mode(raw, mode)
    out.index.name = "Date"
    out = out.dropna(subset=["Open", "Close"]).sort_index()
    return out


def _apply_mode(raw: pd.DataFrame, mode: AdjustMode) -> pd.DataFrame:
    """Build OHLC under the requested adjustment convention.

    The VIX carries no splits/dividends, so ``Adj Close == Close`` and every mode
    collapses to raw — harmless. For the market the three modes behave exactly as
    in Study 02.
    """
    base = raw[[c for c in OHLC_COLS]].copy()
    base.columns = OHLC_COLS

    if mode == "raw":
        out = base
    elif mode == "total_return":
        if "Adj Close" not in raw.columns:
            raise RuntimeError("Adj Close missing; cannot build total_return mode.")
        factor = raw["Adj Close"] / raw["Close"]
        out = base.mul(factor, axis=0)
    elif mode == "split_only":
        split = raw["Stock Splits"].replace(0.0, 1.0) if "Stock Splits" in raw else None
        if split is None or (split == 1.0).all():
            out = base
        else:
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


# --------------------------------------------------------------------------- #
# The two consumable objects: a market frame and a gauge frame
# --------------------------------------------------------------------------- #

def daily_returns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Attach the market return series the event study / benchmark consume.

    Adds (mirrors Study 02 so the two desks line up):
        * ``r_cc``  close-to-close return (the headline daily return)
        * ``r_oc``  open-to-close (intraday) return
        * ``r_co``  previous-close-to-open (overnight gap) return
    """
    out = ohlc.copy()
    prev_close = out["Close"].shift(1)
    out["r_cc"] = out["Close"] / prev_close - 1.0
    out["r_oc"] = out["Close"] / out["Open"] - 1.0
    out["r_co"] = out["Open"] / prev_close - 1.0
    return out


def vix_frame(
    *,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """The gauge: ``^VIX`` close plus the 1-day change that defines a *spike*.

    Columns:
        * ``Close``    VIX level (this is what V1's threshold reads).
        * ``vix_chg``  1-day fractional change ``Close/Close.shift(1) - 1`` — the
          quantity Altucher's "+30% single-day spike" thresholds (V2/V3).
        * ``vix_prev`` prior-day close, so triggers can split a spike by its base.

    The VIX is taken raw (``mode='raw'``); adjustment is meaningless for an index
    of implied volatility.
    """
    vix = fetch(GAUGE, mode="raw", start=start, end=end, use_cache=use_cache)
    out = vix[["Close"]].copy()
    out["vix_prev"] = out["Close"].shift(1)
    out["vix_chg"] = out["Close"] / out["vix_prev"] - 1.0
    return out


def aligned(
    market: str = "^GSPC",
    *,
    mode: AdjustMode = "split_only",
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(market, gauge)`` frames on their common trading days.

    The market frame carries OHLC + the ``r_*`` returns (via :func:`daily_returns`);
    the gauge frame carries ``Close/vix_prev/vix_chg``. Both are reindexed to the
    intersection of their dates, so a VIX trigger and an S&P forward return refer to
    the *same* sessions with no look-ahead. This is the canonical entry point::

        spx, vix = aligned("^GSPC")
        sig = triggers.spike(vix)                 # signal lives on the gauge
        benchmark.conditional_vs_unconditional(spx, sig)   # tested on the market
    """
    mkt = daily_returns(fetch(market, mode=mode, start=start, end=end, use_cache=use_cache))
    vix = vix_frame(start=start, end=end, use_cache=use_cache)
    common = mkt.index.intersection(vix.index)
    return mkt.loc[common].copy(), vix.loc[common].copy()
