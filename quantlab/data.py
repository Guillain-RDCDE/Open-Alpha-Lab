"""Data layer — Yahoo! Finance with a local parquet cache.

The adjustment mode is a *decision, not a detail*: dividends and splits move
return between the night and day legs (a stock trades ex-dividend at the open),
so the mode you pick changes the very thing Knuteson measures. Document your
choice in any figure you publish.

Yahoo's convention, pinned empirically (do not take it on faith)
----------------------------------------------------------------
yfinance with ``auto_adjust=False`` returns OHLC that is **already
split-adjusted**; only ``Adj Close`` differs from ``Close``, and only by
dividends. Checked on QQQ's 2:1 split of 2000-03-20: the reported pre-split
close (2000-03-17) is ~110.8 — the as-traded ~221.6 divided by 2 — and the
close-to-close return across the split is a quiet −2.8%, not a 2x jump.
``tests/test_data.py`` pins this convention with a live (network-marked) test.

Modes
-----
'split_only'   (default) prices adjusted for splits but NOT dividends. Keeps
               the genuine ex-dividend gap in the overnight leg, which is what
               a price-taker actually experiences. Closest to "real life".
               Given Yahoo's convention above this is a NO-OP on the data —
               kept as a named mode so the choice stays explicit. (An earlier
               revision divided by a reconstructed split factor a *second*
               time, manufacturing a ~+100% overnight gap at every split date;
               see ``_apply_mode``.)
'total_return' fully adjusted (splits + dividends). Removes the ex-div gap;
               cleaner academic series but hides a real overnight cost/benefit.
'raw'          as-traded prices, in the units of the day's tape: the
               split-adjusted OHLC is multiplied BACK UP by the cumulative
               factor of all later splits (rebuilt from the 'Stock Splits'
               events). Maximises artefacts across split dates — useful to
               SHOW the artefact problem, never to trade on.
"""

from __future__ import annotations

import os
import warnings
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

# Daily moves beyond this are overwhelmingly bad data, not markets (see
# diagnostics.flag_suspicious_returns); fetch() warns loudly when it sees one.
SUSPICIOUS_RETURN_THRESHOLD = 0.40


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

    Caching decision: ``start``/``end`` are deliberately NOT part of the cache
    key. A cache miss downloads the ticker's FULL history (``period='max'``)
    and stores that; every call then slices ``[start:end]`` from the cached
    frame on return. Two windows share one file, and a narrow request after a
    wide one (or the reverse) gets exactly the window it asked for — an older
    revision returned whatever range happened to be cached, silently. Caches
    written before this convention may hold less than full history; delete the
    parquet (or pass ``use_cache=False``) to refresh.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_{mode}.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = _download(ticker, mode)
        if use_cache:
            df.to_parquet(cache_path)

    out = df.loc[start:end].copy()
    _warn_if_suspicious(out, ticker)
    return out


def _download(ticker: str, mode: str) -> pd.DataFrame:
    """Pull the FULL history from yfinance and apply the chosen adjustment mode."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "yfinance is required for live data; `pip install yfinance` "
            "or use overnight.diagnostics.synthetic_ohlc for offline work."
        ) from exc

    # auto_adjust=False gives us split-adjusted OHLC + 'Adj Close' so we can
    # choose the mode; period='max' so the cache holds the widest history.
    raw = yf.download(
        ticker,
        period="max",
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
    out = out[["Open", "High", "Low", "Close", "Volume"]]

    # Drop bars with no close. Yahoo publishes the CURRENT, unsettled session as a
    # row whose close is NaN (seen live on 2026-08-17 for SPY: one trailing NaN on
    # an otherwise clean 8,444-row tape). A bar without a close is not a bar — it
    # is a placeholder for a session that has not printed yet — and letting it
    # through puts a NaN at the end of every series a study slices, which is both
    # a silent contaminant in any rolling statistic and a false failure in the
    # data-sanity checks studies run on the real tape.
    return out[out["Close"].notna()]


def _apply_mode(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Translate raw yfinance OHLC + 'Adj Close' into the requested series.

    Relies on the convention pinned in the module docstring (and in
    ``tests/test_data.py``): ``auto_adjust=False`` OHLC arrives ALREADY
    split-adjusted, with 'Adj Close' carrying the dividend adjustment on top.

      split_only   -> no-op: the prices are already what this mode means.
                      (The bug this replaces: rebuilding a cumulative split
                      factor from 'Stock Splits' and dividing AGAIN, i.e. a
                      double adjustment — every split date grew a fabricated
                      overnight gap of roughly the split ratio.)
      total_return -> scale OHLC by Adj Close / Close (the dividend factor).
                      A zero Close yields NaN, never inf: bad rows stay
                      visibly bad instead of becoming infinite returns.
      raw          -> as-traded: MULTIPLY OHLC by the cumulative product of
                      all strictly-later split ratios (the split applies from
                      the split date's open, so that day is already in new
                      units). Volume is divided by the same factor, keeping
                      dollar volume invariant.
    """
    df = raw.copy()

    if mode == "split_only":
        return df

    if mode == "total_return":
        # Scale every OHLC by the same Adj-Close/Close factor -> fully adjusted.
        if "Adj Close" not in df.columns:
            return df
        close = df["Close"]
        factor = df["Adj Close"].div(close.where(close != 0))  # NaN on a zero close, not inf
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * factor
        return df

    # mode == "raw": undo Yahoo's split adjustment to recover the as-traded tape.
    splits = df.get("Stock Splits")
    if splits is None or (splits == 0).all():
        return df  # never split: split-adjusted == as-traded
    ratio = splits.replace(0, 1.0).astype(float)
    # Cumulative factor of strictly-future splits: days BEFORE a k:1 split are
    # multiplied back up by k; the split day itself already trades in new units.
    cum = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] * cum
    df["Volume"] = df["Volume"] / cum
    return df


def _warn_if_suspicious(
    df: pd.DataFrame, ticker: str, threshold: float = SUSPICIOUS_RETURN_THRESHOLD
) -> None:
    """Loud automatic guard-rail on every fetch: daily legs beyond ``threshold``.

    Runs the decomposition through ``diagnostics.flag_suspicious_returns`` —
    legitimate daily opens almost never gap ±40%, so a hit is overwhelmingly a
    bad print or a mis-adjustment (the exact failure mode the old split_only
    double-adjustment produced). A warning, not an exception: artefact studies
    fetch deliberately dirty series on purpose.
    """
    try:
        from .decompose import decompose
        from .diagnostics import flag_suspicious_returns

        flagged = flag_suspicious_returns(decompose(df), threshold=threshold)
    except Exception:  # diagnostics must never break a fetch
        return
    if len(flagged):
        dates = ", ".join(str(d)[:10] for d in flagged.index[:5])
        more = "" if len(flagged) <= 5 else f" (+{len(flagged) - 5} more)"
        warnings.warn(
            f"{ticker}: {len(flagged)} daily move(s) beyond ±{threshold:.0%} "
            f"[{dates}{more}] — almost certainly a data/adjustment artefact, "
            "not a market move. Inspect before publishing anything built on this series.",
            stacklevel=3,
        )
