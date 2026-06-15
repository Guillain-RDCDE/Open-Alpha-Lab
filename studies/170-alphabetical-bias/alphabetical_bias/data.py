"""Data layer for Study 170 (Alphabetical-Bias).

Two tapes and one event table:

- ``synthetic_cross`` — a *deterministic, offline* generator.  Produces a
  synthetic cross-section of N stocks with randomised names (first letters drawn
  from the full alphabet) and per-stock returns.  A ``return_bias`` knob plants
  a controlled return premium for early-alphabet (A–C) names; ``return_bias=0``
  is the null (no bias), so tests can assert the detector only fires when we
  plant something real.  A ``volume_bias`` knob similarly plants a turnover
  premium for early-alphabet names (the attention-channel hypothesis).

- ``load_universe`` — loads the pre-staged EDGAR universe and EDGAR-cached
  S&P 500 names with their first letters; daily OHLCV comes from yfinance,
  cache-only by default so the test-suite and the reproducible core never touch
  the network.

Alphabetical ordering effect (Jacobs & Hillert 2015, J. Financial Economics):
stocks that appear at the top of alphabetically sorted broker/screener lists get
more retail attention, higher turnover and volume.  Whether this attention
translates into a *return* premium (early-alphabet earns more) or a *liquidity
discount* (more attention → lower required return) is the question the study
resolves.

Survivorship bias is deliberately named on the Signal axis: the S&P 500 universe
has name-survivorship (companies renamed or delisted drop out), which could
mechanically skew which letters dominate at any point.

No look-ahead is baked in here: forward returns are computed in strategy.py
from prices available *after* the portfolio is formed.
"""

from __future__ import annotations

import hashlib
import os
import string

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Shared repo-level caches (read-only inputs; never written by this study).
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EDGAR_CACHE = os.path.join(REPO_ROOT, "_cache")

# ---------------------------------------------------------------------------
# Alphabet grouping constants (the "early-alphabet" treatment group)
# ---------------------------------------------------------------------------
# Jacobs & Hillert (2015) focus on the very top of alphabetical broker lists.
# We use A–C (first 3 letters) as the treatment, D–Z as the baseline.
# A single-letter split (A-only) would shrink n severely; A–C gives ~10-11%
# of the S&P 500 universe, enough for a meaningful test without over-inclusion.
EARLY_LETTERS = set("ABC")
ALL_LETTERS = set(string.ascii_uppercase)
LATE_LETTERS = ALL_LETTERS - EARLY_LETTERS


# ---------------------------------------------------------------------------
# Synthetic cross-section — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_cross(
    n_stocks: int = 200,
    n_months: int = 120,
    return_bias_bps: float = 0.0,
    volume_bias: float = 0.0,
    daily_vol_bps: float = 100.0,
    start: str = "2015-01-01",
    seed: int = 170,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly cross-section with a planted alphabetical bias.

    Each stock is assigned a random first letter drawn from the full alphabet
    (A–Z, uniform).  Monthly returns are drawn i.i.d. normal; ``return_bias_bps``
    adds a constant monthly premium to stocks whose ticker starts with A, B or C.
    ``volume_bias`` adds a fractional volume premium to the same group.

    Parameters
    ----------
    n_stocks : int
        Number of synthetic tickers.
    n_months : int
        Length of the panel in calendar months.
    return_bias_bps : float
        Monthly return premium (in basis points) for early-alphabet stocks.
        ``0`` → null (no alphabetical return edge); ``>0`` → we planted one.
    volume_bias : float
        Fractional volume premium (e.g. ``0.20`` = 20% more turnover) for
        early-alphabet stocks.  ``0`` → null.
    daily_vol_bps : float
        Monthly return standard deviation in basis points (per stock).
    start : str
        ISO date of the first month.
    seed : int
        Random seed — deterministic.

    Returns
    -------
    panel : pd.DataFrame
        Long-format frame with columns
        ``ticker, date, letter, is_early, ret, volume``.
    truth : dict
        Planted parameters, for test assertions.
    """
    rng = np.random.default_rng(seed)

    # Random ticker names: one letter + two random digits (e.g. "A42", "Z07").
    letters = rng.choice(list(string.ascii_uppercase), size=n_stocks, replace=True)
    numbers = rng.integers(10, 100, size=n_stocks)
    tickers = [f"{L}{n:02d}" for L, n in zip(letters, numbers)]
    is_early = np.array([L in EARLY_LETTERS for L in letters])

    # Monthly dates — calendar months, last calendar day approximate.
    dates = pd.date_range(start=start, periods=n_months, freq="MS")

    # Per-stock, per-month returns: i.i.d. normal + planted bias.
    vol = daily_vol_bps * 1e-4
    ret_matrix = rng.normal(0.0, vol, size=(n_stocks, n_months))
    # Add return bias for early-alphabet stocks.
    ret_matrix[is_early, :] += return_bias_bps * 1e-4

    # Per-stock, per-month volume: log-normal baseline + planted volume bias.
    base_vol = rng.lognormal(mean=12.0, sigma=1.0, size=(n_stocks, n_months))
    base_vol[is_early, :] *= 1.0 + volume_bias

    # Build long-format panel.
    rows = []
    for i, (ticker, letter, early) in enumerate(zip(tickers, letters, is_early)):
        for j, date in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "date": date,
                "letter": letter,
                "is_early": bool(early),
                "ret": float(ret_matrix[i, j]),
                "volume": float(base_vol[i, j]),
            })

    panel = pd.DataFrame(rows)
    truth = {
        "n_stocks": n_stocks,
        "n_months": n_months,
        "return_bias_bps": return_bias_bps,
        "volume_bias": volume_bias,
        "daily_vol_bps": daily_vol_bps,
        "seed": seed,
        "n_early": int(is_early.sum()),
        "n_late": int((~is_early).sum()),
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real data — EDGAR universe + yfinance daily prices
# ---------------------------------------------------------------------------
def _daily_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace(".", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_sp500_names(
    edgar_cache: str = EDGAR_CACHE,
) -> pd.DataFrame:
    """Load the S&P 500 ticker universe from the shared EDGAR cache.

    Returns a DataFrame with columns ``ticker``, ``letter``, ``is_early``
    identifying each symbol's first letter and whether it falls in the A–C
    early-alphabet treatment group.

    Survivorship note: the EDGAR universe is the current S&P 500 list (as-of
    the cache date).  Stocks that were in the index but have since been renamed
    or delisted are absent.  This name-level survivorship is disclosed on the
    Signal axis and is one reason we expect no tradable return edge even if
    the attention channel is real.
    """
    # Use the shared quantlab universe helper if available, otherwise fall back
    # to the EDGAR revenues parquet which has the broadest ticker coverage.
    try:
        import sys
        sys.path.insert(0, REPO_ROOT)
        from quantlab.universe import sp500_symbols
        tickers = list(sp500_symbols(allow_survivorship_bias=True))
    except Exception:
        # Fallback: infer universe from an EDGAR parquet's column names.
        rev_path = os.path.join(edgar_cache, "_edgar_Revenues.parquet")
        if not os.path.exists(rev_path):
            raise FileNotFoundError(
                f"EDGAR revenues cache not found at {rev_path}. "
                "Cannot load S&P 500 universe."
            )
        df = pd.read_parquet(rev_path)
        tickers = list(df.columns)

    records = []
    for t in tickers:
        # First letter of the ticker (strip ^ and other prefix chars).
        clean = t.lstrip("^").lstrip(".")
        if not clean:
            continue
        letter = clean[0].upper()
        if letter not in ALL_LETTERS:
            continue
        records.append({
            "ticker": t,
            "letter": letter,
            "is_early": letter in EARLY_LETTERS,
        })
    return pd.DataFrame(records).drop_duplicates("ticker").reset_index(drop=True)


def fetch_daily(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/``).  Returns a timezone-naïve frame
    indexed by ``date`` with columns ``open, high, low, close, volume``.
    """
    path = _daily_cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def build_monthly_panel(
    universe: pd.DataFrame,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """Build a monthly returns + volume panel from cached daily prices.

    For each ticker in ``universe``, reads the cached daily parquet, computes
    monthly total log-return (close-to-close from last day of prior month to
    last day of current month) and average daily volume, and stacks into a
    long-format panel.

    Returns
    -------
    panel : pd.DataFrame
        Long-format with columns:
        ``ticker, date, letter, is_early, ret, volume, n_days``
        where ``date`` is the month-end date and ``ret`` is the monthly
        log-return (i.e. no look-ahead: formed from closing prices available at
        month-end).
    """
    records = []
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        path = _daily_cache_path(ticker, cache_dir)
        if not os.path.exists(path):
            continue
        try:
            bars = pd.read_parquet(path)
        except Exception:
            continue
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        bars = bars[bars.index >= pd.Timestamp(start)]
        if len(bars) < 20:
            continue
        # Resample to monthly: last close and mean daily volume.
        monthly_close = bars["close"].resample("ME").last().dropna()
        monthly_vol = bars["volume"].resample("ME").mean().dropna()
        log_ret = np.log(monthly_close / monthly_close.shift(1)).dropna()
        for date, ret in log_ret.items():
            if date not in monthly_vol.index:
                continue
            records.append({
                "ticker": ticker,
                "date": date,
                "letter": row["letter"],
                "is_early": row["is_early"],
                "ret": float(ret),
                "volume": float(monthly_vol.get(date, np.nan)),
            })
    panel = pd.DataFrame(records)
    if panel.empty:
        return panel
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of a panel (ret column), for the as-of stamp."""
    arr = panel["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
