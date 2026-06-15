"""Data layer for Study 192 (Tax-Day).

Two tapes, one shape (a daily return series for ^GSPC):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``pre_tax_effect``
  knob lets us dial an additive return differential for the pre-tax-day window (the
  ~10 trading days before April 15) and a ``post_tax_effect`` knob for the days
  immediately after April 15.  Both equal zero produces a pure random walk — the
  null hypothesis — so tests can assert the strategy finds a signal *only* when a
  genuine anomaly is planted.  This is the study's null in a bottle.

- ``fetch_gspc`` — the real ^GSPC daily tape (``yfinance`` / shared repo cache),
  **cache-only** by default so the test-suite and the reproducible core never touch
  the network.  Tries the shared repo cache at ``_cache/^GSPC_split_only.parquet``
  first (1928-onward), then a per-study local cache, then the network only when
  ``fetch=True``.

Calendar arithmetic for the tax deadline:
- The standard US federal-income-tax deadline is April 15.
- If April 15 falls on a Saturday, the deadline shifts to Monday April 17.
- If April 15 falls on a Sunday, the deadline shifts to Monday April 16.
- If the shifted date falls on a public holiday (Patriots' Day, etc.) the IRS
  typically moves to the next business day; we use a simple approximation:
  the *effective deadline* is the next business day on or after April 15 each year.

Pre-deadline window: the ``pre_window`` trading days (default 10) *immediately
preceding* the effective deadline date (the deadline itself is **not** in the window —
the return is the day you're *approaching* the deadline).
Post-deadline window: the ``post_window`` trading days (default 5) *on and after*
the effective deadline date.

No look-ahead: both windows are defined by the calendar date label known before
the open of that day.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Per-study scratch cache (populated by verify.py --fetch if needed).
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Shared repo-level cache (read-only input — do NOT write here).
REPO_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

TICKER = "^GSPC"
_GSPC_SHARED = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")
_GSPC_LOCAL = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")


# ---------------------------------------------------------------------------
# Calendar arithmetic — tax deadline and window labels
# ---------------------------------------------------------------------------
def _effective_tax_deadline(year: int) -> pd.Timestamp:
    """Return the effective US tax filing deadline for *year*.

    April 15, shifted forward to the next business day when it falls on a
    weekend.  We do not model the occasional Patriots' Day (MA) shift because
    the IRS announcement is not systematically predictable from date arithmetic
    alone, and the effect is one day at most.
    """
    d = pd.Timestamp(year=year, month=4, day=15)
    # 5=Saturday, 6=Sunday
    if d.dayofweek == 5:  # Saturday → Monday the 17th
        d += pd.Timedelta(days=2)
    elif d.dayofweek == 6:  # Sunday → Monday the 16th
        d += pd.Timedelta(days=1)
    return d


def label_tax_windows(
    dates: pd.DatetimeIndex,
    pre_window: int = 10,
    post_window: int = 5,
) -> pd.DataFrame:
    """Build calendar label columns for the tax-day effect on a daily index.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Business-day index of trading dates (tz-naive or tz-aware; day-level).
    pre_window : int
        Number of trading days *before* the effective deadline to flag as
        ``pre_tax`` (does not include the deadline day itself).
    post_window : int
        Number of trading days *on and after* the effective deadline to flag as
        ``post_tax`` (includes the deadline day).

    Returns
    -------
    pd.DataFrame with boolean columns ``pre_tax``, ``post_tax``, ``tax_month``
    (the full month of April) indexed on *dates*.

    The columns are mutually exclusive for pre/post: the deadline day itself
    is counted in ``post_tax`` only.
    """
    tz_aware = dates.tz is not None
    years = sorted(set(dates.year))

    pre_tax = np.zeros(len(dates), dtype=bool)
    post_tax = np.zeros(len(dates), dtype=bool)

    # Work on a tz-naive copy for the arithmetic then index back.
    dates_naive = dates.tz_localize(None) if tz_aware else dates

    for yr in years:
        deadline = _effective_tax_deadline(yr)
        # Select all dates in the same year.
        yr_mask = dates_naive.year == yr
        yr_idx = np.where(yr_mask)[0]
        if len(yr_idx) == 0:
            continue

        yr_dates = dates_naive[yr_idx]  # DatetimeIndex subset
        # Positions relative to the deadline.
        # For the pre window: trading days strictly before the deadline.
        n_before = int((yr_dates < deadline).sum())
        n_after = len(yr_idx) - n_before

        # pre_tax: last `pre_window` trading days before the deadline.
        if n_before > 0:
            pre_start_pos = max(0, n_before - pre_window)
            global_pre = yr_idx[pre_start_pos:n_before]
            pre_tax[global_pre] = True

        # post_tax: first `post_window` trading days on or after the deadline.
        if n_after > 0:
            global_post = yr_idx[n_before : n_before + post_window]
            post_tax[global_post] = True

    april = dates_naive.month == 4

    return pd.DataFrame(
        {"pre_tax": pre_tax, "post_tax": post_tax, "tax_month": np.asarray(april)},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 75,
    annual_vol: float = 0.16,
    annual_drift: float = 0.07,
    pre_tax_effect: float = 0.0,
    post_tax_effect: float = 0.0,
    pre_window: int = 10,
    post_window: int = 5,
    seed: int = 192,
    start: str = "1950-01-04",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily GSPC-like tape with a known tax-day anomaly.

    Daily log-returns follow N(drift/252, (vol/sqrt(252))**2) with an additive
    bump of ``pre_tax_effect * daily_sig`` applied on pre-tax-day trading days and
    ``post_tax_effect * daily_sig`` on post-tax-day trading days.

    - Both effects = 0  → pure random walk; tax-window returns are a fair draw.
    - pre_tax_effect > 0 → positive return bias in the pre-deadline window (the
      "IRA contribution / refund money flows in" folk claim).
    - post_tax_effect < 0 → negative return bias just after the deadline (the
      "tax-selling pressure" counter-narrative).

    Returns ``(df, truth)`` where ``df`` has columns
    ``ret, pre_tax, post_tax, tax_month``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(idx)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    ret = rng.normal(daily_mu, daily_sig, n)

    labs = label_tax_windows(idx, pre_window=pre_window, post_window=post_window)
    ret[labs["pre_tax"].values] += pre_tax_effect * daily_sig
    ret[labs["post_tax"].values] += post_tax_effect * daily_sig

    df = pd.DataFrame(
        {
            "ret": ret,
            "pre_tax": labs["pre_tax"].values,
            "post_tax": labs["post_tax"].values,
            "tax_month": labs["tax_month"].values,
        },
        index=idx,
    )
    df.index.name = "date"

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "pre_tax_effect": pre_tax_effect,
        "post_tax_effect": post_tax_effect,
        "pre_window": pre_window,
        "post_window": post_window,
        "n_days": n,
        "n_pre_tax": int(labs["pre_tax"].sum()),
        "n_post_tax": int(labs["post_tax"].sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-only by default
# ---------------------------------------------------------------------------
def _build_daily(raw: pd.DataFrame, pre_window: int = 10, post_window: int = 5) -> pd.DataFrame:
    """Convert a raw OHLCV frame to a daily return frame with tax-day labels."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw.rename(columns=str.lower)["close"].sort_index()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"
    ret = np.log(close / close.shift(1))
    labs = label_tax_windows(close.index, pre_window=pre_window, post_window=post_window)
    df = pd.DataFrame(
        {
            "close": close,
            "ret": ret,
            "pre_tax": labs["pre_tax"].values,
            "post_tax": labs["post_tax"].values,
            "tax_month": labs["tax_month"].values,
        }
    )
    return df.dropna(subset=["ret"])


def fetch_gspc(
    start: str = "1928-01-01",
    fetch: bool = False,
    cache_dir: str | None = None,
    pre_window: int = 10,
    post_window: int = 5,
) -> pd.DataFrame:
    """Daily GSPC return frame with tax-day window labels; cache-only unless ``fetch=True``.

    Tries the shared repo cache first (``_cache/^GSPC_split_only.parquet``);
    falls back to the per-study cache; finally hits the network only when
    ``fetch=True``.

    Columns: ``close, ret, pre_tax, post_tax, tax_month`` (``ret`` is log-return).
    """
    # 1. Shared repo cache (best — covers 1927-onward).
    if os.path.exists(_GSPC_SHARED):
        raw = pd.read_parquet(_GSPC_SHARED)
        df = _build_daily(raw, pre_window=pre_window, post_window=post_window)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 2. Per-study local cache.
    local = _GSPC_LOCAL if cache_dir is None else os.path.join(cache_dir, "gspc_daily.parquet")
    if os.path.exists(local):
        raw = pd.read_parquet(local)
        df = _build_daily(raw, pre_window=pre_window, post_window=post_window)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 3. Network — only with explicit fetch=True.
    if not fetch:
        raise FileNotFoundError(
            f"No GSPC cache found at {_GSPC_SHARED} or {local}. "
            "Call fetch_gspc(fetch=True) to populate the local cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(
        TICKER,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {TICKER}")

    cache_dir_use = cache_dir or DEFAULT_CACHE
    os.makedirs(cache_dir_use, exist_ok=True)
    raw.to_parquet(os.path.join(cache_dir_use, "gspc_daily.parquet"))

    df = _build_daily(raw, pre_window=pre_window, post_window=post_window)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (ret column), for the as-of stamp."""
    arr = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
