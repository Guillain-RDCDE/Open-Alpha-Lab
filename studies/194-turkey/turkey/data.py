"""Data layer for Study 194 (Turkey).

Two tapes, one shape (a daily return series for ^GSPC):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``wed_effect``
  and ``fri_effect`` knob let us dial a mean return differential on the
  Wednesday-before-Thanksgiving and the Friday-after-Thanksgiving (the
  half-day Black Friday session). ``wed_effect = fri_effect = 0`` is the null —
  the Thanksgiving pre/post holiday days are just normal trading days — so
  tests can assert the strategy beats a placebo *only* when a genuine anomaly
  is planted. This is the study's null in a bottle.

- ``fetch_gspc`` — real ^GSPC daily prices (``yfinance``), **cache-only** by
  default so the test-suite and the reproducible core never touch the network.
  We use the shared repo cache at
  ``_cache/^GSPC_split_only.parquet`` (available from 1927-12-30).

Thanksgiving falls on the **fourth Thursday of November** (US). Trading days:
- **Wednesday before** Thanksgiving: the last full session before the holiday.
- **Friday after** (Black Friday): a half-day session (market closes at 13:00 ET,
  i.e., the open is normal but volume is thin); we use the close-to-close return
  as usual and flag it in the data.

The Thanksgiving calendar is derived by pure date arithmetic (no fetch, no
hard-coded table): fourth Thursday of November = the Thursday for which
``((date.day - 1) // 7) == 3`` and ``date.month == 11`` and
``date.weekday() == 3``.

No look-ahead: the calendar date is known before the open. The return is the
close-to-close log-return of that day.

Source:  Brockman, P., & Michayluk, D. (1998). The persistent holiday effect:
additional evidence. Applied Economics Letters, 5(4), 205-209.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Per-study scratch cache (for the study's own fetches if any).
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Shared repo-level cache (read-only input — do NOT write here).
REPO_CACHE = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "_cache")
)

TICKER = "^GSPC"
# Path in the shared repo cache where the GSPC daily bars live.
_GSPC_SHARED = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")
# Fallback: per-study cache (populated by verify.py --fetch).
_GSPC_LOCAL = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")


# ---------------------------------------------------------------------------
# Calendar arithmetic — the only "event table" we need
# ---------------------------------------------------------------------------
def _thanksgiving_year(year: int) -> pd.Timestamp:
    """Return the Thanksgiving date (fourth Thursday of November) for *year*.

    Thanksgiving is the fourth Thursday of November — ``weekday() == 3``.
    We find the first Thursday of November and add 21 days (3 weeks).
    """
    # Find the first Thursday of November of this year.
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    # days_until_thu: number of days from Nov 1 to the first Thursday
    days_until_thu = (3 - nov1.weekday()) % 7
    first_thu = nov1 + pd.Timedelta(days=days_until_thu)
    # Fourth Thursday = first + 21 days
    return first_thu + pd.Timedelta(days=21)


def thanksgiving_dates(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on every Thanksgiving day in *dates*.

    Pure calendar arithmetic — every fourth Thursday of November.
    """
    tsgivings = {_thanksgiving_year(y) for y in dates.year.unique()}
    arr = pd.Series(dates).isin(tsgivings).values
    return pd.Series(arr, index=dates, name="is_thanksgiving")


def is_wed_before_thanksgiving(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on the trading Wednesday before Thanksgiving.

    The Wednesday before Thanksgiving is exactly 1 day before Thanksgiving
    (Thursday). On a normal year that is Thanksgiving_date - 1 calendar day.
    We keep only dates that appear in *dates* (trading days) to avoid
    marking non-trading days.
    """
    tsg_set = {_thanksgiving_year(y) for y in dates.year.unique()}
    # Wednesday before = Thanksgiving - 1 day
    wed_set = {t - pd.Timedelta(days=1) for t in tsg_set}
    arr = pd.Series(dates).isin(wed_set).values
    return pd.Series(arr, index=dates, name="is_wed_before_tsg")


def is_fri_after_thanksgiving(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on the Black Friday (day after Thanksgiving).

    The Friday after Thanksgiving is Thanksgiving_date + 1 calendar day.
    We keep only dates that appear in *dates* (trading days) — Black Friday
    is a short session (closes 13:00 ET) but IS a trading day.
    """
    tsg_set = {_thanksgiving_year(y) for y in dates.year.unique()}
    # Black Friday = Thanksgiving + 1 day
    fri_set = {t + pd.Timedelta(days=1) for t in tsg_set}
    arr = pd.Series(dates).isin(fri_set).values
    return pd.Series(arr, index=dates, name="is_fri_after_tsg")


def is_placebo_tue(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on the Tuesday before Thanksgiving.

    Matched-window placebo: same time of year, same week structure, one day
    earlier than Wednesday. If Tuesday shows the same pattern the signal is
    just a weekly / seasonal effect, not specifically pre-holiday.
    Thanksgiving - 2 calendar days.
    """
    tsg_set = {_thanksgiving_year(y) for y in dates.year.unique()}
    tue_set = {t - pd.Timedelta(days=2) for t in tsg_set}
    arr = pd.Series(dates).isin(tue_set).values
    return pd.Series(arr, index=dates, name="is_tue_before_tsg")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 75,
    annual_vol: float = 0.15,
    annual_drift: float = 0.07,
    wed_effect: float = 0.0,
    fri_effect: float = 0.0,
    seed: int = 194,
    start: str = "1950-01-04",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily GSPC-like tape with known Thanksgiving anomalies.

    Daily log-returns follow N(drift/252, (vol/sqrt(252))**2) with optional
    additive bumps:

    - ``wed_effect`` (in annual-sigma units) applied *only* on the Wednesday
      before Thanksgiving.
    - ``fri_effect`` (in annual-sigma units) applied *only* on the Friday
      after Thanksgiving (Black Friday).

    - Both effects = 0   → pure random walk; the signal is a fair coin.
    - wed_effect > 0     → positive pre-holiday bump on Wednesday.
    - fri_effect > 0     → positive post-holiday bump on Black Friday.

    Returns ``(df, truth)`` where ``df`` has columns:
    ``ret, is_wed_before_tsg, is_fri_after_tsg, is_placebo_tue``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(idx)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    ret = rng.normal(daily_mu, daily_sig, n)

    wed_mask = is_wed_before_thanksgiving(idx).values
    fri_mask = is_fri_after_thanksgiving(idx).values
    tue_mask = is_placebo_tue(idx).values

    # Apply the planted effects.
    ret[wed_mask] += wed_effect * daily_sig
    ret[fri_mask] += fri_effect * daily_sig

    df = pd.DataFrame(
        {
            "ret": ret,
            "is_wed_before_tsg": wed_mask,
            "is_fri_after_tsg": fri_mask,
            "is_placebo_tue": tue_mask,
        },
        index=idx,
    )
    df.index.name = "date"

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "wed_effect": wed_effect,
        "fri_effect": fri_effect,
        "n_days": n,
        "n_wed": int(wed_mask.sum()),
        "n_fri": int(fri_mask.sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-only by default
# ---------------------------------------------------------------------------
def _build_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw OHLCV frame to a daily return frame with calendar labels."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw.rename(columns=str.lower)["close"].sort_index()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"
    ret = np.log(close / close.shift(1))
    idx = close.index

    df = pd.DataFrame(
        {
            "close": close,
            "ret": ret,
            "is_wed_before_tsg": is_wed_before_thanksgiving(idx).values,
            "is_fri_after_tsg": is_fri_after_thanksgiving(idx).values,
            "is_placebo_tue": is_placebo_tue(idx).values,
        }
    )
    return df.dropna(subset=["ret"])


def fetch_gspc(
    start: str = "1928-01-01",
    fetch: bool = False,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Daily GSPC return frame with Thanksgiving calendar labels; cache-only
    unless ``fetch=True``.

    Tries the shared repo cache first (``_cache/^GSPC_split_only.parquet``);
    falls back to the per-study cache; finally hits the network only when
    ``fetch=True``.

    Columns: ``close, ret, is_wed_before_tsg, is_fri_after_tsg,
    is_placebo_tue`` (``ret`` is log-return).
    """
    # 1. Shared repo cache (best — already covers 1927-onward).
    if os.path.exists(_GSPC_SHARED):
        raw = pd.read_parquet(_GSPC_SHARED)
        df = _build_daily(raw)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 2. Per-study local cache.
    local = (
        _GSPC_LOCAL
        if cache_dir is None
        else os.path.join(cache_dir, "gspc_daily.parquet")
    )
    if os.path.exists(local):
        raw = pd.read_parquet(local)
        df = _build_daily(raw)
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

    df = _build_daily(raw)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (ret column), for the as-of stamp."""
    arr = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
