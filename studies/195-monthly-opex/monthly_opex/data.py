"""Data layer for Study 195 (Monthly-OpEx).

Two tapes, one shape (a daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  Two knobs control the
  planted effects: ``ret_premium_bps`` (directional drift on OpEx-week days) and
  ``vol_premium`` (extra fractional range/volume on OpEx-week days).  Both zero yields
  the null — so tests can assert the effect appears only when we plant it.
- ``fetch_daily`` — the real Yahoo! daily tape for SPY (or any ticker), cache-only by
  default.  The real fetch never happens unless ``fetch=True`` is explicit, so the
  test-suite and reproducible core never touch the network.

Monthly OpEx: the 3rd Friday of *every* calendar month is the standard equity option
expiry (since 1973 for equity options; monthly SPY options since 1993).  This is a
superset of the quarterly triple-witching studied in Study 82 — the question here is
whether the *full monthly* cycle has a measurable statistical effect beyond the quarterly
events.

No look-ahead: expiry dates are purely calendar-derived — known before the month begins.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# OpEx-date helpers — pure calendar, no look-ahead
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the third Friday of a given (year, month).

    Standard monthly equity-option expiry falls on the third Friday of each
    calendar month.  This is computed purely from the Gregorian calendar —
    no market data required, so there is no look-ahead.
    """
    first = pd.Timestamp(year=year, month=month, day=1)
    dow = first.weekday()          # Monday=0, Friday=4
    days_to_friday = (4 - dow) % 7
    first_friday = first + pd.Timedelta(days=days_to_friday)
    return first_friday + pd.Timedelta(weeks=2)


def opex_dates(start: str = "1993-01-01", end: str = "2026-12-31") -> pd.DatetimeIndex:
    """All monthly-OpEx dates (3rd Friday of every month) in [start, end].

    Returns a sorted tz-naive DatetimeIndex.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    dates: list[pd.Timestamp] = []
    year = s.year
    while year <= e.year:
        for month in range(1, 13):
            d = _third_friday(year, month)
            if s <= d <= e:
                dates.append(d)
        year += 1
    return pd.DatetimeIndex(sorted(dates))


def opex_week_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask — True for every trading day in an OpEx week (Mon–Fri).

    The OpEx day is the third Friday; the mask flags all five trading days
    of that calendar week (Monday through the OpEx Friday, inclusive).
    Returns a Boolean Series aligned on ``index``.
    """
    odates = opex_dates(
        start=str(index.min().date()), end=str(index.max().date())
    )
    mask = pd.Series(False, index=index, name="opex_week")
    for od in odates:
        mon = od - pd.Timedelta(days=od.weekday())  # Monday of that week
        mask |= (index >= mon) & (index <= od)
    return mask


def opex_day_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask — True only on the OpEx day itself (the 3rd Friday each month)."""
    odates = opex_dates(
        start=str(index.min().date()), end=str(index.max().date())
    )
    mask = pd.Series(False, index=index, name="opex_day")
    for od in odates:
        mask |= index.normalize() == od
    return mask


def is_quarterly_opex(date: pd.Timestamp) -> bool:
    """Return True if the date is a quarterly triple-witching OpEx (Mar/Jun/Sep/Dec)."""
    return date.month in (3, 6, 9, 12)


def quarterly_opex_week_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask — True for OpEx-week days in quarterly months only (Mar/Jun/Sep/Dec).

    Used to separate the Study-82 quarterly effect from the *monthly-only* (non-quarterly)
    OpEx weeks.
    """
    odates = opex_dates(
        start=str(index.min().date()), end=str(index.max().date())
    )
    mask = pd.Series(False, index=index, name="qopex_week")
    for od in odates:
        if is_quarterly_opex(od):
            mon = od - pd.Timedelta(days=od.weekday())
            mask |= (index >= mon) & (index <= od)
    return mask


def monthly_only_opex_week_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask — True for OpEx-week days in *non-quarterly* months only (Jan/Feb/Apr/May/Jul/Aug/Oct/Nov).

    This is the incremental monthly signal: do non-quarterly monthly OpEx weeks
    carry an effect beyond the quarterly events?  If the full monthly mask shows an
    effect but the quarterly-only mask does not, the monthly cycle is the driver.
    """
    odates = opex_dates(
        start=str(index.min().date()), end=str(index.max().date())
    )
    mask = pd.Series(False, index=index, name="monthly_only_opex_week")
    for od in odates:
        if not is_quarterly_opex(od):
            mon = od - pd.Timedelta(days=od.weekday())
            mask |= (index >= mon) & (index <= od)
    return mask


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_daily(
    n_years: int = 30,
    vol_premium: float = 0.20,       # extra range multiplier on OpEx week days (0 = null)
    ret_premium_bps: float = 0.0,    # extra daily return bps on OpEx-week days (0 = null)
    base_vol_ann: float = 0.16,
    seed: int = 195,
    start: str = "1993-01-01",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known OpEx-week vol / return effect.

    Returns are i.i.d. normal with ``base_vol_ann``; on OpEx-week days the return has
    ``ret_premium_bps`` added and the realised range is inflated by ``(1 + vol_premium)``.
    ``vol_premium = 0`` and ``ret_premium_bps = 0`` is the null.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.  The
    ``bars`` DataFrame has columns ``[open, high, low, close, volume]`` and a tz-naive
    DatetimeIndex of business days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_years * 252), name="date")

    sig_d = base_vol_ann / np.sqrt(252)
    w_mask = opex_week_mask(idx)

    # Log returns — add a directional drift on OpEx-week days
    eps = rng.normal(0.0, sig_d, len(idx))
    extra_drift = w_mask.values * ret_premium_bps * 1e-4
    log_ret = eps + extra_drift

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intraday range: base wick ± OpEx vol inflation
    base_wick_frac = rng.exponential(sig_d, len(idx))
    extra = (1.0 + vol_premium * w_mask.values) * base_wick_frac
    hi = np.maximum(open_, close) + extra * close
    lo = np.minimum(open_, close) - extra * close

    # Volume: elevated on OpEx-week days when vol_premium > 0
    base_vol = rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)
    volume = base_vol * (1.0 + vol_premium * 0.5 * w_mask.values)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=idx,
    )
    truth = {
        "vol_premium": vol_premium,
        "ret_premium_bps": ret_premium_bps,
        "base_vol_ann": base_vol_ann,
        "n_years": n_years,
        "n_days": len(idx),
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Returns a DataFrame with columns ``[open, high, low, close, volume]`` and a
    tz-naive DatetimeIndex (business days).  Network is touched only on an explicit
    ``fetch=True``; the result is cached as a parquet under ``_cache/``.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    bars = bars[bars.index >= pd.Timestamp(start)]
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
