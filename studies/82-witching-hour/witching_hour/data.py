"""Data layer for Study 82 (Witching-Hour).

Two tapes, one shape (a daily OHLCV frame with a volume column):

- ``synthetic_daily`` — a *deterministic, offline* generator. The knob is
  ``vol_premium`` (extra fractional volatility on witching days) and
  ``ret_premium`` (extra daily return in bps on witching days). Setting both to
  zero yields a flat null — so tests can assert "the effect appears only when we
  plant it, and not otherwise."
- ``fetch_daily`` — the real Yahoo! daily tape for SPY (or any ticker), cache-only
  by default.  The real fetch never happens unless ``fetch=True`` is explicit, so
  the test-suite and reproducible core never touch the network.

No look-ahead is baked in here — that discipline lives in ``strategy.py``.
Witching dates are computed deterministically from the exchange calendar, which is
known *before* the month begins; every mask is therefore lag-free.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Witching-date helpers — pure calendar, no look-ahead
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the third Friday of a given (year, month).

    The quadruple/triple witching day is the third Friday of March, June,
    September, and December.  This helper computes the date purely from the
    Gregorian calendar — no market data required.
    """
    # First day of month
    first = pd.Timestamp(year=year, month=month, day=1)
    # weekday(): Monday=0 … Friday=4
    dow = first.weekday()
    # Days until first Friday
    days_to_friday = (4 - dow) % 7
    first_friday = first + pd.Timedelta(days=days_to_friday)
    return first_friday + pd.Timedelta(weeks=2)


def witching_dates(start: str = "1990-01-01", end: str = "2026-12-31") -> pd.DatetimeIndex:
    """All third-Friday-of-quarter (witching) dates in [start, end].

    Triple/quadruple witching falls on the 3rd Friday of Mar/Jun/Sep/Dec.
    Returns a sorted DatetimeIndex of those calendar dates (tz-naive).
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    dates = []
    for year in range(s.year, e.year + 1):
        for month in (3, 6, 9, 12):
            d = _third_friday(year, month)
            if s <= d <= e:
                dates.append(d)
    return pd.DatetimeIndex(sorted(dates))


def witching_week_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True for each trading day in the expiry week (Mon–Fri of that week).

    The witching day itself is a Friday; the mask flags all 5 trading days
    of that calendar week (Mon through the witching Fri, inclusive).
    Returns a Boolean Series aligned on ``index``.
    """
    wdates = witching_dates(
        start=str(index.min().date()), end=str(index.max().date())
    )
    # Map each witching Friday to the Monday of that week
    mask = pd.Series(False, index=index, name="witching_week")
    for wd in wdates:
        mon = wd - pd.Timedelta(days=wd.weekday())  # Monday of that week
        fri = wd
        mask |= (index >= mon) & (index <= fri)
    return mask


def witching_day_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True only on the witching day itself (the 3rd Friday)."""
    wdates = witching_dates(
        start=str(index.min().date()), end=str(index.max().date())
    )
    mask = pd.Series(False, index=index, name="witching_day")
    for wd in wdates:
        mask |= index.normalize() == wd
    return mask


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_daily(
    n_years: int = 30,
    vol_premium: float = 0.30,   # extra vol multiplier on witching day (0 = null)
    ret_premium_bps: float = 0.0,  # extra daily return in bps on witching day (0 = null)
    base_vol_ann: float = 0.16,
    seed: int = 82,
    start: str = "1990-01-01",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV-like tape with a known witching-day vol / return effect.

    Returns are i.i.d. normal with ``base_vol_ann``; on witching days the return has
    ``ret_premium_bps`` added and the realised range is inflated by ``(1 + vol_premium)``.
    ``vol_premium = 0`` and ``ret_premium_bps = 0`` is the null — the calendar signal
    is absent.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.  The
    ``bars`` DataFrame has columns ``[open, high, low, close, volume]`` and a
    tz-naive DatetimeIndex of business days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_years * 252), name="date")

    sig_d = base_vol_ann / np.sqrt(252)
    w_day = witching_day_mask(idx)

    # Log returns — extra drift on witching days
    eps = rng.normal(0.0, sig_d, len(idx))
    extra_drift = w_day.values * ret_premium_bps * 1e-4
    log_ret = eps + extra_drift

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intraday range: normal wick ± witching vol inflation
    base_wick_frac = rng.exponential(sig_d, len(idx))
    extra = (1.0 + vol_premium * w_day.values) * base_wick_frac
    hi = np.maximum(open_, close) + extra * close
    lo = np.minimum(open_, close) - extra * close

    # Volume: elevated on witching days only if vol_premium > 0
    # A non-zero vol_premium lifts both the intraday range *and* the volume,
    # mirroring the joint mechanical effect (hedging + roll + unwinding).
    base_vol = rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)
    volume = base_vol * (1.0 + vol_premium * 0.5 * w_day.values)

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
    tz-naive DatetimeIndex (business days).  The ``volume`` column is in shares.
    Network is touched only on an explicit ``fetch=True``; the result is then
    cached as a parquet under ``_cache/``.
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
        import yfinance as yf  # lazy: only when we actually go to the network

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
