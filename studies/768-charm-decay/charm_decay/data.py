"""Data layer for Study 768 (Charm-Decay).

The story under test is the *charm* (delta-decay) dealer-flow narrative popularised by
options-flow desks (SpotGamma, Menthor Q, vol-Twitter): in the last few sessions before
monthly options expiration, the delta of the large book of near-the-money options decays
purely because time is passing (that time-derivative of delta is the Greek *charm*).
Dealers who are short those options must re-hedge in a systematic direction as the deltas
bleed, and the popular claim is that this produces a **directional upward drift into
monthly OpEx** followed by a **give-back the week after** — the "OpEx-week rally / post-OpEx
weakness" pattern.

Two tapes, one shape (a daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  One knob, ``pre_drift_bps``,
  plants a directional drift on the pre-OpEx (charm) window; ``post_drift_bps`` plants the
  mirror give-back.  Both zero yields the null — so tests can assert the effect appears only
  when we plant it.
- ``fetch_daily`` — the real Yahoo! daily tape for SPY (or any ticker), cache-only by
  default.  The real fetch never happens unless ``fetch=True`` is explicit, so the
  test-suite and reproducible core never touch the network.

The windows are anchored on the 3rd Friday of every calendar month (the standard monthly
equity-option expiry) and measured in **trading days**, not calendar days — so market
holidays never smear the window.  All anchoring is pure calendar arithmetic, known before
the month begins: there is no look-ahead in the signal.
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
    """Third Friday of a (year, month) — the standard monthly equity-option expiry.

    Computed purely from the Gregorian calendar, so there is no look-ahead: the
    date is known before the month starts.
    """
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_friday = (4 - first.weekday()) % 7          # Monday=0, Friday=4
    return first + pd.Timedelta(days=days_to_friday) + pd.Timedelta(weeks=2)


def opex_dates(start: str = "1993-01-01", end: str = "2026-12-31") -> pd.DatetimeIndex:
    """All monthly-OpEx dates (3rd Friday of every month) in [start, end]."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    dates: list[pd.Timestamp] = []
    for year in range(s.year, e.year + 1):
        for month in range(1, 13):
            d = _third_friday(year, month)
            if s <= d <= e:
                dates.append(d)
    return pd.DatetimeIndex(sorted(dates))


def is_quarterly_opex(date: pd.Timestamp) -> bool:
    """True if the OpEx is a quarterly triple-witching (Mar/Jun/Sep/Dec)."""
    return date.month in (3, 6, 9, 12)


# ---------------------------------------------------------------------------
# Trading-day windows anchored on OpEx — the charm apparatus
# ---------------------------------------------------------------------------

def _opex_positions(index: pd.DatetimeIndex, shift: int = 0) -> list[int]:
    """Integer positions in ``index`` of the trading day *on or before* each OpEx date.

    The anchor is the OpEx Friday if it traded, else the last session before it (OpEx can
    fall on a market holiday).  ``shift`` displaces every anchor by ``shift`` trading days —
    the hook the placebo/randomisation test uses to build a null of "fake OpEx" calendars.
    """
    idx = pd.DatetimeIndex(index).normalize()
    odates = opex_dates(str(idx.min().date()), str(idx.max().date()))
    pos: list[int] = []
    for od in odates:
        # searchsorted 'right' - 1 gives the last index position <= od
        p = int(idx.searchsorted(od, side="right")) - 1
        p += shift
        if 0 <= p < len(idx):
            pos.append(p)
    return pos


def charm_window_mask(
    index: pd.DatetimeIndex,
    lo: int,
    hi: int,
    shift: int = 0,
    name: str = "window",
) -> pd.Series:
    """Boolean mask — True for trading days whose OpEx offset lies in ``[lo, hi]``.

    Offsets are in **trading days** relative to the OpEx anchor (offset 0 = OpEx session).
    The pre-OpEx "charm week" is ``lo=-4, hi=0`` (the five sessions ending on expiry);
    the post-OpEx give-back window is ``lo=1, hi=5``.  ``shift`` moves the anchor for the
    placebo test.  A trading day can qualify for at most one OpEx anchor per window because
    OpEx dates are a month apart and windows here span at most a week.
    """
    idx = pd.DatetimeIndex(index)
    mask = np.zeros(len(idx), dtype=bool)
    for p in _opex_positions(idx, shift=shift):
        a = max(p + lo, 0)
        b = min(p + hi, len(idx) - 1)
        if a <= b:
            mask[a : b + 1] = True
    return pd.Series(mask, index=idx, name=name)


def pre_opex_mask(index: pd.DatetimeIndex, ndays: int = 5, shift: int = 0) -> pd.Series:
    """The charm window: the ``ndays`` sessions ending on OpEx (inclusive)."""
    return charm_window_mask(index, lo=-(ndays - 1), hi=0, shift=shift, name="pre_opex")


def post_opex_mask(index: pd.DatetimeIndex, ndays: int = 5, shift: int = 0) -> pd.Series:
    """The give-back window: the ``ndays`` sessions after OpEx."""
    return charm_window_mask(index, lo=1, hi=ndays, shift=shift, name="post_opex")


def quarterly_pre_opex_mask(index: pd.DatetimeIndex, ndays: int = 5) -> pd.Series:
    """Pre-OpEx charm window restricted to quarterly (triple-witching) months."""
    idx = pd.DatetimeIndex(index).normalize()
    odates = opex_dates(str(idx.min().date()), str(idx.max().date()))
    mask = np.zeros(len(idx), dtype=bool)
    for od in odates:
        if not is_quarterly_opex(od):
            continue
        p = int(idx.searchsorted(od, side="right")) - 1
        a = max(p - (ndays - 1), 0)
        if 0 <= p < len(idx):
            mask[a : p + 1] = True
    return pd.Series(mask, index=idx, name="pre_opex_quarterly")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_daily(
    n_years: int = 30,
    pre_drift_bps: float = 0.0,     # extra daily return bps on pre-OpEx window (0 = null)
    post_drift_bps: float = 0.0,    # extra daily return bps on post-OpEx window (0 = null)
    base_vol_ann: float = 0.16,
    base_drift_bps: float = 3.0,    # unconditional daily drift (equity risk premium)
    seed: int = 768,
    start: str = "1993-01-01",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known pre/post-OpEx drift effect.

    Returns are i.i.d. normal with ``base_vol_ann`` and ``base_drift_bps``; on pre-OpEx
    (charm) window days ``pre_drift_bps`` is added, on post-OpEx window days
    ``post_drift_bps`` is added.  Both premia zero is the null.  Returns ``(bars, truth)``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_years * 252), name="date")
    sig_d = base_vol_ann / np.sqrt(252)

    pre_m = pre_opex_mask(idx).values
    post_m = post_opex_mask(idx).values

    eps = rng.normal(0.0, sig_d, len(idx))
    drift = (
        base_drift_bps
        + pre_drift_bps * pre_m
        + post_drift_bps * post_m
    ) * 1e-4
    log_ret = eps + drift

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = rng.exponential(sig_d, len(idx))
    hi = np.maximum(open_, close) + wick * close
    lo = np.minimum(open_, close) - wick * close
    volume = rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=idx,
    )
    truth = {
        "pre_drift_bps": pre_drift_bps,
        "post_drift_bps": post_drift_bps,
        "base_drift_bps": base_drift_bps,
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

    Columns ``[open, high, low, close, volume]``, tz-naive DatetimeIndex.  Network is
    touched only on an explicit ``fetch=True``; the result caches as parquet under
    ``_cache/``.  SPY is auto-adjusted (total-return proxy: splits + dividends folded in).
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
