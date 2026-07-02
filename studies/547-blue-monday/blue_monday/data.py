"""Data layer for Study 547 (Blue-Monday) — the 'most depressing day' market-mood test.

'Blue Monday' is the **third Monday of January**, a label invented in 2005 by Cliff Arnall as
a PR stunt (a bogus 'equation' for the most depressing day of the year). The market question is
whether investor mood actually sags then — a *calendar cousin* of the SAD-effect (Study 150):
do returns dip and realised volatility spike on Blue Monday versus an ordinary Monday?

Two tapes, one schema (a daily frame with a ``close`` column):

- ``synthetic_daily`` — a deterministic, offline generator with a single knob, ``blue_dip``.
  ``blue_dip = 0`` is an i.i.d. constant-drift tape (the null: Blue Monday is just a Monday).
  ``blue_dip > 0`` subtracts ``blue_dip`` from the return of the third Monday of every January
  (and, optionally, inflates its vol) — the positive control that plants the mood anomaly.
- ``load_real`` — real daily SPY total-return closes from yfinance, cache-first into this
  study's own ``_cache/`` so the reproducible core never needs the network.

Day-of-week and calendar date are known in advance, so there is **no execution lag**: whether a
given Monday is 'Blue' is a deterministic function of the calendar, fully public before the open.
The one honesty caveat is the *thin sample* — one Blue Monday per year (~33 on the SPY tape),
so the test is under-powered by construction. Named on the SIGNAL axis.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The Blue-Monday calendar rule
# ---------------------------------------------------------------------------
def third_monday_of_january(year: int) -> pd.Timestamp:
    """The date of the third Monday of January in ``year`` — canonical 'Blue Monday'."""
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    # dayofweek: Monday == 0. First Monday on/after Jan 1:
    offset = (0 - jan1.dayofweek) % 7
    first_monday = jan1 + pd.Timedelta(days=offset)
    return first_monday + pd.Timedelta(days=14)


def blue_monday_dates(years) -> pd.DatetimeIndex:
    """Blue-Monday calendar dates for an iterable of years (may be non-trading days)."""
    return pd.DatetimeIndex([third_monday_of_january(int(y)) for y in years])


def is_blue_monday(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask over a DatetimeIndex: True on the exact Blue-Monday calendar dates.

    If Blue Monday itself is a market holiday (e.g. MLK Day — observed on the third Monday of
    January since 1986, with US equity markets closed for it since 1998), the calendar date will
    not appear in a trading-day index and no row is flagged for that year — a deliberately
    conservative convention (we do not roll). The MLK-holiday collision (which strands 30 of 34
    Blue Mondays on non-trading days) is the study's central data limitation; see ``results.md``.
    """
    years = range(int(index.min().year), int(index.max().year) + 1)
    blues = set(blue_monday_dates(years).normalize())
    norm = index.normalize()
    return np.array([d in blues for d in norm])


def is_blue_adjacent(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: the FIRST trading day on/after each Blue-Monday calendar date.

    Because Blue Monday is usually a market holiday (MLK Day), the tradable proxy is the *next*
    open session — typically the Tuesday after Blue Monday. This restores a full ~33-year sample
    (one per year) so the mood claim can actually be tested with power. Named as a proxy: it is
    the market's first chance to price the 'most depressing day', not the day itself.
    """
    years = range(int(index.min().year), int(index.max().year) + 1)
    norm = index.normalize()
    pos_of = {d: i for i, d in enumerate(norm)}
    mask = np.zeros(index.size, dtype=bool)
    sorted_days = norm.sort_values()
    for y in years:
        bm = third_monday_of_january(int(y)).normalize()
        # first trading day on/after bm
        after = sorted_days[sorted_days >= bm]
        if len(after) == 0:
            continue
        # only accept if within a week (guards year boundaries at tape edges)
        if (after[0] - bm).days <= 6:
            mask[pos_of[after[0]]] = True
    return mask


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 40,
    blue_dip: float = 0.0,
    blue_vol_mult: float = 1.0,
    daily_vol: float = 0.010,
    drift: float = 0.08,
    start: str = "1985-01-02",
    seed: int = 547,
) -> tuple:
    """Reproducible daily close series with a tunable Blue-Monday mood anomaly.

    ``blue_dip = 0``: i.i.d. constant-drift tape (null — Blue Monday behaves like any Monday).
    ``blue_dip > 0``: the third Monday of each January has ``blue_dip`` subtracted from its
    return (the mood dip) and its idiosyncratic shock scaled by ``blue_vol_mult`` (the mood
    vol-spike). Everything else is ordinary geometric-random-walk drift + noise.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column on a business-day
    index, exactly the schema ``load_real`` produces.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_years * 252))
    blue = is_blue_monday(idx)

    base_shock = rng.standard_normal(idx.size)
    shock = base_shock.copy()
    if blue_vol_mult != 1.0:
        shock[blue] = base_shock[blue] * blue_vol_mult
    rets = drift / TRADING_DAYS + daily_vol * shock
    if blue_dip != 0.0:
        rets[blue] = rets[blue] - blue_dip

    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"

    truth = {
        "blue_dip": float(blue_dip),
        "blue_vol_mult": float(blue_vol_mult),
        "n_blue": int(blue.sum()),
        "daily_vol": float(daily_vol),
        "drift": float(drift),
        "seed": int(seed),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — yfinance daily SPY total-return closes, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "spy_daily.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-29",
    end: str = "2026-06-27",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
) -> pd.DataFrame:
    """Real daily adjusted-close (total-return) for ``ticker`` as a one-column ``close`` frame.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    raw = _retry(
        lambda: yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False
        )
    )
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("close")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    frame = close.to_frame().dropna()
    os.makedirs(cache_dir, exist_ok=True)
    frame.to_parquet(path)
    return frame


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        col = obj["close"] if "close" in obj.columns else obj.iloc[:, 0]
        arr = np.ascontiguousarray(col.to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, pd.Series):
        arr = np.ascontiguousarray(obj.to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
