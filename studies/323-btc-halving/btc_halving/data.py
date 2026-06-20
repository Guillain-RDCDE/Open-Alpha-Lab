"""Data layer for Study 323 (BTC-Halving).

The folk thesis: Bitcoin moves in a clean **four-year cycle locked to the
mining "halving"** (the protocol event, roughly every 210,000 blocks, that cuts
the block subsidy in half). The lore says the cycle bottoms a bit more than a
year *before* each halving, tops ~12-18 months *after* it, then crashes into the
next bottom — so if you knew the halving date you would (the story goes) have
known when to buy and when to sell.

Three pieces, all offline for the deterministic core:

- ``HALVINGS`` — the four hardcoded historical halving dates (2012-11-28,
  2016-07-09, 2020-05-11, 2024-04-20). These are public, fixed protocol facts;
  hardcoding them keeps the core fully offline and reproducible.

- ``synthetic_daily`` — a deterministic, offline generator of a daily BTC-like
  log-price path with a tunable **cyclic** component locked to a 4-year period
  and *phased to the halving dates*. The ``cycle_amp`` knob plants the lore
  ("price rides a halving-locked sinusoid"); ``cycle_amp = 0`` is the null (a
  pure geometric random walk with drift — no halving structure at all). This is
  the study's positive control AND its null in one bottle: the harness must
  detect the planted cycle when present and find nothing when it is absent.

- ``load_real`` / ``fetch_daily`` — the real Yahoo! ``BTC-USD`` daily close,
  cache-first under ``_cache/btc_usd_daily.parquet`` so the test-suite and the
  reproducible core never touch the network. Falls back gracefully.

**The honest data constraint (named loudly):** Yahoo's ``BTC-USD`` history
*starts 2014-09-17*. So of the four halvings, the **2012 and 2016 events are not
covered** by clean daily data here, the **2020 and 2024** halvings are. A
"4-cycle study" on Yahoo is really a **2-cycle study** (2020, 2024) plus a
partial pre-2016 tail — a tiny *n* that no amount of HAC machinery can rescue.
That scarcity is itself a finding, and it caps the Signal stamp.

No look-ahead in here — the cycle phase at day *t* is a calendar fact known in
advance (the halving schedule is deterministic), so the timing rule in
``strategy.py`` needs no execution lag for the *signal*, only the one-bar fill
lag documented there.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
BTC_CACHE = os.path.join(DEFAULT_CACHE, "btc_usd_daily.parquet")

# Bitcoin block-subsidy halving dates (UTC calendar day of the event).
# Fixed protocol facts; hardcoded to keep the core offline and reproducible.
HALVINGS = (
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
)

# The folk cycle length: ~4 years between halvings (210k blocks @ ~10 min).
CYCLE_DAYS = 1458  # ~3.99 years; the median halving-to-halving gap.
TRADING_DAYS_PER_YEAR = 365  # crypto trades every calendar day.


# ---------------------------------------------------------------------------
# Cycle phase — a calendar-known position within the 4-year halving cycle
# ---------------------------------------------------------------------------
def days_since_last_halving(dates: pd.DatetimeIndex,
                            halvings=HALVINGS) -> np.ndarray:
    """For each date, calendar days since the most recent halving on/before it.

    Dates before the first halving return ``NaN`` (no defined cycle phase yet).
    This is a purely deterministic calendar fact — knowable arbitrarily far in
    advance — which is exactly why the lore feels like a free lunch.
    """
    # Normalise both sides to nanosecond ints so a ms- or ns-resolution index
    # compares correctly against the (ns) halving timestamps.
    d = pd.DatetimeIndex(dates).as_unit("ns")
    hv = np.array([pd.Timestamp(h).as_unit("ns").value
                   for h in sorted(halvings)], dtype="int64")
    out = np.full(len(d), np.nan)
    di = d.asi8
    for i, t in enumerate(di):
        prior = hv[hv <= t]
        if prior.size:
            out[i] = (t - prior[-1]) / (1e9 * 86400.0)  # ns -> days
    return out


def cycle_phase(dates: pd.DatetimeIndex, halvings=HALVINGS,
                cycle_days: int = CYCLE_DAYS) -> np.ndarray:
    """Fractional phase in [0, 1) within the halving cycle (0 = at a halving).

    ``NaN`` before the first halving. Phase wraps every ``cycle_days``.
    """
    dsl = days_since_last_halving(dates, halvings)
    return (dsl % cycle_days) / cycle_days


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (positive control + null)
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 4200,
    cycle_amp: float = 0.0,
    drift: float = 0.0008,
    daily_vol: float = 0.04,
    start: str = "2014-09-17",
    halvings=HALVINGS,
    cycle_days: int = CYCLE_DAYS,
    seed: int = 323,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV BTC-like tape with a planted halving cycle.

    The log-price is a geometric random walk with drift, plus a *deterministic
    sinusoidal* term locked to the halving schedule:

        log P_t = drift * t + sigma * W_t
                  + cycle_amp * sin(2*pi * phase_t)

    where ``phase_t`` is the fractional position in the 4-year halving cycle
    (``cycle_phase``). The sine peaks a quarter-cycle (~1 year) after each
    halving and bottoms a quarter-cycle before the next — the textbook lore.

    - ``cycle_amp = 0``   → pure GBM; **no** halving structure (the null).
    - ``cycle_amp > 0``   → a clean, detectable halving-locked cycle.

    Returns ``(bars, truth)`` recording the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # Use a real DatetimeIndex of calendar days (crypto trades daily). Span is a
    # few thousand days — far under the ns-Timestamp overflow ceiling.
    idx = pd.date_range(start=start, periods=n_days, freq="D", name="date")

    eps = rng.normal(0.0, daily_vol, n_days)
    rw = np.cumsum(eps)
    trend = drift * np.arange(n_days)
    phase = cycle_phase(idx, halvings, cycle_days)
    # Before the first halving in range, phase is NaN -> treat as 0 cyclic term.
    cyc = np.where(np.isnan(phase), 0.0, np.sin(2.0 * np.pi * np.nan_to_num(phase)))
    log_p = trend + rw + cycle_amp * cyc
    log_p = log_p - log_p[0]  # start at 0 -> price 1.0 baseline below
    close = 10000.0 * np.exp(log_p)

    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000, 5_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=idx,
    )
    truth = {
        "cycle_amp": cycle_amp,
        "drift": drift,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "cycle_days": cycle_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo BTC-USD daily, cache-first
# ---------------------------------------------------------------------------
def load_real(cache_path: str = BTC_CACHE) -> pd.DataFrame:
    """Cache-first real BTC-USD daily OHLCV (no network). Raises if no cache.

    Thin wrapper that the notebooks/results call: it only ever reads the cached
    parquet. To populate or refresh, call :func:`fetch_daily` with ``fetch=True``.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached BTC-USD daily tape at {cache_path}. "
            f"Call fetch_daily(fetch=True) once to populate it."
        )
    bars = pd.read_parquet(cache_path)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def fetch_daily(
    ticker: str = "BTC-USD",
    start: str = "2014-01-01",
    fetch: bool = False,
    cache_path: str = BTC_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is then
    cached). Defaults to BTC-USD. Yahoo's BTC-USD history begins 2014-09-17.
    """
    if not fetch:
        return load_real(cache_path)

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(ticker, start=start, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    bars.to_parquet(cache_path)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """Short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
