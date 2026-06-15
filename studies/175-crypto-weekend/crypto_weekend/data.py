"""Data layer for Study 175 (Crypto-Weekend).

Two tapes, one shape (a tz-naive daily OHLCV frame for BTC-USD):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``weekend_boost`` knob
  injects extra drift into Saturday and Sunday, so we can ask: *would the engine detect a
  weekend effect if it were planted?* ``weekend_boost=0`` is a pure random walk — the
  null. This is the study's positive control: the machine should recover the effect when
  it exists, and read nothing when it does not.
- ``fetch_daily`` — the real Yahoo! BTC-USD daily tape (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network. Yahoo
  daily BTC-USD goes back to 2014-09-17, giving ~11.5 years of daily history and
  ~600 weekends — far more than a typical calendar anomaly, though the post-2023
  "banking-day normalisation" sub-sample test shrinks that considerably.

BTC trades 24/7 (no missing weekends), so unlike equity studies there is no survivorship
or calendar-gap issue on the weekend side. The real question is whether the handful of
bps difference in mean returns survives HAC inference and a pre/post-2023 sub-sample
split.

No look-ahead is baked in here — signals use only the calendar date, which is known
in advance; positions are entered at the open of the signalled day.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKER = "BTC-USD"

# The commonly cited "banking normalisation" break-point: Silvergate Bank collapsed
# in March 2023, followed by Signature Bank and Silicon Valley Bank shortly after.
# From roughly 2023-Q2 onward traditional banking rails for stablecoin settlement
# began operating on weekends (FedNow launched July 2023). We test pre/post.
# Source: Auer, Cornelli & Frost (2022); Federal Reserve FedNow press release 2023-07-20.
REGIME_BREAK = pd.Timestamp("2023-01-01")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 11,
    weekend_boost: float = 0.0,
    daily_vol: float = 0.035,
    annual_drift: float = 0.60,
    start: str = "2014-09-17",
    seed: int = 175,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily BTC-like OHLCV tape with an optional weekend-return boost.

    Log returns follow daily drift + i.i.d. noise. On every Saturday and Sunday an extra
    ``weekend_boost`` is added to the daily log-return (so the weekly excess over the
    five weekday days is approximately ``2 * weekend_boost`` in log-return units).
    ``weekend_boost=0`` plants no signal — the null — confirming the engine reads zero
    when nothing is there and finds the planted effect when something is.

    BTC trades every calendar day, so the index is a full calendar-day range. Returns
    ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=int(n_years * 365), freq="D", name="date")

    daily_mu = annual_drift / 365.0
    log_ret = rng.normal(daily_mu, daily_vol, len(idx))

    # Inject the weekend boost on Saturdays (5) and Sundays (6).
    for k, dt in enumerate(idx):
        if dt.dayofweek >= 5:
            log_ret[k] += weekend_boost

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, daily_vol * 0.35, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=idx,
    )
    truth = {
        "weekend_boost": weekend_boost,
        "daily_vol": daily_vol,
        "annual_drift": annual_drift,
        "n_days": len(idx),
        "seed": seed,
        "n_weekends": int((idx.dayofweek >= 5).sum()),
        "n_weekdays": int((idx.dayofweek < 5).sum()),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily BTC-USD, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "btc_usd_daily.parquet")


# Fallback: reuse the study-133 cache if our own is absent.
_STUDY_133_CACHE = os.path.abspath(
    os.path.join(
        HERE, "..", "..", "133-crypto-seasonality", "_cache", "btc_usd_daily.parquet"
    )
)


def fetch_daily(
    ticker: str = TICKER,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for BTC-USD; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``). Yahoo daily BTC-USD starts 2014-09-17, giving roughly
    11.5 years of history and approximately 600 weekend days — a considerably larger
    effective-n than most calendar anomaly studies, though the post-2023 sub-sample
    shrinks it to ~260 weekend days, which the study must own.
    """
    path = _cache_path(cache_dir)

    if not fetch:
        if os.path.exists(path):
            bars = pd.read_parquet(path)
        elif os.path.exists(_STUDY_133_CACHE):
            bars = pd.read_parquet(_STUDY_133_CACHE)
        else:
            raise FileNotFoundError(
                f"No cached daily BTC-USD tape at {path}. "
                f"Call fetch_daily(fetch=True) once to populate the cache."
            )
    else:
        import yfinance as yf  # lazy import: only when we go to the network

        raw = yf.download(
            ticker, period="max", interval="1d", auto_adjust=True, progress=False
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
    return bars.sort_index()


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
