"""Data layer for Study 133 (Crypto-Seasonality).

Two tapes, one shape (a tz-naive daily OHLCV frame for BTC-USD):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``monthly_boost`` knob
  injects extra drift into one target calendar month, so we can ask: *would the engine
  detect the effect if it were planted?* ``monthly_boost=0`` is a pure random walk —
  the null. The single-month effect (12 comparisons on ~11 observations each) is the
  study's statistical power ceiling, and the synthetic world makes that explicit.
- ``fetch_daily`` — the real Yahoo! BTC-USD daily tape (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network. Yahoo
  daily BTC-USD goes back to 2014-09-17, giving ~11 years and ~11 observations per
  calendar month — a tiny effective sample that the study must own.

No look-ahead is baked in: signals are formed on the last close of month *m*; the
seasonal-long trade enters at the open of the first day of month *m+1*.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKER = "BTC-USD"

# Calendar month names (1=Jan .. 12=Dec) — used across the study.
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# The famous seasonal folklore: October ("Uptober") supposedly bullish,
# September supposedly bearish. We test all 12 months impartially.
LORE_MONTHS = {10: "Uptober", 9: "Rektember"}


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 11,
    monthly_boost: float = 0.0,
    boost_month: int = 10,
    daily_vol: float = 0.035,
    annual_drift: float = 0.60,
    start: str = "2014-09-01",
    seed: int = 133,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with an optional single-month return boost.

    Log returns follow daily drift + noise. On every calendar day in ``boost_month``
    (1=Jan..12=Dec) an extra daily return of ``monthly_boost / avg_days_in_month``
    is added, so the total monthly excess is approximately ``monthly_boost`` in
    log-return units. ``monthly_boost=0`` plants no signal — the null — so a test
    can assert the engine detects the planted effect and reads ≈ 0 when absent.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # Use bdate_range to get trading days, but BTC trades 365 days a year.
    # We use a calendar-day index (BTC never closes) but drop weekends for
    # consistency with the real Yahoo tape that occasionally has weekend gaps.
    idx = pd.date_range(start=start, periods=int(n_years * 365), freq="D", name="date")

    daily_mu = annual_drift / 365.0
    log_ret = rng.normal(daily_mu, daily_vol, len(idx))

    # Inject the monthly boost into every day in boost_month.
    avg_days = 30.44  # average calendar days per month
    boost_per_day = monthly_boost / avg_days
    for k, dt in enumerate(idx):
        if dt.month == boost_month:
            log_ret[k] += boost_per_day

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
        "monthly_boost": monthly_boost,
        "boost_month": boost_month,
        "boost_month_name": MONTH_NAMES[boost_month - 1],
        "daily_vol": daily_vol,
        "annual_drift": annual_drift,
        "n_days": len(idx),
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily BTC-USD, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "btc_usd_daily.parquet")


def fetch_daily(
    ticker: str = TICKER,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for BTC-USD; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``). Yahoo daily BTC-USD starts 2014-09-17, giving roughly
    11 years of history and **~11 observations per calendar month** — the study's
    effective-n ceiling that is named loudly rather than papered over.
    """
    path = _cache_path(cache_dir)

    # Fallback: try the study-83 cache if our own cache is absent.
    study83_cache = os.path.abspath(
        os.path.join(HERE, "..", "..", "83-half-life", "_cache", "btc_daily.parquet")
    )

    if not fetch:
        if os.path.exists(path):
            bars = pd.read_parquet(path)
        elif os.path.exists(study83_cache):
            bars = pd.read_parquet(study83_cache)
        else:
            raise FileNotFoundError(
                f"No cached daily tape at {path}. "
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
