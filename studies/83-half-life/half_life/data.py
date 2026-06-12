"""Data layer for Study 83 (Half-Life).

Two tapes, one shape (a tz-naive daily OHLCV frame for BTC-USD):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``halving_boost`` knob
  injects a post-halving drift into the daily log-returns for a configurable window after
  each planted halving date. ``halving_boost=0`` is a pure random walk — the null
  hypothesis. ``halving_boost > 0`` plants exactly the effect the halving thesis claims.
  Only 4 halvings fit in BTC's history (2012, 2016, 2020, 2024), so the synthetic world
  lets us ask: *even if the effect were real*, would 4 events give us statistical power?
- ``fetch_daily`` — the real Yahoo! BTC-USD daily tape (``yfinance``), cache-only by
  default so the test-suite never touches the network.

No look-ahead: signals are formed on the halving date *t*; the 'buy after halving' trade
enters at *t+1*'s open.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The four Bitcoin halving dates (UTC midnight of the block day).
HALVING_DATES = [
    pd.Timestamp("2012-11-28"),  # block 210,000 — reward 50→25 BTC
    pd.Timestamp("2016-07-09"),  # block 420,000 — reward 25→12.5 BTC
    pd.Timestamp("2020-05-11"),  # block 630,000 — reward 12.5→6.25 BTC
    pd.Timestamp("2024-04-20"),  # block 840,000 — reward 6.25→3.125 BTC
]

TICKER = "BTC-USD"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 12,
    halving_boost: float = 0.0,
    boost_window: int = 365,
    daily_vol: float = 0.04,
    annual_drift: float = 0.50,
    start: str = "2012-01-01",
    seed: int = 83,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with an optional post-halving return boost.

    Log returns follow a daily drift-plus-noise process. During the ``boost_window``
    calendar days after each planted halving date, an extra daily drift of
    ``halving_boost / boost_window`` is added — so the total 'event effect' is
    ``halving_boost`` log-return units spread linearly over the window. The control
    case is ``halving_boost=0``: the halvings are structurally present (known dates)
    but carry *zero* additional return signal — only secular drift. This is the study's
    null in a bottle.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters, the
    halving dates used, and n_halvings_in_sample.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_years * 252), name="date")

    daily_mu = annual_drift / 252.0

    # Build daily log-returns with optional post-halving boost
    log_ret = rng.normal(daily_mu, daily_vol, len(idx))

    # Inject boost after each halving date
    for hd in HALVING_DATES:
        boost_per_day = halving_boost / boost_window
        for k, dt in enumerate(idx):
            if hd <= dt < hd + pd.Timedelta(days=boost_window):
                log_ret[k] += boost_per_day

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, daily_vol * 0.4, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(100_000, 5_000_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=idx,
    )

    halvings_in_sample = [h for h in HALVING_DATES if idx[0] <= h <= idx[-1]]
    truth = {
        "halving_boost": halving_boost,
        "boost_window": boost_window,
        "daily_vol": daily_vol,
        "annual_drift": annual_drift,
        "n_days": len(idx),
        "seed": seed,
        "halvings_in_sample": halvings_in_sample,
        "n_halvings": len(halvings_in_sample),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily BTC-USD, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "btc_daily.parquet")


def fetch_daily(
    ticker: str = TICKER,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for BTC-USD; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``). Daily BTC history from Yahoo goes back to 2014-09-17,
    giving us halvings #2 (2016), #3 (2020), and #4 (2024) in full; halving #1 (2012)
    is outside Yahoo's range. That means at most **3 full post-halving windows** are
    observable on the real tape — n=3 events, a power floor the study must own.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape at {path}. "
                f"Call fetch_daily(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
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
        # Strip timezone so index is tz-naive throughout
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
