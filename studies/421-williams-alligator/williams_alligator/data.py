"""Data layer for Study 421 (Williams Alligator).

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  ``edge`` (alias ``trend_persistence``) controls how persistent directional runs are:
  ``edge = 0`` is a pure random walk (a fair coin — the Alligator has nothing real to
  ride), and ``edge > 0`` injects genuine trend persistence (an AR(1) on the *drift*,
  i.e. positive autocorrelation in returns) that a trend-following rule like the
  Alligator *should* be able to harvest. This is the study's null in a bottle and its
  positive control in one function: with ``edge = 0`` the timing rule must NOT beat
  buy-and-hold, and with a large ``edge`` it must.
- ``load_real`` / ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-first.
  Daily bars go back decades, giving high statistical power compared with intraday studies.
  ``load_real`` is cache-first: it reads the parquet if present and only fetches on a miss.

The Alligator's median price = (high + low) / 2, so we carry OHLC. ``auto_adjust=True``
gives split/dividend-adjusted total-return closes — essential for a multi-decade
buy-and-hold comparison.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the three
SMMAs are computed on bars up to *t* AND shifted *forward*, which is conservative; the
position is entered at the *next* bar, one documented lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (null + positive control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    start: str = "2005-01-03",
    seed: int = 421,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a tunable trend-persistence knob.

    The tape is built from **multi-week directional regimes** — runs of ~40 trading days
    that each carry a constant drift of the same sign for a while, the sustained trends a
    trend-following indicator is meant to ride. The ``edge`` knob scales how strong that
    planted drift is relative to the daily noise:

        log_ret_t = mu + ``edge`` * regime_drift_t + daily_vol * eps_t

    - ``edge = 0``   → the regime drift vanishes; returns are i.i.d. noise around a tiny
      constant drift (a near-martingale). The Alligator's fan-out signal is then a fair
      coin and the long/flat timing rule must NOT reliably beat buy-and-hold.
    - ``edge > 0``   → genuine multi-week trends the wake-and-eat rule can harvest; the
      Sharpe-difference t-stat vs buy-and-hold should climb past 2 for a large ``edge``.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    edge = float(max(edge, 0.0))
    mu = 0.0002  # tiny unconditional drift (markets rise slowly over time)
    regime_len = 40           # ~2 trading months per directional run
    regime_drift_mag = 0.0016  # per-day drift magnitude inside a run (before edge scaling)

    regime_drift = np.zeros(n_days)
    i = 0
    while i < n_days:
        L = max(5, int(rng.normal(regime_len, regime_len * 0.4)))
        # symmetric up/down runs: B&H earns ~the constant mu, while a both-sided
        # trend follower can harvest the directional runs themselves.
        sign = rng.choice([-1.0, 1.0])
        regime_drift[i:i + L] = sign * regime_drift_mag
        i += L

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = mu + edge * regime_drift + eps

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 80_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "edge": edge,
        "trend_persistence": edge,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"alligator_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = "1993-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). On rate-limit the caller (``load_real``) retries with a
    small backoff. Uses ``auto_adjust=True`` (split- and dividend-adjusted total-return
    closes) — essential for a multi-decade buy-and-hold comparison.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real({ticker!r}) once (with network) to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
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
    return bars.dropna(subset=["close"])


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Cache-first loader: read the parquet if present, else fetch and cache.

    Re-runs are fully offline once the cache exists. On a cache miss the fetch is retried
    a couple of times with a short backoff to survive transient yfinance rate-limits.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        return fetch_daily(ticker, fetch=False, cache_dir=cache_dir)

    import time

    last = None
    for k in range(retries):
        try:
            return fetch_daily(ticker, start=start, end=end, fetch=True, cache_dir=cache_dir)
        except Exception as exc:  # transient rate-limit / network
            last = exc
            time.sleep(2.0 * (k + 1))
    raise RuntimeError(f"Could not load {ticker} after {retries} tries: {last}")


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
