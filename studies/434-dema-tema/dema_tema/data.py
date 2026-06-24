"""Data layer for Study 434 — DEMA & TEMA (double / triple exponential moving averages).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

* **Synthetic.** ``synthetic_panel`` — a deterministic, fixed-seed generator with a
  *planted-edge* knob (``edge``).  The price path is a random walk plus an optional
  slow, persistent **trend component** whose strength is ``edge``.  At ``edge = 0`` the
  log-return series is a pure martingale — any moving-average trend rule (SMA / DEMA /
  TEMA alike) is then a fair coin and must NOT manufacture a Sharpe.  At ``edge > 0`` a
  genuine low-frequency trend is present, which a faster-reacting smoother (DEMA / TEMA)
  *should* harvest a touch better than a laggy SMA — the positive control that proves
  the harness can tell the smoothers apart when there is something to find.

* **Real tape.** ``load_real`` — the Yahoo daily tape (``yfinance``), **cache-first**:
  it reads a cached parquet under ``_cache/`` and only touches the network on an explicit
  cache miss (with a small retry/backoff), then caches the parquet so re-runs are offline.
  Daily history is long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the trend
signal is formed on closes up to *t*, and the position earns the return of *t+1*
(one documented ``shift``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2006-01-03",
    seed: int = 434,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a *planted* low-frequency trend.

    The log return is ``r_t = mu_t + eps_t`` where ``eps_t`` is i.i.d. normal with daily
    standard deviation ``annual_vol / sqrt(252)`` and ``mu_t`` is a slow, mean-reverting
    **trend state** (an Ornstein-Uhlenbeck process) scaled by ``edge``:

    - ``edge = 0``  → ``mu_t = 0``: a pure martingale.  Every MA trend rule is a fair coin;
      no smoother (SMA, DEMA, TEMA) can bank a Sharpe out of noise.
    - ``edge > 0``  → a persistent drift that *flips sign slowly*.  A trend-following rule
      can harvest it, and a faster-reacting smoother (less lag) catches the turns sooner —
      the exact thing the DEMA/TEMA claim asserts.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Slow OU trend state: mean 0, slow reversion -> persistent regimes that flip sign.
    phi = 0.985                       # high persistence => long trends
    trend_innov = 0.0006              # innovation scale of the trend state
    mu = np.zeros(n_days)
    s = 0.0
    for i in range(n_days):
        s = phi * s + rng.normal(0.0, trend_innov)
        mu[i] = edge * s

    log_ret = mu + rng.normal(0.0, daily_vol, size=n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(50_000, 500_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "edge": edge,
        "annual_vol": annual_vol,
        "n_days": n_days,
        "seed": seed,
        "trend_persistence": phi,
    }
    return bars, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def _normalise(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = True,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; **cache-first**.

    If a cached parquet exists it is returned with no network access.  On a cache miss,
    and only if ``fetch=True``, the bars are downloaded from Yahoo! (``yfinance``,
    ``auto_adjust=True`` -> total-return-ish adjusted closes), retried a couple of times
    with a small backoff, then cached as a parquet so all later runs are offline.

    Set ``fetch=False`` to force cache-only (raises on a miss) — used by the
    reproducible core and tests so they never touch the network.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        return _normalise(pd.read_parquet(path))
    if not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )
    import yfinance as yf  # lazy: only on a real cache miss

    raw = None
    for k in range(retries):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                break
        except Exception:
            raw = None
        time.sleep(1.5 * (k + 1))
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    bars = _normalise(bars)
    os.makedirs(cache_dir, exist_ok=True)
    bars.to_parquet(path)
    return bars


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
