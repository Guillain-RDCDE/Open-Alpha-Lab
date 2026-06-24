"""Data layer for Study 449 (Renko-Charts).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  The knob is ``trend`` (an AR(1)-style drift-persistence coefficient): at ``trend = 0``
  log returns are a pure random walk (a fair coin — no trend a moving-average crossover
  could harvest), while ``trend > 0`` plants serial correlation in returns that a
  trend-following crossover *can* harvest.  This lets the harness assert that a
  Renko/MA crossover beats buy-and-hold *only* when persistence is planted — the
  positive control that proves the engine can detect an edge.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first so the
  test-suite and the reproducible core never touch the network.  Daily history is long
  (20+ years) and free of the 60-day cap that affects sub-hourly bars.

Renko is the subject: a chart type that strips time and plots a new "brick" only when
price moves a fixed amount (the brick size).  The folk claim is that this noise-filtered
brick series makes a *cleaner* trend signal — so a moving-average crossover on the brick
series should beat the same crossover on raw closes.  We encode that as the tightest
falsifiable mechanical rule (see ``strategy.py``).

No look-ahead is baked in here — that discipline lives in ``strategy.py``: signals are
formed on closes up to *t*, positions held from *t+1* onward (one execution lag).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (planted-edge knob)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 3000,
    trend: float = 0.0,
    annual_vol: float = 0.18,
    annual_drift: float = 0.0,
    start: str = "2005-01-03",
    seed: int = 449,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of return-persistence.

    Log returns follow an AR(1):  ``r_t = trend * r_{t-1} + eps_t + mu``, where ``eps_t``
    is i.i.d. normal with daily standard deviation ``annual_vol / sqrt(252)`` and ``mu``
    is the daily drift ``annual_drift / 252``.  ``trend`` is the *only* forecastable
    structure beyond drift:

    - ``trend = 0``   → returns are i.i.d.; a trend-following crossover is a fair coin.
    - ``trend > 0``   → momentum/persistence a crossover (raw or Renko) can harvest.
    - ``trend < 0``   → mean-reversion; a trend follower is on the *wrong* side.

    Bars are stamped on consecutive business days.  Returns ``(data, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    mu = annual_drift / 252.0
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        r = trend * prev + eps + mu
        log_ret[i] = r
        prev = r - mu  # persistence acts on the de-meaned innovation

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(50_000, 500_000, close.size).astype(float)

    data = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "trend": trend,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "n_days": n_days,
        "seed": seed,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str,
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch=True``.

    On a cache miss with ``fetch=True`` we pull from yfinance (retrying a couple of
    times with a small backoff against rate-limits) and cache the parquet, so every
    re-run after the first is fully offline.  Daily history goes back to 2005 for the
    basket here, a meaningful ~21-year power budget.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
            except Exception:  # noqa: BLE001 — retry transient network errors
                raw = None
            if raw is not None and not raw.empty:
                break
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
