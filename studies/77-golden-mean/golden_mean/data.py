"""Data layer for Study 77 (Golden-Mean).

Two tapes, one shape (a tz-aware daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A mean-reversion
  knob (``mean_rev``) lets us dial in whether price respects support/resistance
  levels. ``mean_rev=0`` is a pure random walk — so a test can assert that
  Fibonacci-level bounces only appear when we *plant* the effect, and not on a
  martingale. This is the study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network.
  Daily history is long (~25 years for SPY), giving genuine statistical power
  for an event-study approach.

No look-ahead is baked in here — that discipline lives in ``strategy.py``
(levels are identified from price data strictly before the test window).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Fibonacci ratios used throughout the study
FIB_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.786)

# The "classic" Fibonacci trio most Fibonacci traders quote
FIB_KEY = (0.382, 0.500, 0.618)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    mean_rev: float = 0.0,
    annual_vol: float = 0.18,
    start: str = "2020-01-02",
    seed: int = 77,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable mean-reversion tendency.

    Log returns follow: ``r_t = -mean_rev * (price_t / ema_t - 1) + eps_t``
    with ``eps_t`` i.i.d. normal. The mean-reversion term anchors price back
    toward a slow EMA, simulating the effect that Fibonacci believers claim
    drives bounces at key levels. ``mean_rev`` is the *only* forecastable
    structure in the tape:

    - ``mean_rev = 0``  → a martingale; level-based entries are a fair coin.
    - ``mean_rev > 0``  → mean reversion; prices are pulled back toward the
      EMA, which Fibonacci levels proxy.
    - ``mean_rev < 0``  → momentum; prices accelerate away (anti-Fibonacci).

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    bar_vol = annual_vol / np.sqrt(252)
    eps = rng.normal(0.0, bar_vol, n_days)

    log_price = np.empty(n_days)
    log_price[0] = np.log(100.0)

    # Slow EMA (span=50 trading days) as the "anchor" — Fibonacci believers
    # often measure retracements from peak to trough; we approximate this by
    # using a slow exponential moving average as the attractor.
    alpha_ema = 2.0 / (50 + 1)
    ema_val = 100.0
    ema = np.empty(n_days)
    ema[0] = ema_val

    for t in range(1, n_days):
        px = np.exp(log_price[t - 1])
        ema_val = alpha_ema * px + (1 - alpha_ema) * ema_val
        ema[t] = ema_val
        # Mean-reversion: if price is above the EMA, nudge down; vice versa
        reversion = -mean_rev * (px / ema_val - 1.0) if ema_val > 0 else 0.0
        log_price[t] = log_price[t - 1] + reversion + eps[t]

    close = np.exp(log_price)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, bar_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(5_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "mean_rev": mean_rev,
        "annual_vol": annual_vol,
        "n_days": n_days,
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
    ticker: str,
    period: str = "25y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/``). Daily history stretches back ~25
    years for liquid instruments, giving substantial power for an event-study
    approach. Index timezone is not set (daily bars are date-stamped, not
    intraday).
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
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
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
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
