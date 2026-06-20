"""Data layer for Study 302 (Lithium-Boom).

Two tapes, one shape (a tz-naive daily OHLCV frame):

- ``synthetic_theme`` — a *deterministic, offline* generator. A thematic price path made of
  three regimes glued together: a long sleepy drift, a parabolic **boom**, then a slow
  **crash** that gives most of the boom back. A single knob, ``trend_strength``, dials how
  *persistent* daily returns are (a positive AR(1) on log returns) — the only thing a trend
  / time-series-momentum rule can possibly harvest. ``trend_strength=0`` is a pure random
  walk with the *same* boom-bust envelope, so a test can assert that the rule beats
  buy-and-hold *only* when there is genuine autocorrelation to ride, and not merely because
  the price went up then down. This is the study's positive control and its null in one.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``, ``auto_adjust=True``,
  total-return), cache-only by default so the test-suite and the reproducible core never
  touch the network.

The thematic basket: **LIT** (Global X Lithium & Battery Tech ETF, the headline theme),
plus two pure-play miners **ALB** (Albemarle) and **SQM** (Sociedad Química y Minera) for
breadth, and **SPY** as the benchmark a believer would have to beat.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (one ``shift``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The thematic basket and the benchmark.
THEME_TICKERS = ["LIT", "ALB", "SQM"]
BENCH_TICKER = "SPY"
ALL_TICKERS = THEME_TICKERS + [BENCH_TICKER]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_theme(
    n_days: int = 4000,
    trend_strength: float = 0.0,
    daily_vol: float = 0.022,
    start: str = "2010-07-22",
    seed: int = 302,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a *boom-bust envelope* and tunable trend.

    The price has three deterministic regimes (a believer's caricature of the battery-metals
    decade): a long flat-ish drift, a parabolic boom, then a protracted crash that hands most
    of the boom back. On top of that envelope, daily log-returns carry an AR(1) component:

        r_t = drift_t + trend_strength * (r_{t-1} - drift_{t-1}) + eps_t

    so ``trend_strength`` is the only thing a trend rule can exploit:

    - ``trend_strength = 0``   → a random walk *with the same boom-bust shape*; a trend rule
      cannot beat buy-and-hold here except by luck (and timing the crash is then a coin flip).
    - ``trend_strength > 0``   → genuine momentum/persistence the rule can ride and, crucially,
      can use to *step aside* during the crash leg.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Deterministic drift envelope: sleepy -> boom -> crash, in fractions of the sample.
    t = np.arange(n_days) / max(n_days - 1, 1)
    drift = np.empty(n_days)
    boom_lo, boom_hi = 0.45, 0.70  # the parabolic boom window
    for i in range(n_days):
        x = t[i]
        if x < boom_lo:
            drift[i] = 0.0001                      # sleepy positive drift
        elif x < boom_hi:
            drift[i] = 0.0016                      # the boom: strong positive drift
        else:
            drift[i] = -0.0011                     # the crash: give it back, slowly

    log_ret = np.empty(n_days)
    prev_excess = 0.0
    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        excess = trend_strength * prev_excess + eps
        log_ret[i] = drift[i] + excess
        prev_excess = excess

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(100_000, 5_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "trend_strength": trend_strength,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
        "boom_window": (boom_lo, boom_hi),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2010-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True`` (total-return).

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). ``auto_adjust=True`` gives a total-return (dividends +
    splits) series — labelled total-return everywhere it is used. The trailing in-progress
    bar (a same-day NaN close) is dropped, so the tape never carries a partial session.
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
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars = bars.dropna(subset=["close"])  # drop the in-progress trailing bar
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars = bars.dropna(subset=["close"])
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
