"""Data layer for Study 420 (Awesome Oscillator).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  (``edge``). The Awesome Oscillator is a *trend / regime* indicator (the spread of a fast
  vs slow SMA of the midpoint price), and the structure a long/flat trend rule can actually
  harvest is **persistent bear markets it can step out of**: ``edge`` controls a two-state
  bull/bear regime separation. At ``edge = 0`` both states have the same drift and vol — a
  single-regime random walk, so the trend rule has nothing to time and cannot beat
  buy-and-hold. At ``edge > 0`` the bear state has negative drift and high vol; a slow trend
  filter falls below zero during the bear and ducks to cash, beating buy-and-hold on Sharpe
  — the **positive control**. ``edge < 0`` inverts the regimes (the rule times them backwards).
  Returns ``(data, truth)``.

- ``load_real`` — the real Yahoo! daily OHLC tape (``yfinance``), **cache-first**: it reads
  the parquet under ``_cache/`` and only touches the network on an explicit cache miss
  (``fetch=True``), retrying a couple of times with a small backoff. Daily history is long
  (20+ years) and free of the 60-day cap that affects sub-hourly bars. ``auto_adjust=True``
  gives split/dividend-adjusted total-return OHLC — essential for a multi-decade race.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the AO/MACD
state is formed on closes up to day *t*, and the position is entered at *t+1* (we proxy the
next open by the next close, the standard daily-data approximation for a slow timing rule).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (positive / null control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 6000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    annual_drift: float = 0.08,
    start: str = "2002-01-02",
    seed: int = 420,
    p_bull_to_bear: float = 0.04,
    p_bear_to_bull: float = 0.20,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a *planted* bull/bear regime.

    A two-state Markov regime drives the daily log-return: a bull state (drift
    ``annual_drift``, vol ``annual_vol``) and a bear state whose drift/vol are pushed
    away from the bull by the ``edge`` knob:

    - ``edge = 0``   → bear ≡ bull (single regime); the trend rule has nothing to time
      and cannot beat buy-and-hold (the **null**).
    - ``edge > 0``   → the bear state has negative drift and elevated vol; a slow trend
      filter (the AO) falls below zero during the bear and steps to cash, **dodging the
      drawdown** and beating buy-and-hold on Sharpe — the **positive control**.
    - ``edge < 0``   → regimes inverted; the rule times them backwards.

    Bars are stamped on consecutive business days. The high/low wick is built around the
    open/close so that HL2 = (high+low)/2 (the AO's input) is well defined. Returns
    ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    drift_bull = annual_drift
    vol_bull = annual_vol
    # The edge separates the bear state from the bull state.
    drift_bear = annual_drift - edge * 0.45      # edge=1 → bear drift ≈ -0.37
    vol_bear = annual_vol + edge * 0.22          # edge=1 → bear vol ≈ 0.38

    d_bull = drift_bull / TRADING_DAYS_PER_YEAR
    d_bear = drift_bear / TRADING_DAYS_PER_YEAR
    s_bull = vol_bull / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_bear = vol_bear / np.sqrt(TRADING_DAYS_PER_YEAR)

    MONTH = 21
    state = 0  # 0 = bull, 1 = bear
    log_ret = np.empty(n_days)
    regime = np.zeros(n_days, dtype=int)
    for i in range(n_days):
        if i % MONTH == 0 and i > 0:
            if state == 0:
                state = 1 if rng.random() < p_bull_to_bear else 0
            else:
                state = 0 if rng.random() < p_bear_to_bull else 1
        regime[i] = state
        if state == 0:
            log_ret[i] = rng.normal(d_bull, s_bull)
        else:
            log_ret[i] = rng.normal(d_bear, s_bear)

    close = 100.0 * np.exp(np.cumsum(log_ret))
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intraday wick: a fraction of the daily vol, symmetric around the bar.
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
        "annual_drift": annual_drift,
        "drift_bear": drift_bear,
        "vol_bear": vol_bear,
        "bear_frac": float(regime.mean()),
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily OHLC, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; **cache-first**.

    Reads the parquet under ``cache_dir`` if present. Network is touched only on an
    explicit ``fetch=True`` *or* a cache miss — and then with a couple of retries and a
    small backoff, after which the result is cached so all re-runs are offline.
    ``auto_adjust=True`` → split/dividend-adjusted total-return OHLC.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:  # network / rate-limit
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(
                f"yfinance returned no daily bars for {ticker}"
                + (f" (last error: {last_err})" if last_err else "")
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.to_datetime(bars.index)
    bars.index.name = "date"
    bars = bars.dropna(subset=["close"])
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
