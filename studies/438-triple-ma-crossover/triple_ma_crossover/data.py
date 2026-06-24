"""Data layer for Study 438 (Triple-MA-Crossover).

Two tapes, one shape (a tz-naive daily close frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  Price follows a two-state Markov regime (a trending bull and a choppy bear); the
  ``edge`` knob dials how separable the regimes are. At ``edge = 0`` the log-return
  series is a pure random walk (a fair coin) — any moving-average ribbon timing rule
  must then be indistinguishable from cash-adjusted buy-and-hold. At ``edge > 0`` a
  persistent, trend-following structure appears that a fast>mid>slow MA stack can ride.
  This is the positive control: zero edge must NOT manufacture significance; a planted
  edge must light up.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first** by default
  so the test-suite and the reproducible core never touch the network. Daily history is
  long (30+ years on ^GSPC) and free of the 60-day cap that affects intraday bars.
  ``auto_adjust=True`` gives split/dividend-adjusted **total-return** closes.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the MA stack
is read on the close of day *t*, and the resulting position earns the return of *t+1*
(one ``shift``, applied once).
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
# Synthetic tape — the deterministic offline core (positive control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 8000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "1994-01-03",
    seed: int = 438,
    p_stay_bull: float = 0.985,
    p_stay_bear: float = 0.965,
    drift_bull: float = 0.18,
    drift_bear: float = -0.20,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close tape with a known amount of trend structure.

    Two latent regimes drive the daily drift:

    - ``edge = 0``   → both regimes collapse to a single zero-drift state; log returns are
      i.i.d. (a martingale). Any MA-ribbon timing rule is a fair coin — no false positive
      is allowed.
    - ``edge > 0``   → a persistent bull (positive drift) / bear (negative drift) Markov
      chain. ``edge`` scales the drift separation, so a trend-following fast>mid>slow stack
      can lean into the bull and step aside in the bear.

    Returns ``(data, truth)`` where ``data`` has a single ``close`` column and ``truth``
    records the planted parameters and the realised bear fraction.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)

    d_bull = edge * drift_bull / TRADING_DAYS_PER_YEAR
    d_bear = edge * drift_bear / TRADING_DAYS_PER_YEAR

    regime = np.zeros(n_days, dtype=int)  # 0 = bull, 1 = bear
    state = 0
    for i in range(n_days):
        if state == 0:
            state = 0 if rng.random() < p_stay_bull else 1
        else:
            state = 1 if rng.random() < p_stay_bear else 0
        regime[i] = state

    drift = np.where(regime == 0, d_bull, d_bear)
    log_ret = drift + rng.normal(0.0, daily_vol, n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    data = pd.DataFrame(
        {"close": close}, index=pd.DatetimeIndex(dates, name="date")
    )
    truth = {
        "edge": edge,
        "annual_vol": annual_vol,
        "n_days": n_days,
        "seed": seed,
        "bear_frac": float(regime.mean()),
        "drift_bull": drift_bull,
        "drift_bear": drift_bear,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"close_{safe}_1d.parquet")


def load_real(
    ticker: str = "^GSPC",
    start: str = "1993-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily close for ``ticker``; **cache-first**, network only on a cache miss.

    With ``fetch=False`` (default) the parquet under ``_cache/`` is read if present, and a
    missing cache raises ``FileNotFoundError`` so the offline core never silently hits the
    network. With ``fetch=True`` the tape is downloaded (a few retries with backoff on
    rate-limit) and cached. ``auto_adjust=True`` → split/dividend-adjusted total-return
    closes, which matters for a multi-decade backtest.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch and os.path.exists(path):
        df = pd.read_parquet(path)
    elif not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(2.0 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df


def have_real(ticker: str = "^GSPC", cache_dir: str = DEFAULT_CACHE) -> bool:
    """True if the real-tape cache for ``ticker`` is present (no network)."""
    return os.path.exists(_cache_path(ticker, cache_dir))


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
