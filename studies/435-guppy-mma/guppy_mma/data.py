"""Data layer for Study 435 (Guppy Multiple MA).

Two tapes, one shape (a tz-naive daily close frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  A two-state Markov bull/bear regime drives the drift; the ``edge`` knob blends the
  world from a flat random walk (``edge = 0`` → the GMMA ribbon has *nothing* to read,
  the null) toward a fully separated regime (``edge > 0`` → sustained trends the ribbon
  *can* ride). The seed is fixed, so the test-suite and the reproducible core are fully
  deterministic and offline. The returned ``truth`` dict records the planted parameters
  (this is the positive control: a harness that can't bank a planted edge proves nothing
  by finding nothing on the real tape).

- ``load_real`` — daily total-return closes from Yahoo! Finance (``yfinance``), cache-first
  into ``_cache/``. ``fetch=False`` (the default) reads the cached parquet and never touches
  the network; on an explicit cache miss with ``fetch=True`` it downloads with a small
  retry/backoff and caches the result. ``auto_adjust=True`` → split/dividend-adjusted
  total-return closes (essential for a multi-decade backtest).

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the ribbon is
formed on closes up to day *t*; the position is acted on the return of day *t+1* (one
``shift``, applied once).
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
    n_years: int = 25,
    edge: float = 0.0,
    drift_bull: float = 0.18,
    drift_bear: float = -0.30,
    vol_bull: float = 0.12,
    vol_bear: float = 0.32,
    p_bull_to_bear: float = 0.06,
    p_bear_to_bull: float = 0.25,
    null_drift: float = 0.06,
    start: str = "2000-01-03",
    seed: int = 435,
) -> tuple[pd.DataFrame, dict]:
    """A daily close series with a tunable trend (the GMMA's only food).

    The ``edge`` knob blends a two-state Markov bull/bear world toward a flat
    random walk:

    - ``edge = 0`` → drift and vol collapse to a single state; the price path is
      a near-martingale and the GMMA ribbon carries no usable information (the null).
    - ``edge = 1`` → full regime separation; sustained bull/bear trends the ribbon
      crossover can ride and the spread can read as "conviction".

    Because the ribbon is a trend filter, the planted structure is *persistent drift*
    (regimes), not mean reversion. Returns ``(data, truth)`` where ``data`` has a
    single ``close`` column (date-indexed) and ``truth`` records the planted
    parameters and the realised bear fraction.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Blend toward the null as edge -> 0. At edge=0 both states collapse to a single
    # mild-positive-drift, single-vol random walk (a martingale-ish tape the ribbon
    # cannot time); at edge=1 the full bull/bear separation is restored.
    stationary_bear = p_bull_to_bear / (p_bull_to_bear + p_bear_to_bull)
    avg_vol = (1 - stationary_bear) * vol_bull + stationary_bear * vol_bear

    d_bull = ((1 - edge) * null_drift + edge * drift_bull) / TRADING_DAYS_PER_YEAR
    d_bear = ((1 - edge) * null_drift + edge * drift_bear) / TRADING_DAYS_PER_YEAR
    s_bull = ((1 - edge) * avg_vol + edge * vol_bull) / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_bear = ((1 - edge) * avg_vol + edge * vol_bear) / np.sqrt(TRADING_DAYS_PER_YEAR)

    MONTH = 21
    regime = np.zeros(n_days, dtype=int)  # 0 = bull, 1 = bear
    state = 0
    for i in range(n_days):
        if i % MONTH == 0 and i > 0:
            if state == 0:
                state = 1 if rng.random() < p_bull_to_bear else 0
            else:
                state = 0 if rng.random() < p_bear_to_bull else 1
        regime[i] = state

    log_ret = np.where(
        regime == 0,
        rng.normal(d_bull, s_bull, n_days),
        rng.normal(d_bear, s_bear, n_days),
    )
    close = 100.0 * np.exp(np.cumsum(log_ret))

    data = pd.DataFrame(
        {"close": close}, index=pd.DatetimeIndex(dates, name="date")
    )
    truth = {
        "edge": edge,
        "drift_bull": drift_bull,
        "drift_bear": drift_bear,
        "vol_bull": vol_bull,
        "vol_bear": vol_bear,
        "p_bull_to_bear": p_bull_to_bear,
        "p_bear_to_bull": p_bear_to_bull,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "bear_frac": float(regime.mean()),
    }
    return data, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo! Finance daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Daily adjusted-close prices for ``ticker``; cache-first.

    ``fetch=False`` (default) reads the cached parquet and never touches the network;
    it raises ``FileNotFoundError`` if the cache is absent. ``fetch=True`` downloads
    via ``yfinance`` (with a short retry/backoff on transient rate-limits) and caches
    the parquet so all subsequent runs are offline. ``auto_adjust=True`` → split- and
    dividend-adjusted total-return closes.

    Returns a frame with a single ``close`` column, tz-naive date index.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily prices for {ticker} at {path}. "
                f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
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
            except Exception:  # noqa: BLE001 — transient network/rate-limit
                pass
            time.sleep(2.0 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker} after {retries} tries")
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


def fingerprint(df: pd.DataFrame) -> str:
    """A short sha1 content fingerprint of a price series (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
