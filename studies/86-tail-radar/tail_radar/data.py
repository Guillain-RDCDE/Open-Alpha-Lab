"""Data layer for Study 86 (Tail-Radar).

Two tapes, one shape (a tz-naive daily OHLCV/Close frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``skew_signal`` knob
  controls how much predictive information the synthetic SKEW analogue carries about
  forward SPY-like returns.  ``skew_signal=0`` produces a pure noise SKEW with no
  forecasting power — the null in a bottle.  ``skew_signal>0`` plants a genuine
  negative correlation between SKEW percentile and the next-day return, so the test
  machinery can be verified against a positive control.

- ``fetch_daily`` — the real Yahoo! daily closes for ``^SKEW``, ``^VIX``, and ``SPY``,
  cache-only by default so the test-suite and the reproducible core never touch the
  network.  Daily history is long (SKEW goes back to 1990 on Yahoo), giving us real
  statistical power.

No look-ahead is baked in here — SKEW and VIX measured at close of day *t* are only
used to form signals that trade at close of day *t+1* or later (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default tickers for the real-tape fetch.
TICKERS = ["^SKEW", "^VIX", "SPY"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 2000,
    skew_signal: float = 0.0,
    spy_vol: float = 0.01,
    start: str = "2012-01-03",
    seed: int = 86,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of SKEW predictive power.

    The synthetic tape mimics three daily series — SKEW, VIX, and SPY forward
    returns — with the following structure:

    - ``SKEW_synth`` is drawn from a bounded uniform [100, 150] with autocorrelation
      (AR(1) with coefficient 0.97) — matching the sluggish, mean-reverting real SKEW.
    - ``VIX_synth`` is an independent AR(1) process (coefficient 0.95) in log-space,
      roughly matching the 10–80 range of real VIX.
    - ``SPY_ret`` is generated as::

          spy_ret_t = -skew_signal * skew_rank_{t-1} + eps_t

      where ``skew_rank`` is the rolling percentile (out-of-sample) of SKEW over the
      last 252 days, and ``eps_t`` is i.i.d. normal of standard deviation ``spy_vol``.
      ``skew_signal=0`` makes SPY a pure random walk; ``skew_signal>0`` plants a genuine
      *negative* relationship between high SKEW and forward SPY returns.

    Returns ``(df, truth)`` where ``df`` has columns
    ``[SKEW, VIX, SPY_close, SPY_ret]`` indexed by ``pd.DatetimeIndex`` (tz-naive
    business days), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # --- Synthetic SKEW: bounded AR(1) in [100, 150] ---
    skew = np.empty(n_days)
    skew[0] = 120.0
    phi_skew = 0.97
    sigma_skew = 3.5
    for i in range(1, n_days):
        skew[i] = phi_skew * skew[i - 1] + (1 - phi_skew) * 120.0 + rng.normal(0, sigma_skew)
        skew[i] = np.clip(skew[i], 100.0, 160.0)

    # --- Synthetic VIX: AR(1) in log space, range ~[10, 80] ---
    log_vix = np.empty(n_days)
    log_vix[0] = np.log(18.0)
    phi_vix = 0.95
    sigma_vix = 0.08
    for i in range(1, n_days):
        log_vix[i] = phi_vix * log_vix[i - 1] + (1 - phi_vix) * np.log(18.0) + rng.normal(0, sigma_vix)
    vix = np.exp(log_vix)

    # --- Rolling SKEW percentile rank (252-day lookback) ---
    skew_s = pd.Series(skew, index=idx)
    skew_rank = skew_s.rolling(252, min_periods=63).rank(pct=True)

    # --- Synthetic SPY returns: -skew_signal * lagged_rank + noise ---
    eps = rng.normal(0.0, spy_vol, n_days)
    spy_ret = np.empty(n_days)
    spy_ret[0] = eps[0]
    for i in range(1, n_days):
        lagged_rank = skew_rank.iloc[i - 1]
        if np.isnan(lagged_rank):
            spy_ret[i] = eps[i]
        else:
            spy_ret[i] = -skew_signal * (lagged_rank - 0.5) + eps[i]

    spy_close = 200.0 * np.exp(np.cumsum(spy_ret))

    df = pd.DataFrame(
        {"SKEW": skew, "VIX": vix, "SPY_close": spy_close, "SPY_ret": spy_ret},
        index=idx,
    )
    truth = {
        "skew_signal": skew_signal,
        "spy_vol": spy_vol,
        "n_days": n_days,
        "seed": seed,
        "start": start,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily closes, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    """Canonical parquet path for the combined SKEW/VIX/SPY daily cache."""
    return os.path.join(cache_dir, "daily_skew_vix_spy.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1990-01-01",
) -> pd.DataFrame:
    """Real daily SKEW, VIX and SPY data; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  With ``fetch=False`` (the default) a
    ``FileNotFoundError`` is raised if the cache is absent — consistent with the desk
    convention that tests and the reproducible core must never hit the network.

    The returned frame has columns ``[SKEW, VIX, SPY_close, SPY_ret]`` with a tz-naive
    ``DatetimeIndex`` named ``date``.  Rows where SKEW or SPY is NaN are dropped.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape at {path}. "
                f"Call fetch_daily(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            TICKERS,
            start=start,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no daily bars")
        # Multi-ticker download returns a MultiIndex — flatten to Close only.
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw[["Close"]].copy()

        closes.columns = [c.replace("^", "") for c in closes.columns]
        closes = closes.rename(columns={"SKEW": "SKEW", "VIX": "VIX", "SPY": "SPY_close"})
        closes.index.name = "date"
        closes = closes[["SKEW", "VIX", "SPY_close"]].dropna(subset=["SKEW", "SPY_close"])
        closes["SPY_ret"] = np.log(closes["SPY_close"]).diff()
        closes = closes.dropna(subset=["SPY_ret"])
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)

        os.makedirs(cache_dir, exist_ok=True)
        closes.to_parquet(path)
        df = closes

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a daily tape (SPY_close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["SPY_close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
