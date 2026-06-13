"""Data layer for Study 111 (VIX-Term-Structure).

Two tapes, one shape (a tz-naive daily frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``contango_signal``
  knob controls how much predictive power the synthetic VIX/VIX3M slope carries about
  forward SPY-like returns.  ``contango_signal=0`` is the null: the slope carries no
  information and the timing signal is a fair coin.  ``contango_signal>0`` plants a
  genuine positive relationship between contango (VIX < VIX3M) and forward returns,
  so the test machinery can be verified against a planted effect.

- ``fetch_daily`` — the real Yahoo! daily closes for ``^VIX``, ``^VIX3M``, and
  ``SPY``, cache-only by default so the test-suite and the reproducible core never
  touch the network.  ``^VIX3M`` was introduced by CBOE in January 2008, giving us
  ~17 years of daily history — far more power than the 5-minute studies.

No look-ahead is baked in here — VIX and VIX3M measured at the close of day *t* are
only used to form signals that trade at the close of day *t+1* or later (see
``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default real-tape tickers.
TICKERS = ["^VIX", "^VIX3M", "SPY"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 3000,
    contango_signal: float = 0.0,
    spy_vol: float = 0.01,
    start: str = "2008-01-03",
    seed: int = 111,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of VIX-slope predictive power.

    The synthetic tape mimics three daily series — VIX, VIX3M, and SPY forward returns
    — with the following structure:

    - ``VIX_synth`` is an AR(1) process in log-space (coefficient 0.95, mean-reverting
      around log(18)), roughly matching the 10–80 range and persistence of real VIX.
    - ``VIX3M_synth`` is a *smoother* AR(1) process (coefficient 0.97) that reverts to
      the same long-run mean.  The term structure is naturally contango (VIX < VIX3M)
      most of the time, with VIX spikes creating transient backwardation.
    - ``SPY_ret`` is generated as::

          spy_ret_t = contango_signal * slope_rank_{t-1} + eps_t

      where ``slope_rank`` is the rolling percentile (out-of-sample) of the slope
      ``log(VIX3M/VIX)`` over the last 252 days, and ``eps_t`` is i.i.d. normal of
      standard deviation ``spy_vol``.  ``contango_signal=0`` makes SPY a pure random
      walk; ``contango_signal>0`` plants a genuine *positive* relationship between
      contango (positive slope) and forward SPY returns.

    Returns ``(df, truth)`` where ``df`` has columns
    ``[VIX, VIX3M, slope, SPY_close, SPY_ret]`` indexed by a tz-naive
    ``pd.DatetimeIndex`` of business days, and ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # --- Synthetic VIX: AR(1) in log space, range ~[10, 80] ---
    log_vix = np.empty(n_days)
    log_vix[0] = np.log(18.0)
    phi_vix = 0.95
    sigma_vix = 0.08
    for i in range(1, n_days):
        log_vix[i] = phi_vix * log_vix[i - 1] + (1 - phi_vix) * np.log(18.0) + rng.normal(0, sigma_vix)
    vix = np.exp(log_vix)

    # --- Synthetic VIX3M: smoother AR(1) with mean = long-run VIX + small premium ---
    # VIX3M reverts to log(20) — slightly above VIX long-run of log(18) —
    # producing natural contango on average.
    log_vix3m = np.empty(n_days)
    log_vix3m[0] = np.log(20.0)
    phi_vix3m = 0.97
    sigma_vix3m = 0.05
    for i in range(1, n_days):
        log_vix3m[i] = (phi_vix3m * log_vix3m[i - 1]
                        + (1 - phi_vix3m) * np.log(20.0)
                        + rng.normal(0, sigma_vix3m))
    vix3m = np.exp(log_vix3m)

    # --- Slope: log(VIX3M / VIX) > 0 is contango, < 0 is backwardation ---
    slope = np.log(vix3m / vix)

    # --- Rolling slope percentile rank (252-day lookback, out-of-sample) ---
    slope_s = pd.Series(slope, index=idx)
    slope_rank = slope_s.rolling(252, min_periods=63).rank(pct=True)

    # --- Synthetic SPY returns: contango_signal * lagged_rank + noise ---
    eps = rng.normal(0.0, spy_vol, n_days)
    spy_ret = np.empty(n_days)
    spy_ret[0] = eps[0]
    for i in range(1, n_days):
        lagged_rank = slope_rank.iloc[i - 1]
        if np.isnan(lagged_rank):
            spy_ret[i] = eps[i]
        else:
            spy_ret[i] = contango_signal * (lagged_rank - 0.5) + eps[i]

    spy_close = 200.0 * np.exp(np.cumsum(spy_ret))

    df = pd.DataFrame(
        {
            "VIX": vix,
            "VIX3M": vix3m,
            "slope": slope,
            "SPY_close": spy_close,
            "SPY_ret": spy_ret,
        },
        index=idx,
    )
    truth = {
        "contango_signal": contango_signal,
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
    """Canonical parquet path for the combined VIX/VIX3M/SPY daily cache."""
    return os.path.join(cache_dir, "daily_vix_vix3m_spy.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2008-01-01",
) -> pd.DataFrame:
    """Real daily VIX, VIX3M and SPY data; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  With ``fetch=False`` (the default) a
    ``FileNotFoundError`` is raised if the cache is absent — consistent with the desk
    convention that tests and the reproducible core must never hit the network.

    The returned frame has columns ``[VIX, VIX3M, slope, SPY_close, SPY_ret]`` with a
    tz-naive ``DatetimeIndex`` named ``date``.  ``slope = log(VIX3M / VIX)`` is the
    raw term-structure signal: positive = contango, negative = backwardation.  Rows
    where VIX or VIX3M is NaN are dropped; SPY_ret is the log daily return.
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

        # Strip the ^ from index names and rename columns.
        closes.columns = [str(c).replace("^", "") for c in closes.columns]
        closes = closes.rename(columns={"VIX3M": "VIX3M", "VIX": "VIX", "SPY": "SPY_close"})
        closes.index.name = "date"

        # Require both VIX indices and SPY.
        closes = closes[["VIX", "VIX3M", "SPY_close"]].dropna(subset=["VIX", "VIX3M", "SPY_close"])
        closes["slope"] = np.log(closes["VIX3M"] / closes["VIX"])
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
