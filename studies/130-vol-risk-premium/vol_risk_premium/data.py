"""Data layer for Study 130 (Vol-Risk-Premium).

Two tapes, one shape (a tz-naive daily frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``vrp_signal`` knob
  controls how much predictive power the synthetic VRP (implied minus realised vol)
  carries about forward SPY-like returns.  ``vrp_signal=0`` is the null: the VRP
  carries no information and the timing signal is a fair coin.  ``vrp_signal>0`` plants
  a genuine positive relationship between a high VRP and forward returns (the hypothesis
  that high IV-RV spread → calm risk environment → good for equities).  A negative
  ``vrp_signal`` plants the reverse.

- ``fetch_daily`` — the real Yahoo! daily data for ``^VIX`` and ``SPY``, cache-only by
  default so the test-suite and the reproducible core never touch the network.  The VIX
  goes back to January 1990 and SPY to January 1993, giving ~33 years of daily history
  for the premium measurement and ~28 years once the 21-day realised vol warmup is paid.

No look-ahead is baked in here — VIX measured at the close of day *t* is compared to
the *trailing* 21-day realised vol (also known at close *t*), forming a signal that
trades at the open of day *t+1* or later (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Realised vol lookback (trading days).
RV_WINDOW = 21


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 4000,
    vrp_signal: float = 0.0,
    spy_vol: float = 0.01,
    base_iv: float = 18.0,
    start: str = "1994-01-03",
    seed: int = 130,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of VRP predictive power.

    The synthetic tape mimics VIX and SPY returns with the following structure:

    - ``VIX_synth`` is an AR(1) process in log-space (coefficient 0.95, mean-reverting
      around ``log(base_iv)``), roughly matching the 10–80 range and persistence of the
      real VIX (which quotes annualised vol in percent; ``base_iv`` is already in those
      units, default 18 %).
    - Realised vol (``RV``) is the trailing 21-day standard deviation of SPY log-returns
      multiplied by ``sqrt(252) * 100`` to match VIX units.
    - The VRP is defined as ``VIX - RV`` in annualised-percent units; it is positive on
      average in the data (IV > RV, sellers earn a premium).
    - ``SPY_ret`` is generated as::

          spy_ret_t = vrp_signal * vrp_rank_{t-1} + eps_t

      where ``vrp_rank`` is the rolling percentile (out-of-sample) of the VRP over the
      last 252 days, and ``eps_t`` is i.i.d. normal of standard deviation ``spy_vol``.
      ``vrp_signal=0`` makes SPY a pure random walk; ``vrp_signal>0`` plants a genuine
      positive relationship between a high VRP and subsequent SPY returns.

    Returns ``(df, truth)`` where ``df`` has columns
    ``[VIX, RV, VRP, SPY_close, SPY_ret]`` indexed by a tz-naive
    ``pd.DatetimeIndex`` of business days, and ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # --- Synthetic VIX: AR(1) in log space ---
    # ``base_iv`` is the long-run VIX level in annualised-percent units (default 18).
    log_vix = np.empty(n_days)
    log_vix[0] = np.log(base_iv)
    phi_vix = 0.95
    sigma_vix = 0.08
    for i in range(1, n_days):
        log_vix[i] = (phi_vix * log_vix[i - 1]
                      + (1 - phi_vix) * np.log(base_iv)
                      + rng.normal(0, sigma_vix))
    vix = np.exp(log_vix)

    # --- Synthetic SPY returns: vrp_signal * lagged VRP rank + noise ---
    # First, generate raw SPY returns and compute trailing RV.
    eps = rng.normal(0.0, spy_vol, n_days)
    spy_ret_raw = eps.copy()  # warm-up round (no signal yet)
    rv_warmup = np.full(n_days, np.nan)
    for i in range(RV_WINDOW, n_days):
        rv_warmup[i] = float(np.std(spy_ret_raw[i - RV_WINDOW:i], ddof=1) * np.sqrt(252) * 100)

    vrp_raw = vix - rv_warmup  # positive on average (IV > RV)
    vrp_s = pd.Series(vrp_raw, index=idx)
    vrp_rank = vrp_s.rolling(252, min_periods=63).rank(pct=True)

    eps2 = rng.normal(0.0, spy_vol, n_days)
    spy_ret = np.empty(n_days)
    spy_ret[0] = eps2[0]
    for i in range(1, n_days):
        lagged_rank = vrp_rank.iloc[i - 1]
        if np.isnan(lagged_rank):
            spy_ret[i] = eps2[i]
        else:
            spy_ret[i] = vrp_signal * (lagged_rank - 0.5) + eps2[i]

    spy_close = 100.0 * np.exp(np.cumsum(spy_ret))

    # Compute final RV and VRP from the generated returns.
    rv = np.full(n_days, np.nan)
    for i in range(RV_WINDOW, n_days):
        rv[i] = float(np.std(spy_ret[i - RV_WINDOW:i], ddof=1) * np.sqrt(252) * 100)
    vrp = vix - rv

    df = pd.DataFrame(
        {
            "VIX": vix,
            "RV": rv,
            "VRP": vrp,
            "SPY_close": spy_close,
            "SPY_ret": spy_ret,
        },
        index=idx,
    )
    truth = {
        "vrp_signal": vrp_signal,
        "spy_vol": spy_vol,
        "base_iv_pct": base_iv,
        "n_days": n_days,
        "seed": seed,
        "start": start,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily closes, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    """Canonical parquet path for the combined VIX/SPY/RV daily cache."""
    return os.path.join(cache_dir, "daily_vix_spy_vrp.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1990-01-01",
) -> pd.DataFrame:
    """Real daily VIX and SPY data with computed RV and VRP; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  With ``fetch=False`` (the default) a
    ``FileNotFoundError`` is raised if the cache is absent — consistent with the desk
    convention that tests and the reproducible core must never hit the network.

    The returned frame has columns ``[VIX, SPY_close, SPY_ret, RV, VRP]`` with a
    tz-naive ``DatetimeIndex`` named ``date``.

    - ``VIX`` is the CBOE Volatility Index (annualised percent), sourced from Yahoo
      as ``^VIX``.
    - ``RV`` (realised vol) is the trailing 21-day standard deviation of SPY log-returns,
      annualised in the same units as VIX: ``std(ret[-21:], ddof=1) * sqrt(252) * 100``.
    - ``VRP`` = ``VIX - RV`` in annualised-percent points.  A positive VRP means implied
      vol exceeds realised vol: option sellers earn a gross premium.
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
            ["^VIX", "SPY"],
            start=start,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no daily bars")

        # Multi-ticker download returns a MultiIndex — extract Close.
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw[["Close"]].copy()

        closes.columns = [str(c).replace("^", "") for c in closes.columns]
        closes = closes.rename(columns={"VIX": "VIX", "SPY": "SPY_close"})
        closes.index.name = "date"
        closes = closes[["VIX", "SPY_close"]].dropna(subset=["VIX", "SPY_close"])
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)

        closes["SPY_ret"] = np.log(closes["SPY_close"]).diff()
        closes = closes.dropna(subset=["SPY_ret"])

        # Trailing 21-day realised vol in annualised-pct units (matching VIX).
        closes["RV"] = (
            closes["SPY_ret"]
            .rolling(RV_WINDOW, min_periods=RV_WINDOW)
            .std(ddof=1)
            * np.sqrt(252)
            * 100
        )
        closes["VRP"] = closes["VIX"] - closes["RV"]
        closes = closes.dropna(subset=["RV"])

        os.makedirs(cache_dir, exist_ok=True)
        closes.to_parquet(path)
        df = closes

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a daily tape (VIX column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["VIX"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
