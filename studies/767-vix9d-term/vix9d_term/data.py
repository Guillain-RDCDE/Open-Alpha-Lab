"""Data layer for Study 767 (VIX9D-Term).

The *short end* of the VIX curve: ^VIX9D (9-day implied vol) against ^VIX (30-day).
When ^VIX9D < ^VIX the front end is *upward-sloping* — contango, the normal state —
and consensus reads it as calm. When ^VIX9D spikes *above* ^VIX the very short end
inverts into backwardation, which the community reads as acute near-term fear and a
risk-off timer for equities. This is the shorter-dated cousin of Study 111's ^VIX/^VIX3M
slope, and the front end reacts faster and inverts more often.

Two tapes, one shape (a tz-naive daily frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``contango_signal``
  knob controls how much predictive power the synthetic slope carries about forward
  SPY-like returns.  ``contango_signal=0`` is the null: the slope carries no
  information and the timing signal is a fair coin.  ``contango_signal>0`` plants a
  genuine positive relationship between contango (VIX9D < VIX) and forward returns,
  so the test machinery can be verified against a planted effect.

- ``fetch_daily`` — the real Yahoo! daily closes for ``^VIX9D``, ``^VIX``, and
  ``SPY``, cache-only by default so the test-suite and the reproducible core never
  touch the network.  ``^VIX9D`` was launched by CBOE in 2011, giving us ~15 years
  of daily history.

The slope is defined ``slope = log(^VIX / ^VIX9D)``: **positive = contango** (9-day
below 30-day, the normal upward front end), **negative = backwardation** (9-day above
30-day, the near-term stress signal) — the same orientation as Study 111 so the whole
strategy layer is shared verbatim.

No look-ahead is baked in here — VIX and VIX9D measured at the close of day *t* are
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
TICKERS = ["^VIX9D", "^VIX", "SPY"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 3000,
    contango_signal: float = 0.0,
    spy_vol: float = 0.01,
    start: str = "2011-01-03",
    seed: int = 767,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of VIX9D-slope predictive power.

    The synthetic tape mimics three daily series — VIX (30-day), VIX9D (9-day), and
    SPY forward returns — with the following structure:

    - ``VIX_synth`` is an AR(1) process in log-space (coefficient 0.95, mean-reverting
      around log(18)), roughly matching the 10–80 range and persistence of real VIX.
    - ``VIX9D_synth`` is a *more reactive* AR(1) process (coefficient 0.90, noisier)
      that reverts to a slightly *lower* long-run mean (log(16.5)).  The front end is
      naturally in contango (VIX9D < VIX) most of the time, but its faster dynamics
      make it overshoot *above* VIX during shocks — the transient backwardation the
      claim keys on.
    - ``SPY_ret`` is generated as::

          spy_ret_t = contango_signal * slope_rank_{t-1} + eps_t

      where ``slope_rank`` is the rolling percentile (out-of-sample) of the slope
      ``log(VIX/VIX9D)`` over the last 252 days, and ``eps_t`` is i.i.d. normal of
      standard deviation ``spy_vol``.  ``contango_signal=0`` makes SPY a pure random
      walk; ``contango_signal>0`` plants a genuine *positive* relationship between
      contango (positive slope) and forward SPY returns.

    Returns ``(df, truth)`` where ``df`` has columns
    ``[VIX9D, VIX, slope, SPY_close, SPY_ret]`` indexed by a tz-naive
    ``pd.DatetimeIndex`` of business days, and ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # --- Synthetic VIX (30-day): AR(1) in log space, range ~[10, 80] ---
    log_vix = np.empty(n_days)
    log_vix[0] = np.log(18.0)
    phi_vix = 0.95
    sigma_vix = 0.08
    for i in range(1, n_days):
        log_vix[i] = phi_vix * log_vix[i - 1] + (1 - phi_vix) * np.log(18.0) + rng.normal(0, sigma_vix)
    vix = np.exp(log_vix)

    # --- Synthetic VIX9D (9-day): more reactive AR(1), slightly lower mean ---
    # VIX9D reverts to log(16.5) — a touch below VIX's log(18) — producing natural
    # contango on average, but its lower persistence (phi=0.90) and larger shock
    # variance let it spike *above* VIX during stress, inverting the front end.
    log_vix9d = np.empty(n_days)
    log_vix9d[0] = np.log(16.5)
    phi_vix9d = 0.90
    sigma_vix9d = 0.13
    for i in range(1, n_days):
        # Front end is partly driven by the same shock as VIX (correlated), amplified.
        common = 0.6 * (log_vix[i] - phi_vix * log_vix[i - 1] - (1 - phi_vix) * np.log(18.0))
        log_vix9d[i] = (phi_vix9d * log_vix9d[i - 1]
                        + (1 - phi_vix9d) * np.log(16.5)
                        + 1.4 * common
                        + rng.normal(0, sigma_vix9d))
    vix9d = np.exp(log_vix9d)

    # --- Slope: log(VIX / VIX9D) > 0 is contango, < 0 is backwardation ---
    slope = np.log(vix / vix9d)

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
            "VIX9D": vix9d,
            "VIX": vix,
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
    """Canonical parquet path for the combined VIX9D/VIX/SPY daily cache."""
    return os.path.join(cache_dir, "daily_vix9d_vix_spy.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2011-01-01",
) -> pd.DataFrame:
    """Real daily VIX9D, VIX and SPY data; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  With ``fetch=False`` (the default) a
    ``FileNotFoundError`` is raised if the cache is absent — consistent with the desk
    convention that tests and the reproducible core must never hit the network.

    The returned frame has columns ``[VIX9D, VIX, slope, SPY_close, SPY_ret]`` with a
    tz-naive ``DatetimeIndex`` named ``date``.  ``slope = log(VIX / VIX9D)`` is the
    raw short-end term-structure signal: positive = contango, negative = backwardation.
    Rows where VIX or VIX9D is NaN are dropped; SPY_ret is the log daily return.
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
        closes = closes.rename(columns={"VIX9D": "VIX9D", "VIX": "VIX", "SPY": "SPY_close"})
        closes.index.name = "date"

        # Require both VIX indices and SPY.
        closes = closes[["VIX9D", "VIX", "SPY_close"]].dropna(subset=["VIX9D", "VIX", "SPY_close"])
        closes["slope"] = np.log(closes["VIX"] / closes["VIX9D"])
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
