"""Data layer for Study 132 (Yield-Curve-Steepener).

Two tapes, one shape (a tz-naive daily frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``slope_signal``
  knob controls how much predictive power the synthetic yield-curve slope carries about
  forward TLT returns.  ``slope_signal=0`` is the null: the slope carries no information
  and the timing overlay is a fair coin.  ``slope_signal>0`` plants a genuine positive
  relationship between a steep curve (long yield > short yield) and forward TLT returns,
  so the test machinery can be verified against a planted effect.

- ``fetch_daily`` — the real Yahoo! daily closes for TLT (iShares 20+ Year Treasury
  Bond ETF), IEF (7-10 Year Treasury Bond ETF), SHY (1-3 Year Treasury Bond ETF),
  ^TNX (10-year Treasury yield index), and ^IRX (3-month Treasury bill rate).
  Cache-only by default so the test-suite and reproducible core never touch the network.
  TLT's history starts in 2002, giving ~23 years of daily data.

No look-ahead is baked in here: the slope measured at the close of day *t* is only used
to form signals that trade at the close of day *t+1* or later (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default real-tape tickers fetched and stored in one combined parquet.
TICKERS = ["TLT", "IEF", "SHY", "^TNX", "^IRX"]

# Slope is the spread between the 10-year yield proxy and the short-rate proxy.
# Positive => steep curve (long yields >> short yields), "steepener" regime.
# Negative => inverted or flat curve ("flattener" / inversion regime).


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 5000,
    slope_signal: float = 0.0,
    tlt_vol: float = 0.007,
    start: str = "2003-01-02",
    seed: int = 132,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of curve-slope predictive power.

    The synthetic tape mimics five daily series — a yield-curve slope proxy,
    TLT, IEF, SHY (as price levels), and ^TNX/^IRX (as synthetic yield levels)
    — with the following structure:

    - ``slope_synth`` is an AR(1) process (coefficient 0.97) mean-reverting around
      zero, spanning roughly −2.5 % to +3.5 % (in percentage points), mimicking the
      10Y minus 3M spread.  Positive = steep, negative = inverted.
    - ``TLT_ret`` is generated as::

          tlt_ret_t = slope_signal * slope_rank_{t-1} + eps_t

      where ``slope_rank`` is the rolling percentile (out-of-sample) of the slope
      over the last 252 days, and ``eps_t`` is i.i.d. normal with standard deviation
      ``tlt_vol``.  ``slope_signal=0`` makes TLT a pure random walk;
      ``slope_signal>0`` plants a genuine *positive* relationship between a steep
      curve and forward TLT returns.

    - IEF and SHY price series are driven by related but noisier AR(1) yield
      processes, so the slope computed from ETF returns is similarly structured.

    Returns ``(df, truth)`` where ``df`` has columns
    ``[slope, TNX, IRX, TLT_close, IEF_close, SHY_close, TLT_ret]``
    indexed by a tz-naive ``pd.DatetimeIndex`` of business days, and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # --- Synthetic 10Y yield: AR(1) in level space, range ~[1.5%, 8%] ---
    tnx = np.empty(n_days)
    tnx[0] = 4.0  # start at ~4%
    phi_tnx = 0.97
    sigma_tnx = 0.06  # daily std in %-point
    mu_tnx = 4.0
    for i in range(1, n_days):
        tnx[i] = phi_tnx * tnx[i - 1] + (1 - phi_tnx) * mu_tnx + rng.normal(0, sigma_tnx)
    tnx = np.clip(tnx, 0.5, 9.0)

    # --- Synthetic 3M T-bill: smoother AR(1), lags the 10Y ---
    irx = np.empty(n_days)
    irx[0] = 2.0
    phi_irx = 0.985
    sigma_irx = 0.04
    mu_irx = 2.5
    for i in range(1, n_days):
        # Mean-reversion target slowly tracks 10Y with a lag premium
        irx[i] = phi_irx * irx[i - 1] + (1 - phi_irx) * (tnx[i - 1] - 1.5) + rng.normal(0, sigma_irx)
    irx = np.clip(irx, 0.01, 8.5)

    # --- Curve slope: 10Y minus 3M (in percentage points) ---
    slope = tnx - irx  # positive = steep, negative = inverted

    # --- Rolling slope percentile rank (252-day lookback, out-of-sample) ---
    slope_s = pd.Series(slope, index=idx)
    slope_rank = slope_s.rolling(252, min_periods=63).rank(pct=True)

    # --- Synthetic TLT returns: slope_signal * lagged rank + noise ---
    eps = rng.normal(0.0, tlt_vol, n_days)
    tlt_ret = np.empty(n_days)
    tlt_ret[0] = eps[0]
    for i in range(1, n_days):
        lagged_rank = slope_rank.iloc[i - 1]
        if np.isnan(lagged_rank):
            tlt_ret[i] = eps[i]
        else:
            tlt_ret[i] = slope_signal * (lagged_rank - 0.5) + eps[i]

    tlt_close = 100.0 * np.exp(np.cumsum(tlt_ret))

    # IEF and SHY are simpler random-walk price series with smaller vol
    ief_ret = rng.normal(0.0, tlt_vol * 0.55, n_days)
    ief_close = 100.0 * np.exp(np.cumsum(ief_ret))
    shy_ret = rng.normal(0.0, tlt_vol * 0.12, n_days)
    shy_close = 100.0 * np.exp(np.cumsum(shy_ret))

    df = pd.DataFrame(
        {
            "slope": slope,
            "TNX": tnx,
            "IRX": irx,
            "TLT_close": tlt_close,
            "IEF_close": ief_close,
            "SHY_close": shy_close,
            "TLT_ret": tlt_ret,
        },
        index=idx,
    )
    truth = {
        "slope_signal": slope_signal,
        "tlt_vol": tlt_vol,
        "n_days": n_days,
        "seed": seed,
        "start": start,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily closes, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    """Canonical parquet path for the combined curve/bond daily cache."""
    return os.path.join(cache_dir, "daily_yield_curve_steepener.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2002-07-01",
) -> pd.DataFrame:
    """Real daily yield-curve and bond-ETF data; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  With ``fetch=False`` (the default) a
    ``FileNotFoundError`` is raised if the cache is absent — consistent with the desk
    convention that tests and the reproducible core must never hit the network.

    The returned frame has columns
    ``[TLT_close, IEF_close, SHY_close, TNX, IRX, slope, TLT_ret, IEF_ret]``
    with a tz-naive ``DatetimeIndex`` named ``date``.

    ``slope = TNX - IRX`` is the raw term-structure spread in percentage points:
    positive = steep / normal curve, negative = inverted curve.
    ``TLT_ret`` and ``IEF_ret`` are log daily returns.

    Rows where TNX or IRX is NaN (before either series starts) are dropped;
    the TLT / IEF / SHY ETF closes are forward-filled for at most 5 trading days
    (holidays) but dropped where still missing.
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

        # Strip the ^ prefix from yield tickers and rename cleanly.
        closes.columns = [str(c).replace("^", "") for c in closes.columns]
        closes.index.name = "date"

        # Forward-fill ETF prices for holidays (max 5 days), then drop rows
        # where we still lack ETF data or the yield indices.
        for col in ["TLT", "IEF", "SHY"]:
            if col in closes.columns:
                closes[col] = closes[col].ffill(limit=5)

        closes = closes.rename(columns={
            "TLT": "TLT_close",
            "IEF": "IEF_close",
            "SHY": "SHY_close",
        })
        closes = closes.dropna(subset=["TLT_close", "TNX", "IRX"])

        closes["slope"] = closes["TNX"] - closes["IRX"]
        closes["TLT_ret"] = np.log(closes["TLT_close"]).diff()
        closes["IEF_ret"] = np.log(closes["IEF_close"]).diff()
        closes = closes.dropna(subset=["TLT_ret"])

        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)

        os.makedirs(cache_dir, exist_ok=True)
        closes.to_parquet(path)
        df = closes

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a daily tape (TLT_close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["TLT_close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
