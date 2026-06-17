"""Data layer for Study 221 (Mayer-Multiple).

Two tapes, one shape (a daily BTC price frame with the Mayer Multiple attached):

- ``synthetic_btc`` — a *deterministic, offline* generator.  A ``signal_strength``
  knob controls how strongly the Mayer Multiple predicts forward returns.  At
  ``signal_strength=0`` price is a pure random walk (the null in a bottle); at
  ``signal_strength=1.0`` below-1 multiples genuinely predict positive next-30-day
  returns (the claim's world).  This separation lets tests verify the engine only
  fires when the signal is real.

- ``fetch_btc`` — the real BTC-USD daily tape (``yfinance``), cache-only by default.
  We re-use the Study-174 sibling cache if present, then fall back to the local
  ``_cache/``.  Data runs from 2014-09-17 (earliest reliable Yahoo Finance data)
  through the as-of date.

The Mayer Multiple is fully determined by the 200-day simple moving average of close
price — a parameter-free quantity known in real time.  The conventional "cheap" threshold
is MM < 1.0 (price below the 200-day SMA); the conventional "trim/sell" threshold is
MM > 2.4 (price more than 2.4x above the 200-day SMA).  These two numbers were
popularised by Trace Mayer in 2018; they are the thresholds we test.

Note on the 200-day window: the SMA requires at least 200 calendar days of history
(~140 trading days) before it is valid.  During the warm-up period all derived columns
are NaN.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The two canonical threshold levels popularised by Trace Mayer (2018).
BUY_THRESHOLD = 1.0    # MM < 1.0: price below 200-day SMA -- "cheap" / accumulate
SELL_THRESHOLD = 2.4   # MM > 2.4: price more than 2.4x above 200-day SMA -- "trim"

# SMA window (days).  The canonical Mayer Multiple uses 200 calendar-day SMA.
SMA_WINDOW = 200


# ---------------------------------------------------------------------------
# Mayer Multiple computation — fully deterministic from close prices
# ---------------------------------------------------------------------------
def compute_mayer_multiple(close: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    """Price / N-day simple moving average.

    The standard Mayer Multiple uses N=200.  Returns NaN for the first ``window-1``
    rows (the warm-up period when the SMA is not yet valid).
    """
    sma = close.rolling(window, min_periods=window).mean()
    mm = close / sma
    mm.name = "mayer_multiple"
    return mm


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_btc(
    n_days: int = 3_000,
    signal_strength: float = 1.0,
    mu_daily: float = 0.001,          # slight daily drift (BTC-like uptrend)
    vol_daily: float = 0.04,          # daily vol
    mean_reversion: float = 0.005,    # per-unit pull-back strength when MM is elevated
    seed: int = 221,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily BTC-like tape where Mayer Multiple has tunable predictive power.

    Price is generated as a geometric random walk with a mild mean-reversion term
    that scales with ``signal_strength``:

        r_t = mu_daily - signal_strength * mean_reversion * (MM_{t-1} - 1) + noise_t

    When ``signal_strength=0`` the price is a pure random walk and the Mayer Multiple
    carries no information.  When ``signal_strength=1`` there is a genuine but modest
    pull-back when MM is elevated (mean-reversion towards the 200-day MA), which is
    exactly what the Mayer Multiple claims to capture.

    Returns ``(df, truth)`` where ``df`` has columns
    ``[close, log_return, mayer_multiple, sma_200]`` and ``truth`` records the
    planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2014-09-17", periods=n_days, name="date")

    price = np.empty(n_days)
    price[0] = 500.0   # Starting price

    # We need to track a rolling SMA; do it in a simple forward loop.
    for t in range(1, n_days):
        # SMA at t-1 (use available window; NaN before we have enough history).
        lo = max(0, t - SMA_WINDOW)
        sma_prev = price[lo:t].mean() if (t >= SMA_WINDOW) else price[:t].mean()
        mm_prev = price[t - 1] / sma_prev if sma_prev > 0 else 1.0

        # Return with optional mean-reversion signal.
        reversion = signal_strength * mean_reversion * (mm_prev - 1.0)
        r = mu_daily - reversion + rng.normal(0.0, vol_daily)
        price[t] = price[t - 1] * np.exp(r)

    close = pd.Series(price, index=dates, name="close")
    sma = close.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    mm = compute_mayer_multiple(close, SMA_WINDOW)
    log_ret = np.log(close / close.shift(1))

    df = pd.DataFrame(
        {
            "close": close,
            "log_return": log_ret,
            "sma_200": sma,
            "mayer_multiple": mm,
        },
        index=dates,
    )

    truth = {
        "n_days": n_days,
        "signal_strength": signal_strength,
        "mu_daily": mu_daily,
        "vol_daily": vol_daily,
        "mean_reversion": mean_reversion,
        "seed": seed,
        "buy_threshold": BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — BTC-USD daily, cache-only by default
# ---------------------------------------------------------------------------
def _local_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "btc_usd_daily.parquet")


def _shared_cache_path() -> str:
    """Path to the Study-174 sibling BTC cache, preferred when present."""
    return os.path.abspath(
        os.path.join(HERE, "..", "..", "174-bitcoin-rainbow", "_cache", "btc_usd_daily.parquet")
    )


def fetch_btc(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real BTC-USD daily OHLCV from Yahoo Finance; cache-only unless ``fetch=True``.

    Returns a DataFrame with columns ``[close, log_return, sma_200, mayer_multiple]``,
    indexed by date.  The first 199 rows of ``sma_200`` and ``mayer_multiple`` are NaN
    (the 200-day warm-up period).

    Lookup order:
      1. Local study ``_cache/btc_usd_daily.parquet``
      2. Study-174 sibling cache (shared BTC tape, already downloaded)
      3. Network (only with ``fetch=True``)
    """
    local_path = _local_cache_path(cache_dir)
    shared_path = _shared_cache_path()

    if not fetch:
        if os.path.exists(local_path):
            raw = pd.read_parquet(local_path)
        elif os.path.exists(shared_path):
            raw = pd.read_parquet(shared_path)
        else:
            raise FileNotFoundError(
                f"No cached BTC daily tape found at {local_path} or {shared_path}. "
                f"Call fetch_btc(fetch=True) once to populate the cache."
            )
    else:
        import yfinance as yf

        raw = yf.download(
            "BTC-USD", period="max", interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no daily bars for BTC-USD")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        raw.index.name = "date"
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        raw.to_parquet(local_path)

    raw = raw.copy()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw = raw[raw["close"] > 0].copy()

    raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))
    raw["sma_200"] = raw["close"].rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    raw["mayer_multiple"] = compute_mayer_multiple(raw["close"], SMA_WINDOW)

    return raw.dropna(subset=["close"])


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
