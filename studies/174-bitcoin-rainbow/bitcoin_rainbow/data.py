"""Data layer for Study 174 (Bitcoin-Rainbow).

Two tapes, one shape (a daily BTC price frame with a log-time regressor):

- ``synthetic_btc`` — a *deterministic, offline* generator. A ``signal_strength``
  knob controls how tightly log-price tracks the log-time trend the Rainbow Chart
  exploits.  At ``signal_strength=0`` the price is pure random walk (the null in a
  bottle); at ``signal_strength=1.0`` it follows the planted log-time trend exactly
  plus noise (the Rainbow's world).  This separation lets tests verify the engine
  works only when the signal is real.

- ``fetch_btc`` — the real BTC-USD daily tape (``yfinance``), cache-only by default.
  We re-use the canonical cache from Study 84 (Moon-Math) if present, then fall back
  to the local ``_cache/``.  Data runs from 2014-09-17 (earliest reliable Yahoo data)
  through the as-of date.

The Rainbow Chart's geometry is *fully determined* by fitting a linear regression of
log(price) on log(days since genesis).  Bands are placed at ±k·σ_residual offsets
(k = 1,2,3,4 above and below the regression line) — a parameter-free family that
nonetheless suffers catastrophic look-ahead bias because the regression is trained on
the *whole* available history, so the bands always look "right" in-sample.

The canonical Bitcoin genesis block date is 2009-01-03 (block 0).  This is hardcoded
as the time origin of the log-time regressor; no price data is needed to construct it.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Canonical reference: Nakamoto (2008), "Bitcoin: A Peer-to-Peer Electronic Cash
# System"; block 0 timestamp 2009-01-03 18:15:05 UTC.
GENESIS_DATE = pd.Timestamp("2009-01-03")

# Band labels in the original Blockchaincenter chart (7 colours), used for narrative.
BAND_LABELS = [
    "Maximum bubble territory",   # > +4 σ
    "Sell. Seriously, SELL!",      # +3 to +4 σ
    "FOMO intensifies",            # +2 to +3 σ
    "Is this a bubble?",           # +1 to +2 σ
    "HODL!",                       # −1 to +1 σ  (fair-value band)
    "Still cheap",                 # −2 to −1 σ
    "Accumulate",                  # −3 to −2 σ
    "BUY!",                        # −4 to −3 σ
    "Fire sale",                   # < −4 σ
]

# Strategy band thresholds: σ offsets for buy/sell signals (used in strategy.py).
BUY_THRESHOLD_SIGMA = -1.0   # enter long when log-price < regression − 1σ ("cheap" band)
SELL_THRESHOLD_SIGMA = +1.5  # exit long when log-price > regression + 1.5σ ("expensive" band)


# ---------------------------------------------------------------------------
# Log-time regressor — fully deterministic from the genesis date
# ---------------------------------------------------------------------------
def log_days_since_genesis(dates: pd.DatetimeIndex) -> pd.Series:
    """Log of the number of days since the BTC genesis block.

    This is the *only* regressor the Rainbow Chart uses.  It is computed
    deterministically from the genesis date — no price data, no estimation.
    The log transformation stretches the early years where BTC moved from $0
    to $1,000 and compresses the later bull runs, making the visual bands appear
    to track price history in-sample (they always will, by construction).
    """
    days = (dates - GENESIS_DATE).days.astype(float)
    days = np.maximum(days, 1.0)  # avoid log(0)
    return pd.Series(np.log(days), index=dates, name="log_days")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_btc(
    n_days: int = 2_500,
    signal_strength: float = 1.0,
    base_alpha: float = -17.0,
    base_beta: float = 2.8,
    noise_vol: float = 0.05,
    start: str = "2014-09-17",
    seed: int = 174,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily BTC-like tape with a controlled Rainbow-Chart relationship.

    Log-price is generated as:
        ``log_price_t = signal_strength * (base_alpha + base_beta * log_days_t) + noise_t``
    where ``noise_t`` is cumulative random-walk noise scaled by ``noise_vol``.

    - ``signal_strength = 1.0`` — price follows the planted log-time trend (the rainbow's
      assumed world).
    - ``signal_strength = 0.0`` — pure random walk with no log-time component (the null):
      in-sample the rainbow regression will still look great because *both series trend*,
      but out-of-sample (walk-forward) it will fail.

    Returns ``(df, truth)`` where ``df`` has columns
    ``[close, log_close, log_days, band_label_insample]`` and ``truth`` records the
    planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, name="date")

    ld = log_days_since_genesis(dates).to_numpy(dtype=float)
    noise = np.cumsum(rng.normal(0.0, noise_vol, n_days))  # random walk noise
    log_px = signal_strength * (base_alpha + base_beta * ld) + noise
    # Clamp to a plausible price floor to avoid negative/tiny prices.
    log_px = np.maximum(log_px, -4.0)

    # In-sample (look-ahead-biased) regression for the band column.
    alpha_is, beta_is, sigma_is = _fit_ols(log_px, ld)
    resid_is = log_px - (alpha_is + beta_is * ld)
    band_sigma_is = resid_is / sigma_is if sigma_is > 0 else np.zeros_like(resid_is)

    df = pd.DataFrame(
        {
            "close": np.exp(log_px),
            "log_close": log_px,
            "log_days": ld,
            "band_sigma_insample": band_sigma_is,
        },
        index=dates,
    )
    truth = {
        "n_days": n_days,
        "signal_strength": signal_strength,
        "base_alpha": base_alpha,
        "base_beta": base_beta,
        "noise_vol": noise_vol,
        "start": start,
        "seed": seed,
        "alpha_insample": float(alpha_is),
        "beta_insample": float(beta_is),
        "sigma_insample": float(sigma_is),
    }
    return df, truth


def _fit_ols(log_price: np.ndarray, log_days: np.ndarray) -> tuple[float, float, float]:
    """OLS fit of log_price ~ alpha + beta*log_days.  Returns (alpha, beta, sigma_resid)."""
    n = len(log_price)
    X = np.column_stack([np.ones(n), log_days])
    # Normal equations (NumPy least squares)
    params, _, _, _ = np.linalg.lstsq(X, log_price, rcond=None)
    alpha, beta = float(params[0]), float(params[1])
    resid = log_price - (alpha + beta * log_days)
    sigma = float(resid.std(ddof=2)) if n > 2 else 1.0
    return alpha, beta, sigma


# ---------------------------------------------------------------------------
# Real tape — BTC-USD daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "btc_usd_daily.parquet")


def _shared_cache_path() -> str:
    """Path to the Study-84 shared BTC cache, which we prefer when present."""
    root = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))
    # The root _cache is the global scratch; moon-math has a study-local one.
    moon_cache = os.path.abspath(
        os.path.join(HERE, "..", "..", "84-moon-math", "_cache", "btc_usd_daily.parquet")
    )
    return moon_cache


def fetch_btc(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real BTC-USD daily OHLCV from Yahoo Finance; cache-only unless ``fetch=True``.

    Returns a DataFrame with columns ``[close, log_close, log_days]``, indexed by date.
    The ``log_days`` column is the Rainbow Chart's deterministic regressor.

    Lookup order for the cache:
      1. Local study ``_cache/btc_usd_daily.parquet``
      2. Study-84 sibling cache (shared BTC tape, already downloaded)
      3. Network (only with ``fetch=True``)

    Network is touched only on an explicit ``fetch=True`` (then the result is saved under
    the local ``_cache/``).  With ``fetch=False`` and no cache, raises ``FileNotFoundError``.
    """
    local_path = _cache_path(cache_dir)
    shared_path = _shared_cache_path()

    if not fetch:
        # Try local first, then shared sibling.
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
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            "BTC-USD", period="max", interval="1d",
            auto_adjust=True, progress=False
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

    # Normalise the index.
    raw = raw.copy()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw = raw[raw["close"] > 0].copy()

    # Attach the deterministic log-time regressor.
    raw["log_close"] = np.log(raw["close"])
    raw["log_days"] = log_days_since_genesis(raw.index).values

    return raw.dropna(subset=["log_close", "log_days"])


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
