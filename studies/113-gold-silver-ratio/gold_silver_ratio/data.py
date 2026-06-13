"""Data layer for Study 113 (Gold-Silver-Ratio).

Two tapes, one shape (a daily OHLCV frame for each of the two metals):

- ``synthetic_daily`` — a *deterministic, offline* generator. The gold/silver ratio is
  modelled as a mean-reverting Ornstein-Uhlenbeck process with a tunable speed
  (``mean_rev_speed``). ``mean_rev_speed=0`` is a pure random walk — the null where
  reversion-based signals cannot work. Positive speeds introduce genuine reversion that
  the z-score strategy can harvest. The two underlying price series are reconstructed
  from the ratio + an independent gold price random walk, ensuring the synthetic tape
  has a realistic structure.
- ``fetch_daily`` — the real Yahoo! daily tape for GLD and SLV (ETFs) and optionally
  GC=F / SI=F (futures), cache-only by default so the test-suite and the reproducible
  core never touch the network. Daily history goes back many years, giving reasonable
  statistical power.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on closes up to *t*, positions entered at the next open, approximated as the
*t+1* close for daily data).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default tickers — liquid ETFs tracking gold and silver.
GOLD_TICKER = "GLD"
SILVER_TICKER = "SLV"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 1500,
    mean_rev_speed: float = 0.0,
    ratio_mean: float = 70.0,
    ratio_vol: float = 8.0,
    gold_vol: float = 0.012,
    start: str = "2018-01-02",
    seed: int = 113,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible pair of daily price series sharing a mean-reverting ratio.

    The gold/silver ratio ``R_t`` evolves as a discrete-time OU process:
    ``R_t = ratio_mean + rho * (R_{t-1} - ratio_mean) + eps_t`` where
    ``rho = 1 - mean_rev_speed`` and ``eps_t ~ N(0, ratio_vol^2 * (1-rho^2))``.
    When ``mean_rev_speed = 0``, rho = 1 and the ratio is a pure random walk (the
    null hypothesis). When ``mean_rev_speed > 0`` the ratio mean-reverts, which is
    exactly the structure the z-score strategy can exploit.

    Gold prices follow an independent log-random-walk with daily vol ``gold_vol``.
    Silver prices are derived as ``gold / ratio`` so the ratio is internally
    consistent.

    Returns ``(gold_bars, silver_bars, truth)`` where each frame has columns
    ``[open, high, low, close, volume]`` indexed by business-day dates, and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # 1. Generate the ratio path via OU
    rho = 1.0 - mean_rev_speed
    # Stationary variance = ratio_vol^2; for the increments use sqrt(1 - rho^2) * ratio_vol
    # When rho = 1 (random walk), increment std = ratio_vol directly.
    if abs(rho) < 1.0:
        inc_std = ratio_vol * np.sqrt(max(1.0 - rho**2, 0.0))
    else:
        inc_std = ratio_vol  # random walk: each step has std ratio_vol
    ratio = np.empty(n_days)
    ratio[0] = ratio_mean
    eps = rng.normal(0.0, inc_std, n_days - 1)
    for i in range(1, n_days):
        ratio[i] = ratio_mean + rho * (ratio[i - 1] - ratio_mean) + eps[i - 1]
    ratio = np.maximum(ratio, 5.0)  # floor: ratio can't go below 5

    # 2. Generate gold log-prices (random walk)
    gold_log_ret = rng.normal(0.0, gold_vol, n_days)
    gold_log_ret[0] = 0.0
    gold_close = 150.0 * np.exp(np.cumsum(gold_log_ret))  # GLD-like starting level

    # 3. Silver derived from gold / ratio
    silver_close = gold_close / ratio

    # 4. Build synthetic OHLCV for each (simple: open = prior close, wick ±0.3%)
    def _ohlcv(close: np.ndarray) -> pd.DataFrame:
        open_ = np.empty_like(close)
        open_[0] = close[0] * 0.9995
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(0.0, 0.003, close.size)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        lo = np.maximum(lo, 1e-3)
        vol = rng.integers(1_000_000, 10_000_000, close.size).astype(float)
        return pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=dates,
        )

    gold_bars = _ohlcv(gold_close)
    silver_bars = _ohlcv(silver_close)
    truth = {
        "mean_rev_speed": mean_rev_speed,
        "ratio_mean": ratio_mean,
        "ratio_vol": ratio_vol,
        "gold_vol": gold_vol,
        "n_days": n_days,
        "seed": seed,
        "planted_half_life": float(np.log(2) / mean_rev_speed) if mean_rev_speed > 0 else float("inf"),
    }
    return gold_bars, silver_bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2010-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Daily history on Yahoo goes back many years, so this
    study has real statistical power unlike the 5-minute cousins.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        bars.index = pd.to_datetime(bars.index)
        # Drop rows with NaN closes
        bars = bars.dropna(subset=["close"])
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    bars.index = pd.to_datetime(bars.index)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
