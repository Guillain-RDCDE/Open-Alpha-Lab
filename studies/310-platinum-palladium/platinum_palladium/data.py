"""Data layer for Study 310 (Platinum-Palladium).

Platinum (PL=F) and palladium (PA=F) are the two great *autocatalyst* metals: both
scrub NOx and CO out of car exhaust. For decades platinum traded far above palladium,
and the folk trade was "the ratio always reverts — when palladium gets rich, sell it
and buy platinum." Then 2018–2022 happened: tightening petrol-engine emissions rules
sent palladium demand through the roof, palladium overtook platinum (the ratio inverted
below 1.0), and the "mean" the trade was supposed to revert to ceased to exist. This is
exactly the regime-shift hazard a naive ratio-reversion trade ignores.

Two tapes, one shape (a daily OHLCV frame for each of the two metals):

- ``synthetic_daily`` — a *deterministic, offline* generator. The platinum/palladium
  ratio is modelled as a mean-reverting Ornstein-Uhlenbeck process with a tunable speed
  (``mean_rev_speed``). ``mean_rev_speed=0`` is a pure random walk — the null where
  reversion-based signals cannot work. Positive speeds plant genuine reversion the
  z-score strategy can harvest (the positive control). Platinum prices are an independent
  log-random-walk; palladium is reconstructed as ``platinum / ratio`` so the tape is
  internally consistent.
- ``fetch_daily`` — the real Yahoo! daily tape for PL=F and PA=F (COMEX/NYMEX futures),
  cache-only by default so the test-suite and the reproducible core never touch the
  network. ``load_real`` is the cache-first convenience wrapper with graceful fallback.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on closes up to *t*, the spread held from entry to exit close).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default tickers — COMEX/NYMEX continuous front-month futures on Yahoo.
PLATINUM_TICKER = "PL=F"
PALLADIUM_TICKER = "PA=F"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 1500,
    mean_rev_speed: float = 0.0,
    ratio_mean: float = 1.6,
    ratio_vol: float = 0.45,
    plat_vol: float = 0.013,
    start: str = "2018-01-02",
    seed: int = 310,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible pair of daily price series sharing a mean-reverting ratio.

    The platinum/palladium ratio ``R_t`` evolves as a discrete-time OU process:
    ``R_t = ratio_mean + rho * (R_{t-1} - ratio_mean) + eps_t`` where
    ``rho = 1 - mean_rev_speed`` and ``eps_t ~ N(0, ratio_vol^2 * (1-rho^2))``.
    When ``mean_rev_speed = 0``, rho = 1 and the ratio is a pure random walk (the null
    hypothesis). When ``mean_rev_speed > 0`` the ratio mean-reverts, which is exactly the
    structure the z-score strategy can exploit (the positive control).

    Platinum prices follow an independent log-random-walk with daily vol ``plat_vol``.
    Palladium prices are derived as ``platinum / ratio`` so the ratio is internally
    consistent. (``ratio_mean = 1.6`` reflects the pre-2018 era when platinum sat well
    above palladium.)

    Returns ``(plat_bars, pall_bars, truth)`` where each frame has columns
    ``[open, high, low, close, volume]`` indexed by business-day dates, and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # 1. Generate the ratio path via OU
    rho = 1.0 - mean_rev_speed
    if abs(rho) < 1.0:
        inc_std = ratio_vol * np.sqrt(max(1.0 - rho**2, 0.0))
    else:
        inc_std = ratio_vol  # random walk: each step has std ratio_vol
    ratio = np.empty(n_days)
    ratio[0] = ratio_mean
    eps = rng.normal(0.0, inc_std, n_days - 1)
    for i in range(1, n_days):
        ratio[i] = ratio_mean + rho * (ratio[i - 1] - ratio_mean) + eps[i - 1]
    ratio = np.maximum(ratio, 0.2)  # floor: ratio can't collapse to zero

    # 2. Generate platinum log-prices (random walk)
    plat_log_ret = rng.normal(0.0, plat_vol, n_days)
    plat_log_ret[0] = 0.0
    plat_close = 950.0 * np.exp(np.cumsum(plat_log_ret))  # USD/oz, PL-like level

    # 3. Palladium derived from platinum / ratio
    pall_close = plat_close / ratio

    # 4. Build synthetic OHLCV for each (open = prior close, wick ±0.3%)
    def _ohlcv(close: np.ndarray) -> pd.DataFrame:
        open_ = np.empty_like(close)
        open_[0] = close[0] * 0.9995
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(0.0, 0.003, close.size)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        lo = np.maximum(lo, 1e-3)
        vol = rng.integers(1_000, 100_000, close.size).astype(float)
        return pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=dates,
        )

    plat_bars = _ohlcv(plat_close)
    pall_bars = _ohlcv(pall_close)
    truth = {
        "mean_rev_speed": mean_rev_speed,
        "ratio_mean": ratio_mean,
        "ratio_vol": ratio_vol,
        "plat_vol": plat_vol,
        "n_days": n_days,
        "seed": seed,
        "planted_half_life": (
            float(np.log(2) / mean_rev_speed) if mean_rev_speed > 0 else float("inf")
        ),
    }
    return plat_bars, pall_bars, truth


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
    a parquet under ``_cache/``). PL=F / PA=F have daily history on Yahoo back to ~2010,
    giving the study real statistical power for a slow ratio signal.
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
        bars = bars.dropna(subset=["close"])
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def load_real(
    fetch: bool = False, cache_dir: str = DEFAULT_CACHE
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Cache-first convenience loader for the PL=F / PA=F pair.

    Returns ``(platinum_bars, palladium_bars)`` if both tapes are available (from the
    shared cache, or fetched when ``fetch=True``); returns ``None`` on any failure so a
    notebook or script can fall back to the synthetic tape without crashing offline.
    """
    try:
        plat = fetch_daily(PLATINUM_TICKER, fetch=fetch, cache_dir=cache_dir)
        pall = fetch_daily(PALLADIUM_TICKER, fetch=fetch, cache_dir=cache_dir)
        return plat, pall
    except (FileNotFoundError, RuntimeError, ImportError):
        return None


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
