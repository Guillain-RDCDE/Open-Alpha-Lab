"""Data layer for Study 110 (Faber-Timing).

Two tapes, one shape (a date-indexed daily OHLCV close frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A calendar of simulated
  monthly returns drawn from a two-state Markov regime: a bull market with positive
  drift and moderate vol, and a bear market with negative drift and high vol. The
  ``signal_strength`` knob controls how predictable the regime transition is — at
  ``signal_strength=0`` the SMA rule has nothing to read (the null); at higher values
  it correctly identifies the regime and can duck bear-market drawdowns. The seed is
  fixed so tests and the demo are fully deterministic and offline.
- ``fetch_prices`` — daily adjusted closes from Yahoo! Finance (``yfinance``), cache-only
  by default. ``fetch=False`` raises ``FileNotFoundError`` if the parquet is absent,
  so the test-suite and offline core never touch the network. ``fetch=True`` downloads
  and caches. Uses ``auto_adjust=True`` for split/dividend-adjusted total-return closes.

The Faber rule is a *monthly* rule in the original 2007 paper (compare price to the
end-of-month 10-month SMA). We implement it on *daily* closes (compare to the 200-day
SMA) for better real-data power. Either periodicity is available through
``resample_monthly``.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals
formed on data up to day ``t``; positions entered at the open of day ``t+1``, or at the
next month's first close for the monthly variant).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 30,
    drift_bull: float = 0.10,          # annualised bull-market return
    drift_bear: float = -0.25,         # annualised bear-market return
    vol_bull: float = 0.12,            # annualised bull-market vol
    vol_bear: float = 0.30,            # annualised bear-market vol
    p_bull_to_bear: float = 0.10,      # monthly prob of switching to bear
    p_bear_to_bull: float = 0.30,      # monthly prob of recovering
    signal_strength: float = 1.0,      # 0 = flat vol (null), 1 = full regime
    start: str = "1994-01-03",
    seed: int = 110,
    tbill_rate_ann: float = 0.04,      # cash yield credited when out of market
) -> tuple[pd.DataFrame, dict]:
    """A daily equity price series with a two-state Markov bull/bear regime.

    The ``signal_strength`` knob blends the two-regime world toward a flat-vol
    null (``signal_strength=0``):

    - ``signal_strength = 1`` → full regime separation; the 200-day SMA can see
      the bear and move to cash, earning the T-bill.
    - ``signal_strength = 0`` → drift and vol collapse to a single state (the
      regime still exists structurally but the vols equalize, so the SMA cross
      carries no net information).

    Returns ``(prices, truth)`` where ``prices`` has columns ``close`` and
    ``tbill`` (daily T-bill accrual factor), and ``truth`` records the planted
    parameters and the fraction of time spent in the bear regime.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    dates = pd.bdate_range(start=start, periods=n_days)

    # --- blend regime vols toward the null -----------------------------------
    # vol_bull_eff = weighted average so that at signal_strength=0 both regimes
    # have the same vol (the SMA has nothing useful to time).
    stationary_bear = p_bull_to_bear / (p_bull_to_bear + p_bear_to_bull)
    avg_vol = (1 - stationary_bear) * vol_bull + stationary_bear * vol_bear
    vol_bull_eff = vol_bull + signal_strength * (vol_bull - avg_vol)  # noqa: F841 — kept for symmetry
    vol_bear_eff = vol_bear + signal_strength * (vol_bear - avg_vol)
    vol_bull_eff = (1 - signal_strength) * avg_vol + signal_strength * vol_bull
    vol_bear_eff = (1 - signal_strength) * avg_vol + signal_strength * vol_bear

    drift_bull_eff = (1 - signal_strength) * 0.05 + signal_strength * drift_bull  # pull bear toward mild
    drift_bear_eff = (1 - signal_strength) * 0.05 + signal_strength * drift_bear

    d_bull = drift_bull_eff / TRADING_DAYS_PER_YEAR
    d_bear = drift_bear_eff / TRADING_DAYS_PER_YEAR
    s_bull = vol_bull_eff / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_bear = vol_bear_eff / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Monthly regime transitions (approx 21 trading days per month).
    MONTH = 21
    regime = np.zeros(n_days, dtype=int)   # 0 = bull, 1 = bear
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
    # Daily T-bill accrual (compounded simple rate)
    tbill_daily = (1.0 + tbill_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    tbill_idx = np.cumprod(np.full(n_days, tbill_daily))

    prices = pd.DataFrame({"close": close, "tbill": tbill_idx}, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "signal_strength": signal_strength,
        "drift_bull": drift_bull,
        "drift_bear": drift_bear,
        "vol_bull": vol_bull,
        "vol_bear": vol_bear,
        "p_bull_to_bear": p_bull_to_bear,
        "p_bear_to_bull": p_bear_to_bull,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "tbill_rate_ann": tbill_rate_ann,
        "bear_frac": float(regime.mean()),
        "vol_bull_eff": float(vol_bull_eff),
        "vol_bear_eff": float(vol_bear_eff),
    }
    return prices, truth


def resample_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """End-of-month close series, for the original Faber monthly variant.

    Returns a frame with columns matching ``prices`` (``close``, and ``tbill`` if
    present), indexed by end-of-month dates. The monthly T-bill factor is the product
    of daily factors within the month.
    """
    cols = {c: ("last" if c == "close" else "prod") for c in prices.columns}
    return prices.resample("ME").agg(cols)


# ---------------------------------------------------------------------------
# Real tape — Yahoo! Finance daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch_prices(
    ticker: str,
    start: str = "1993-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily adjusted close prices for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Returns a frame with a single ``close`` column indexed
    by date. Uses ``auto_adjust=True`` (split- and dividend-adjusted total-return close)
    — this matters for a multi-decade backtest; unadjusted prices would overstate
    buy-and-hold performance during dividend-paying periods.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily prices for {ticker} at {path}. "
                f"Call fetch_prices({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of a price series (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(prices["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
