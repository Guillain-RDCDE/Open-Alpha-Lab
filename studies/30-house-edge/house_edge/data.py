"""Price data for the leverage study — an offline synthetic path, and a cached real fetch.

The desk's offline/cache split:

  * :func:`synthetic_market` — fully **offline, deterministic**. A daily equity index with
    realistic *volatility clustering* (a GARCH-like vol process) and occasional *crashes*, plus
    a slow upward drift. This is enough to demonstrate the whole point: a vol-targeted dip-buyer
    cuts drawdown, and the broker's markup — full-notional on a CFD — eats the return. Seeded —
    same seed, same path.
  * :func:`fetch_index` — the real S&P 500 (``^GSPC``, total-return proxy) and a short-rate
    (``^IRX``, 13-week T-bill) from Yahoo, **cache-first**; network only on an explicit cache miss
    with ``fetch=True``. The real 36-year run lives in ``docs/results.md``.

Data choice: daily closes, the financing rate is the 3-month T-bill (what a broker funds at) plus
a markup; dividends modelled as a constant yield credited to a long index position. Stated, not hidden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketTruth:
    n_days: int
    seed: int
    drift_ann: float
    base_vol_ann: float
    n_crashes: int


def synthetic_market(n_days: int = 252 * 24, drift_ann: float = 0.09, base_vol_ann: float = 0.15,
                     garch_alpha: float = 0.08, garch_beta: float = 0.90, crash_lambda: float = 0.5,
                     seed: int = 30) -> tuple[pd.Series, pd.Series, MarketTruth]:
    """A synthetic daily equity index with volatility clustering and crashes (offline, seeded).

    Returns ``(price, short_rate, truth)``:
      * ``price`` — a daily close path. Equity-like drift ``drift_ann``; daily variance follows a
        stationary **GARCH(1,1)** (``alpha + beta < 1``) around ``base_vol_ann``, so calm and stormy
        regimes cluster; plus Poisson-timed downward *crashes* (a fat left tail). This is the minimal
        world in which a vol-targeting overlay and the financing-cost point are both real.
      * ``short_rate`` — a slow, mean-reverting annualised financing rate (what leverage is funded
        at), so the financing drag varies through time as it does in reality.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2000-01-03", periods=n_days, name="date")
    mu = drift_ann / TRADING_DAYS
    base_var = (base_vol_ann / np.sqrt(TRADING_DAYS)) ** 2
    omega = base_var * (1 - garch_alpha - garch_beta)     # stationary GARCH(1,1)
    # Persistent BEAR REGIMES: ~crash_lambda per year, each a multi-week decline with elevated
    # vol — long enough that a vol-targeting overlay and a trend gate have time to react (a 3-day
    # crash would simply not be a fair test of risk management). Drift turns negative inside one.
    drift_path = np.full(n_days, mu)
    vol_mult = np.ones(n_days)
    n_crashes = 0
    for _ in range(rng.poisson(crash_lambda * n_days / TRADING_DAYS)):
        length = int(rng.integers(25, 70)); start = int(rng.integers(0, max(1, n_days - length)))
        drift_path[start:start + length] = -rng.uniform(0.0015, 0.0035)   # daily bear drift
        vol_mult[start:start + length] = rng.uniform(1.8, 2.6)             # stress vol
        n_crashes += 1
    z = rng.standard_normal(n_days)
    var = np.empty(n_days); ret = np.empty(n_days)
    var[0] = base_var; ret[0] = drift_path[0] + np.sqrt(var[0]) * z[0]
    for t in range(1, n_days):
        resid = ret[t - 1] - drift_path[t - 1]
        var[t] = omega + garch_alpha * resid ** 2 + garch_beta * var[t - 1]
        ret[t] = drift_path[t] + np.sqrt(var[t] * vol_mult[t]) * z[t]
    price = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx, name="price")
    # mean-reverting short rate in [0, ~8%], slow
    r = np.empty(n_days); r[0] = 0.03
    for t in range(1, n_days):
        r[t] = np.clip(r[t - 1] + 0.02 * (0.03 - r[t - 1]) / TRADING_DAYS
                       + 0.004 / np.sqrt(TRADING_DAYS) * rng.standard_normal(), 0.0, 0.10)
    short_rate = pd.Series(r, index=idx, name="short_rate")
    return price, short_rate, MarketTruth(n_days=n_days, seed=seed, drift_ann=drift_ann,
                                          base_vol_ann=base_vol_ann, n_crashes=n_crashes)


def fetch_index(cache_dir: str = DEFAULT_CACHE, fetch: bool = False
                ) -> tuple[pd.Series, pd.Series]:
    """Real ``(price, short_rate)`` — S&P 500 close and 13-week T-bill — cache-first.

    **Cache-only** unless ``fetch=True`` (which downloads ``^GSPC`` and ``^IRX`` from Yahoo).
    Returns empty Series on a cache miss with ``fetch=False`` so offline runs never hit the network.
    """
    cache = os.path.join(cache_dir, "house_edge_index.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df["price"], df["short_rate"]
    if not fetch:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    import yfinance as yf  # lazy
    def _close(t):
        raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].dropna(); s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s.astype(float)
    price = _close("^GSPC").rename("price")
    rate = (_close("^IRX") / 100.0).reindex(price.index).ffill().fillna(0.03).rename("short_rate")
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame({"price": price, "short_rate": rate}).to_parquet(cache)
    return price, rate
