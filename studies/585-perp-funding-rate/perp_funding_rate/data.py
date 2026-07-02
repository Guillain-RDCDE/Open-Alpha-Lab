"""Data layer for Study 585 (Perp-Funding-Rate) — funding-rate contrarian signal.

## The claim

Crypto perpetual swaps have no expiry; they are tethered to spot by a periodic **funding rate**
paid between longs and shorts (typically every 8h). When the perp trades *above* spot the funding
rate is **positive** — longs pay shorts — which happens when the crowd is aggressively long and
levered. The folklore: **extreme positive funding = over-crowded longs = a contrarian short
signal** (a flush is due), and symmetrically extreme negative funding marks capitulated shorts, a
contrarian long. It is a *mean-reversion / positioning* signal, not a trend signal.

## Why the core is synthetic-only

A no-key retail stack (yfinance) reaches BTC/ETH **prices** — but *not* the historical
**funding-rate tape**. Funding history lives behind paid or rate-limited exchange APIs
(Binance/Bybit fapi, Coinglass, Amberdata, Kaiko). Without the funding series there is no signal
to test on a real tape, so:

- The reproducible **core is synthetic**: :func:`synthetic_panel` builds a deterministic,
  seeded funding-rate + forward-return panel with ONE knob (``contrarian_beta``) that plants the
  effect (``> 0`` = the contrarian folklore is true; ``= 0`` = the null). This is the study's
  positive control and null in one bottle, and runs fully offline.
- :func:`fetch_btc_eth_prices` is a cache-first yfinance loader for BTC/ETH **prices only** — it
  exists so a reader can *see* the price tape the funding rate would tether to, and to make the
  data-availability limitation concrete (prices are free; funding is not). It never produces a
  funding series, so it can never lift the study to `REAL`.

The synthetic-only status caps the Signal axis at `WEAK`/`NONE` and is named openly there.

## No look-ahead

In :func:`synthetic_panel`, the funding rate at time ``t`` is observed at ``t`` and paired with the
**forward** return over ``(t, t+h]``. The signal is entered at the close where funding is observed
(a documented same-bar convention) and the forward window is strictly future — no future data
enters the signal.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

SEED = 585

# Free-data reachable prices (for the price tape only; funding is NOT free).
PRICE_TICKERS = ["BTC-USD", "ETH-USD"]

# Perp funding convention: 8h funding intervals -> 3 per day -> ~1095 per year.
FUNDING_PER_DAY = 3


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic funding panel."""

    contrarian_beta: float  # forward-return response per unit of standardized funding (<0 desired)

    @property
    def has_effect(self) -> bool:
        return self.contrarian_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_periods: int = 2000,
    contrarian_beta: float = -0.012,
    funding_ar: float = 0.85,
    funding_scale: float = 0.0004,
    idio_vol: float = 0.030,
    horizon: int = 3,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible funding-rate + forward-return series with a tunable contrarian effect.

    A realistic **funding rate** is a slow, mean-reverting (AR(1)) process — funding regimes
    persist for days/weeks — centered near zero with occasional hot/cold extremes:

        f_t = funding_ar * f_{t-1} + funding_scale * eps_t        (eps ~ N(0,1))

    The **forward** simple return over the next ``horizon`` funding periods is a market drift plus
    a contrarian response to *standardized* funding plus idiosyncratic noise:

        fwd_ret_t = drift + contrarian_beta * z(f_t) + idio_t

    With ``contrarian_beta < 0`` a *high* (hot) funding rate predicts a *lower* forward return —
    the contrarian folklore (crowded longs get flushed). ``contrarian_beta = 0`` is the null
    (funding is unrelated to forward returns). The returned frame is one row per period with the
    columns the strategy consumes: ``funding`` (the observed rate), ``z_funding`` (its trailing
    standardized value) and the realised ``fwd_ret``.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_periods) + int(horizon)

    # AR(1) funding rate.
    eps = rng.standard_normal(n)
    f = np.empty(n, dtype=float)
    f[0] = funding_scale * eps[0]
    for t in range(1, n):
        f[t] = funding_ar * f[t - 1] + funding_scale * eps[t]

    # Standardized funding (full-sample z of the observed rate).
    fmean, fstd = f.mean(), f.std(ddof=1)
    z_full = (f - fmean) / fstd if fstd > 0 else np.zeros_like(f)

    # Forward simple return over (t, t+horizon]: a horizon-scaled drift plus a contrarian response
    # to funding-z observed at t plus horizon-scaled idiosyncratic noise. The forward window is
    # strictly future, so the funding-z is fully public at the entry close (no look-ahead).
    per_period_drift = 0.30 / (FUNDING_PER_DAY * 365.0)  # ~30%/yr crypto drift, per 8h period
    scale = np.sqrt(horizon)
    idio = (idio_vol * scale) * rng.standard_normal(n_periods)
    fwd_ret = (per_period_drift * horizon) + contrarian_beta * z_full[:n_periods] + idio

    idx = pd.RangeIndex(n_periods, name="period")
    panel = pd.DataFrame(
        {
            "funding": f[:n_periods],
            "z_funding": z_full[:n_periods],
            "fwd_ret": fwd_ret,
        },
        index=idx,
    )
    return panel, WorldTruth(contrarian_beta=contrarian_beta)


# ---------------------------------------------------------------------------
# Real tape — BTC/ETH PRICES ONLY (funding is not free), cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "btc_eth_prices.parquet")


def fetch_btc_eth_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2018-01-01",
    end: str = "2026-06-30",
) -> pd.DataFrame:
    """Daily adjusted-close BTC/ETH prices, cache-first — the price tape ONLY.

    Funding-rate history is NOT reachable from free retail data; this loader exists purely so a
    reader can see the price tape the funding rate would tether to, and to make the
    data-availability limitation concrete. It never yields a funding series and cannot lift the
    study to `REAL`.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    raw = yf.download(
        PRICE_TICKERS, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
