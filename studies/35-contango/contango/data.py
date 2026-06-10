"""Data for the commodity carry / roll-yield study — an offline synthetic term-structure panel, plus the
real-tape hook (which needs the deferred contracts this sandbox does not cache).

The carry of a commodity future is its **roll yield**: as a long position rolls from the front contract
toward expiry it slides along the term-structure curve. A **backwardated** curve (front > deferred) rolls
*up* — you sell the expiring contract dear and buy the next one cheap, banking a positive roll. A
**contangoed** curve (front < deferred) rolls *down* — a negative roll, the storage/convenience tax. The
documented premium (Gorton-Rouwenhorst 2006; Erb-Harvey 2006; Koijen et al. "Carry" 2018) is that
backwardated commodities systematically out-return contangoed ones, so a book long the most-backwardated
and short the most-contangoed harvests a real carry. The data layer keeps the desk's offline/cache split:

  * :func:`synthetic_term_structure` — fully **offline, deterministic**. A panel of commodities, each with
    a persistent latent **carry state** (roll yield) drifting slowly between backwardation and contango;
    that carry *predicts* the commodity's own return (positive control: high-carry commodities out-earn).
    ``carry_strength`` sets how strongly roll yield maps into return; ``carry_strength = 0`` is the
    **null** (a carry signal disconnected from returns — nothing to harvest). Returns
    ``(returns, roll_yield, truth)``.
  * :func:`fetch_curve` — the real-tape hook. Computing roll yield needs the **term structure** — at least
    the front and first-deferred contract price for each commodity, every week. yfinance does not reliably
    serve the deferred contracts (it gives a single front-month *continuous* series), and the desk's cache
    holds only that front-month tape. So this is a **cache-first stub**: it returns ``{}`` on the (current)
    cache miss, exactly like Study 27 (Steamroller) before its FRED fetch. See ``docs/results.md``.
  * :func:`load_front_month_basket` — loads the cached ``commodity_futures_weekly.parquet`` (12 commodities,
    **front-month continuous returns only**). Useful to illustrate the basket and *cross-check the universe*,
    but it is **not enough to compute roll yield** — that needs the deferred leg this file lacks.

Data choice, named up front: roll yield is a slow, weekly-to-monthly carry signal, so a weekly term
structure is the right horizon. The synthetic uses weekly bars to match the cached front-month tape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
FRONT_MONTH_CACHE = os.path.join(DEFAULT_CACHE, "commodity_futures_weekly.parquet")

WEEKS_PER_YEAR = 52


@dataclass(frozen=True)
class CarryTruth:
    """What the synthetic generator baked in, so a test can check the book recovers it."""
    n_commodities: int
    n_weeks: int
    carry_strength: float     # how strongly roll yield maps into return; 0 == disconnected null

    @property
    def has_carry(self) -> bool:
        return self.carry_strength != 0.0


def synthetic_term_structure(n_commodities: int = 12, n_weeks: int = 52 * 20,
                             carry_strength: float = 0.9, carry_persist: float = 0.985,
                             carry_spread: float = 0.0006, idio: float = 0.032,
                             mkt_vol: float = 0.012, trend_strength: float = 0.10, seed: int = 35
                             ) -> tuple[pd.DataFrame, pd.DataFrame, CarryTruth]:
    """A weekly commodity panel where **high roll-yield (backwardated) names out-return**, by construction.

    For commodity ``i`` the latent **roll yield** ``y_{i,t}`` follows a slow AR(1) around a fixed
    commodity-specific mean ``μ_i`` (some structurally backwardated, some structurally contangoed),
    ``y_{i,t} = carry_persist·y_{i,t-1} + (1-carry_persist)·μ_i + small noise``. The weekly return is::

        r_{i,t} = β_i·mkt_t  +  carry_strength · y_{i,t-1}  +  trend_strength · trend_{i,t-1}  +  idio·ε_{i,t}

    so a commodity carrying a positive roll yield (backwardation) earns it as a drift, lagged so the signal
    is tradable. A separate, *independent* slow trend component ``trend_{i,t}`` (driving the momentum sleeve
    tested in the beat-7 extension) makes the panel carry both a carry premium and a lowly-correlated trend
    premium, exactly as real commodities do. ``carry_strength`` is the slope from roll yield to return;
    ``carry_strength = 0`` keeps the roll-yield signal but **disconnects it from returns** — the null,
    nothing to harvest. Returns ``(returns, roll_yield, truth)`` as weekly ``weeks × commodity`` frames,
    deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2006-01-06", periods=n_weeks, freq="W-FRI", name="date")
    cols = [f"CMD{i:02d}" for i in range(n_commodities)]

    # structural per-commodity roll-yield mean: spread evenly from deep contango to deep backwardation
    mu = np.linspace(-carry_spread, carry_spread, n_commodities)
    rng.shuffle(mu)
    betas = np.clip(rng.normal(1.0, 0.2, n_commodities), 0.4, 1.6)

    mkt = 0.0004 + mkt_vol * rng.standard_normal(n_weeks)

    # latent roll yield (the term-structure carry), a slow AR(1) around each commodity's structural mean
    y = np.empty((n_weeks, n_commodities))
    y[0] = mu
    carry_noise = 0.0006
    for t in range(1, n_weeks):
        y[t] = carry_persist * y[t - 1] + (1.0 - carry_persist) * mu + carry_noise * rng.standard_normal(n_commodities)

    # an independent slow trend per commodity (drives the momentum sleeve; uncorrelated with carry by
    # construction, since it has its own driving noise and mean-zero target)
    trend = np.zeros((n_weeks, n_commodities))
    trend_persist, trend_noise = 0.97, 0.012
    for t in range(1, n_weeks):
        trend[t] = trend_persist * trend[t - 1] + trend_noise * rng.standard_normal(n_commodities)

    rets = np.empty((n_weeks, n_commodities))
    eps = idio * rng.standard_normal((n_weeks, n_commodities))
    for t in range(n_weeks):
        carry_drift = carry_strength * (y[t - 1] if t > 0 else y[0])   # lagged roll yield → return
        trend_drift = trend_strength * (trend[t - 1] if t > 0 else trend[0])
        rets[t] = betas * mkt[t] + carry_drift + trend_drift + eps[t]

    returns = pd.DataFrame(rets, index=idx, columns=cols)
    roll_yield = pd.DataFrame(y, index=idx, columns=cols)
    return returns, roll_yield, CarryTruth(n_commodities, n_weeks, carry_strength)


def load_front_month_basket(cache: str = FRONT_MONTH_CACHE) -> pd.DataFrame:
    """Load the cached **front-month continuous** weekly returns for the 12-commodity basket.

    This illustrates the universe and lets the verify hook confirm the basket is present — but it carries
    **no term structure**, so it cannot price roll yield. Returns an empty frame on a cache miss.
    """
    if not os.path.exists(cache):
        return pd.DataFrame()
    df = pd.read_parquet(cache)
    df.index.name = "date"
    return df


def fetch_curve(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """Real-tape hook: return ``{'front': DataFrame, 'deferred': DataFrame}`` of weekly term-structure
    prices, cache-first — **or ``{}`` on the (current) cache miss**.

    Computing roll yield needs at least the **front and first-deferred** contract for each commodity, every
    week — the slope of the curve. The desk's cache holds only the front-month *continuous* series
    (:func:`load_front_month_basket`); yfinance does not reliably serve the individual deferred contracts.
    So this hook reads a cached ``commodity_term_structure.parquet`` if one is ever populated (front_*/def_*
    columns), and otherwise returns ``{}`` — the real run is **pending a term-structure fetch**, exactly as
    Study 27 (Steamroller) was pending its FRED download. The offline synthetic core is the validated proof
    meanwhile. ``fetch=True`` is reserved for a future curve source; today it still returns ``{}`` because no
    free source serves the deferred contracts in this environment.
    """
    cache = os.path.join(cache_dir, "commodity_term_structure.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        front_cols = [c for c in df.columns if c.startswith("front_")]
        def_cols = [c for c in df.columns if c.startswith("def_")]
        if front_cols and def_cols:
            front = df[front_cols].rename(columns=lambda c: c[6:])
            deferred = df[def_cols].rename(columns=lambda c: c[4:])
            return {"front": front, "deferred": deferred}
    # cache miss: the deferred leg is not available in this environment — pending fetch.
    return {}
