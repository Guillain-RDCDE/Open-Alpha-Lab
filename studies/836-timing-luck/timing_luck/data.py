"""Data layer for Study 836 (Rebalance Timing Luck) — the world the demo rebalances.

The pitfall under test (Hoffstein, Sober & Vezeris, *"Rebalance Timing Luck: The
(Dumb) Luck of Smart Beta"*): the **same** monthly (or annual) strategy, rebalanced on
a **different day of the period**, produces materially different equity curves and
Sharpe ratios — a *phantom dispersion* that is pure luck, not skill. Nothing about the
strategy changed; only the arbitrary choice of *which* day you rebalance did. The fix
is **tranching / overlapping portfolios** — spread the rebalance across every offset so
the choice averages out — which collapses the dispersion to a single curve.

Like the desk's other research-method demos, the underlying tape is **built on
purpose** so the interpretation is unambiguous:

- ``synthetic_panel(mom_edge=0.0, ...)`` — the **null**: a cross-section of assets with
  a common (dollar-neutral-cancelling) market factor and pure idiosyncratic noise, so a
  cross-sectional **momentum** sort has **no genuine edge**. Trailing returns still vary
  across names, so the momentum book still takes real positions and each rebalance
  offset still traces a different equity curve — but the Sharpe *dispersion* across
  offsets is manifestly luck: there is nothing real underneath.
- ``synthetic_panel(mom_edge>0, ...)`` — the **positive control**: each name carries a
  persistent latent "trend" ``s_i`` that both shapes its trailing return (so the
  momentum sort proxies it) and — only when ``mom_edge > 0`` — genuinely predicts its
  forward return. A real momentum premium the *tranched* portfolio should harvest,
  proving the machinery detects real edge and is not itself the artefact.

Every asset shares ``beta = 1`` on the common market factor, so a dollar-neutral
long-short book cancels the market **exactly** — the null has no accidental beta tilt
masquerading as edge. Everything is deterministic and offline (fixed seed = 836); the
tests never touch the network, and there is **no real-data fetch**: real free data can
never certify "zero momentum edge", so the study is synthetic-only and capped at
``NONE`` on the SIGNAL axis (stated openly, like the desk's sharpe-hacking /
backtest-overfitting demos).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
AS_OF = "2026-06-30"        # publication stamp (the synthetic tape is calendar-agnostic)


@dataclass(frozen=True)
class WorldTruth:
    """The planted truth for a synthetic cross-section."""

    mom_edge: float   # strength of the planted momentum premium (0 = the null)

    @property
    def has_edge(self) -> bool:
        return self.mom_edge != 0.0


def synthetic_panel(
    mom_edge: float = 0.0,
    seed: int = 836,
    n_assets: int = 30,
    n_days: int = 2600,
    idio_vol: float = 0.015,
    mkt_vol: float = 0.010,
    mkt_drift: float = 0.0002,
    trend_rho: float = 0.98,
    prem: float = 0.0012,
    start: str = "2012-01-02",
) -> tuple[pd.DataFrame, WorldTruth]:
    """A deterministic daily **return panel** with a tunable momentum premium.

    Construction (all assets share ``beta = 1`` on the market, so a dollar-neutral
    long-short book cancels the market exactly):

        market[t] ~ N(mkt_drift, mkt_vol)
        s_i[t]     = trend_rho * s_i[t-1] + sqrt(1 - trend_rho^2) * eps   (unit-var AR(1))
        r[i,t]     = market[t] + mom_edge * prem * s_i[t] + idio_vol * noise

    The latent trend ``s_i`` is **persistent** (``trend_rho`` close to 1), so a trailing
    return proxies it. When ``mom_edge > 0`` the trend both drives past returns *and*
    predicts forward returns — a genuine momentum premium. When ``mom_edge = 0`` the
    trend is absent from returns entirely: trailing returns still vary across names
    (idiosyncratic noise), so the momentum sort still trades, but it predicts **nothing**
    — the null. Business-day index; span well below the pandas ns-timestamp horizon.

    Returns ``(returns, truth)`` where ``returns`` is a daily ``pd.DataFrame`` (index =
    business-day calendar, columns = ``A00 .. A{n-1}``).
    """
    rng = np.random.default_rng(seed)

    market = rng.normal(mkt_drift, mkt_vol, n_days)

    innov_sd = np.sqrt(1.0 - trend_rho ** 2)
    s = np.empty((n_days, n_assets))
    s[0] = rng.normal(0.0, 1.0, n_assets)
    eps = rng.normal(0.0, innov_sd, (n_days, n_assets))
    for t in range(1, n_days):
        s[t] = trend_rho * s[t - 1] + eps[t]

    noise = rng.normal(0.0, idio_vol, (n_days, n_assets))
    r = market[:, None] + mom_edge * prem * s + noise

    idx = pd.bdate_range(start, periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(r, index=idx, columns=cols)
    return returns, WorldTruth(mom_edge=mom_edge)


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
