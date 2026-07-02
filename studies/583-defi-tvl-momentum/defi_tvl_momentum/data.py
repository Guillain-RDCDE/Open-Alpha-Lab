"""Data layer for Study 583 (DeFi-TVL-Momentum) — the on-chain "flows follow price" claim.

The folklore, stated plainly: when money floods into a DeFi protocol — its **total value
locked** (TVL) climbs — the protocol's governance token keeps *pumping*. TVL is treated as a
leading, on-chain "smart money" flow signal: locked capital is a public, real-time proxy for
demand that (the story goes) front-runs the token's price.

The tradable version is a cross-sectional **TVL-momentum sort**: each month, rank protocols by
their trailing TVL growth, go long the top basket (fastest inflows) and short the bottom
(outflows), and collect the forward token-return spread.

### Why this study is synthetic-only

There is no free, no-key tape that a retail research stack can *cleanly* reach for the real
version of this test. It needs a **survivorship-free monthly panel** of per-protocol TVL joined
to the same protocols' **token total returns** — with dead protocols (rugs, exploits, abandoned
forks) still present, and TVL measured point-in-time (not back-filled after the fact). DefiLlama
exposes per-protocol TVL, but: (a) its free history is a moving, revised series (protocols are
re-categorised and re-based), (b) matching each protocol to a clean token-return series is a
manual, error-prone join, and (c) the universe that mattered in any given month is riddled with
tokens that later went to zero — exactly the names a live index quietly drops. Any panel we could
scrape today would be **survivorship-selected upward** and **look-ahead-contaminated** by
revisions. So this study is honest about that: it builds a deterministic synthetic world, proves
the engine catches a planted TVL→return effect, and — per the desk rule for the lego-returns /
whisky-cask / sneaker-resale kind of study — **caps the Signal at WEAK** because no robust real
tape backs it.

One schema, two knobs of one generator:

- ``synthetic_panel(tvl_alpha>0)`` — plants the effect: protocols with faster trailing TVL growth
  earn higher forward token returns (the positive control).
- ``synthetic_panel(tvl_alpha=0)`` — the null: forward token returns are unrelated to TVL flow.

No look-ahead: TVL growth is measured over a *trailing* formation window ending at the ranking
date; the position is entered at that date and held over the *forward* window. No future TVL or
price enters the signal. The one-period entry lag (rank on month *t*'s close, hold *t*→*t+h*) is
the single documented execution convention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 583
MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic DeFi-TVL panel."""

    tvl_alpha: float  # forward-return premium per unit of trailing TVL-growth z-score

    @property
    def has_effect(self) -> bool:
        return self.tvl_alpha != 0.0


def synthetic_panel(
    n_protocols: int = 60,
    n_months: int = 48,
    tvl_alpha: float = 0.05,
    tvl_vol: float = 0.35,
    ret_vol: float = 0.22,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible monthly DeFi panel with a tunable TVL-momentum effect.

    Each protocol carries a slowly-drifting *fundamental demand* state that drives **both** its
    TVL and (optionally) its forward token return. Monthly:

    - **TVL flow** ``g_{i,t}`` = trailing log-growth of TVL — the observable ranking signal — is a
      noisy read on the demand state plus a persistent protocol-specific drift.
    - **Forward token return** ``r_{i,t+1}`` = a common crypto-beta market factor + a protocol
      idiosyncratic return + ``tvl_alpha`` × (this month's TVL-growth z-score). With
      ``tvl_alpha > 0`` inflows predict outperformance (the claim); ``= 0`` is the null.

    The returned long frame has one row per (protocol, month) with the columns the strategy
    consumes: ``tvl_growth`` (trailing, the signal) and ``fwd_ret`` (next-month token total
    return, the target). Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)

    # Persistent per-protocol demand drift (some protocols structurally grow, some fade).
    proto_drift = 0.02 * rng.standard_normal(n_protocols)

    # Latent monthly demand shocks, mildly autocorrelated (money chases momentum on-chain).
    shocks = rng.standard_normal((n_months + 1, n_protocols))
    demand = np.zeros((n_months + 1, n_protocols))
    rho = 0.4
    for t in range(1, n_months + 1):
        demand[t] = rho * demand[t - 1] + shocks[t]

    # Common crypto market factor (a rising tide) + idiosyncratic token returns.
    market = 0.01 + 0.08 * rng.standard_normal(n_months + 1)  # monthly market factor
    beta = np.clip(1.0 + 0.3 * rng.standard_normal(n_protocols), 0.2, 2.0)

    rows = []
    for t in range(1, n_months):
        # Observable trailing TVL growth (the ranking signal at month t).
        tvl_growth = proto_drift + tvl_vol * demand[t]
        gz = _zscore(tvl_growth)  # cross-sectional z of the flow signal

        # Forward (t -> t+1) token total return.
        idio = ret_vol * rng.standard_normal(n_protocols)
        fwd_ret = beta * market[t + 1] + idio + tvl_alpha * gz

        for i in range(n_protocols):
            rows.append(
                {
                    "month": t,
                    "protocol": f"P{i:03d}",
                    "tvl_growth": float(tvl_growth[i]),
                    "fwd_ret": float(fwd_ret[i]),
                }
            )

    panel = pd.DataFrame(rows)
    return panel, WorldTruth(tvl_alpha=tvl_alpha)


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if not sd or sd <= 0:
        return x * 0.0
    return (x - x.mean()) / sd


# ---------------------------------------------------------------------------
# Real tape — intentionally unavailable on a no-key retail stack (see module docstring)
# ---------------------------------------------------------------------------
def fetch_panel(fetch: bool = False) -> pd.DataFrame:
    """Real per-protocol TVL x token-return panel — **not available** here.

    A faithful real test needs a survivorship-free, point-in-time monthly panel of per-protocol
    DeFi TVL joined to the same protocols' token total returns, including the protocols that later
    went to zero. That cannot be assembled cleanly from a free, no-key source (see the module
    docstring). This function therefore always returns an empty frame; the study is synthetic-only
    by construction and the Signal axis is capped at WEAK accordingly.

    Kept as an explicit, cache-first stub so the study's shape matches its real-tape siblings.
    """
    return pd.DataFrame()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.select_dtypes("number").fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
