"""Data layer for Study 587 (NFT-Floor-Beta) — synthetic NFT floor index vs ETH returns.

The claim under test: **blue-chip NFT floor-price momentum is a risk-appetite signal that leads
crypto returns.** The sceptical prior (the "expected lean"): NFT floors are a *high-beta,
slightly-lagged* echo of ETH — they *follow* the market with extra noise rather than *leading* it,
so floor momentum has no predictive edge on forward ETH returns once you account for the lag.

## Why this study is SYNTHETIC-ONLY (data-availability limitation)

A clean, free, no-key retail tape does *not* exist for this question:

- **NFT floor prices** live behind marketplace / aggregator APIs (OpenSea, Blur, Reservoir,
  NFTGo, DappRadar). They are rate-limited, key-gated, discontinuous (collections launch, die,
  wash-trade), denominated in ETH (so they inherit ETH beta mechanically), and have only a few
  years of history. There is no ``yfinance``-style free daily floor series for "blue-chip NFTs".
- Even where floor snapshots exist, they are riddled with **wash trading**, thin liquidity, and
  survivorship (dead collections vanish), so a naive floor index is not a trustworthy tape.

So this study builds a **deterministic synthetic world** with a single knob that plants (or does
not plant) a genuine *lead* from NFT floor momentum to forward ETH returns, and asks whether the
engine can tell "leads" from "just lagged high-beta noise". Per the desk rule, a synthetic-only
study can **never** earn a REAL signal (that needs a robust ``t >= 2`` on a REAL tape); it is
capped at WEAK/NONE and the data-availability limitation is named on the SIGNAL axis. This mirrors
the desk's other synthetic-tape alt-data studies (lego-returns, whisky-cask, sneaker-resale).

## The synthetic world (one schema, one knob)

``synthetic_series`` returns a daily panel with three columns:

- ``eth_ret``    — ETH daily log return (the crypto risk-appetite tape).
- ``floor_ret``  — NFT blue-chip floor-index daily log return. By construction it is a
  **high-beta, lagged** function of ETH plus heavy idiosyncratic noise:
      floor_ret_t = beta * eth_ret_{t-1} + idio_t
  i.e. floor prices *follow* ETH with a one-day lag and amplify it (``beta`` > 1). This is the
  null-world mechanism: floors are lagged high-beta ETH noise.
- ``eth_fwd``    — the *forward* ETH return the signal is trying to predict (built inside the
  strategy from ``eth_ret``; carried here only for convenience).

The knob is ``lead_alpha``. When ``lead_alpha != 0`` a *genuine lead* is planted: today's floor
momentum carries information about *future* ETH returns beyond ETH's own autocorrelation —

      eth_ret_t += lead_alpha * floor_mom_{t-1}

where ``floor_mom`` is a trailing floor-index momentum. ``lead_alpha = 0`` is the null (floors are
pure lagged high-beta noise, no predictive lead). ``lead_alpha > 0`` is the positive control the
engine must catch; ``lead_alpha = 0`` must stay flat.

No look-ahead: the momentum signal at day ``t`` uses only floor data through day ``t``; the target
is the *forward* ETH return over ``(t, t+h]``. The one-bar execution lag (enter at the close on
which the signal is known, hold the forward window) is documented in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 587
TRADING_DAYS = 365  # crypto trades every day


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic series."""

    lead_alpha: float  # predictive lead of floor momentum on forward ETH return (0 = null)
    beta: float        # NFT floor beta to lagged ETH (the high-beta-noise mechanism)

    @property
    def has_lead(self) -> bool:
        return self.lead_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_days: int = 1500,
    lead_alpha: float = 0.0,
    beta: float = 1.6,
    mom_lookback: int = 14,
    eth_vol: float = 0.045,
    floor_idio_vol: float = 0.06,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily ETH + NFT-floor world with a tunable predictive *lead*.

    Construction (all offline, deterministic given ``seed``):

    1. Draw an ETH innovation series ``eps`` (fat-ish tails via Student-t) and build a *base* ETH
       return with mild momentum/AR(1) structure (crypto is weakly trending).
    2. Build the NFT floor return as a **lagged high-beta** echo of ETH plus heavy idiosyncratic
       noise: ``floor_ret_t = beta * eth_ret_{t-1} + idio_t``. This is the null mechanism — floors
       *follow* ETH.
    3. Compute a trailing floor **momentum** ``floor_mom`` (mean of the last ``mom_lookback`` floor
       returns), known at day ``t``.
    4. If ``lead_alpha != 0`` inject a genuine *lead*: add ``lead_alpha * floor_mom_{t-1}`` to
       today's ETH return — floor momentum now carries information about *future* ETH beyond ETH's
       own dynamics. ``lead_alpha = 0`` leaves ETH untouched (the null).

    Because the injection is done iteratively (ETH at ``t`` depends on floor momentum through
    ``t-1``, and floor at ``t`` depends on ETH at ``t-1``), the causal ordering is clean: the
    signal never sees its own future.

    Returns ``(frame, truth)`` where ``frame`` has columns ``eth_ret``, ``floor_ret``,
    ``floor_mom`` indexed by a daily ``DatetimeIndex``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days, freq="D")  # daily incl. weekends

    # ETH innovations: Student-t for mild fat tails, scaled to eth_vol
    eps = rng.standard_t(df=5, size=n_days) * (eth_vol / np.sqrt(5 / 3))
    ar = 0.05          # weak ETH autocorrelation (its own, honest, momentum)
    drift = 0.0003     # tiny positive daily drift

    eth = np.zeros(n_days)
    floor = np.zeros(n_days)
    floor_idio = floor_idio_vol * rng.standard_normal(n_days)

    # rolling buffer of floor returns for the trailing momentum
    for t in range(n_days):
        # base ETH: drift + AR(1) on its own past + innovation
        e = drift + ar * (eth[t - 1] if t > 0 else 0.0) + eps[t]
        # planted lead: floor momentum through t-1 predicts today's ETH
        if lead_alpha != 0.0 and t >= mom_lookback + 1:
            mom_prev = floor[t - mom_lookback - 1:t - 1].mean()
            e += lead_alpha * mom_prev
        # defensive clip: keep the coupled system finite even for large planted alpha
        # (a runaway feedback loop is an artefact of the knob, not the economics).
        eth[t] = float(np.clip(e, -1.0, 1.0))
        # NFT floor: lagged high-beta echo of ETH + heavy idio noise
        floor[t] = beta * (eth[t - 1] if t > 0 else 0.0) + floor_idio[t]

    df = pd.DataFrame({"eth_ret": eth, "floor_ret": floor}, index=dates)
    df["floor_mom"] = df["floor_ret"].rolling(mom_lookback).mean()
    truth = WorldTruth(lead_alpha=lead_alpha, beta=beta)
    return df, truth


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True, default=float).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
