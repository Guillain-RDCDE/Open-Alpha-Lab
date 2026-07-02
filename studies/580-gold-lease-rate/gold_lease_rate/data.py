"""Data layer for Study 580 (Gold-Lease-Rate) — the microstructure lead-lag on gold.

The claim under test: the **gold lease rate** (the cost to borrow bullion, classically GOFO-implied
as ``LIBOR − GOFO``) *leads* the gold price. This study is **synthetic-only** by necessity — the
LBMA discontinued the daily GOFO benchmark on 2015-01-30 and no free, continuous long-history
lease-rate tape exists — so ``data.py`` is a deterministic generator, not a fetch layer.

One knob, ``lead_beta``, plants the effect:

    gold_fwd_ret[t] = drift + lead_beta * lease_z[t-lag] + noise

- ``lead_beta > 0`` = the folklore signal: a high lease rate today foreshadows a *positive* forward
  gold return ``lag`` periods later (the positive control).
- ``lead_beta = 0`` = the null: the lease rate carries no information about future gold moves.

The lease-rate series itself is a persistent, mean-reverting (AR(1)) process with occasional
*spikes* (physical-scarcity scrambles / backwardation), matching how the real series behaves — but
its *predictive* content is governed solely by ``lead_beta``. The gold return series it produces is
a monthly total-return-style series; everything runs offline and deterministic (seed = 580).

There is a cache-first ``fetch_series`` stub for interface parity with the desk's real-data studies,
but it deliberately returns an empty frame: **the real data does not exist for free**. The honest
consequence — no real tape — is stated on the SIGNAL axis (capped at WEAK).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

TRADING_MONTHS = 12
SEED = 580


@dataclass(frozen=True)
class WorldTruth:
    """The planted lead-lag for the synthetic world."""

    lead_beta: float  # forward-gold sensitivity to lagged lease-rate z (0 = null)
    lag: int          # how many periods the lease rate leads

    @property
    def has_signal(self) -> bool:
        return self.lead_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = 300,
    lead_beta: float = 0.020,
    lag: int = 1,
    ar: float = 0.85,
    spike_prob: float = 0.06,
    gold_vol: float = 0.045,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible monthly (lease-rate, gold-return) panel with a tunable lead-lag.

    The lease rate is a mean-reverting AR(1) around a small positive level with occasional upward
    *spikes* (physical-scarcity scrambles). Its standardised value ``lease_z`` drives the forward
    gold return ``lag`` months ahead with slope ``lead_beta``:

        gold_ret[t] = drift + lead_beta * lease_z[t-lag] + gold_vol * eps[t]

    Columns returned (one row per month, ``DatetimeIndex``):

    - ``lease_rate`` — the raw synthetic lease rate (annualised, decimal, e.g. 0.015 = 1.5%)
    - ``lease_z``    — its standardised (z-scored) value, the actual predictor
    - ``gold_ret``   — the realised monthly gold total return

    With ``lead_beta = 0`` the gold return is pure noise around the drift (the null). Returns
    ``(frame, truth)``.
    """
    rng = np.random.default_rng(seed)
    lease = np.empty(n_months)
    level = 0.012                                   # ~1.2% baseline lease rate
    lease[0] = level
    for t in range(1, n_months):
        shock = 0.0025 * rng.standard_normal()
        spike = (0.02 + 0.02 * rng.random()) if rng.random() < spike_prob else 0.0
        lease[t] = level + ar * (lease[t - 1] - level) + shock + spike
    lease = np.clip(lease, -0.005, 0.15)            # lease rates can go slightly negative

    lease_z = (lease - lease.mean()) / (lease.std(ddof=1) + 1e-12)

    drift = 0.004                                   # ~4.9%/yr baseline gold drift
    eps = rng.standard_normal(n_months)
    gold_ret = np.full(n_months, drift, dtype=float)
    for t in range(n_months):
        src = t - lag
        lead_term = lead_beta * lease_z[src] if src >= 0 else 0.0
        gold_ret[t] = drift + lead_term + gold_vol * eps[t]

    idx = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    frame = pd.DataFrame(
        {"lease_rate": lease, "lease_z": lease_z, "gold_ret": gold_ret},
        index=idx,
    )
    frame.index.name = "date"
    return frame, WorldTruth(lead_beta=lead_beta, lag=lag)


# ---------------------------------------------------------------------------
# Real tape — deliberately absent (the data does not exist for free)
# ---------------------------------------------------------------------------
def fetch_series(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Cache-first stub for a *real* lease-rate tape — returns empty by design.

    The LBMA discontinued the daily GOFO benchmark on 2015-01-30, and no free, continuous,
    long-history gold-lease-rate series is reachable on a no-key retail stack. This function
    exists only for interface parity with the desk's real-data studies: it returns whatever a
    reader may have manually dropped into ``_cache/`` and otherwise an **empty frame**. ``fetch``
    is accepted but ignored — there is nothing to download.
    """
    path = os.path.join(cache_dir, "gold_lease_real.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df
    return pd.DataFrame()  # no free real data — synthetic-only study


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
