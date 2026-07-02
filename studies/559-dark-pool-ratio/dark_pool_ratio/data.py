"""Data layer for Study 559 (Dark-Pool-Ratio) — a synthetic microstructure panel.

The claim under test: a stock's **dark-pool ratio** (DPR) — the share of its total volume that
prints off-exchange, in ATSs / dark pools / internalisers rather than on the lit tape — is a
tell of *informed accumulation*. Patient institutions supposedly work large buy orders in the dark
to avoid moving the price, so a rising DPR is smart money loading up, and the stock drifts up next.

**Why synthetic-only.** A true per-name daily DPR joined to forward returns is not reachable from a
free, no-key retail stack:

- FINRA publishes ATS ("dark pool") share volume only **weekly**, aggregated, and released on a
  **two-to-four-week lag** — no clean point-in-time daily panel.
- The large *off-exchange, non-ATS* internalised flow (wholesalers / single-dealer platforms) is
  not in the FINRA ATS file at all, so "off-exchange %" cannot even be fully reconstructed from it.
- Vendors who sell a clean daily DPR (e.g. the ones behind the retail "dark pool index" charts)
  are paywalled.

So there is nothing honest to cache-fetch. This module ships a **deterministic synthetic
generator** only; the real-data limitation is named on the SIGNAL axis and caps the study at
`WEAK`/`NONE` (a `REAL` stamp needs a robust t ≥ 2 on a real tape, which does not exist here).

``synthetic_panel`` has a single knob, ``dpr_alpha``, planting the effect: forward return loads on
the *cross-sectionally demeaned* dark-pool ratio with coefficient ``dpr_alpha`` (positive = the
folklore, high-DPR names drift up; ``0`` = the null, DPR carries no information). This is the
study's positive control and null in one bottle. Everything runs offline, seeded on 559.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 559
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    dpr_alpha: float  # forward-return loading per unit of demeaned dark-pool ratio

    @property
    def has_effect(self) -> bool:
        return self.dpr_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 300,
    dpr_alpha: float = 0.06,
    idio_vol: float = 0.09,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-section of dark-pool ratios vs forward returns.

    Each firm draws a **dark-pool ratio** ``dpr`` (share of volume printed off-exchange) from a Beta
    distribution centred near a realistic ~40% off-exchange mean, with a right tail of very-dark
    names. It also gets two confounders a naive tester would worry about — a size proxy and a
    liquidity proxy (both correlated with DPR in reality, since large liquid names internalise more)
    — so the panel is not a trivially clean signal.

    Forward return is::

        forward_ret = drift + dpr_alpha * (dpr - dpr_bar) + size_load * size_z + idio

    With ``dpr_alpha > 0`` the high-dark-pool names drift **up** (the folklore); ``dpr_alpha = 0``
    is the null (DPR carries no forward information, the confounders still move returns). The frame
    has one row per firm with the columns the strategy consumes.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)

    # Dark-pool ratio: most names 30-50% off-exchange, a right tail of very-dark names.
    dpr = np.clip(rng.beta(4.0, 6.0, size=n_stocks), 0.02, 0.95)   # mean ~0.40
    dpr_bar = dpr.mean()

    # Confounders correlated with DPR (bigger, more-liquid names internalise more off-exchange).
    size_z = (dpr - dpr_bar) / dpr.std() * 0.35 + rng.standard_normal(n_stocks)
    liq_z = (dpr - dpr_bar) / dpr.std() * 0.3 + rng.standard_normal(n_stocks)

    drift = 0.04
    size_load = 0.015  # a real, non-DPR return driver (the confound a naive DPR sort mistakes for signal)
    idio = idio_vol * rng.standard_normal(n_stocks)
    forward_ret = drift + dpr_alpha * (dpr - dpr_bar) + size_load * size_z + idio

    tickers = [f"S{j:03d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {
            "dark_pool_ratio": dpr,
            "size_z": size_z,
            "liq_z": liq_z,
            "forward_ret": forward_ret,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(dpr_alpha=dpr_alpha)


def synthetic_null_panel(n_stocks: int = 300, seed: int = SEED):
    """Convenience: the null world (``dpr_alpha = 0``) — DPR carries no forward information."""
    return synthetic_panel(n_stocks=n_stocks, dpr_alpha=0.0, seed=seed)


# ---------------------------------------------------------------------------
# Real tape — deliberately unavailable (documented above); kept as an honest stub
# ---------------------------------------------------------------------------
def fetch_panel(fetch: bool = False) -> pd.DataFrame:
    """Real per-name daily dark-pool-ratio panel — **not available** on a free retail stack.

    See the module docstring: FINRA ATS volume is weekly and lagged, off-ATS internalised flow is
    excluded, and clean daily DPR feeds are paywalled. There is nothing honest to cache-fetch, so
    this returns an empty frame regardless of ``fetch``. The synthetic panel is the whole study;
    the tape gap is stated on the SIGNAL axis and caps the verdict below `REAL`.
    """
    return pd.DataFrame()


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
