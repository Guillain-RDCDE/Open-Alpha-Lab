"""Data layer for Study 571 (Pension-Underfunding) — the Franzoni-Marin funding-status sort.

The claim (Franzoni & Marin 2006, *"Pension Plan Funding and Stock Market Efficiency,"* JF): a
firm's defined-benefit pension plan can be badly **underfunded** — the projected benefit
obligation (PBO) exceeds plan assets — and that shortfall is a real, senior, off-balance-sheet
liability. Franzoni-Marin argue the market **under-reacts** to it: the most-underfunded firms go on
to earn *anomalously low* returns (a negative alpha of roughly −7%/yr in their sample), as if
investors ignore the hidden leverage until it bites.

The natural funding measure is the *funded status scaled by market cap*:

    funding_ratio = plan_assets / PBO          (< 1 = underfunded)
    funding_gap   = (plan_assets - PBO) / mktcap  (< 0 = a hole, sized by how big it is vs equity)

The prediction: firms with the **largest holes** (most negative ``funding_gap``) earn the
**lowest** forward returns.

## Data availability — why this study is synthetic-only

The funded status lives in the pension footnote of the 10-K (Compustat items ``PPROBF``/``PBPRO``
and, post-SFAS 158, the balance-sheet net funded status). A no-key retail stack (yfinance) exposes
*none* of it: ``.balance_sheet`` carries no pension line, and there is no free point-in-time
pension-footnote panel. Reconstructing Franzoni-Marin honestly would need Compustat. So this study
ships a **deterministic synthetic panel only**, and states the limitation on the SIGNAL axis:

  > A synthetic-only study can NEVER be `REAL` (that needs a robust t >= 2 on a real tape). The
  > engine here proves it *would* catch the effect if the data existed; the real-tape verdict is
  > left open (`NONE` — not evidenced on a real tape), not asserted.

## Two tapes, one schema

- ``synthetic_panel`` — a deterministic, offline generator. A single knob, ``underfunding_alpha``,
  plants the anomaly: the deeper a firm's pension hole, the **lower** its forward alpha
  (``underfunding_alpha < 0`` reproduces the Franzoni-Marin puzzle; ``= 0`` is the null). The
  study's positive control and null in one bottle. Tests never touch the network.
- ``real_panel`` — a stub. The real point-in-time pension-funding panel is not free; this returns
  an empty frame and documents where the data would come from (Compustat pension items). It exists
  so the notebooks' ``HAVE_REAL`` gate is always ``False`` here, by construction.
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

SEED = 571  # study number = master seed


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    underfunding_alpha: float  # forward alpha per unit of pension hole (negative = the puzzle)

    @property
    def has_anomaly(self) -> bool:
        return self.underfunding_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 300,
    underfunding_alpha: float = -0.05,
    idio_vol: float = 0.18,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-section with a tunable pension-underfunding anomaly.

    Each firm carries a defined-benefit pension plan with a projected benefit obligation ``PBO``
    and plan ``assets``. The observable funding measures are:

        funding_ratio = assets / PBO                     (< 1 = underfunded)
        funding_gap   = (assets - PBO) / mktcap          (< 0 = a hole scaled by equity)

    The forward return is:

        forward_ret = market_drift
                    + underfunding_alpha * hole_z        (hole_z = z-scored size of the hole)
                    + size_tilt + idio

    where ``hole_z`` is large-positive for the most-underfunded firms, so with
    ``underfunding_alpha < 0`` those firms earn the *lowest* forward return (the Franzoni-Marin
    puzzle) and ``underfunding_alpha = 0`` is the null (return unrelated to the pension hole).

    A mild, orthogonalised ``size_tilt`` is added so the panel is not a single clean regressor —
    it mimics the confound Franzoni-Marin control for (underfunded firms skew larger/older), and
    forces the placebo/robustness to work for their significance rather than getting it for free.

    Returns ``(panel, truth)`` with one row per firm and the columns the strategy consumes:
    ``funding_ratio``, ``funding_gap``, ``mktcap``, ``forward_ret`` (plus a hidden ``true_hole``).
    """
    rng = np.random.default_rng(seed)

    # Market cap (log-normal): most firms mid-cap, a right tail of mega-caps.
    log_mktcap = rng.normal(8.0, 1.1, size=n_stocks)            # ln($mn)
    mktcap = np.exp(log_mktcap)

    # PBO scaled by market cap: pension obligations are a meaningful fraction of equity for
    # old-economy names, negligible for young/tech names. Right-skewed.
    pbo_over_cap = rng.gamma(shape=1.4, scale=0.28, size=n_stocks)   # PBO / mktcap
    pbo = pbo_over_cap * mktcap

    # Funding ratio: centred a touch below 1 (a typical DB plan is modestly underfunded), with a
    # left tail of badly underfunded plans.
    funding_ratio = np.clip(0.72 + 0.22 * rng.standard_normal(n_stocks), 0.30, 1.35)
    assets = funding_ratio * pbo

    # The hole scaled by equity: negative = underfunded. This is the signal Franzoni-Marin price.
    funding_gap = (assets - pbo) / mktcap                       # <= 0 for underfunded firms

    # The "size of the hole" the anomaly acts on: how deep, z-scored across the panel. Deeper hole
    # (more negative funding_gap) => larger hole_z.
    hole = -funding_gap                                          # >= 0, bigger = deeper hole
    hole_z = (hole - hole.mean()) / hole.std()

    drift = 0.08
    # A mild size confound, orthogonalised against hole_z so it is a genuine nuisance, not a proxy.
    sz = (log_mktcap - log_mktcap.mean()) / log_mktcap.std()
    beta_hz = float(np.dot(sz, hole_z) / np.dot(hole_z, hole_z))
    sz_orth = sz - beta_hz * hole_z
    size_tilt = 0.015 * sz_orth

    idio = idio_vol * rng.standard_normal(n_stocks)
    forward_ret = drift + underfunding_alpha * hole_z + size_tilt + idio

    tickers = [f"P{j:04d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {
            "funding_ratio": funding_ratio,
            "funding_gap": funding_gap,
            "pbo_over_cap": pbo_over_cap,
            "mktcap": mktcap,
            "forward_ret": forward_ret,
            "true_hole": hole,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(underfunding_alpha=underfunding_alpha)


# ---------------------------------------------------------------------------
# Real tape — a documented stub (the point-in-time pension panel is not free)
# ---------------------------------------------------------------------------
def real_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """The real pension-funding cross-section — NOT AVAILABLE on a free retail stack.

    The Franzoni-Marin funding measure needs point-in-time pension-footnote data (Compustat
    pension items: the projected benefit obligation and plan assets, or the post-SFAS-158
    balance-sheet net funded status), scaled by market cap. yfinance exposes no pension line and
    there is no free point-in-time pension-footnote panel, so this returns an **empty frame** by
    construction. The notebooks' ``HAVE_REAL`` gate is therefore always ``False`` for this study,
    and the SIGNAL axis is capped accordingly.

    ``fetch`` is accepted for interface symmetry with the desk's cache-first loaders but is a
    no-op: there is no free endpoint to fetch from.
    """
    _ = (cache_dir, fetch)
    return pd.DataFrame()  # real point-in-time pension footnotes are not reachable for free


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
