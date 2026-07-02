"""Data layer for Study 575 (CDS-Equity-Basis) — the cross-asset credit-vs-equity gap.

The claim, at full strength: for a single name at time *t*, the credit market prices its default
risk as a **CDS spread** ``s_t`` (bp/yr of protection), while its equity implies a default risk of
its own — the market cap and equity volatility feed a **Merton (1974) structural model** that
backs out an equity-implied credit spread ``c_t``. When these two disagree, the *basis*

    basis_t = s_t - c_t

is folklore-supposed to predict **convergence**: if CDS is "too wide" relative to what the stock
says (``basis > 0``), the credit market is pricing more distress than the equity market — and the
folklore trade is to expect the *equity* to fall (the stock is complacent and will catch down),
i.e. a **negative** relation between the basis and forward equity return. The opposite sign
(``basis < 0``, CDS complacent) predicts the equity rises. This is a cross-asset *lead-lag* claim.

Why this study is **synthetic-only.** Single-name CDS spreads live on an OTC, licensed market
(Markit/IHS, Bloomberg CDSW) — there is no free, no-key retail feed of clean per-name 5-year CDS.
A structural equity-implied spread additionally needs point-in-time liabilities. Neither is on a
yfinance-class stack. So there is **no real tape** to certify the effect: this study builds the
machinery on a deterministic synthetic world, proves the engine catches a *planted* basis→return
effect and stays flat at the null, and states the data gap openly on the SIGNAL axis. A
synthetic-only study can never be REAL (that needs a robust ``t >= 2`` on a real tape) — it is
capped at WEAK / NONE.

Two tapes, one schema (a per-name-per-date **panel** of ``basis`` and ``forward_ret``):

- ``synthetic_panel`` — a reproducible, offline generator. A single knob, ``convergence_beta``,
  plants the effect: the forward equity return loads on the (lagged) basis with that coefficient
  (``convergence_beta < 0`` = the folklore convergence sign; ``= 0`` = the null). The positive
  control and the null in one bottle. Tests never touch the network.
- ``fetch_panel`` — **absent by design.** There is no free real CDS tape; the function exists only
  to document that and return an empty frame, so the notebooks' ``HAVE_REAL`` gate is always False
  and every real-tape cell falls back to the frozen synthetic-control numbers.

No look-ahead: the basis is measured at the *close* of month *t* (public then); the forward return
is the equity return over month *t+1*. The one-month execution lag is explicit — the basis at *t*
predicts the return realised over the *next* month, never the contemporaneous one.
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

SEED = 575
MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    convergence_beta: float  # loading of forward equity return on the lagged basis

    @property
    def has_effect(self) -> bool:
        return self.convergence_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 60,
    n_months: int = 96,
    convergence_beta: float = -0.90,
    idio_vol: float = 0.055,
    basis_vol_bp: float = 120.0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible name-by-month panel with a tunable CDS-equity-basis effect.

    For each name we simulate two coupled risk views:

    - a **CDS spread** ``s`` (bp) that mean-reverts around a name-specific credit level, and
    - an **equity-implied spread** ``c`` (bp) that tracks ``s`` but with its own transient
      dislocations (the equity market lags/leads credit by an AR(1) noise).

    Their difference is the **basis** ``s - c`` (bp). The *next* month's equity return then loads
    on the *current* basis:

        fwd_ret_{t+1} = market_drift + convergence_beta * (basis_t / 10000) + idio

    With ``convergence_beta < 0`` a *wide* basis (CDS more worried than equity) predicts the equity
    **falls** next month — the folklore convergence sign. ``convergence_beta = 0`` is the null
    (forward return unrelated to the basis). The returned frame is one row per (name, month) with
    the columns the strategy consumes: ``name``, ``month``, ``cds_bp``, ``eq_impl_bp``, ``basis_bp``
    and the realised ``forward_ret`` — exactly the schema ``fetch_panel`` *would* produce.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)

    # Name-specific baseline credit level (bp) — a right-skewed cross-section (most IG, a HY tail).
    base_cds = 40.0 + 260.0 * rng.beta(1.6, 5.0, size=n_names)

    rows = []
    for j in range(n_names):
        # CDS spread path: AR(1) mean-reverting around the name's baseline.
        s = np.empty(n_months)
        c = np.empty(n_months)
        s_prev = base_cds[j]
        # equity-implied spread carries its own AR(1) dislocation vs the CDS
        disloc_prev = 0.0
        for t in range(n_months):
            shock = basis_vol_bp * 0.18 * rng.standard_normal()
            s_t = 0.85 * s_prev + 0.15 * base_cds[j] + shock
            s_t = max(5.0, s_t)
            # equity-implied spread = CDS minus a mean-reverting dislocation (the basis source)
            disloc = 0.6 * disloc_prev + basis_vol_bp * rng.standard_normal()
            c_t = max(5.0, s_t - disloc)
            s[t] = s_t
            c[t] = c_t
            s_prev = s_t
            disloc_prev = disloc

        basis = s - c  # bp; >0 => CDS wider than equity-implied (credit more worried)

        # forward equity return loads on the *lagged* basis (one-month execution lag)
        drift = 0.008  # ~10%/yr monthly drift
        idio = idio_vol * rng.standard_normal(n_months)
        fwd = drift + convergence_beta * (basis / 10000.0) + idio
        # month t's forward_ret is realised over t->t+1; last month has no forward obs
        for t in range(n_months - 1):
            rows.append(
                {
                    "name": f"N{j:03d}",
                    "month": t,
                    "cds_bp": float(s[t]),
                    "eq_impl_bp": float(c[t]),
                    "basis_bp": float(basis[t]),
                    "forward_ret": float(fwd[t + 1]),
                }
            )

    panel = pd.DataFrame(rows)
    return panel, WorldTruth(convergence_beta=convergence_beta)


# ---------------------------------------------------------------------------
# Real tape — absent by design (no free single-name CDS feed)
# ---------------------------------------------------------------------------
def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real CDS-equity panel — **not available** on a no-key retail stack.

    Single-name CDS is an OTC, licensed market (Markit/IHS, Bloomberg): there is no free per-name
    5-year CDS feed, and a structural equity-implied spread additionally needs point-in-time
    liabilities. This function documents that gap and always returns an empty frame, so the
    reproducible core and the notebooks stay fully offline and every real-tape gate is False.

    ``fetch`` is accepted for signature parity with cache-first loaders elsewhere in the repo but
    is ignored — there is nothing to fetch.
    """
    return pd.DataFrame()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.select_dtypes("number").fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
