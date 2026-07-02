"""Data layer for Study 557 (Borrow-Fee-Signal) — the securities-lending-fee cross-section.

The claim (Cohen, Diether & Malloy 2007, *"Supply and Demand Shifts in the Shorting Market"*,
JF; D'Avolio 2002): the **cost to borrow** a stock — the securities-lending fee — is a
demand-driven price for shorting. When shorting demand outstrips lendable supply, the fee spikes,
the name is *special* / *hard-to-borrow*, and the crowd of informed short sellers pushing that
demand predicts **negative future returns**. Crucially the fee is claimed to carry information
**beyond raw short interest**: two stocks with the same short-% of float can have very different
borrow fees (one has ample lendable supply, one is on-special), and it is the *fee* — the supply
side — that sharpens the signal.

Data-availability limitation (stated openly on the SIGNAL axis): a clean, historical, free
panel of daily stock-borrow fees does **not** exist for a retail stack. The lending market is
opaque; the good data (Markit / IHS / DataExplorers, S3 Partners) is expensive and private.
yfinance exposes short *interest* (via ``.info['shortPercentOfFloat']``) but **not** the borrow
fee. So this study is **synthetic-only**: we build a deterministic borrow-fee panel with a single
knob that plants the effect, prove the engine catches it, and cap the verdict at WEAK/NONE
(a synthetic-only study can never earn REAL — that needs a robust t >= 2 on a real tape).

Distinct from Study 262 (Short-Interest): 262 sorts on short **% of float** (the *quantity*
demanded). This study sorts on the borrow **fee** (the *price* of that demand), and explicitly
carries a raw-short-interest column so the engine can test whether the fee adds signal *beyond*
short interest (a partial-correlation / double-sort argument).

Two entry points, one schema:

- ``synthetic_panel`` — a deterministic, offline generator. A single knob, ``fee_alpha``, plants
  the anomaly: the higher a firm's borrow fee (relative to the cross-section), the **lower** its
  forward return (``fee_alpha < 0`` reproduces the claim; ``= 0`` is the null). Short interest is
  generated as a correlated-but-distinct variable so the fee's *incremental* signal is testable.
- ``fetch_panel`` — real free borrow-fee data does not exist; this raises with a clear message
  pointing at the synthetic core. (Kept for schema parity with the rest of the desk; the study's
  reproducible core is fully offline.)

No look-ahead: the borrow fee and short interest are *as-of the formation date* (public that day);
the forward return is measured strictly *after* formation. The single execution lag (form on the
snapshot, enter at that close, hold the forward window) is documented in ``strategy.py``.
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

SEED = 557
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    fee_alpha: float  # forward-return alpha per unit of standardized borrow fee (negative = claim)

    @property
    def has_anomaly(self) -> bool:
        return self.fee_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 120,
    fee_alpha: float = -0.035,
    si_alpha: float = 0.0,
    idio_vol: float = 0.18,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible borrow-fee cross-section with a tunable shorting anomaly.

    Each firm has a latent *shorting-demand pressure* ``d`` (how crowded the short trade is).
    From ``d`` we derive the two **observable** microstructure variables:

    - ``borrow_fee`` — the annualised securities-lending fee (%). Most names are *general
      collateral* (a low, floor-y fee); a right tail is *special* / hard-to-borrow with a fee that
      rises steeply in ``d`` and in a scarcity of lendable supply. This is the study's signal.
    - ``short_interest`` — short % of float. Also rises with ``d`` but is a *noisier, distinct*
      read (Study 262's variable): a name can be heavily shorted yet easy to borrow (ample
      supply → low fee), or lightly shorted yet on-special (thin float → high fee). We inject an
      independent supply component into the fee so the two variables are correlated but not
      collinear — that gap is exactly where the fee's *incremental* signal lives.

    Forward return:

        forward_ret = drift + fee_alpha * z(borrow_fee) + si_alpha * z(short_interest) + idio

    With ``fee_alpha < 0`` the most-expensive-to-borrow firms earn the *lowest* return (the claim);
    ``fee_alpha = 0`` is the null. ``si_alpha`` lets a positive control plant signal in *short
    interest instead* — used to show the fee-sort does **not** fire on a pure-SI world (the
    dedup-from-262 argument).

    Returns ``(panel, truth)`` with columns ``borrow_fee`` (%), ``short_interest`` (%),
    ``forward_ret`` and the latent ``true_d`` — the same schema a real fee tape would produce.
    """
    rng = np.random.default_rng(seed)

    # Latent shorting-demand pressure: most names quiet, a right tail crowded.
    d = rng.beta(1.7, 6.0, size=n_stocks)                       # in [0,1], right-skewed
    d_z = (d - d.mean()) / d.std(ddof=1)

    # Lendable-supply scarcity: an INDEPENDENT driver of the fee (thin float, concentrated
    # holders) that short interest does not see. This is what separates fee from SI.
    supply_scarcity = rng.beta(2.0, 5.0, size=n_stocks)
    supply_z = (supply_scarcity - supply_scarcity.mean()) / supply_scarcity.std(ddof=1)

    # Borrow fee (annualised %): a low general-collateral floor plus a special component that
    # rises steeply in demand AND in supply scarcity, so a right tail reaches realistic
    # hard-to-borrow levels (double-digit fees for the most special names — GameStop-2021 territory
    # is >100%, we stay milder). Right-skewed, non-negative, in percent.
    fee_raw = (
        0.30
        + 18.0 * np.maximum(d - 0.15, 0.0) ** 1.5           # demand-driven special component
        + 6.0 * np.maximum(supply_scarcity - 0.35, 0.0) ** 1.3  # scarcity-driven component
    )
    borrow_fee = np.clip(fee_raw * np.exp(0.25 * rng.standard_normal(n_stocks)), 0.05, 120.0)

    # Short interest (% of float): rises with demand, its own noise, NO supply component.
    short_interest = np.clip(
        2.0 + 22.0 * d + 2.5 * rng.standard_normal(n_stocks), 0.1, 60.0
    )

    # Standardise the observables the alpha loads on (so fee_alpha is per-cross-section-sigma).
    def _z(x):
        return (x - x.mean()) / x.std(ddof=1)

    fee_zz = _z(borrow_fee)
    si_zz = _z(short_interest)

    drift = 0.06
    idio = idio_vol * rng.standard_normal(n_stocks)
    forward_ret = drift + fee_alpha * fee_zz + si_alpha * si_zz + idio

    tickers = [f"S{j:03d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {
            "borrow_fee": borrow_fee,
            "short_interest": short_interest,
            "forward_ret": forward_ret,
            "true_d": d,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(fee_alpha=fee_alpha)


# ---------------------------------------------------------------------------
# Real tape — does not exist for free; schema-parity stub
# ---------------------------------------------------------------------------
def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real borrow-fee panel — unavailable on a free retail stack.

    A clean historical panel of daily stock-borrow fees is private, paid data (Markit / IHS
    DataExplorers, S3 Partners). yfinance exposes short *interest* but not the borrow *fee*. So
    this returns an empty frame (``fetch=False``, the default) or raises a clear message on an
    explicit ``fetch=True`` — the reproducible core is the deterministic ``synthetic_panel``.
    """
    if not fetch:
        return pd.DataFrame()
    raise RuntimeError(
        "No free historical stock-borrow-fee panel exists for a retail stack "
        "(the data is private: Markit/IHS DataExplorers, S3 Partners). "
        "Study 557 is synthetic-only by construction; use synthetic_panel(). "
        "See docs/results.md — the verdict is capped at WEAK/NONE for want of a real tape."
    )


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
