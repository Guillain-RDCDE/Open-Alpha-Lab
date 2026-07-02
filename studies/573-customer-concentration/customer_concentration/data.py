"""Data layer for Study 573 (Customer-Concentration) — the supply-chain-fragility sort.

The claim (Patatoukas 2012; Dhaliwal, Judd, Serfling & Shaikh 2016; Hertzel, Li, Officer & Rodgers
2008): a firm that sells a large share of its revenue to a *few* large customers is fundamentally
fragile — one customer defecting, going bankrupt, or renegotiating can gut its cash flows. The two
opposing predictions this study weighs:

- **Risk story (higher forward volatility).** Concentrated firms have lumpier, less-diversified
  demand, so their forward cash-flow / return *volatility* is higher. This part is the most robust
  in the literature.
- **Return story (a premium *or* a discount).** If that fragility is a priced systematic risk, the
  concentrated names should earn a *premium* (compensation). But several papers find a *discount*
  (concentrated firms earn *lower* returns / higher valuation multiples in some cuts) — investors
  under-price the tail, or the concentration coincides with efficiency gains. The *sign* is the
  open question the return leg tests.

There is **no free, point-in-time customer-concentration dataset**. The measure comes from 10-K
"major customer" disclosures (SFAS 131 segment reporting: any customer > 10% of revenue) and
Compustat's segment files — paywalled, hand-collected, or both. A no-key retail stack cannot reach
it. So this study is **synthetic-only**: the real tape does not exist here, and the SIGNAL axis is
capped at `WEAK`/`NONE` (a `REAL` stamp needs a robust ``t >= 2`` on a real tape, by house rule).

The synthetic generator is the whole game. A single knob family plants the effect:

- ``vol_beta`` — how much a firm's concentration raises its forward return *volatility*
  (``vol_beta > 0`` = the risk story; ``= 0`` = risk-null).
- ``ret_alpha`` — the concentration return premium/discount per unit of concentration
  (``ret_alpha > 0`` = a priced premium; ``< 0`` = a fragility discount; ``= 0`` = the return-null).

With every knob at zero the panel is the pure null (concentration predicts neither risk nor
return); turning a knob on is the positive control. Tests never touch the network.

No look-ahead: concentration is an *as-of* firm characteristic (the last reported 10-K customer
mix), and forward return / realised forward vol are measured over the *subsequent* window. In the
synthetic panel these are separate columns by construction, so no future data enters the signal.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

SEED = 573


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core (the only tape here)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 400,
    vol_beta: float = 0.35,
    ret_alpha: float = 0.04,
    base_ret: float = 0.08,
    base_vol: float = 0.22,
    idio_ret_vol: float = 0.06,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible cross-section with tunable concentration → (risk, return) effects.

    Each firm draws a **customer-concentration** score. We model it as a Herfindahl-style index of
    revenue shares across a firm's customers: a firm with one dominant customer scores near 1, a
    firm with many small customers scores near 0. We draw it from a Beta(1.7, 4.0) — most firms are
    diversified, with a right tail of dangerously concentrated names.

    From the concentration ``c`` we build the two observables the strategy tests:

        forward_vol = base_vol * (1 + vol_beta * (c - c_bar)) + small idio-vol noise   [RISK leg]
        forward_ret = base_ret + ret_alpha * (c - c_bar) + N(0, forward_vol^2)          [RETURN leg]

    - ``vol_beta > 0`` plants the **risk story**: concentrated firms have higher forward volatility.
    - ``ret_alpha > 0`` plants a **priced premium**; ``< 0`` a **fragility discount**; ``= 0`` no
      return effect. Crucially the return draw's own dispersion scales with the firm's ``forward_vol``
      so the risk and return legs are internally consistent (a fragile firm is genuinely noisier).

    All knobs at zero (``vol_beta = 0``, ``ret_alpha = 0``) is the pure null. Returns
    ``(panel, truth)`` where ``panel`` has one row per firm with columns ``concentration``,
    ``forward_vol`` and ``forward_ret``, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    c = rng.beta(1.7, 4.0, size=n_stocks)  # customer concentration (Herfindahl-like), right-skewed
    c_bar = float(c.mean())

    fvol = base_vol * (1.0 + vol_beta * (c - c_bar))
    fvol = fvol + idio_ret_vol * np.abs(rng.standard_normal(n_stocks)) * 0.15
    fvol = np.clip(fvol, 0.03, 2.0)

    fret = base_ret + ret_alpha * (c - c_bar) + fvol * rng.standard_normal(n_stocks)

    tickers = [f"C{j:04d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {
            "concentration": c,
            "forward_vol": fvol,
            "forward_ret": fret,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    truth = {
        "n_stocks": n_stocks,
        "vol_beta": vol_beta,
        "ret_alpha": ret_alpha,
        "base_ret": base_ret,
        "base_vol": base_vol,
        "seed": seed,
        "has_risk": vol_beta != 0.0,
        "has_return": ret_alpha != 0.0,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — DOES NOT EXIST for a no-key retail stack (stated openly)
# ---------------------------------------------------------------------------
def fetch_panel(fetch: bool = False) -> pd.DataFrame:
    """There is no free customer-concentration tape — always returns an empty frame.

    Customer concentration comes from 10-K "major customer" (SFAS 131) disclosures and Compustat
    segment files: paywalled and/or hand-collected. A cache-first retail stack cannot reach it, so
    this study is synthetic-only and this loader is a deliberate no-op that documents the gap. The
    ``fetch`` flag is accepted for signature-parity with the desk's real-tape studies but never
    downloads anything.
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
