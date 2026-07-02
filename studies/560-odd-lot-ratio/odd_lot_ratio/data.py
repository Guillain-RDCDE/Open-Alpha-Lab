"""Data layer for Study 560 (Odd-Lot-Ratio) — a deterministic synthetic odd-lot tape.

The odd-lot theory needs one series the strategy fades: a weekly **odd-lot ratio** (odd-lot buy
volume as a share of total odd-lot volume, or equivalently an odd-lot *buy/sell* imbalance),
aligned to the *forward* return of a broad index. The folklore: when the odd-lot crowd is buying
heavily (a high ratio, euphoric small-money), the market is near a top and the next period is weak;
when they are dumping (a low ratio), fade the panic and the next period is strong.

Why synthetic-only. A clean, long, *point-in-time* odd-lot ratio series is proprietary
(TAQ / consolidated-tape odd-lot flags); odd lots were not even on the SIP until 2013–14, and
post-decimalization most odd-lot flow is algorithmic order-slicing, not retail. No free, no-key
retail feed exposes it. So — like the desk's other unmeasurable-asset studies (lego-returns,
whisky-cask, sneaker-resale) — the real free data does not exist, the core is synthetic-only, and
the SIGNAL axis is capped at `WEAK`/`NONE`.

The generator plants the effect with a single knob:

    forward_ret = drift + fade_alpha * (-1) * z(odd_lot_ratio_lagged) + idio

- ``fade_alpha > 0`` reproduces the **old 'dumb-money' world**: a high (lagged) odd-lot ratio →
  *lower* forward return (fading their buying pays). Positive control.
- ``fade_alpha = 0`` is the **modern post-decimalization null**: the odd-lot ratio carries no
  information about forward returns (order-slicing has scrambled it). The honest default.

Everything is seeded (seed = 560) and offline. There is no ``fetch`` path: there is no free real
tape to cache, and inventing one would violate the desk's #1 rule (never publish a number you did
not compute from real, reproducible data). The absence of a real tape is the finding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 560
TRADING_WEEKS = 52


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic odd-lot tape."""

    fade_alpha: float  # forward-return sensitivity to fading the odd-lot ratio (>0 = fade pays)

    @property
    def has_edge(self) -> bool:
        return self.fade_alpha != 0.0


def _z(a: np.ndarray) -> np.ndarray:
    sd = a.std(ddof=1)
    return (a - a.mean()) / sd if sd > 0 else a * 0.0


def synthetic_series(
    n_weeks: int = 1040,
    fade_alpha: float = 0.0,
    idio_vol: float = 0.021,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible weekly odd-lot-ratio / forward-return tape with a tunable fade edge.

    Columns (one row per week, a ~20-year weekly sample by default):

    - ``odd_lot_ratio`` — the observable sentiment gauge in [0, 1] (fraction of odd-lot volume that
      is *buying*). Persistent (crowds stay euphoric/panicked for stretches) via an AR(1), so the
      series has realistic autocorrelation — which is exactly what makes naive t-tests overstate
      significance and why the engine uses an autocorrelation-robust statistic.
    - ``forward_ret`` — the index's *next-week* total return. Under the planted edge it depends on
      the *contrarian* (negated) standardised odd-lot ratio of the **prior** week (a one-week
      execution lag: the ratio is observed at week *t* close, the position is held over week *t+1*).

    ``fade_alpha`` is the single knob: ``> 0`` plants the old 'dumb-money' fade edge (high odd-lot
    buying → weak next week); ``= 0`` is the modern null (no relation). Returns ``(frame, truth)``.
    """
    rng = np.random.default_rng(seed)

    # Persistent odd-lot ratio: AR(1) in logit space so it stays in (0,1) and clusters.
    phi = 0.85
    x = np.zeros(n_weeks)
    for t in range(1, n_weeks):
        x[t] = phi * x[t - 1] + rng.standard_normal() * 0.55
    ratio = 1.0 / (1.0 + np.exp(-x))          # logistic -> (0,1), mean ~0.5

    # The signal the strategy actually trades on is the PRIOR week's standardised ratio (lag 1).
    z_prior = np.empty(n_weeks)
    z_prior[:] = np.nan
    zr = _z(ratio)
    z_prior[1:] = zr[:-1]                       # week t sees week t-1's z-ratio

    drift = 0.0015                              # ~8%/yr weekly index drift
    idio = idio_vol * rng.standard_normal(n_weeks)
    # Fade edge: high prior odd-lot buying -> LOWER forward return => coefficient is -fade_alpha.
    edge = np.where(np.isnan(z_prior), 0.0, -fade_alpha * np.nan_to_num(z_prior))
    forward_ret = drift + edge + idio

    weeks = pd.date_range("2005-01-07", periods=n_weeks, freq="W-FRI")
    frame = pd.DataFrame(
        {
            "odd_lot_ratio": ratio,
            "z_ratio_prior": z_prior,      # standardised prior-week ratio (the traded signal)
            "forward_ret": forward_ret,
        },
        index=pd.Index(weeks, name="week"),
    )
    # Drop the first row (no prior-week signal).
    frame = frame.iloc[1:].copy()
    return frame, WorldTruth(fade_alpha=fade_alpha)


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
