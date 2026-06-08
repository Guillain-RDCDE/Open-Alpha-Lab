"""Robustness — the ways "the VIX runs on an 80-day clock" can be a mirage.

Four questions decide whether a cycle is a *clock* or a story:

1. **Does the period stay put?** A real fixed cycle has a *constant* period; a curve-fit
   has one that drifts and must be re-tuned every few months. :func:`period_stability` slides
   the spectral window and reads how far the dominant in-band period wanders. A wide range is
   the tell — the theorist re-draws the cycle precisely *because* it isn't fixed.

2. **Does it clear red noise at the claimed periods?** :func:`red_noise_pvalues` runs the
   Monte-Carlo test (`spectral.test_period`) at every period the tweet names, in one table.

3. **Does the forecast ever work?** :func:`subperiod_skill` re-runs the walk-forward direction
   test decade by decade — so a flat full-sample number can't hide a regime where it *did*
   pay (or expose one where it briefly did, by luck).

4. **Noise.** A finite sample of trades has a wide Sharpe. :func:`bootstrap_sharpe` (the
   desk standard, from ``quantlab.stats``) closes the loop on the tradeable expression.

:func:`detection_recall` is the offline mirror: on the synthetic series, the detector *must*
recover the injected periods, or we can't trust a null on the real one.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from quantlab.stats import sharpe_ci_bootstrap        # noqa: E402  (desk-standard CI)

from . import backtest, spectral                       # noqa: E402


def period_stability(
    x,
    band=spectral.DEFAULT_BAND,
    window: int = 1000,
    step: int = 125,
) -> pd.DataFrame:
    """Rolling dominant in-band period — does the 'cycle' hold its length or wander?

    Slides a ``window``-session window every ``step`` sessions and records the period with the
    most periodogram power inside ``band``. A fixed clock returns near-constant values; a
    drifting column is the curve-fit signature. Columns: ``dominant_period`` indexed by the
    window's end date.
    """
    s = pd.Series(x, dtype=float).dropna()
    rows = []
    for end in range(window, len(s) + 1, step):
        seg = s.iloc[end - window:end]
        rows.append({"end": seg.index[-1], "dominant_period": backtest.dominant_period(seg, band)})
    return pd.DataFrame(rows).set_index("end")


def red_noise_pvalues(
    x,
    targets=(40.0, 80.0, 100.0, 250.0, 1000.0),
    band_frac: float = 0.15,
    n_sim: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Monte-Carlo red-noise p-value at each claimed period, in one table.

    ``targets`` default to the tweet's own clocks: VIX 40d & 80d, plus stocks' 20-week
    (~100d), the yearly (~250d) and 4-year (~1000d) cycles. Columns: ``data_peak_power,
    p_value`` indexed by target period; ``p_value`` below 0.05 would be a cycle red noise
    struggles to fake.
    """
    rows = []
    for tp in targets:
        r = spectral.test_period(x, tp, band_frac=band_frac, n_sim=n_sim, seed=seed)
        rows.append({"target_period": tp, "data_peak_power": r["data_peak_power"],
                     "p_value": r["p_value"]})
    return pd.DataFrame(rows).set_index("target_period")


def subperiod_skill(
    x,
    bounds=("1990", "2000", "2010", "2020", "2027"),
    band=spectral.DEFAULT_BAND,
    horizon: int = 20,
    min_train: int = 500,
    step: int = 10,
    n_null: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Walk-forward direction skill, computed within each consecutive date bucket.

    Splits the sample at ``bounds`` and re-runs :func:`backtest.oos_direction_skill` inside
    each slice, so a dead full-sample number can't mask a live decade (or vice versa).
    Columns: ``skill, null_mean, p_value, n_calls``.
    """
    s = pd.Series(x, dtype=float).dropna()
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = s.loc[lo:hi]
        if len(seg) < min_train + horizon + step:
            rows.append({"period": f"{lo}-{hi}", "skill": np.nan, "null_mean": np.nan,
                         "p_value": np.nan, "n_calls": 0})
            continue
        r = backtest.oos_direction_skill(seg, band=band, horizon=horizon, min_train=min_train,
                                         step=step, n_null=n_null, seed=seed)
        rows.append({"period": f"{lo}-{hi}", "skill": r["skill"], "null_mean": r["null_mean"],
                     "p_value": r["p_value"], "n_calls": r["n_calls"]})
    return pd.DataFrame(rows).set_index("period")


def bootstrap_sharpe(daily: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI for the annualized Sharpe of the cycle trade — the desk-standard read."""
    return sharpe_ci_bootstrap(daily.dropna(), n_boot=n_boot, seed=seed)


def detection_recall(env: dict, injected, q: float = 0.99, tol_frac: float = 0.15) -> float:
    """Fraction of injected synthetic cycles recovered as significant peaks (offline check).

    For each ground-truth period, succeed if a peak clearing the ``q`` red-noise envelope sits
    within ``tol_frac`` of it. Asserts the detector finds real cycles before we trust its
    silence on the VIX. ``env`` is a :func:`spectral.red_noise_envelope` result.
    """
    peaks = spectral.significant_peaks(env, q=q)
    if len(peaks) == 0:
        return 0.0
    hit = 0
    for c in injected:
        P = c.period
        if (np.abs(peaks["period"] - P) <= tol_frac * P).any():
            hit += 1
    return hit / len(injected)
