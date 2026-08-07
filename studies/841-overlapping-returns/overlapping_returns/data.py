"""Data layer for Study 841 (Overlapping-Returns Inflation) — the synthetic monthly world.

This is a research-method demonstration, so the world is built on purpose. The whole point of the
overlapping-returns trap is that a predictive regression can print an *enormous* t-stat and R² at
long horizons **even when the predictor has zero forecasting power** — so the tape must be a
controlled null in which any long-horizon "predictability" is, by construction, an artefact of the
overlap and nothing else.

``simulate_world`` builds one monthly world with two ingredients, both drawn from the classic
predictive-regression literature (Stambaugh 1999; Hodrick 1992):

* a **persistent predictor** ``x_t`` — an AR(1) with persistence ``rho`` (≈ 0.95 monthly, the order
  of magnitude of a valuation ratio like the dividend yield or CAPE), because the size distortion of
  overlapping long-horizon regressions is worst exactly when the regressor is highly persistent;
* **monthly returns** ``r_{t+1} = ret_mean + beta * x_t + eps_{t+1}`` whose innovation ``eps`` is
  correlated with the predictor innovation (``delta`` ≈ −0.9, the Stambaugh feedback that makes
  valuation ratios notorious). The single knob that carries genuine predictability is ``beta``:

  - ``beta = 0`` is **the null** — the predictor is pure noise, so any long-horizon t-stat or R²
    above the nominal 5% level is spurious. This is the world the whole demo runs on.
  - ``beta > 0`` plants a genuine one-period edge — the **positive control** that proves the
    overlap-robust standard errors (Hodrick 1992; Newey-West) still *detect real predictability*
    (power), not just tame the null (size).

Everything is deterministic and offline (default seed 841). There is **no real-data fetch**: real
free data can never certify "zero predictability", so — exactly like the desk's other method demos
(344 backtest-overfitting, 590 sharpe-hacking) — the study is synthetic-only and capped at ``NONE``
on the SIGNAL axis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class WorldTruth:
    """The planted truth for one synthetic monthly world."""

    beta: float   # true one-period predictive slope of r_{t+1} on x_t (0 = the null the demo runs on)
    rho: float    # AR(1) persistence of the predictor

    @property
    def has_edge(self) -> bool:
        return self.beta != 0.0


def simulate_world(
    n_months: int = 600,
    beta: float = 0.0,
    rho: float = 0.95,
    delta: float = -0.9,
    ret_vol: float = 0.043,
    ret_mean: float = 0.005,
    pred_sd: float = 1.0,
    seed: int = 841,
) -> tuple[pd.DataFrame, WorldTruth]:
    """One monthly world: a persistent predictor ``x`` and monthly returns ``r``.

    The data-generating process (Stambaugh 1999 form):

        x_{t+1} = rho * x_t + u_{t+1}                      (persistent predictor)
        r_{t+1} = ret_mean + beta * x_t + eps_{t+1}        (one-period return)
        Corr(eps_{t+1}, u_{t+1}) = delta                   (Stambaugh feedback)

    with ``u`` scaled so ``x`` is a stationary AR(1) of unconditional s.d. ``pred_sd`` and ``eps`` of
    s.d. ``ret_vol``. The regressor ``x[t]`` is known at the close of month ``t`` and forecasts the
    return of month ``t+1`` onward — the one-period execution lag is baked into the alignment.

    ``beta = 0`` is the **null** (no predictability — any long-horizon t/R² is an overlap artefact);
    ``beta > 0`` plants a genuine edge (the positive control). Returns ``(df, truth)`` where ``df``
    has columns ``x`` and ``r`` on a monthly ``DatetimeIndex``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_months)

    # Predictor innovation s.d. so that Var(x) = pred_sd**2 in the stationary AR(1).
    u_sd = pred_sd * np.sqrt(max(1e-12, 1.0 - rho**2))
    u = rng.standard_normal(n) * u_sd

    # Return innovation eps with marginal s.d. ret_vol and Corr(eps, u) = delta (Stambaugh feedback).
    z = rng.standard_normal(n)
    u_std = u / u_sd  # unit-variance version of the predictor innovation
    eps = ret_vol * (delta * u_std + np.sqrt(max(0.0, 1.0 - delta**2)) * z)

    # Persistent predictor, started at its stationary distribution. The AR(1) recursion
    # x[t] = rho*x[t-1] + u[t] with x[0] as the seed is computed vectorised via an IIR filter
    # (identical to the explicit loop, but ~50x faster for the Monte Carlo).
    x0 = rng.standard_normal() * pred_sd
    u_seq = u.copy()
    u_seq[0] = x0                                    # so the filter's first output equals x0
    x = lfilter([1.0], [1.0, -rho], u_seq)

    # One-period returns: r_t = ret_mean + beta * x_{t-1} + eps_t  (x[t-1] predicts r[t]).
    r = np.empty(n)
    r[0] = ret_mean + eps[0]
    r[1:] = ret_mean + beta * x[:-1] + eps[1:]

    # PeriodIndex (not date_range) so a large ``n`` of monthly points cannot overflow
    # the pandas nanosecond-Timestamp horizon (~year 2262) on the CI's pandas build —
    # the index is a decorative month label, never used for calendar arithmetic.
    idx = pd.period_range("1970-01", periods=n, freq="M")
    df = pd.DataFrame({"x": x, "r": r}, index=idx)
    return df, WorldTruth(beta=float(beta), rho=float(rho))


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp (matches the desk's other demos)."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
