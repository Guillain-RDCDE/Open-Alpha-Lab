"""Data layer for Study 840 — Clustered Standard Errors (cross-sectional dependence).

This is a *research-method* study, so the "data" is a controlled null panel we build
ourselves. The pitfall under the microscope is **cross-sectional correlation** — the
Petersen (2009) *time effect*: in a given period, a common shock hits every firm at once,
so the residuals of a pooled regression are correlated **across firms within a period**.
Treat the ``N × T`` observations as if they were independent and the ordinary OLS standard
error is far too small, so the *t*-statistic **overstates significance** — and, crucially,
clustering on the **wrong** dimension (by firm) does **not** fix it. Fama-MacBeth (or
two-way / time clustering) does.

The generator plants the two ingredients of a Petersen time effect, both in closed form:

* **A common time factor in the regressor.** ``x_{it} = sqrt(rho_x) * f_t +
  sqrt(1-rho_x) * u_{it}`` — a shared per-period draw ``f_t`` (same for every firm) plus an
  idiosyncratic part. ``x`` then has unit variance and an **intra-period correlation of
  exactly ``rho_x``**.
* **A common time factor in the residual.** ``e_{it} = sqrt(rho_e) * g_t +
  sqrt(1-rho_e) * v_{it}`` — same structure, intra-period correlation ``rho_e``.

The observed outcome is ``y_{it} = beta * x_{it} + e_{it}``. The two time factors ``f_t``
and ``g_t`` are drawn **independently**, so in the population ``x`` and ``e`` are
uncorrelated and the pooled slope ``beta_hat`` is centred at the true ``beta``: the point
estimate is fine. The damage is **entirely in the standard error**. Under the null
``beta = 0`` there is *nothing to find*; the naive OLS *t* nonetheless rejects far more than
5% of the time because it under-states its own sampling variability.

The variance inflation has an exact closed form — the **Moulton (1986) factor** for equal
cluster sizes:

    tau = 1 + (N - 1) * rho_x * rho_e            (variance ratio, true / naive)
    naive-t inflation = sqrt(tau)

With ``N = 50`` firms and ``rho_x = rho_e = 0.5`` this is ``sqrt(1 + 49*0.25) = sqrt(13.25)
= 3.64`` — the naive OLS SE is understated ~3.6x, purely from the shared time shock, with
**no real effect present**.

``beta = 0`` is the null (the whole demonstration); ``beta > 0`` plants a genuine effect
(the positive control that proves Fama-MacBeth still *fires* when there really is
something). Everything here is pure numpy + scipy + stdlib, deterministic under a fixed
seed. No network, no market data, no cache — a simulation study end to end.
"""

from __future__ import annotations

import hashlib

import numpy as np

AS_OF = "2026-06-30"        # frozen stamp for the headline Monte-Carlo run

# The frozen simulation configuration behind docs/results.md (the "tape" of a sim study).
CONFIG = {
    "seed": 840,
    "n_reps": 2000,         # Monte-Carlo replications for the false-positive experiment
    "n_firms": 50,          # N — firms observed each period (the cross-section)
    "n_periods": 50,        # T — number of periods (the time-series length)
    "rho_x": 0.5,           # intra-period correlation of the regressor  (fraction from f_t)
    "rho_e": 0.5,           # intra-period correlation of the residual   (fraction from g_t)
    "crit": 1.96,           # two-sided 5% Gaussian critical value
    "control_beta": 0.06,   # planted slope for the positive control (Fama-MacBeth power)
    "ret_scale": 0.01,      # nominal return scale for the costed notional timer
    "rho_e_grid": [0.0, 0.2, 0.4, 0.6, 0.8],   # residual intra-period corr dial
    "n_firms_grid": [2, 5, 10, 25, 50, 100],   # cluster-size dial (N)
}

__all__ = [
    "AS_OF", "CONFIG",
    "panel", "one_panel",
    "theoretical_moulton",
    "fingerprint", "config_fingerprint",
]


# --------------------------------------------------------------------------- #
# Monte-Carlo panels  (n_reps x n_periods x n_firms) — the experiment input
# --------------------------------------------------------------------------- #
def panel(
    n_reps: int,
    n_periods: int,
    n_firms: int,
    rho_x: float = 0.5,
    rho_e: float = 0.5,
    beta: float = 0.0,
    seed: int = 840,
) -> tuple[np.ndarray, np.ndarray]:
    """``n_reps`` independent Petersen *time-effect* panels, fully vectorised.

    Returns ``(X, Y)`` each of shape ``(n_reps, n_periods, n_firms)``. In every period a
    single common factor draw (``f_t`` for the regressor, ``g_t`` for the residual, drawn
    **independently**) is shared across all firms, giving an exact intra-period correlation
    of ``rho_x`` in ``x`` and ``rho_e`` in the residual ``e``. The outcome is
    ``y = beta*x + e``. ``beta = 0`` is the null (nothing to find); ``beta > 0`` plants a
    real slope. Both ``x`` and ``e`` have unit variance by construction.
    """
    if not (0.0 <= rho_x <= 1.0 and 0.0 <= rho_e <= 1.0):
        raise ValueError("rho_x, rho_e must be in [0, 1]")
    if n_firms < 2 or n_periods < 2:
        raise ValueError("need n_firms >= 2 and n_periods >= 2")
    rng = np.random.default_rng(seed)
    R, T, N = n_reps, n_periods, n_firms

    f = rng.standard_normal((R, T, 1))          # common time factor in the regressor
    u = rng.standard_normal((R, T, N))          # idiosyncratic regressor part
    X = np.sqrt(rho_x) * f + np.sqrt(1.0 - rho_x) * u

    g = rng.standard_normal((R, T, 1))          # common time factor in the residual
    v = rng.standard_normal((R, T, N))          # idiosyncratic residual part
    E = np.sqrt(rho_e) * g + np.sqrt(1.0 - rho_e) * v

    Y = beta * X + E
    return X, Y


def one_panel(
    n_periods: int,
    n_firms: int,
    rho_x: float = 0.5,
    rho_e: float = 0.5,
    beta: float = 0.0,
    seed: int = 840,
) -> tuple[np.ndarray, np.ndarray]:
    """A single ``(n_periods, n_firms)`` panel (for worked illustrations / the timer)."""
    X, Y = panel(1, n_periods, n_firms, rho_x, rho_e, beta, seed)
    return X[0], Y[0]


# --------------------------------------------------------------------------- #
# Closed-form inflation factor (the ground truth the sim must reproduce)
# --------------------------------------------------------------------------- #
def theoretical_moulton(n_firms: int, rho_x: float, rho_e: float) -> float:
    """Moulton (1986) naive-*t* inflation factor for a single (time) grouping of equal size.

    ``sqrt(1 + (N - 1) * rho_x * rho_e)`` — the factor by which the true SE of the pooled
    OLS slope exceeds the naive i.i.d. SE when the regressor has intra-cluster correlation
    ``rho_x`` and the residual ``rho_e``, with ``N`` firms per period.
    """
    return float(np.sqrt(1.0 + (n_firms - 1) * rho_x * rho_e))


# --------------------------------------------------------------------------- #
# Fingerprints for the as-of stamp
# --------------------------------------------------------------------------- #
def fingerprint(arr: np.ndarray) -> str:
    """Short content fingerprint of a numeric array (for reproducibility stamps)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=float).ravel())
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def config_fingerprint(cfg: dict = CONFIG) -> str:
    """Stable label of the simulation config (seeds + params) — the sim's 'data stamp'."""
    payload = repr(sorted((k, repr(v)) for k, v in cfg.items())).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]
