"""Data layer for Study 838 — HAC Necessity (Newey-West necessity).

This is a *research-method* study, so the "data" is a controlled null world we build
ourselves. The whole point is to hand the inference machinery a return series that has
**zero true mean** but **strong serial correlation** — exactly the shape a real
overlapping-signal strategy produces — and watch the naive (i.i.d.) OLS *t* fabricate
significance while the HAC (Newey-West) *t* stays honest.

Two canonical ways to inject autocorrelation into a mean-zero series, both closed-form:

* **Overlapping window (the hero).** Draw i.i.d. daily innovations and report the trailing
  ``window``-day *rolling mean*. This is precisely the "overlapping returns" object of a
  daily long-short with a ``window``-day formation window (Hansen-Hodrick 1980): each day's
  value shares ``window-1`` innovations with its neighbour. The result is an MA(``window``-1)
  process with a beautiful, exact long-run-variance identity:

      gamma0 = sigma**2 / window                 (naive per-obs variance)
      LRV    = sigma**2                           (true long-run variance)
      LRV / gamma0 = window

  So the naive standard error understates the truth by a factor ``sqrt(window)`` — a 21-day
  overlap inflates the naive *t* by ``sqrt(21) ~= 4.58x`` **with no real effect present.**

* **AR(1) (the tunable dial).** A first-order autoregression with autocorrelation ``rho``.
  Here ``LRV / gamma0 = (1 + rho) / (1 - rho)``, so the naive-*t* inflation factor is
  ``sqrt((1+rho)/(1-rho))`` — a single knob to sweep the pitfall from mild to severe.

Both generators expose a ``mean`` knob: ``mean = 0`` is the **null** (nothing to find — the
whole demonstration); ``mean > 0`` plants a genuine effect (the **positive control** that
proves the HAC machinery still *fires* when there really is something there).

Everything here is pure numpy + scipy + stdlib, deterministic under a fixed seed. No network,
no market data, no cache — a simulation study end to end.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.signal import lfilter

AS_OF = "2026-06-30"        # frozen stamp for the headline Monte-Carlo run

# The frozen simulation configuration behind docs/results.md (the "tape" of a sim study).
CONFIG = {
    "seed": 838,
    "n_reps": 600,          # Monte-Carlo replications for the false-positive experiment
    "n_days": 2520,         # ~10 trading years per replication
    "window": 21,           # the overlapping formation window (the hero case)
    "daily_vol": 0.01,      # innovation scale (irrelevant to a t-stat, fixed for the record)
    "nw_lags": 42,          # Newey-West Bartlett bandwidth (>= overlap length; Bartlett downweighting wants a generous span)
    "crit": 1.96,           # two-sided 5% Gaussian critical value
    "control_mean": 6e-4,   # planted daily mean for the positive control (machinery power)
    "rho_grid": [0.0, 0.2, 0.4, 0.6, 0.8],   # AR(1) dial for the inflation curve
    "window_grid": [1, 5, 10, 21, 42, 63],   # overlap dial for the inflation curve
}

__all__ = [
    "AS_OF", "CONFIG",
    "overlap_returns", "ar1_returns",
    "overlap_matrix", "ar1_matrix",
    "theoretical_inflation_overlap", "theoretical_inflation_ar1",
    "fingerprint", "config_fingerprint",
]


# --------------------------------------------------------------------------- #
# Single paths (for notebook figures / worked illustrations)
# --------------------------------------------------------------------------- #
def _bday_index(n: int) -> pd.DatetimeIndex:
    """Decorative business-day labels without overflowing ns-Timestamps (CI pitfall)."""
    idx = pd.period_range("1990-01-01", periods=n, freq="D").to_timestamp()
    return pd.DatetimeIndex(idx, name="date")


def overlap_returns(
    n_days: int = 2520,
    window: int = 21,
    seed: int = 838,
    daily_vol: float = 0.01,
    mean: float = 0.0,
) -> pd.Series:
    """One mean-``mean`` overlapping-window series (trailing ``window``-day rolling mean of
    i.i.d. innovations). ``mean = 0`` is the null; ``mean > 0`` plants a real effect."""
    x = overlap_matrix(1, n_days, window, seed, daily_vol, mean)[0]
    return pd.Series(x, index=_bday_index(n_days), name="ret")


def ar1_returns(
    n_days: int = 2520,
    rho: float = 0.6,
    seed: int = 838,
    daily_vol: float = 0.01,
    mean: float = 0.0,
) -> pd.Series:
    """One mean-``mean`` AR(1) series with autocorrelation ``rho`` (process SD ``daily_vol``)."""
    x = ar1_matrix(1, n_days, rho, seed, daily_vol, mean)[0]
    return pd.Series(x, index=_bday_index(n_days), name="ret")


# --------------------------------------------------------------------------- #
# Monte-Carlo matrices (n_reps x n_days) — the false-positive experiment input
# --------------------------------------------------------------------------- #
def overlap_matrix(
    n_reps: int,
    n_days: int,
    window: int = 21,
    seed: int = 838,
    daily_vol: float = 0.01,
    mean: float = 0.0,
) -> np.ndarray:
    """``n_reps`` independent overlapping-window null paths, vectorised.

    Each row is a trailing ``window``-day rolling MEAN of i.i.d. N(0, ``daily_vol``)
    innovations, plus ``mean``. An MA(``window``-1) process: strong positive
    autocorrelation out to lag ``window``-1, exact per-obs variance ``daily_vol**2/window``
    and exact long-run variance ``daily_vol**2`` (ratio = ``window``).
    """
    rng = np.random.default_rng(seed)
    if window < 1:
        raise ValueError("window must be >= 1")
    innov = rng.standard_normal((n_reps, n_days + window - 1)) * daily_vol
    csum = np.cumsum(innov, axis=1)
    # rolling sum of width `window` -> length n_days
    out = csum[:, window - 1:].copy()
    out[:, 1:] -= csum[:, : n_days - 1]
    out /= window
    return out + mean


def ar1_matrix(
    n_reps: int,
    n_days: int,
    rho: float = 0.6,
    seed: int = 838,
    daily_vol: float = 0.01,
    mean: float = 0.0,
) -> np.ndarray:
    """``n_reps`` independent AR(1) paths x_t = rho*x_{t-1} + eps_t, vectorised via lfilter.

    The innovation SD is scaled so the stationary process SD equals ``daily_vol``; the
    stationary autocorrelation at lag ``l`` is ``rho**l`` and ``LRV/gamma0 = (1+rho)/(1-rho)``.
    """
    rng = np.random.default_rng(seed)
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    innov_sd = daily_vol * np.sqrt(1.0 - rho * rho)
    eps = rng.standard_normal((n_reps, n_days)) * innov_sd
    x = lfilter([1.0], [1.0, -rho], eps, axis=1)
    return x + mean


# --------------------------------------------------------------------------- #
# Closed-form inflation factors (the ground truth the sim must reproduce)
# --------------------------------------------------------------------------- #
def theoretical_inflation_overlap(window: int) -> float:
    """Naive-*t* inflation factor for a ``window``-day overlap: ``sqrt(window)``."""
    return float(np.sqrt(window))


def theoretical_inflation_ar1(rho: float) -> float:
    """Naive-*t* inflation factor for AR(1) with autocorrelation ``rho``:
    ``sqrt((1+rho)/(1-rho))``."""
    return float(np.sqrt((1.0 + rho) / (1.0 - rho)))


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
