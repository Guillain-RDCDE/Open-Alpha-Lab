"""Data layer for Study 839 (The t > 3 Threshold) — the factor zoo the demo mines.

The whole point of the *multiple-testing parable* is that the underlying candidate
factors carry **no genuine edge**: any *t*-stat that clears the bar in the null world is,
by construction, a false discovery. So the tape is built on purpose.

- ``synthetic_zoo(n_true=0, ...)`` — the **null**: ``n_factors`` candidate long-short
  factors, each a zero-mean iid return stream. There is *nothing to find*; every factor
  that posts ``|t| > 2`` did so by luck. This is the world Harvey-Liu-Zhu (2016) warn
  about — data-mine hundreds of factors and roughly 5% will clear the conventional
  ``t > 2`` bar from noise alone.
- ``synthetic_zoo(n_true>0, ...)`` — the **positive control**: the same zoo with a known
  subset of ``n_true`` genuinely-priced factors (a planted per-period mean sized so the
  *expected* single-test ``|t|`` equals ``expected_t``). The corrections must *keep the
  real ones while dropping the fakes* — otherwise a "correction" that simply rejects
  everything would look great on the null and be useless in practice.

Because the per-factor *t*-stat is scale-free (``mean / (sd/sqrt(T))``), the return
volatility ``vol`` is cosmetic; the planted mean is set relative to it so ``expected_t``
is the only knob that matters. Everything is deterministic and offline (fixed seed 839).
There is **no real-data fetch**: real free factor data can never certify "zero edge", so
the study is synthetic-only and capped at ``NONE`` on the SIGNAL axis (stated openly,
like the desk's backtest-overfitting / sharpe-hacking method demos).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 20 years of monthly long-short factor returns — HLZ's cross-sectional regime.
DEFAULT_T = 240


@dataclass(frozen=True)
class ZooTruth:
    """The planted truth for a synthetic factor zoo."""

    n_factors: int      # total candidate factors data-mined
    n_true: int         # how many carry a genuine effect (0 = the pure-noise null)
    expected_t: float   # the single-test |t| a true factor is sized to
    n_periods: int      # length of each factor's return history

    @property
    def has_edge(self) -> bool:
        return self.n_true > 0

    @property
    def frac_true(self) -> float:
        return self.n_true / self.n_factors if self.n_factors else 0.0


def synthetic_zoo(
    n_factors: int = 1000,
    n_periods: int = DEFAULT_T,
    n_true: int = 0,
    expected_t: float = 4.0,
    vol: float = 0.04,
    seed: int = 839,
) -> tuple[np.ndarray, np.ndarray, ZooTruth]:
    """A deterministic ``(T x N)`` matrix of candidate-factor per-period returns.

    Each column is one data-mined long-short factor's return history. Null factors are
    iid ``N(0, vol)``; the first ``n_true`` columns (deterministic labels) are true, with
    a planted per-period mean ``mu = expected_t * vol / sqrt(T)`` so a single-test
    *t*-stat has expectation ``expected_t``.

    - ``n_true = 0`` → the **null**: nothing is real, so any ``|t| > 2`` is a false
      discovery. This is the world the whole demo runs on.
    - ``n_true > 0`` → the **positive control**: a known real subset the corrections must
      retain while purging the noise.

    Returns ``(returns, is_true, truth)`` where ``returns`` is ``float64`` ``(T, N)``,
    ``is_true`` is a boolean length-``N`` mask, and ``truth`` is the planted
    :class:`ZooTruth`.
    """
    rng = np.random.default_rng(seed)
    T, N = int(n_periods), int(n_factors)
    R = rng.normal(0.0, vol, size=(T, N))
    is_true = np.zeros(N, dtype=bool)
    n_true = int(max(0, min(n_true, N)))
    if n_true > 0:
        is_true[:n_true] = True
        mu = expected_t * vol / np.sqrt(T)
        R[:, :n_true] += mu
    truth = ZooTruth(
        n_factors=N, n_true=n_true, expected_t=float(expected_t), n_periods=T
    )
    return R, is_true, truth


def fingerprint(obj) -> str:
    """A short content fingerprint of a returns matrix / array, for the as-of stamp."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = obj.to_numpy()
    arr = np.ascontiguousarray(np.asarray(obj, dtype=float).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
