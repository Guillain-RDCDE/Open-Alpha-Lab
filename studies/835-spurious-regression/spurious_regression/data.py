"""Data layer for Study 835 (Spurious Regression) — the driftless worlds the demo dissects.

Granger & Newbold (1974) showed that regressing one **independent random walk** on
another routinely produces a large, "significant" *t*-statistic and a high R² — a
textbook relation that does not exist. The whole point of a *nonstationarity parable*
is that the two series must be **built with no genuine link**: any significance the
level regression prints is manufactured by the trending (unit-root) structure, not
harvested from a real relationship. So every world here is generated on purpose:

- ``independent_walks(...)`` — the **null / pitfall world**: two *independent* Gaussian
  random walks ``x`` and ``y`` (each ``I(1)``, driftless by default). There is **no**
  relation between them; a correctly sized test should reject "no relation" ~5% of the
  time. The level OLS instead rejects far more often — that inflation *is* the pitfall.

- ``stationary_pairs(...)`` — the **specificity control**: two *independent* stationary
  series (white noise, or a mild stationary AR(1)). OLS on levels here is correctly
  sized (~5% rejection): the pitfall is a property of **nonstationarity**, not of OLS.

- ``cointegrated_pairs(...)`` — the **positive control**: two series that share a common
  stochastic trend, so ``y - beta*x`` is stationary (a *genuine* long-run relation). The
  Engle-Granger cointegration test should *reject* the no-cointegration null here (and
  should NOT reject on the independent walks) — proving the machinery tells a real
  relation from a spurious one.

Everything is deterministic and offline (base seed = 835). Tests never touch the
network, and there is **no real-data fetch**: a research-method demo cannot certify "no
relation" from real prices, so the study is synthetic-only and capped at ``NONE`` on the
SIGNAL axis (stated openly, like the desk's other method demos — 344, 590).
"""

from __future__ import annotations

import hashlib

import numpy as np

TRADING_DAYS = 252
AS_OF = "2026-06-30"        # method-demo stamp (no partial-month tape)
BASE_SEED = 835


# --------------------------------------------------------------------------- #
# The null / pitfall world — two INDEPENDENT random walks
# --------------------------------------------------------------------------- #
def independent_walks(
    n_pairs: int,
    n_obs: int = 250,
    sigma: float = 1.0,
    drift: float = 0.0,
    seed: int = BASE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """``n_pairs`` pairs of **independent** Gaussian random walks, each length ``n_obs``.

    ``x[i]`` and ``y[i]`` are built from two **disjoint** iid Gaussian shock streams and
    cumulatively summed, so each is ``I(1)`` and the two carry **zero** relationship. A
    non-zero ``drift`` adds a deterministic trend to both (the "trending series" case).
    Returns ``(X, Y)`` as ``(n_pairs, n_obs)`` float arrays. Fully vectorised.
    """
    rng = np.random.default_rng(seed)
    ex = rng.normal(0.0, sigma, size=(n_pairs, n_obs))
    ey = rng.normal(0.0, sigma, size=(n_pairs, n_obs))
    if drift:
        ex = ex + drift
        ey = ey + drift
    X = np.cumsum(ex, axis=1)
    Y = np.cumsum(ey, axis=1)
    return X, Y


# --------------------------------------------------------------------------- #
# The specificity control — two INDEPENDENT stationary series
# --------------------------------------------------------------------------- #
def stationary_pairs(
    n_pairs: int,
    n_obs: int = 250,
    phi: float = 0.0,
    sigma: float = 1.0,
    seed: int = BASE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """``n_pairs`` pairs of **independent stationary** series (AR(1), ``|phi|<1``).

    ``phi = 0`` is white noise; a mild ``phi`` gives autocorrelated-but-stationary
    regressors. Because both series are stationary, level OLS is correctly sized — this
    is the control that isolates the pitfall to **nonstationarity**, not to OLS itself.
    Returns ``(X, Y)`` as ``(n_pairs, n_obs)`` arrays. Vectorised over pairs.
    """
    rng = np.random.default_rng(seed)
    ex = rng.normal(0.0, sigma, size=(n_pairs, n_obs))
    ey = rng.normal(0.0, sigma, size=(n_pairs, n_obs))
    if phi == 0.0:
        return ex, ey
    X = np.empty_like(ex)
    Y = np.empty_like(ey)
    X[:, 0] = ex[:, 0]
    Y[:, 0] = ey[:, 0]
    for t in range(1, n_obs):
        X[:, t] = phi * X[:, t - 1] + ex[:, t]
        Y[:, t] = phi * Y[:, t - 1] + ey[:, t]
    return X, Y


# --------------------------------------------------------------------------- #
# The positive control — a GENUINE cointegrating relation (common trend)
# --------------------------------------------------------------------------- #
def cointegrated_pairs(
    n_pairs: int,
    n_obs: int = 250,
    beta: float = 1.0,
    sigma: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = BASE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """``n_pairs`` pairs sharing a **common stochastic trend** (genuinely cointegrated).

    A shared random walk ``w`` drives both series; each also carries its own *stationary*
    noise::

        w = cumsum(iid shocks)                       # the common I(1) trend
        x = w              + noise_x                  # stationary deviation
        y = beta * w       + noise_y                  # stationary deviation

    Then ``y - beta*x = noise_y - beta*noise_x`` is **stationary** — a real long-run
    relation. This is the positive control: the Engle-Granger test SHOULD reject
    no-cointegration here (and must NOT on :func:`independent_walks`). Returns
    ``(X, Y)`` as ``(n_pairs, n_obs)`` arrays. Vectorised.
    """
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.normal(0.0, sigma, size=(n_pairs, n_obs)), axis=1)
    nx = rng.normal(0.0, noise_sd, size=(n_pairs, n_obs))
    ny = rng.normal(0.0, noise_sd, size=(n_pairs, n_obs))
    X = w + nx
    Y = beta * w + ny
    return X, Y


# --------------------------------------------------------------------------- #
# Config fingerprint for the as-of stamp
# --------------------------------------------------------------------------- #
def fingerprint(config: dict) -> str:
    """A short content fingerprint of a simulation config (seeds + params)."""
    payload = repr(sorted(config.items())).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


__all__ = [
    "TRADING_DAYS", "AS_OF", "BASE_SEED",
    "independent_walks", "stationary_pairs", "cointegrated_pairs", "fingerprint",
]
