"""Data layer for Study 837 — Look-Ahead Standardization.

The pitfall under test is a **specific, ubiquitous** form of look-ahead bias: z-scoring /
normalising a predictive feature with the **full-sample** mean & standard deviation — the whole
history *including the future* — instead of an **expanding / point-in-time** window that only ever
sees the past. It is the single most common way a research backtest quietly leaks the future, and it
is *distinct* from the generic look-ahead of [347](../../347-look-ahead-bias/) (aligning a signal to
a return it could not have known): here the leak hides inside an innocuous preprocessing step that
almost every feature pipeline runs.

This is a **research-method demonstration**: synthetic / simulation only, no network, no real market
data, no cache. The whole point of a method parable is that the world must be built so we *know* the
ground truth — so any backtest "edge" the leak manufactures is, by construction, an artefact.

Three deterministic seeded worlds:

* ``null_stationary`` — a **stationary** feature (an AR(1) that reverts to a fixed mean) paired with
  **iid noise returns**. There is genuinely nothing to find, *and* full-sample standardisation of a
  stationary series is essentially a per-name affine rescale, so it leaks (almost) nothing. Both
  standardisations must read ~0. This is the *contrast* case that pins the leak to non-stationarity.

* ``null_nonstationary`` — the trap. A **random-walk** (price/level-like, **non-stationary**)
  feature paired with a **forward return equal to the feature's own future change** over ``horizon``
  days. Because a random walk's increments are iid, the forward return is **genuinely unpredictable
  from anything known at time t** — an efficient-market null. An expanding-window z-score correctly
  finds ~0. But the full-sample mean sits *in the middle of the eventual path*, so subtracting it
  tells you whether today is above or below where the series will *end up on average* — and a random
  walk mechanically drifts back toward its own sample mean. That peeked-at centring manufactures a
  large spurious IC and a gorgeous fake Sharpe out of pure noise.

* ``planted_edge`` — the **positive control**. A stationary feature whose *current* value genuinely
  predicts the next return (``r_{t+1} = beta * f_t + noise``) — a real, point-in-time-tradeable edge.
  The honest expanding-window standardisation **must** recover it (proving the machinery is not
  simply always-zero); the leak does not need to help.

Everything is pure numpy + pandas + stdlib and deterministic (base seed 837).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical simulation configuration (the study's fingerprinted "as-of" world)
# --------------------------------------------------------------------------- #
BASE_SEED = 837
N_NAMES = 60          # cross-section width
N_DAYS = 1000         # ~4 trading years
HORIZON = 10          # forward-return horizon (days) for the non-stationary null
MIN_PERIODS = 60      # burn-in before an expanding z-score is defined
AS_OF = "2026-06-30"  # stamp (a synthetic-only method demo — no partial months)
START = "2018-01-02"  # business-day index origin for the synthetic panel

__all__ = [
    "BASE_SEED", "N_NAMES", "N_DAYS", "HORIZON", "MIN_PERIODS", "AS_OF", "START",
    "null_stationary", "null_nonstationary", "planted_edge",
    "as_frame", "config_fingerprint", "fingerprint",
]


def _bdays(n: int) -> pd.DatetimeIndex:
    """A business-day index of length ``n`` (well below the pandas ns horizon)."""
    return pd.bdate_range(START, periods=n)


# --------------------------------------------------------------------------- #
# World 1 — the STATIONARY null (the contrast: full-sample z barely leaks here)
# --------------------------------------------------------------------------- #
def null_stationary(
    seed: int = BASE_SEED,
    n_names: int = N_NAMES,
    n_days: int = N_DAYS,
    phi: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary AR(1) feature, iid-noise returns — a genuine null with no leak.

    ``X[t, i] = phi * X[t-1, i] + sqrt(1 - phi**2) * e`` (unit-variance stationary AR(1)); the return
    ``R[t, i]`` is independent standard-normal noise (the last row is NaN — no forward return). Because
    the feature is stationary, its full-sample mean/std are stable estimates of fixed constants, so
    full-sample standardisation is (to leading order) a per-name affine rescale that leaks almost
    nothing. Both standardisations should read IC ~ 0. Returns ``(X, R)`` each shaped ``(n_days,
    n_names)``.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_days, n_names))
    X[0] = rng.normal(0.0, 1.0, n_names)
    innov = rng.normal(0.0, np.sqrt(1.0 - phi * phi), (n_days, n_names))
    for t in range(1, n_days):
        X[t] = phi * X[t - 1] + innov[t]
    R = np.full((n_days, n_names), np.nan)
    R[:-1] = rng.normal(0.0, 1.0, (n_days - 1, n_names))
    return X, R


# --------------------------------------------------------------------------- #
# World 2 — the NON-STATIONARY null (the trap the whole study is about)
# --------------------------------------------------------------------------- #
def null_nonstationary(
    seed: int = BASE_SEED,
    n_names: int = N_NAMES,
    n_days: int = N_DAYS,
    horizon: int = HORIZON,
    daily_vol: float = 0.01,
    noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk feature; forward return = the feature's own future change (an efficient null).

    ``X[t, i]`` is a driftless random walk (cumulative iid ``N(0, daily_vol)`` increments) — a
    price/level-like, **non-stationary** feature. The forward return is its realised change over the next
    ``horizon`` days plus noise:

        R[t, i] = (X[t + horizon, i] - X[t, i]) + noise * N(0, 1)

    Since a random walk's increments are iid, ``R[t]`` is **independent of everything observable at or
    before t** — there is genuinely no point-in-time-tradeable edge. An expanding-window z-score must
    therefore read ~0. The **full-sample** mean/std, however, are computed over the whole path
    (including ``X[t+1 .. T]``), so ``X[t] - full_mean`` encodes whether the walk is above or below
    where it eventually averages — and a random walk drifts back toward its own sample mean. That
    peeked-at centring is the leak. Rows within ``horizon`` of the end are NaN (no forward return).
    Returns ``(X, R)`` shaped ``(n_days, n_names)``.
    """
    rng = np.random.default_rng(seed)
    incr = rng.normal(0.0, daily_vol, (n_days, n_names))
    X = np.cumsum(incr, axis=0)
    R = np.full((n_days, n_names), np.nan)
    if n_days > horizon:
        R[: n_days - horizon] = (
            X[horizon:] - X[: n_days - horizon]
            + daily_vol * noise * rng.normal(0.0, 1.0, (n_days - horizon, n_names))
        )
    return X, R


# --------------------------------------------------------------------------- #
# World 3 — the PLANTED real edge (positive control: expanding MUST find it)
# --------------------------------------------------------------------------- #
def planted_edge(
    seed: int = BASE_SEED,
    n_names: int = N_NAMES,
    n_days: int = N_DAYS,
    beta: float = 0.10,
    phi: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary feature whose CURRENT value predicts the NEXT return — a real, tradeable edge.

    Same stationary AR(1) feature as :func:`null_stationary`, but the forward return genuinely loads
    on the feature known at ``t``: ``R[t] = beta * X[t] + N(0,1)``. This edge is available in real
    time, so the honest **expanding-window** standardisation must recover a positive IC — the proof
    that the machinery is *unbiased* (silent on the nulls, alive on a real effect), not merely
    always-zero. Returns ``(X, R)`` shaped ``(n_days, n_names)``.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_days, n_names))
    X[0] = rng.normal(0.0, 1.0, n_names)
    innov = rng.normal(0.0, np.sqrt(1.0 - phi * phi), (n_days, n_names))
    for t in range(1, n_days):
        X[t] = phi * X[t - 1] + innov[t]
    R = np.full((n_days, n_names), np.nan)
    R[:-1] = beta * X[:-1] + rng.normal(0.0, 1.0, (n_days - 1, n_names))
    return X, R


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def as_frame(mat: np.ndarray) -> pd.DataFrame:
    """Wrap a ``(T, N)`` matrix as a business-day-indexed DataFrame (columns SYN00..)."""
    T, N = mat.shape
    cols = [f"SYN{i:02d}" for i in range(N)]
    return pd.DataFrame(mat, index=_bdays(T), columns=cols)


def config_fingerprint(
    base_seed: int = BASE_SEED,
    n_names: int = N_NAMES,
    n_days: int = N_DAYS,
    horizon: int = HORIZON,
    min_periods: int = MIN_PERIODS,
    n_seeds: int = 20,
) -> str:
    """A short, stable fingerprint of the simulation configuration for the as-of stamp."""
    cfg = f"seed={base_seed}|N={n_names}|T={n_days}|H={horizon}|minp={min_periods}|seeds={n_seeds}"
    return hashlib.sha1(cfg.encode()).hexdigest()[:12]


def fingerprint(obj) -> str:
    """Content fingerprint of a numpy array / DataFrame (for reproducibility checks)."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.fillna(0.0).to_numpy(dtype=float)
    arr = np.ascontiguousarray(np.asarray(obj, dtype=float))
    arr = np.nan_to_num(arr)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
