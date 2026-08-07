"""Data layer for Study 834 (Minimum Backtest Length) — the synthetic tapes the demo needs.

This is a *research-method demonstration*, so there is **no real market data and no network**:
the whole point is to show what a backtest of a *known* truth looks like, and only a
constructed world lets us fix that truth. Two ingredients, both deterministic and offline.

* ``synthetic_returns(sr_ann, ...)`` — a daily (or monthly) return series with a **known
  annualised Sharpe** ``sr_ann`` and a **known return-distribution shape**. ``sr_ann = 0`` is
  the **null** the demo lives on: a driftless series with *zero* genuine edge, on which any
  gaudy in-sample Sharpe is luck. ``sr_ann > 0`` is the **positive control**: a genuinely
  skilful series, used to prove real edge is eventually detectable — but only past its MinTRL.

* Two distribution shapes, both with *closed-form* population moments so the tests can assert
  the formula against the truth:
  - ``dist="normal"`` — Gaussian shocks (skew 0, kurtosis 3);
  - ``dist="fat_left"`` — a standardised **negated-gamma** shock with **negative skew** and
    **excess kurtosis** (fat left tail), the realistic hedge-fund shape that *lengthens* the
    required track record. For shape ``k`` the shock has population skew ``-2/sqrt(k)`` and
    excess kurtosis ``6/k`` (so kurtosis ``3 + 6/k``) — exact, not estimated.

Everything is seeded (base seed 834). ``fingerprint`` hashes the simulation config for the
as-of stamp. Real free data can never certify "true Sharpe = 0", so the study is synthetic-only
and capped at ``NONE`` on the Signal axis, stated openly (like 344 Backtest-Overfitting and
590 Sharpe-Hacking).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252
MONTHS = 12
BASE_SEED = 834
ANN_VOL = 0.15  # a plausible annualised volatility; Sharpe is scale-free in it


@dataclass(frozen=True)
class WorldTruth:
    """The planted truth for a synthetic tape."""

    sr_ann: float          # true annualised Sharpe (0 = the null the demo lives on)
    freq: int              # observations per year (252 daily, 12 monthly)
    skew: float            # population skewness of the per-observation returns
    kurt: float            # population kurtosis (Gaussian = 3)

    @property
    def has_edge(self) -> bool:
        return self.sr_ann != 0.0


def _gamma_shape_for(skew: float) -> float:
    """Gamma shape ``k`` whose negated-standardised shock has population skew ``skew`` (<0)."""
    if skew >= 0:
        raise ValueError("fat_left shock needs a negative target skew")
    return 4.0 / (skew * skew)  # skew = -2/sqrt(k)  ->  k = 4/skew**2


def _fat_left_shock(k: float, shape, rng: np.random.Generator) -> np.ndarray:
    """Standardised negated-gamma shock: mean 0, var 1, skew -2/sqrt(k), exkurt 6/k."""
    g = rng.gamma(k, 1.0, size=shape)      # mean k, var k, skew +2/sqrt(k)
    return -(g - k) / np.sqrt(k)           # negate -> negative skew, unit variance


def moments_of(dist: str, skew_target: float):
    """Return the (skew, kurtosis) a shape produces — closed form, no estimation."""
    if dist == "normal":
        return 0.0, 3.0
    k = _gamma_shape_for(skew_target)
    return -2.0 / np.sqrt(k), 3.0 + 6.0 / k


def synthetic_returns(
    sr_ann: float = 0.0,
    n_years: float = 5.0,
    freq: int = TRADING_DAYS,
    dist: str = "normal",
    skew_target: float = -1.0,
    ann_vol: float = ANN_VOL,
    seed: int = BASE_SEED,
) -> tuple[np.ndarray, WorldTruth]:
    """A deterministic return series with a **known** annualised Sharpe and distribution shape.

    The per-observation mean is set so the *population* annualised Sharpe is exactly ``sr_ann``:

        SR_ann = mean_per * sqrt(freq) / sd_per   =>   mean_per = sr_ann * sd_per / sqrt(freq)

    with ``sd_per = ann_vol / sqrt(freq)``. ``dist="normal"`` uses Gaussian shocks; ``dist=
    "fat_left"`` uses a standardised negated-gamma shock with the requested negative ``skew_target``
    and the matching excess kurtosis (the realistic left-tail shape). Returns ``(returns, truth)``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(n_years * freq))
    sd_per = ann_vol / np.sqrt(freq)
    mean_per = sr_ann * sd_per / np.sqrt(freq)
    if dist == "normal":
        shock = rng.standard_normal(n)
        skew, kurt = 0.0, 3.0
    elif dist == "fat_left":
        k = _gamma_shape_for(skew_target)
        shock = _fat_left_shock(k, n, rng)
        skew, kurt = -2.0 / np.sqrt(k), 3.0 + 6.0 / k
    else:
        raise ValueError(f"unknown dist {dist!r}")
    ret = mean_per + sd_per * shock
    return ret, WorldTruth(sr_ann=sr_ann, freq=freq, skew=skew, kurt=kurt)


def synthetic_panel(
    sr_ann: float = 0.0,
    n_sims: int = 4000,
    n_years: float = 2.0,
    freq: int = TRADING_DAYS,
    dist: str = "normal",
    skew_target: float = -1.0,
    ann_vol: float = ANN_VOL,
    seed: int = BASE_SEED,
) -> tuple[np.ndarray, WorldTruth]:
    """``n_sims`` independent backtests of the same known world — a ``(n_sims, n_obs)`` array.

    Vectorised: one draw of the whole ``(n_sims, n_obs)`` shock matrix, so a Monte-Carlo over
    thousands of backtests is a few array ops (no python loop over sims/dates). Used to measure
    how often a *worthless* strategy (``sr_ann = 0``) posts a gaudy Sharpe by luck, and how often
    a *genuine* one (``sr_ann > 0``) is actually confirmed at a given track length.
    """
    rng = np.random.default_rng(seed)
    n = int(round(n_years * freq))
    sd_per = ann_vol / np.sqrt(freq)
    mean_per = sr_ann * sd_per / np.sqrt(freq)
    if dist == "normal":
        shock = rng.standard_normal((n_sims, n))
        skew, kurt = 0.0, 3.0
    elif dist == "fat_left":
        k = _gamma_shape_for(skew_target)
        shock = _fat_left_shock(k, (n_sims, n), rng)
        skew, kurt = -2.0 / np.sqrt(k), 3.0 + 6.0 / k
    else:
        raise ValueError(f"unknown dist {dist!r}")
    ret = mean_per + sd_per * shock
    return ret, WorldTruth(sr_ann=sr_ann, freq=freq, skew=skew, kurt=kurt)


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj, dtype=float)
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]


def config_fingerprint() -> str:
    """Fingerprint of the frozen simulation config (seeds + params) for docs/results.md."""
    cfg = dict(
        base_seed=BASE_SEED, ann_vol=ANN_VOL, freq_daily=TRADING_DAYS, freq_monthly=MONTHS,
        conf=0.95, sr_grid=(2.0, 1.0, 0.5, 0.25), n_sims=4000,
    )
    return hashlib.sha1(repr(sorted(cfg.items())).encode()).hexdigest()[:12]
