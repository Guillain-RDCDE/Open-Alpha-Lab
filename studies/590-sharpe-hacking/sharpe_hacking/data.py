"""Data layer for Study 590 (Sharpe-Hacking) — the zero-alpha tape the demo games.

The whole point of a *metric-gaming parable* is that the underlying series must carry **no genuine
edge**: any Sharpe you print after "financial engineering" it is manufactured, not harvested. So the
tape is built on purpose:

- ``synthetic_series(alpha=0.0, ...)`` — the **null**: daily returns whose *true* Sharpe is a modest
  benign number set by ``base_sharpe`` (a plain buy-and-hold-style series with realistic
  volatility clustering), with **zero timing/selection alpha**. This is the honest baseline the
  transforms are measured against — nothing about smoothing, leverage or vol-targeting adds real
  return, so any Sharpe *gain* over this baseline is pure measurement inflation.
- ``synthetic_series(alpha>0, ...)`` — the **positive control**: the same generator with a genuinely
  higher risk-adjusted return baked in, so the seed-robust control can prove the honest
  (autocorrelation-corrected) Sharpe *tracks real edge* and is not itself gamed by the transforms.

Volatility clustering is real (a light GARCH-ish AR(1) on the conditional variance), because
volatility targeting only "does something" when volatility varies over time — a fair test of the
vol-targeting lever must give it time-varying vol to exploit.

Everything is deterministic and offline (fixed seed = 590). Tests never touch the network, and there
is **no real-data fetch**: real free data can never certify "zero edge", so the study is
synthetic-only and capped at ``NONE`` on the SIGNAL axis (stated openly, like the repo's
lego-returns / whisky-cask / genetic-algo-overfit studies).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted truth for a synthetic tape."""

    alpha: float          # genuine annualised excess drift (0 = the null the demo games)
    base_sharpe: float    # the honest annualised Sharpe of the raw (un-gamed) series

    @property
    def has_edge(self) -> bool:
        return self.alpha != 0.0


def synthetic_series(
    n_days: int = 3000,
    alpha: float = 0.0,
    base_sharpe: float = 0.45,
    ann_vol: float = 0.16,
    vol_persistence: float = 0.94,
    vol_of_vol: float = 0.35,
    seed: int = 590,
) -> tuple[pd.Series, WorldTruth]:
    """A deterministic daily return series with realistic volatility clustering and a tunable edge.

    The raw series has:

    - an honest annualised Sharpe of ``base_sharpe`` at ``alpha = 0`` (a plausible buy-and-hold
      number — *not* zero, because the demo is about *inflating* an already-honest Sharpe, not
      conjuring one from a driftless walk),
    - **plus** an extra annualised drift of ``alpha`` (the positive-control knob; ``alpha = 0`` keeps
      the series at its honest baseline with no added edge),
    - time-varying conditional volatility (an AR(1) on log-variance around ``ann_vol``), so the
      vol-targeting lever has something real to act on.

    The returns are drawn iid-conditional-on-vol (no serial correlation in the *mean*), so the raw
    series has honest, non-overlapping-annualisable Sharpe — the transforms in ``strategy.py`` are
    the only thing that can distort it.

    Returns ``(returns, truth)`` where ``returns`` is a daily ``pd.Series`` indexed by a business-day
    calendar.
    """
    rng = np.random.default_rng(seed)

    # Time-varying volatility: AR(1) on log-variance (a light stochastic-vol clock).
    daily_vol = ann_vol / np.sqrt(TRADING_DAYS)
    log_v = np.log(daily_vol**2)
    logv = np.empty(n_days)
    state = log_v
    for i in range(n_days):
        state = log_v + vol_persistence * (state - log_v) + vol_of_vol * rng.standard_normal()
        logv[i] = state
    sigma_t = np.sqrt(np.exp(logv))

    # Drift: honest baseline Sharpe on the raw series + planted alpha.
    # We tie the daily drift to *that day's* volatility so the conditional Sharpe is constant
    # (mu_t / sigma_t = base_sharpe / sqrt(252)); this makes the realised annual Sharpe a stable
    # ~base_sharpe instead of a high-variance estimate swamped by the vol-of-vol, and adds NO serial
    # correlation in the mean (mu_t depends only on the contemporaneous vol clock, not past returns).
    daily_sr = base_sharpe / np.sqrt(TRADING_DAYS)
    daily_mu_alpha = alpha / TRADING_DAYS
    mu_t = daily_sr * sigma_t + daily_mu_alpha

    z = rng.standard_normal(n_days)
    ret = mu_t + sigma_t * z

    idx = pd.bdate_range("2008-01-02", periods=n_days)
    series = pd.Series(ret, index=idx, name="ret")
    return series, WorldTruth(alpha=alpha, base_sharpe=base_sharpe)


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
