"""Data layer for Study 842 (Implementation Shortfall) — the tape the cost gap runs on.

An implementation-shortfall parable needs a strategy whose **paper** (0-cost) performance
is genuinely good, so that whatever the cost model then eats is a real loss and not a
non-edge. So the tape is built on purpose, with two independent knobs:

- ``edge`` — the **planted predictive strength**. Each name carries a persistent latent
  signal ``s_{i,t}`` (a standardised AR(1)); the next-day return loads on the *lagged*
  signal, ``r_{i,t} = edge * s_{i,t-1} + noise``. With ``edge > 0`` a cross-sectional sort
  on the signal known at ``t-1`` earns a genuine gross spread — the **paper alpha** that
  dazzles at 0 cost. With ``edge = 0`` the signal predicts nothing: the **null**, where
  even the gross book earns ~0, so any "edge" a naive backtest reports is noise.

- ``persistence`` (the AR(1) coefficient ``phi``) — the **turnover lever**, and the whole
  point of the study. The latent signal is standardised to unit variance for *every*
  ``phi`` (``s_t = phi*s_{t-1} + sqrt(1-phi^2)*eps_t``), so the *gross* edge is held fixed
  while the **turnover changes**: as ``phi -> 1`` the signal barely moves, the sort barely
  rotates, turnover is low; as ``phi -> 0`` the signal is white noise, the ranks reshuffle
  every day, turnover is high. Costs scale with turnover, so this is exactly the axis on
  which the paper alpha lives or dies.

Everything is deterministic and offline (fixed seed = 842). Tests never touch the network,
and there is **no real-data fetch**: real free data can never certify a clean planted gross
edge with a known turnover, so the study is synthetic-only and capped at ``NONE`` on the
SIGNAL axis (stated openly, like the repo's backtest-overfitting / sharpe-hacking demos).
No look-ahead is baked into the generator — the return on day ``t`` loads on the signal
known at the close of ``t-1``; the one execution lag lives in ``strategy`` (a single shift).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
AS_OF = "2026-06-30"        # stamp date (a synthetic tape carries no calendar of its own)


@dataclass(frozen=True)
class WorldTruth:
    """The planted truth for a synthetic tape."""

    edge: float           # planted predictive strength (0 = the null, nothing gross to find)
    persistence: float    # AR(1) phi of the latent signal — sets the turnover
    n_assets: int
    n_days: int

    @property
    def has_edge(self) -> bool:
        return self.edge != 0.0


def synthetic_panel(
    edge: float = 0.0005,
    persistence: float = 0.96,
    n_assets: int = 30,
    n_days: int = 2520,
    daily_vol: float = 0.015,
    seed: int = 842,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A deterministic (T x N) return panel with a planted, turnover-tunable gross edge.

    Model, per name ``i``:

        s_{i,t}   = phi * s_{i,t-1} + sqrt(1 - phi^2) * eps_{i,t}   (unit-variance AR(1))
        r_{i,t}   = edge * s_{i,t-1} + daily_vol * eta_{i,t}

    The latent signal ``s`` is standardised to unit variance for every ``phi`` so the gross
    predictive strength is a function of ``edge`` alone, *independent of the turnover* that
    ``phi`` controls. The tradeable signal on day ``t`` is ``s_{i,t-1}`` (last night's
    value); the sort in :mod:`strategy` shifts by one row so a day-``t`` position is formed
    on information known at the close of ``t-1`` — the single execution lag.

    Returns ``(returns, signal, truth)``:
      * ``returns`` — (T x N) DataFrame of daily simple returns, business-day index.
      * ``signal``  — (T x N) DataFrame of the *contemporaneous* latent ``s_{i,t}``; the
        book uses ``signal.shift(1)`` so it never peeks. Exposed for tests/diagnostics.
      * ``truth``   — the planted parameters (ground truth for the tests).
    """
    rng = np.random.default_rng(seed)
    phi = float(persistence)
    innov_sd = np.sqrt(max(1.0 - phi * phi, 0.0))

    # Latent unit-variance AR(1) signal, vectorised across names via a time recursion.
    s = np.empty((n_days, n_assets))
    s[0] = rng.standard_normal(n_assets)                 # stationary start ~ N(0,1)
    eps = rng.standard_normal((n_days, n_assets))
    for t in range(1, n_days):
        s[t] = phi * s[t - 1] + innov_sd * eps[t]

    # Return loads on the LAGGED signal (known at t-1); day 0 has no lagged signal -> no load.
    s_lag = np.vstack([np.zeros((1, n_assets)), s[:-1]])
    noise = daily_vol * rng.standard_normal((n_days, n_assets))
    r = edge * s_lag + noise

    idx = pd.bdate_range("2010-01-04", periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(r, index=idx, columns=cols)
    signal = pd.DataFrame(s, index=idx, columns=cols)
    truth = WorldTruth(edge=edge, persistence=phi, n_assets=n_assets, n_days=n_days)
    return returns, signal, truth


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp (of a returns/signal panel)."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0.0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
