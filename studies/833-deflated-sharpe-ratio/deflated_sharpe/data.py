"""Data layer for Study 833 — the Deflated Sharpe Ratio.

This is a *research-method* study, so the data layer's only job is to manufacture a world
in which an honest backtest is **guaranteed to find nothing** — and then, separately, a
single strategy that is **guaranteed to be genuinely good**. Whatever a "best-of-N" search
turns up on the first world is, with certainty, luck; the second world is the positive
control that a faithful correction must spare.

Everything is deterministic, offline (fixed seed = 833), pure numpy + pandas. There is **no
real-data fetch and no network**: real free data can never *certify* zero edge, so — exactly
like the repo's [344 Backtest-Overfitting](../../344-backtest-overfitting/) null core and
[590 Sharpe-Hacking](../../590-sharpe-hacking/) — the study is synthetic-only and capped at
`NONE` on the SIGNAL axis, stated openly.

Two generators:

* ``null_panel(n_strategies, n_days, ...)`` — the NULL world: an ``(T × N)`` matrix whose
  every column is an **independent** strategy with a *true* Sharpe of **exactly zero** (iid
  Gaussian daily returns, no drift). This is the object the expected-maximum-Sharpe formula
  is built for: ``N`` genuinely-empty, independent trials. The best column's *sample* Sharpe
  is pure selection luck, and it climbs with ``N``.

* ``honest_strategy(n_days, true_ann_sharpe, ...)`` — the POSITIVE CONTROL: a single stream
  of daily returns with a genuine positive drift, so its *true* annualised Sharpe is
  ``true_ann_sharpe`` (> 0). A faithful deflation must leave this one standing.

Why iid *independent* columns (not N timing rules on one price path)? Bailey & López de
Prado's expected-maximum-Sharpe result assumes **independent trials**; N random rules on a
shared random-walk market are cross-correlated through the common tape, which shrinks the
*effective* number of trials and muddies the formula. Independent columns are the pure,
faithful demonstration — the sibling study [344](../../344-backtest-overfitting/) runs the
correlated crossover-grid version.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

TRADING_DAYS = 252
AS_OF = "2026-06-30"        # stamp for parity with the desk's dated runs (a sim label here)


# --------------------------------------------------------------------------- #
# The NULL world — N independent, genuinely-empty strategies
# --------------------------------------------------------------------------- #
def null_panel(
    n_strategies: int = 1000,
    n_days: int = 1260,
    ann_vol: float = 0.15,
    seed: int = 833,
) -> np.ndarray:
    """An ``(T × N)`` matrix of iid Gaussian daily returns — every column a **true-zero-edge**
    strategy.

    Each of the ``n_strategies`` columns is an *independent* draw of ``n_days`` iid
    ``N(0, sigma^2)`` daily returns with ``sigma = ann_vol / sqrt(252)`` and **zero drift**,
    so its population Sharpe is exactly ``0``. Nothing in this panel is real: any column with
    a high *sample* Sharpe won it by luck, and the best-of-``N`` sample Sharpe inflates with
    ``N`` per the expected-maximum-Sharpe formula. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    sigma = ann_vol / np.sqrt(TRADING_DAYS)
    return rng.standard_normal((n_days, n_strategies)) * sigma


def honest_strategy(
    n_days: int = 1260,
    true_ann_sharpe: float = 1.0,
    ann_vol: float = 0.15,
    seed: int = 833,
) -> np.ndarray:
    """A single daily-return stream with a **genuine** positive edge (the positive control).

    Daily returns are ``mu + sigma * z`` with ``sigma = ann_vol / sqrt(252)`` and a drift
    ``mu`` chosen so the *population* annualised Sharpe equals ``true_ann_sharpe``
    (``mu = true_ann_sharpe * sigma / sqrt(252)``). This is an honestly-good *single*
    hypothesis — not the survivor of a search — and a faithful deflation must keep its DSR
    high. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    sigma = ann_vol / np.sqrt(TRADING_DAYS)
    mu = true_ann_sharpe * sigma / np.sqrt(TRADING_DAYS)
    return mu + sigma * rng.standard_normal(n_days)


def planted_in_pool(
    n_strategies: int = 1000,
    n_days: int = 1260,
    true_ann_sharpe: float = 2.0,
    ann_vol: float = 0.15,
    seed: int = 833,
) -> tuple[np.ndarray, int]:
    """A NULL panel with **one genuinely-good column planted** inside it (a harder control).

    Column ``0`` carries a real annualised Sharpe of ``true_ann_sharpe``; the other
    ``n_strategies - 1`` columns are true-zero-edge nulls. Used to show that a *strong* real
    strategy survives the deflation **even when buried among N trials** (its DSR stays high),
    while the empty winner of the same pool does not. Returns ``(panel, planted_col)``.
    """
    panel = null_panel(n_strategies, n_days, ann_vol, seed)
    sigma = ann_vol / np.sqrt(TRADING_DAYS)
    mu = true_ann_sharpe * sigma / np.sqrt(TRADING_DAYS)
    panel = panel.copy()
    panel[:, 0] = panel[:, 0] + mu     # turn column 0 into a genuine edge
    return panel, 0


def fingerprint(obj) -> str:
    """A short content fingerprint of the sim (for the as-of stamp / results header)."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        arr = np.ascontiguousarray(np.asarray(obj, dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj.astype(float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]


def config_fingerprint(
    n_strategies: int = 1000, n_days: int = 1260, ann_vol: float = 0.15, seed: int = 833,
) -> str:
    """A stable label of the sim configuration (seeds + params) for the Fingerprint line."""
    key = f"null|N={n_strategies}|T={n_days}|vol={ann_vol}|seed={seed}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]
