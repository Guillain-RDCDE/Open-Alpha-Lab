"""Strategy + inference for Study 377 — Bid-Ask-Bounce (Roll 1984).

The claim under test: short-horizon ("one-day") **mean reversion** is a real, harvestable edge.
The rival explanation (Roll, 1984): it is the **bid-ask bounce** in disguise — observed
transaction returns carry a *mechanical* negative lag-1 autocovariance ``cov1 = -(s/2)**2`` that
a reversal book cannot capture, because capturing it means crossing the very spread ``s`` that
creates it.

This module supplies:

  * ``lag1_autocorr`` / ``autocov1`` — the apparent one-day reversion in a return series.
  * ``roll_spread`` — Roll's spread estimator ``s = 2*sqrt(-cov1)`` (defined only when cov1<0),
    which *reads back* the spread that the bounce implies.
  * ``reversal_book`` — the textbook one-day reversal: short yesterday's winners / long
    yesterday's losers (cross-sectional, equal weight, dollar-neutral), with a **1-day execution
    lag** and **one-way costs = half the spread per leg per rebalance** (the honest cost of a
    daily-turnover book that trades against the bounce).
  * Inference: a **Welch t** of the book's daily PnL, a **block-bootstrap / placebo** null, a
    **win-rate vs the 50% base rate**, and a **gross-vs-net** decomposition that is the whole
    point — gross looks like an edge, net of the spread it is gone.

The decisive object is not whether ``cov1 < 0`` (it always is, by the bounce) but whether the
reversal book survives the spread. On Roll's model with ``edge = 0`` it does not; only a
*planted genuine* reversion survives. That is the line between "real edge" and "the bounce."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Apparent reversion + Roll's spread estimator
# --------------------------------------------------------------------------- #
def autocov1(r: np.ndarray) -> float:
    """Lag-1 autocovariance of a return series (NaNs dropped). Negative => apparent reversion."""
    x = np.asarray(r, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    return float((x[:-1] * x[1:]).mean())


def lag1_autocorr(r: np.ndarray) -> float:
    """Lag-1 autocorrelation (NaNs dropped). The headline 'mean-reversion' number."""
    x = np.asarray(r, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    denom = (x * x).mean()
    if denom == 0:
        return float("nan")
    return float((x[:-1] * x[1:]).mean() / denom)


def roll_spread(r: np.ndarray) -> float:
    """Roll (1984) implied effective spread: ``s = 2*sqrt(-cov1)``.

    Defined only when the lag-1 autocovariance is negative (the bounce regime); returns NaN
    otherwise (a positive cov1 means momentum, not a bounce). The estimate is the round-trip
    spread in the same units as the returns (here, log-return units ~ fractional spread).
    """
    c1 = autocov1(r)
    if not np.isfinite(c1) or c1 >= 0:
        return float("nan")
    return float(2.0 * np.sqrt(-c1))


# --------------------------------------------------------------------------- #
# The one-day reversal book (cross-sectional, the real-tape strategy)
# --------------------------------------------------------------------------- #
def reversal_book(ret: pd.DataFrame, cost_halfspread: float = 0.0,
                  lag: int = 1) -> pd.Series:
    """Equal-weight, dollar-neutral one-day cross-sectional reversal PnL.

    Each day the signal is *minus yesterday's return* cross-sectionally demeaned (long the
    losers, short the winners), normalised so gross exposure sums to 1 (half long, half short).
    The position formed on the close of ``t-lag`` earns day ``t``'s return — one execution lag,
    applied once. ``cost_halfspread`` charges, per name per day, the **one-way** cost of the
    turnover: a daily-flip reversal book turns over its whole book each day, so the per-name
    per-day cost is ~ ``cost_halfspread`` times the absolute weight change (here ≈ the weight
    itself, since the book re-forms daily). Returns the daily net PnL series.
    """
    R = ret.copy()
    sig = -(R.shift(lag))                       # yesterday's return, reversed (1-day lag)
    sig = sig.sub(sig.mean(axis=1), axis=0)     # cross-sectionally demean -> dollar-neutral
    gross = sig.abs().sum(axis=1)
    w = sig.div(gross.where(gross > 0), axis=0).fillna(0.0)   # |weights| sum to 1
    pnl_gross = (w * R).sum(axis=1)
    # one-way cost: book re-forms daily, turnover per name ~ |w_t - w_{t-1}|.
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    cost = cost_halfspread * turnover
    pnl = pnl_gross - cost
    return pnl.dropna()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``mean(sample) - mu0`` (NaN if < 2 obs). The book's daily-PnL t-stat."""
    x = np.asarray(sample, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    if se == 0:
        return float("nan")
    return float((x.mean() - mu0) / se)


def block_bootstrap_p(pnl: np.ndarray, block: int = 10, n_draws: int = 5_000,
                      seed: int = 377) -> dict:
    """Block-bootstrap null: is the mean daily PnL distinguishable from zero?

    Resamples the demeaned PnL in blocks (preserving short-horizon dependence), re-centres each
    draw, and returns the two-sided p-value P[|bootstrap mean| >= |observed mean|]. Block
    resampling (not i.i.d.) because reversal PnL is itself autocorrelated.
    """
    x = np.asarray(pnl, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < block + 1:
        return {"mean": float("nan"), "p_value": float("nan")}
    obs = float(x.mean())
    xc = x - obs                                  # null: mean zero
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_draws)
    starts_hi = n - block
    for i in range(n_draws):
        starts = rng.integers(0, starts_hi + 1, size=n_blocks)
        draw = np.concatenate([xc[s:s + block] for s in starts])[:n]
        means[i] = draw.mean()
    p = float((np.abs(means) >= abs(obs)).mean())
    return {"mean": obs, "p_value": p}


def book_sharpe(ret: pd.DataFrame, cost_halfspread: float = 0.0, ann: int = 252) -> float:
    """Annualised Sharpe of the reversal book at a cost — fast, no bootstrap.

    For cost sweeps / capacity curves where only the Sharpe is needed (the bootstrap p-value
    in :func:`summarize_book` is what makes that call expensive)."""
    x = reversal_book(ret, cost_halfspread=cost_halfspread).values
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


def summarize_book(ret: pd.DataFrame, cost_halfspread: float = 0.0,
                   ann: int = 252) -> dict:
    """Headline stats for the reversal book at a given one-way half-spread cost.

    Returns mean daily PnL, daily win-rate (vs the 50% base rate), annualised Sharpe, the
    one-sample t and the block-bootstrap p-value.
    """
    pnl = reversal_book(ret, cost_halfspread=cost_halfspread)
    x = pnl.values
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0, "mean_bps": float("nan"), "win": float("nan"),
                "sharpe": float("nan"), "t": float("nan"), "p": float("nan")}
    bb = block_bootstrap_p(x)
    sd = x.std(ddof=1)
    return {
        "n": int(len(x)),
        "mean_bps": float(x.mean() * 1e4),
        "win": float((x > 0).mean()),
        "sharpe": float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan"),
        "t": welch_t(x),
        "p": bb["p_value"],
    }


def avg_lag1(ret: pd.DataFrame) -> float:
    """Mean per-name lag-1 autocorrelation across the panel (the apparent reversion)."""
    vals = [lag1_autocorr(ret[c].values) for c in ret.columns]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def avg_roll_spread(ret: pd.DataFrame) -> float:
    """Mean per-name Roll-implied effective spread across the panel (round-trip, fractional)."""
    vals = [roll_spread(ret[c].values) for c in ret.columns]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")
