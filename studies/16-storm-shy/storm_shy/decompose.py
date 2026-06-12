"""The teardown that earns the stamps — and, just as carefully, bounds them.

Four legs, in the order an honest investigation runs:

  1. :func:`spanning_alpha` — the Moreira–Muir test. Regress the managed return on buy-&-hold:
     ``managed = α + β·market + ε``. A positive **α**, significant under autocorrelation-robust
     (Newey–West) errors, means the overlay expands the mean–variance frontier — it adds something
     a static position cannot replicate. This is the leg that makes Signal `REAL`.
  2. :func:`sharpe_gain_bootstrap` — a paired **circular block** bootstrap CI on
     ``Sharpe(managed) − Sharpe(buy-hold)`` (blocks preserve the very vol-clustering the study
     trades; an i.i.d. resample would understate the interval), so the headline gain carries an
     honest interval, not just a point.
  3. :func:`certainty_equivalent` — the honest counter (Cederburg–O'Doherty–Wang–Yan 2020). A
     risk-averse CRRA investor doesn't care about Sharpe per se; they care about utility *after*
     paying for the leverage the overlay needs in calm times. We lever **both** series to the same
     unconditional vol and compare mean–variance certainty-equivalents. The managed book still
     wins on a clean clustered tape — but by the smaller, real margin, not the headline.
  4. :func:`equal_risk_return` — a blunt restatement: at matched full-sample vol, what extra annual
     return does the overlay deliver? The number a practitioner actually banks.

Leg 1 implements its own Newey–West OLS (``_ols_nw``) because the desk's ``quantlab.analytics``
HAC helper tests a *mean*, and here we need the HAC standard error of a regression *intercept*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import managed_returns, summary

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Newey–West OLS for y = a + b x  (HAC t-stats on both coefficients)
# --------------------------------------------------------------------------- #

def _ols_nw(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey–West (Bartlett) HAC covariance.

    Returns the intercept and slope with their HAC standard errors and t-stats. The sandwich is the
    textbook ``(X'X)^{-1} M (X'X)^{-1}`` with ``M`` the Bartlett-weighted autocovariance of the
    score ``X_t e_t`` — the same long-run-variance idea as ``quantlab.analytics.mean_tstat_hac``,
    generalised from a mean to a regression.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * resid[:, None]                       # n x 2 score
    M = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)               # Bartlett weight
        G = u[k:].T @ u[:-k]
        M += w * (G + G.T)
    cov = XtX_inv @ M @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {
        "alpha": float(beta[0]), "beta": float(beta[1]),
        "alpha_se": float(se[0]), "beta_se": float(se[1]),
        "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
        "beta_t": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
        "lags": int(lags), "n": int(n),
    }


def spanning_alpha(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **managed_kw,
) -> dict:
    """Moreira–Muir spanning regression: does the overlay expand the mean–variance frontier?

    Regresses the vol-managed return on buy-&-hold and reports the **annualised alpha** with its
    Newey–West t-stat. ``alpha_t`` comfortably above 2 (with ``alpha > 0``) is the `REAL` signal:
    the managed factor earns a return a static long position cannot span. ``managed_kw`` flow to
    :func:`storm_shy.strategy.managed_returns` (target_vol_ann, window, max_leverage, cost_bps).
    """
    m = managed_returns(returns, **managed_kw)
    bh = pd.Series(returns).astype(float).reindex(m.index)
    reg = _ols_nw(m.to_numpy(), bh.to_numpy())
    reg["alpha_ann_pct"] = float(((1.0 + reg["alpha"]) ** periods_per_year - 1.0) * 100.0)
    reg["alpha_bps_day"] = float(reg["alpha"] * 1e4)
    return reg


# --------------------------------------------------------------------------- #
# Bootstrap CI on the Sharpe gain
# --------------------------------------------------------------------------- #

def sharpe_gain_bootstrap(
    returns: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    block_size: int | None = None,
    method: str = "cbb",
    **managed_kw,
) -> dict:
    """Paired **circular block** bootstrap CI for ``Sharpe(managed) − Sharpe(buy-hold)``.

    Resamples the *aligned* (buy-hold, managed) daily pairs, preserving their contemporaneous
    link, and reports the point gain, the (1−alpha) percentile interval, and the fraction of
    resamples where the overlay *loses* — a p-value-like read on "is the gain real?".

    Daily returns are not i.i.d. — volatility clusters, which is the very effect this study
    trades — so an i.i.d. resample destroys the serial dependence and yields intervals that are
    too *narrow*. The default is therefore the house circular block bootstrap (Politis & Romano
    1994; same convention as ``quantlab.stats.sharpe_ci_bootstrap``): each resample concatenates
    blocks of ``block_size`` consecutive *pairs*, with starts drawn uniformly and indices
    wrapping around the end of the sample. ``block_size=None`` uses the rate-optimal
    ``round(n ** (1/3))`` rule; ``method='iid'`` reproduces the legacy independent resampling
    (block length 1), kept for comparison — expect it to understate the interval.
    """
    m = managed_returns(returns, **managed_kw)
    bh = pd.Series(returns).astype(float).reindex(m.index)
    a = bh.to_numpy(); b = m.to_numpy()
    n = a.size
    rng = np.random.default_rng(seed)

    if method == "iid":
        blk = 1
    elif method == "cbb":
        blk = int(block_size) if block_size is not None else max(1, round(n ** (1.0 / 3.0)))
    else:
        raise ValueError("method must be 'cbb' or 'iid'")
    blk = max(1, min(blk, n))

    def _sr(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0

    point = _sr(b) - _sr(a)
    n_blocks = int(np.ceil(n / blk))
    offsets = np.arange(blk)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        if blk == 1:
            idx = rng.integers(0, n, n)
        else:
            starts = rng.integers(0, n, n_blocks)
            idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = _sr(b[idx]) - _sr(a[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe_gain": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n),
        "n_boot": int(n_boot),
        "block_size": int(blk),
        "method": method,
    }


# --------------------------------------------------------------------------- #
# The honest counter: CRRA certainty-equivalent at matched risk (Cederburg et al.)
# --------------------------------------------------------------------------- #

def _lever_to_vol(r: pd.Series, target_ann: float, periods_per_year: int) -> pd.Series:
    """Apply a single constant leverage so ``r`` has full-sample annualised vol ``target_ann``."""
    cur = r.std(ddof=1) * np.sqrt(periods_per_year)
    k = target_ann / cur if cur > 0 else 1.0
    return r * k


def certainty_equivalent(
    returns: pd.Series,
    gamma: float = 5.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **managed_kw,
) -> dict:
    """Mean–variance certainty-equivalent of buy-hold vs managed, **levered to the same vol**.

    The Cederburg et al. (2020) discipline: a Sharpe gain is only an *investor* gain if it survives
    the leverage you must take to realise it. We scale both series to the same unconditional
    volatility (so neither wins by simply running more risk), then score each by a CRRA
    mean–variance certainty-equivalent, annualised::

        CE ≈ (mean − ½ γ variance) · periods_per_year

    A positive ``ce_gain_pct`` means a risk-averse investor is genuinely better off — the honest,
    smaller number behind the headline Sharpe. ``gamma=5`` is a standard moderate risk aversion.
    """
    m = managed_returns(returns, **managed_kw)
    bh = pd.Series(returns).astype(float).reindex(m.index)
    target = bh.std(ddof=1) * np.sqrt(periods_per_year)        # match to buy-hold's own vol

    def _ce(r):
        r = _lever_to_vol(r, target, periods_per_year)
        return (r.mean() - 0.5 * gamma * r.var(ddof=1)) * periods_per_year

    ce_bh, ce_m = _ce(bh), _ce(m)
    return {
        "gamma": float(gamma),
        "ce_buy_hold_pct": float(ce_bh * 100.0),
        "ce_managed_pct": float(ce_m * 100.0),
        "ce_gain_pct": float((ce_m - ce_bh) * 100.0),
        "matched_vol_ann": float(target),
    }


def equal_risk_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **managed_kw,
) -> dict:
    """Extra annualised return the overlay delivers at **matched full-sample volatility**.

    Levers both books to the buy-&-hold's vol and compares CAGR — the practitioner's bottom line:
    same risk, how much more return does re-timing it buy you?
    """
    m = managed_returns(returns, **managed_kw)
    bh = pd.Series(returns).astype(float).reindex(m.index)
    target = bh.std(ddof=1) * np.sqrt(periods_per_year)
    m_lev = _lever_to_vol(m, target, periods_per_year)
    s_bh, s_m = summary(bh, periods_per_year), summary(m_lev, periods_per_year)
    return {
        "matched_vol_ann": float(target),
        "buy_hold_cagr": s_bh["cagr"],
        "managed_cagr": s_m["cagr"],
        "excess_cagr_pct": float((s_m["cagr"] - s_bh["cagr"]) * 100.0),
        "buy_hold_maxdd": s_bh["max_drawdown"],
        "managed_maxdd": s_m["max_drawdown"],
    }
