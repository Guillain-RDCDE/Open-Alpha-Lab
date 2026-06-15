"""Signal and honest controls — Study 197 (Dividend-Payout-Ratio).

Arnott & Asness (2003) claim that a *higher* aggregate dividend payout ratio predicts
*higher* subsequent real earnings growth — the opposite of the standard reinvestment
narrative. Low payout signals empire-building managers who invest at low returns, not
high-returning reinvestment.

We test three forward outcomes:

1. **10-year real earnings growth** (the primary Arnott-Asness claim).
2. **10-year real price return** (can payout time the equity market?).

The honest control for each is an **unconditional baseline** — the long-run mean — so
we can ask whether payout ratio adds any predictive power above and beyond "buy and hold
forever." We also run an **in-sample / out-of-sample split** (split at 1953) to flag
data-mining risk.

Key inference tool: overlapping-window OLS with Newey-West HAC standard errors
(120 lags), following Hodrick (1992) for long-horizon regressions. Non-overlapping
annual observations are used as a robustness check.

No look-ahead: payout ratio at *t* is used to predict outcomes from *t* to *t+120*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regression engine
# ---------------------------------------------------------------------------
def ols_hac(
    x: np.ndarray,
    y: np.ndarray,
    maxlags: int = 120,
) -> dict:
    """OLS with Newey-West HAC standard errors for an overlapping-window regression.

    Parameters
    ----------
    x : predictor (1-d array, payout ratio).
    y : outcome (1-d array, forward earnings growth or price return).
    maxlags : Bartlett-kernel lag truncation; 120 for a 10-year horizon.

    Returns slope, intercept, their HAC t-stats, R², and sample size.
    """
    assert x.shape == y.shape
    n = x.size
    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x])
    # OLS estimates
    bhat = np.linalg.lstsq(X, y, rcond=None)[0]  # [intercept, slope]
    resid = y - X @ bhat
    # HAC (Newey-West) covariance matrix
    meat = np.zeros((2, 2))
    e = resid
    S = np.einsum("i,ij,ik->jk", e**2, X, X) / n  # lag-0
    for k in range(1, maxlags + 1):
        w = 1.0 - k / (maxlags + 1.0)
        Sk = np.einsum("i,ij,ik->jk", e[k:] * e[:-k], X[k:], X[:-k]) / n
        meat += 2.0 * w * Sk
    meat += S  # add lag-0
    meat = (meat + meat.T) / 2.0  # symmetrise
    sandwich = np.linalg.solve(X.T @ X / n, meat) @ np.linalg.solve(X.T @ X / n, np.eye(2)) / n
    se = np.sqrt(np.diag(sandwich))
    tstat = bhat / se
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "intercept": float(bhat[0]),
        "slope": float(bhat[1]),
        "t_intercept": float(tstat[0]),
        "t_slope": float(tstat[1]),
        "se_slope": float(se[1]),
        "r_squared": float(r2),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Primary analysis: payout ratio → forward outcomes
# ---------------------------------------------------------------------------
def regress_payout(
    df: pd.DataFrame,
    outcome_col: str = "fwd_10y_eg",
    payout_col: str = "Payout",
    maxlags: int = 120,
) -> dict:
    """HAC regression of payout ratio on a forward outcome over the full sample.

    Parameters
    ----------
    df : DataFrame from ``data.load_shiller()`` (or the synthetic tape).
    outcome_col : one of ``"fwd_10y_eg"`` or ``"fwd_10y_price"``.
    payout_col : name of the payout-ratio column.
    maxlags : Newey-West lag truncation (default 120 for 10-year horizon).

    Returns the full regression dict from :func:`ols_hac`, plus unconditional
    baseline stats (mean outcome, its HAC standard error, its t-stat for H0=0).
    """
    sub = df[[payout_col, outcome_col]].dropna()
    x = sub[payout_col].to_numpy(dtype=float)
    y = sub[outcome_col].to_numpy(dtype=float)
    reg = ols_hac(x, y, maxlags=maxlags)

    # Unconditional baseline: mean outcome (the buy-and-hold prediction)
    n = y.size
    e = y - y.mean()
    lrv = float(e @ e) / n
    for k in range(1, min(maxlags + 1, n)):
        w = 1.0 - k / (maxlags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se_uncond = np.sqrt(max(lrv, 0.0) / n)
    t_uncond = float(y.mean() / se_uncond) if se_uncond > 0 else float("nan")

    reg["unconditional_mean"] = float(y.mean())
    reg["unconditional_se"] = float(se_uncond)
    reg["unconditional_t"] = float(t_uncond)
    return reg


# ---------------------------------------------------------------------------
# In-sample / out-of-sample split
# ---------------------------------------------------------------------------
def is_oos_split(
    df: pd.DataFrame,
    outcome_col: str = "fwd_10y_eg",
    payout_col: str = "Payout",
    split_year: int = 1953,
    maxlags: int = 120,
) -> dict:
    """Repeat the regression in two non-overlapping windows split at ``split_year``.

    ``split_year`` = 1953 mirrors the Arnott & Asness (2003) roughly in-sample period.
    If the signal is an artefact of data-mining, it should not replicate out of sample.
    Returns dicts for the ``"is"`` (in-sample) and ``"oos"`` (out-of-sample) periods.
    """
    is_df = df[df.index.year <= split_year]
    oos_df = df[df.index.year > split_year]
    return {
        "is": regress_payout(is_df, outcome_col, payout_col, maxlags),
        "oos": regress_payout(oos_df, outcome_col, payout_col, maxlags),
        "split_year": split_year,
    }


# ---------------------------------------------------------------------------
# Non-overlapping robustness (annual January observations)
# ---------------------------------------------------------------------------
def non_overlapping_regression(
    df: pd.DataFrame,
    outcome_col: str = "fwd_10y_eg",
    payout_col: str = "Payout",
) -> dict:
    """Robust check on non-overlapping annual (January) observations.

    10-year windows overlap massively at monthly sampling, inflating the effective
    sample and making OLS standard errors too small even with Newey-West. Taking
    one observation per year (January) gives a clean non-overlapping series where
    heteroskedasticity-robust (HC3) errors are a valid alternative.
    Returned dict includes ``n``, ``slope``, ``t_hc3``, ``r_squared``.
    """
    ann = df[df.index.month == 1].copy()
    sub = ann[[payout_col, outcome_col]].dropna()
    x = sub[payout_col].to_numpy(dtype=float)
    y = sub[outcome_col].to_numpy(dtype=float)
    n = x.size
    X = np.column_stack([np.ones(n), x])
    bhat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ bhat
    # HC3 sandwich
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)  # hat values
    e2 = (resid / (1 - h)) ** 2
    meat = np.einsum("i,ij,ik->jk", e2, X, X)
    sandwich = np.linalg.inv(X.T @ X) @ meat @ np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sandwich))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(bhat[1]),
        "intercept": float(bhat[0]),
        "t_hc3": float(bhat[1] / se[1]) if se[1] > 0 else float("nan"),
        "se_slope": float(se[1]),
        "r_squared": float(r2),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------
def payout_quintile_means(
    df: pd.DataFrame,
    outcome_col: str = "fwd_10y_eg",
    payout_col: str = "Payout",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Mean forward outcome by payout-ratio quintile.

    A non-parametric complement to the regression: if high payout genuinely
    predicts higher forward outcomes, the quintile means should be monotonically
    increasing in payout.
    """
    sub = df[[payout_col, outcome_col]].dropna().copy()
    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    sub["quintile"] = pd.qcut(sub[payout_col], n_quantiles, labels=labels)
    out = (
        sub.groupby("quintile")[[ payout_col, outcome_col]]
        .mean()
        .rename(columns={payout_col: "mean_payout", outcome_col: "mean_outcome"})
    )
    return out


# ---------------------------------------------------------------------------
# Synthetic-tape regression helper (offline, for tests)
# ---------------------------------------------------------------------------
def regress_synthetic(
    df: pd.DataFrame,
    outcome_col: str = "fwd_outcome",
    payout_col: str = "payout",
) -> dict:
    """Thin wrapper around :func:`ols_hac` for the synthetic tape.

    Used by tests: ``payout_signal > 0`` must produce a positive, significant slope;
    ``payout_signal = 0`` must produce a slope indistinguishable from zero.
    """
    sub = df[[payout_col, outcome_col]].dropna()
    x = sub[payout_col].to_numpy(dtype=float)
    y = sub[outcome_col].to_numpy(dtype=float)
    # For synthetic: use standard OLS t (no overlapping window), maxlags=1
    return ols_hac(x, y, maxlags=1)


# ---------------------------------------------------------------------------
# Summary helper (mirrors study-72 style)
# ---------------------------------------------------------------------------
def summarize(reg: dict) -> dict:
    """Extract the decisive reporting numbers from a regression dict."""
    return {
        "slope": reg["slope"],
        "t_slope": reg["t_slope"],
        "r_squared": reg["r_squared"],
        "n": reg["n"],
        "unconditional_mean": reg.get("unconditional_mean", float("nan")),
        "unconditional_t": reg.get("unconditional_t", float("nan")),
    }
