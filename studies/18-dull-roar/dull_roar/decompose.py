"""The teardown that earns the stamps — and, just as carefully, bounds them.

The raw scoreboard in :mod:`strategy` is suggestive but not decisive: a low-vol book can look good for
the wrong reason (it simply ran less market risk), and a long-short can look good on a "free short"
assumption that no desk gets. Four legs settle it, in the order an honest investigation runs:

  1. :func:`capm_alpha` — regress any book on the market: ``r = alpha + beta·market + e``. A positive
     **alpha**, significant under autocorrelation-robust (Newey-West) errors, is return a static market
     position can't replicate. This is the leg that makes Signal `REAL`.
  2. :func:`beta_neutral_bab` — the Frazzini-Pedersen (2014) construction: lever the low-beta leg *up*
     and the high-beta leg *down* so the long-short is **beta-neutral**, removing the structural short-
     the-market drag that hides the anomaly in the naive dollar-neutral book. Its alpha is the clean
     measure of the effect.
  3. :func:`beta_tilt_test` — the honest counter for the *long-only* book. Lever the low-vol portfolio
     to **beta 1** (so it runs the same market risk as just holding the index) and ask what excess
     return survives. Most of a low-vol book's "outperformance" is simply *lower beta* — a risk
     reduction you could buy with cash, not alpha — and this leg prices exactly how much.
  4. :func:`leg_attribution` — split the effect across the two legs. The literature (Ang et al. 2006)
     finds the anomaly concentrates in the **short** (high-vol) leg — the one you can't cheaply borrow.
     If most of the alpha lives there, the harvestable (long-only) part is the smaller, defensive slice.

Leg 1 carries its own Newey-West OLS (``_ols_nw``) because we need the HAC standard error of a
regression *intercept*, not of a mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import build, summary

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Newey-West OLS for y = a + b x  (HAC t-stats on both coefficients)
# --------------------------------------------------------------------------- #

def _ols_nw(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey-West (Bartlett) HAC covariance.

    Returns intercept and slope with their HAC standard errors and t-stats — the sandwich
    ``(X'X)^{-1} M (X'X)^{-1}`` with ``M`` the Bartlett-weighted autocovariance of the score
    ``X_t e_t``, the same long-run-variance idea the desk's ``quantlab.analytics`` uses for a mean,
    generalised to a regression intercept.
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
    u = X * resid[:, None]
    M = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
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


def capm_alpha(
    port: pd.Series,
    market: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """CAPM regression of a book on the market: annualised alpha with its Newey-West t-stat and beta.

    ``alpha_t`` above ~2 (with ``alpha > 0``) is the `REAL` signal — return the market doesn't span.
    ``beta`` is the honesty check: a deeply *negative* beta (naive long-short) or a *sub-1* beta
    (low-vol long-only) flags that a chunk of the raw return is just a market-exposure choice.
    """
    p = pd.Series(port).astype(float).dropna()
    m = pd.Series(market).astype(float).reindex(p.index)
    ok = p.notna() & m.notna()
    reg = _ols_nw(p[ok].to_numpy(), m[ok].to_numpy())
    reg["alpha_ann_pct"] = float(reg["alpha"] * periods_per_year * 100.0)
    reg["alpha_bps_day"] = float(reg["alpha"] * 1e4)
    return reg


# --------------------------------------------------------------------------- #
# The beta-neutral BAB construction (Frazzini-Pedersen)
# --------------------------------------------------------------------------- #

def beta_neutral_bab(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    cost_bps: float = 1.0,
    borrow_bps_ann: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **build_kw,
) -> dict:
    """Beta-neutralise the long-short (lever low-beta up, high-beta down) and price its alpha.

    The naive dollar-neutral book is structurally *short the market* (long low-beta, short high-beta),
    so in a rising market its beta drag masks the anomaly. Frazzini-Pedersen fix this by scaling each
    leg by ``1/beta`` so the combination is beta-neutral; the result is the clean BAB factor. Returns
    its CAPM alpha (HAC), realised beta, Sharpe and the two leg betas. Charges the same costs as the
    naive book, including the short-leg borrow.
    """
    b = build(panel, market=market, cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann, **build_kw)
    mkt = b["market"]
    beta_low = capm_alpha(b["low_only"], mkt, periods_per_year)["beta"]
    beta_high = capm_alpha(b["high"], mkt, periods_per_year)["beta"]

    # daily borrow drag on the high (short) leg, levered with it
    borrow = borrow_bps_ann * 1e-4 / periods_per_year
    bab = (b["low_only"] / beta_low - (b["high"] + borrow) / beta_high).rename("bab")

    reg = capm_alpha(bab, mkt, periods_per_year)
    s = summary(bab, periods_per_year)
    return {
        "alpha_ann_pct": reg["alpha_ann_pct"],
        "alpha_t": reg["alpha_t"],
        "beta": reg["beta"],
        "sharpe": s["sharpe"],
        "beta_low_leg": float(beta_low),
        "beta_high_leg": float(beta_high),
        "stream": bab,
        "n_days": s["n_days"],
    }


# --------------------------------------------------------------------------- #
# The honest counter: is the long-only book just low beta?
# --------------------------------------------------------------------------- #

def beta_tilt_test(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    cost_bps: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **build_kw,
) -> dict:
    """Lever the low-vol long-only book to **beta 1** and ask what excess return survives.

    A low-vol portfolio runs a sub-1 market beta, so part of its higher Sharpe is simply that it took
    *less* market risk — a reduction any investor can buy by holding cash beside the index, not alpha.
    We scale the low-vol stream by ``1/beta`` to match the market's risk, then compare CAGR and Sharpe.
    The leftover excess CAGR is the part that is *not* just a beta choice — the honest, smaller number.
    """
    b = build(panel, market=market, cost_bps=cost_bps, **build_kw)
    mkt = b["market"]
    reg = capm_alpha(b["low_only"], mkt, periods_per_year)
    beta = reg["beta"] if reg["beta"] > 1e-6 else 1.0
    levered = (b["low_only"] / beta).rename("low_beta1")

    s_mkt = summary(mkt, periods_per_year)
    s_low = summary(b["low_only"], periods_per_year)
    s_lev = summary(levered, periods_per_year)
    return {
        "low_beta": float(reg["beta"]),
        "alpha_ann_pct": reg["alpha_ann_pct"],
        "alpha_t": reg["alpha_t"],
        "low_only_sharpe": s_low["sharpe"],
        "market_sharpe": s_mkt["sharpe"],
        "sharpe_gain": float(s_low["sharpe"] - s_mkt["sharpe"]),
        "levered_cagr": s_lev["cagr"],
        "market_cagr": s_mkt["cagr"],
        "excess_cagr_at_beta1_pct": float((s_lev["cagr"] - s_mkt["cagr"]) * 100.0),
        "low_only_maxdd": s_low["max_drawdown"],
        "market_maxdd": s_mkt["max_drawdown"],
    }


# --------------------------------------------------------------------------- #
# Where does the effect live — long (calm) leg or short (wild) leg?
# --------------------------------------------------------------------------- #

def leg_attribution(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    cost_bps: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **build_kw,
) -> dict:
    """Split the anomaly's alpha into the **long (low-vol)** leg and the **short (high-vol)** leg.

    Each leg's CAPM alpha measures its own contribution. If the bulk sits in the high-vol leg's
    *negative* alpha — the names you'd have to short, and can't cheaply — then the harvestable
    long-only slice is the smaller, defensive one, and the tradability stamp must say so.
    """
    b = build(panel, market=market, cost_bps=cost_bps, **build_kw)
    mkt = b["market"]
    low = capm_alpha(b["low_only"], mkt, periods_per_year)
    high = capm_alpha(b["high"], mkt, periods_per_year)
    # the long-short alpha is (low alpha) - (high alpha); attribute shares by magnitude
    a_low, a_high = low["alpha_ann_pct"], high["alpha_ann_pct"]
    spread = a_low - a_high
    share_short = abs(a_high) / (abs(a_low) + abs(a_high)) if (abs(a_low) + abs(a_high)) > 1e-9 else np.nan
    return {
        "low_leg_alpha_pct": float(a_low),
        "low_leg_alpha_t": float(low["alpha_t"]),
        "low_leg_beta": float(low["beta"]),
        "high_leg_alpha_pct": float(a_high),
        "high_leg_alpha_t": float(high["alpha_t"]),
        "high_leg_beta": float(high["beta"]),
        "long_short_alpha_pct": float(spread),
        "share_in_short_leg": float(share_short),
    }


# --------------------------------------------------------------------------- #
# Bootstrap CI on the long-only Sharpe gain over the market
# --------------------------------------------------------------------------- #

def sharpe_gain_bootstrap(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    cost_bps: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **build_kw,
) -> dict:
    """Paired bootstrap CI for ``Sharpe(low-vol long-only) - Sharpe(market)``.

    Resamples the aligned (market, low-vol) daily pairs with replacement, preserving their
    contemporaneous link, and reports the point gain, the (1-alpha) interval, and the fraction of
    resamples where the low-vol book *loses* — a p-value-like read on the defensive tilt's reality.
    """
    b = build(panel, market=market, cost_bps=cost_bps, **build_kw)
    m = b["market"].to_numpy()
    lo = b["low_only"].reindex(b["market"].index).to_numpy()
    n = m.size
    rng = np.random.default_rng(seed)

    def _sr(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0

    point = _sr(lo) - _sr(m)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = _sr(lo[idx]) - _sr(m[idx])
    lo_ci, hi_ci = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe_gain": float(point),
        "ci_low": float(lo_ci),
        "ci_high": float(hi_ci),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n),
        "n_boot": int(n_boot),
    }
