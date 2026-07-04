"""Tested rules + inference for Study 623 — IPO Long-Run Underperformance.

THE CLAIM (Ritter 1991, JF; Loughran & Ritter 1995, JF; Ritter & Welch 2002, JF): IPOs
underperform seasoned stocks for 3-5 years after listing — the first-day pop is for the
flippers, the multi-year drift is for the bagholders. Two tracks:

* **Track A — the published cohort record, re-run through our stats.** Ritter's Table 19
  gives one *average 3-year buy-and-hold abnormal return* per cohort year, 1980-2024. We
  treat the cohort years as the sample and test the mean market-adjusted (and style-adjusted)
  BHAR against zero with a **Newey-West (HAC) t**: consecutive cohorts hold overlapping
  3-year windows, so the series is serially correlated by construction — lag = 3 covers the
  overlap. Both the equal-weighted cohort mean and Ritter's own N-weighted aggregate are
  reported. This is *published* data re-framed — by desk law it carries the literature story,
  not the REAL stamp.

* **Track B — the live tape (the judge).** Renaissance IPO ETF monthly total returns vs
  SPY / IWM / IWO, Nov 2013 onward. The decisive statistics are HAC t's on (i) the simple
  long-short spread IPO−benchmark (self-financing, so the risk-free leg cancels) and (ii)
  the **CAPM-style alpha** of IPO excess returns on benchmark excess returns (Newey-West
  intercept t). REAL requires |t| >= 2 *here*.

* **Third axis — is it small-growth beta rather than IPO-ness?** Ritter's own style
  adjustment (size + B/M matched firms) cuts the drag from −20.5% to −8.9% per 3 years.
  Live analogue: re-benchmark the ETF against IWO (Russell 2000 Growth) and against a
  two-factor market + size-growth-spread model, and see how much of the drag survives.

Execution & costs convention (documented once, used everywhere): the claim is a *passive
holding* claim — hold (or short) a published, rules-based basket. There is no signal to lag:
the rule is calendar-known (buy-and-hold both legs, monthly rebalanced to $1/$1), so the
effective lag is zero by construction, stated here explicitly. The tradable expression of
"IPOs drift down" is SHORT the IPO ETF / LONG the benchmark: the short leg pays borrow
(charged per month), plus one-way costs x NAV on the monthly rebalancing turnover. ETF
expense ratios are already inside the adjusted NAV tape (net-of-fee total returns).

All random baselines in the synthetic control are averaged over >= 20 seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0


# --------------------------------------------------------------------------- #
# HAC (Newey-West) machinery
# --------------------------------------------------------------------------- #
def nw_tstat(x: np.ndarray, lags: int = 6) -> tuple[float, float, float]:
    """Newey-West t of the mean of ``x``: returns (mean, HAC se, t)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    u = x - x.mean()
    gamma0 = float(u @ u) / n
    s = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(u[k:] @ u[:-k]) / n
    se = np.sqrt(max(s, 0.0) / n)
    if se == 0:
        return float(x.mean()), 0.0, float("nan")
    return float(x.mean()), float(se), float(x.mean() / se)


def hac_alpha(y: np.ndarray, X: np.ndarray, lags: int = 6) -> dict:
    """OLS of ``y`` on [1, X] with Newey-West (Bartlett) standard errors.

    ``y`` is the dependent excess-return series; ``X`` an (n,) or (n,k) matrix of factor
    excess returns. Returns the intercept (alpha, monthly), its HAC se and t, plus betas.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    Z = np.column_stack([np.ones(n), X])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    b = ZtZ_inv @ (Z.T @ y)
    u = y - Z @ b
    g = Z * u[:, None]                          # moment conditions
    S = (g.T @ g) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = (g[k:].T @ g[:-k]) / n
        S += w * (gk + gk.T)
    cov = ZtZ_inv @ (n * S) @ ZtZ_inv
    se = np.sqrt(np.diag(cov))
    return {
        "alpha": float(b[0]), "alpha_se": float(se[0]),
        "alpha_t": float(b[0] / se[0]) if se[0] > 0 else float("nan"),
        "betas": b[1:].tolist(), "beta_ses": se[1:].tolist(),
        "n": int(n),
    }


# --------------------------------------------------------------------------- #
# Track A — Ritter cohort table through our stats
# --------------------------------------------------------------------------- #
def cohort_stats(cohorts: pd.DataFrame, col: str, lags: int = 3) -> dict:
    """Mean cohort-year BHAR (%/3yr) with a Newey-West t across cohort years.

    Consecutive cohorts hold overlapping 3-year windows -> serial correlation by
    construction; ``lags=3`` covers the overlap. Also reports Ritter's N-weighted mean
    and the share of negative cohort years.
    """
    x = cohorts[col].values.astype(float)
    mean, se, t = nw_tstat(x, lags=lags)
    w = cohorts["n_ipos"].values.astype(float)
    return {
        "n_cohorts": int(len(x)),
        "mean_pct": mean, "se_pct": se, "t_nw": t,
        "weighted_mean_pct": float(np.average(x, weights=w)),
        "share_negative": float((x < 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Track B — the live tape
# --------------------------------------------------------------------------- #
def spread_stats(panel: pd.DataFrame, bench: str = "SPY", lags: int = 6) -> dict:
    """Monthly long-IPO / short-``bench`` spread: mean (bps/mo), annualised %, HAC t.

    Self-financing $1/$1 spread, so the risk-free leg cancels — an excess-vs-excess race
    by construction. Gross of trading costs and borrow (see :func:`short_overlay_stats`).
    """
    d = (panel["IPO"] - panel[bench]).dropna().values
    mean, se, t = nw_tstat(d, lags=lags)
    return {
        "bench": bench, "n_months": int(len(d)),
        "mean_bps": mean * 1e4, "t_nw": t,
        "ann_pct": (np.expm1(np.log1p(mean) * MONTHS)) * 100.0,
    }


def alpha_stats(panel: pd.DataFrame, bench: str = "SPY", lags: int = 6) -> dict:
    """HAC-t CAPM-style alpha of IPO excess returns on ``bench`` excess returns."""
    y = (panel["IPO"] - panel["rf"]).values
    x = (panel[bench] - panel["rf"]).values
    r = hac_alpha(y, x, lags=lags)
    r.update({
        "bench": bench,
        "alpha_bps": r["alpha"] * 1e4,
        "alpha_ann_pct": (np.expm1(np.log1p(r["alpha"]) * MONTHS)) * 100.0,
        "beta": r["betas"][0],
    })
    return r


def two_factor_alpha(panel: pd.DataFrame, lags: int = 6) -> dict:
    """Alpha on market + size/growth spread: IPO_exc ~ SPY_exc + (IWO − SPY).

    The second factor is the small-growth-minus-market spread (self-financing), the live
    analogue of Ritter's style adjustment: if the IPO drag is really small-growth beta,
    loading on this factor should absorb it.
    """
    y = (panel["IPO"] - panel["rf"]).values
    X = np.column_stack([
        (panel["SPY"] - panel["rf"]).values,
        (panel["IWO"] - panel["SPY"]).values,
    ])
    r = hac_alpha(y, X, lags=lags)
    r.update({
        "alpha_bps": r["alpha"] * 1e4,
        "alpha_ann_pct": (np.expm1(np.log1p(r["alpha"]) * MONTHS)) * 100.0,
        "beta_mkt": r["betas"][0], "beta_style": r["betas"][1],
    })
    return r


def cumulative_growth(panel: pd.DataFrame, col: str) -> float:
    """Growth of $1 (total return) over the panel."""
    return float(np.exp(np.log1p(panel[col].values).sum()))


def short_overlay_stats(panel: pd.DataFrame, bench: str = "SPY",
                        cost_bps: float = 5.0, borrow_ann_pct: float = 1.0,
                        lags: int = 6) -> dict:
    """The tradable expression: SHORT the IPO ETF / LONG the benchmark, net of frictions.

    Monthly net spread = (bench − IPO) − borrow/12 − 2 × cost × turnover, where the short
    leg pays ``borrow_ann_pct``/yr (charged monthly) and the $1/$1 monthly re-hedge trades
    ~|r_IPO − r_bench| of NAV per month on each leg (one-way costs × NAV, both legs
    counted). No signal lag exists: the rule is calendar-known (always-on spread).
    """
    d = (panel[bench] - panel["IPO"]).dropna().values
    turnover = np.abs(d)                        # per-leg re-hedge, fraction of NAV
    c = cost_bps / 1e4
    net = d - borrow_ann_pct / 100.0 / 12.0 - 2.0 * c * turnover
    g_mean, _, g_t = nw_tstat(d, lags=lags)
    n_mean, _, n_t = nw_tstat(net, lags=lags)
    return {
        "bench": bench, "cost_bps": cost_bps, "borrow_ann_pct": borrow_ann_pct,
        "gross_bps": g_mean * 1e4, "gross_t": g_t,
        "net_bps": n_mean * 1e4, "net_t": n_t,
        "net_ann_pct": (np.expm1(np.log1p(n_mean) * MONTHS)) * 100.0,
        "worst_drawdown_pct": _max_drawdown(net) * 100.0,
    }


def _max_drawdown(monthly: np.ndarray) -> float:
    nav = np.exp(np.cumsum(np.log1p(monthly)))
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# Synthetic control — seed-averaged machinery proof
# --------------------------------------------------------------------------- #
def synthetic_control(edge: float, n_seeds: int = 20, lags: int = 6,
                      base_seed: int = 623) -> dict:
    """Run the HAC-alpha detector on ``n_seeds`` synthetic worlds with a planted ``edge``.

    Averaged over >= 20 seeds per desk law (no single-seed baselines). Returns the mean
    measured alpha, the mean HAC t and the |t| >= 2 rejection rate. With ``edge = 0`` the
    rejection rate must sit near the nominal 5%; with a large planted drag it must fire.
    """
    from . import data as _data

    alphas, ts = [], []
    for s in range(n_seeds):
        w = _data.synthetic_world(edge=edge, seed=base_seed + s)
        r = alpha_stats(w, bench="SPY", lags=lags)
        alphas.append(r["alpha_bps"])
        ts.append(r["alpha_t"])
    ts = np.asarray(ts)
    return {
        "edge_bps": edge * 1e4, "n_seeds": int(n_seeds),
        "mean_alpha_bps": float(np.mean(alphas)),
        "mean_t": float(np.mean(ts)),
        "reject_rate": float((np.abs(ts) >= 2.0).mean()),
    }
