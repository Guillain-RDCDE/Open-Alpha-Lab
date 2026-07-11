"""Strategy + inference for Study 664 — ESG Premium.

The claim: **ESG investing pays a premium** ("doing well by doing good") — screening out
fossil fuels, tobacco and controversial weapons and tilting toward high ESG-rated names
should either boost risk-adjusted return (the "good companies are better-run companies"
story) or at least not cost anything. The skeptical counter-claim: any ESG-fund outperformance
is just a **relabelled large-cap growth/quality tilt** — ESG screens mechanically underweight
energy/utilities and overweight tech/services, so once you control for the growth-value and
quality factors any residual "ESG alpha" should evaporate.

Measurements:

* **Tracking difference & Sharpe** — ESGU vs SPY (2016-12 → as-of) and SUSA vs IVV
  (2005-01 → as-of): CAGR, annualized vol, excess-of-cash Sharpe (both legs against the same
  ^IRX risk-free proxy — never a raw-Sharpe-vs-excess-Sharpe race), tracking error (annualized
  std of the daily active return) and information ratio.
* **Active-return spread test** — daily (ESG fund − benchmark) return, Newey-West (HAC) t of
  the mean (the planned primary; daily active returns are autocorrelated), a Welch t
  cross-check, net of one documented cost convention.
* **Factor decomposition** — regress the ESG fund's daily return on the benchmark, a
  growth-value spread (IVW − IVE) and a quality spread (QUAL − SPY), Newey-West (HAC) SEs.
  The residual intercept ("ESG alpha" once the style tilts are priced in) is the decisive
  number for the "is it just a growth tilt" question.
* **Third axis (myth-check)** — does a long-ESG / short-benchmark spread beat a coin flip
  (mean daily active return significantly different from zero, after the one documented cost
  convention: entry/exit one-way costs on both legs plus a modest short-borrow drag on the
  benchmark leg)?

The decisive numbers are the Newey-West *t* on the raw active-return spread and the
Newey-West *t* on the factor-regression alpha — the gap between the two is the entire story.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int = 5) -> dict:
    """OLS with a Newey-West (Bartlett kernel) HAC covariance. X excludes the intercept
    column (added here). Returns beta (incl. intercept, index 0) and its HAC t-stats."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    keep = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y, X = y[keep], X[keep]
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ y)
    u = y - Xd @ beta
    s = Xd * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = beta / np.where(se > 0, se, np.nan)
    return {"beta": beta, "se": se, "t": t, "n": n}


def one_sample_nw_t(x: np.ndarray, lags: int = 5) -> dict:
    """Newey-West HAC t of the mean of ``x`` (regression on a constant only)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    r = ols_hac(x, np.zeros((len(x), 0)), lags=lags)
    return {"mean": float(r["beta"][0]), "t": float(r["t"][0]), "n": r["n"]}


# --------------------------------------------------------------------------- #
# Return frames
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def rf_daily(irx: pd.Series) -> pd.Series:
    """Daily risk-free rate from the ^IRX 13-week T-bill discount yield (percent points,
    annualized). rf_daily = (yield / 100) / 252 — a standard simple-annualization proxy."""
    return (irx / 100.0) / TRADING_DAYS


# --------------------------------------------------------------------------- #
# Tracking difference & Sharpe
# --------------------------------------------------------------------------- #
def tracking_stats(fund_ret: pd.Series, bench_ret: pd.Series, rf: pd.Series) -> dict:
    """CAGR, vol, excess-of-cash Sharpe (both legs), tracking error & information ratio."""
    idx = fund_ret.index.intersection(bench_ret.index).intersection(rf.index)
    f, b, r = fund_ret.reindex(idx), bench_ret.reindex(idx), rf.reindex(idx)
    n = len(idx)
    years = n / TRADING_DAYS

    def cagr(x):
        return float((1.0 + x).prod() ** (1.0 / years) - 1.0)

    f_ex, b_ex = f - r, b - r
    active = f - b

    def sharpe(ex):
        sd = ex.std(ddof=1)
        return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ir = float(active.mean() * TRADING_DAYS / te) if te > 0 else float("nan")
    return {
        "n": n, "start": str(idx.min().date()), "end": str(idx.max().date()), "years": years,
        "fund_cagr": cagr(f), "bench_cagr": cagr(b),
        "fund_vol": float(f.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "bench_vol": float(b.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "fund_sharpe_xs": sharpe(f_ex), "bench_sharpe_xs": sharpe(b_ex),
        "tracking_error": te, "information_ratio": ir,
        "active_mean_ann": float(active.mean() * TRADING_DAYS),
    }


# --------------------------------------------------------------------------- #
# The headline spread test
# --------------------------------------------------------------------------- #
def spread_test(fund_ret: pd.Series, bench_ret: pd.Series, nw_lags: int = 5,
                 cost_bps_oneway: float = 5.0, borrow_bps_ann: float = 30.0) -> dict:
    """Daily active return (fund - bench): NW t (primary), Welch t (cross-check), net of
    one documented cost convention (2 legs x one-way cost, amortized over the sample, plus
    an annualized short-borrow drag on the benchmark leg)."""
    idx = fund_ret.index.intersection(bench_ret.index)
    active = (fund_ret.reindex(idx) - bench_ret.reindex(idx)).dropna()
    n = len(active)
    nw = one_sample_nw_t(active.values, lags=nw_lags)
    half = np.zeros(0)
    welch = welch_t(fund_ret.reindex(idx).values, bench_ret.reindex(idx).values)
    entry_exit_bps = 2 * 2 * cost_bps_oneway   # 2 legs x one-way x (entry+exit), amortized once
    borrow_drag_ann = borrow_bps_ann / 1e4
    gross_ann = nw["mean"] * TRADING_DAYS
    net_ann = gross_ann - borrow_drag_ann - (entry_exit_bps / 1e4) / (n / TRADING_DAYS)
    return {
        "n": n, "mean_daily_bps": nw["mean"] * 1e4, "gross_ann_pct": gross_ann * 100,
        "net_ann_pct": net_ann * 100, "nw_t": nw["t"], "welch_t": welch,
        "hit_rate": float((active > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Factor decomposition — is the gap just a growth/quality tilt?
# --------------------------------------------------------------------------- #
def factor_decomposition(fund_ret: pd.Series, mkt_ret: pd.Series,
                          growth_val_spread: pd.Series, quality_spread: pd.Series,
                          nw_lags: int = 5) -> dict:
    """Regress fund_ret on [mkt_ret, growth_val_spread, quality_spread] with an intercept
    (Newey-West HAC SEs). The intercept, annualized, is the "ESG alpha" once the market and
    the two style tilts are priced in."""
    idx = (fund_ret.index.intersection(mkt_ret.index)
           .intersection(growth_val_spread.index).intersection(quality_spread.index))
    y = fund_ret.reindex(idx).values
    X = np.column_stack([mkt_ret.reindex(idx).values, growth_val_spread.reindex(idx).values,
                          quality_spread.reindex(idx).values])
    r = ols_hac(y, X, lags=nw_lags)
    beta, t = r["beta"], r["t"]
    return {
        "n": r["n"],
        "alpha_daily": float(beta[0]), "alpha_ann_pct": float(beta[0]) * TRADING_DAYS * 100,
        "alpha_t": float(t[0]),
        "beta_mkt": float(beta[1]), "beta_mkt_t": float(t[1]),
        "beta_growth_value": float(beta[2]), "beta_growth_value_t": float(t[2]),
        "beta_quality": float(beta[3]), "beta_quality_t": float(t[3]),
    }


def raw_vs_factor_alpha(fund_ret: pd.Series, bench_ret: pd.Series,
                         mkt_ret: pd.Series, growth_val_spread: pd.Series,
                         quality_spread: pd.Series, nw_lags: int = 5) -> dict:
    """Convenience: the raw active-return NW t next to the factor-model alpha NW t, so the
    "does controlling for style kill it" contrast is one function call."""
    raw = spread_test(fund_ret, bench_ret, nw_lags=nw_lags)
    fac = factor_decomposition(fund_ret, mkt_ret, growth_val_spread, quality_spread,
                                nw_lags=nw_lags)
    return {"raw_t": raw["nw_t"], "raw_ann_pct": raw["gross_ann_pct"],
            "factor_alpha_t": fac["alpha_t"], "factor_alpha_ann_pct": fac["alpha_ann_pct"],
            "beta_growth_value": fac["beta_growth_value"],
            "beta_growth_value_t": fac["beta_growth_value_t"],
            "beta_quality": fac["beta_quality"], "beta_quality_t": fac["beta_quality_t"]}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, nw_lags: int = 5) -> dict:
    """Run the headline active-return NW-t test on a synthetic paired (fund, bench) world."""
    active = (world["fund_ret"] - world["bench_ret"]).values
    return one_sample_nw_t(active, lags=nw_lags)
