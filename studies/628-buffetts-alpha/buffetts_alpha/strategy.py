"""Stats for Study 628 — Buffett's Alpha.

The tested claim (Frazzini, Kabiller & Pedersen, *Buffett's Alpha*, FAJ 2018): Berkshire's
four-decade CAPM alpha is real (their 1976-2011 sample: ~19% ann. excess, alpha *t* > 3),
is largely explained by quality + low-beta exposure financed with cheap insurance-float
leverage — and has been fading.

What lives here:

* ``ols_hac``     — multivariate OLS with Newey-West (Bartlett-kernel) standard errors,
  the desk's serial-correlation-robust inference for monthly alpha regressions.
* ``capm``        — excess-vs-excess CAPM (brk-rf on mkt-rf) over any window: annualised
  alpha, HAC alpha *t*, beta, R², Sharpe of both legs (excess-vs-excess race), IR.
* ``rolling_alpha`` — the decay curve: 10-year (120m) rolling CAPM alpha + HAC *t*.
* ``decade_table``  — per-decade CAPM rows.
* ``factor_lite``   — the FKP attribution with investable proxies: alpha before/after
  adding QUAL + USMV excess returns (ETF era only, honestly labeled).

No portfolio is being timed here: the only "strategy" is buy-and-hold BRK, so there is no
signal to lag — the one timing convention (documented) is that the risk-free leg uses the
PREVIOUS month-end T-bill yield, the rate actually lockable at the start of each month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Newey-West OLS
# --------------------------------------------------------------------------- #
def nw_lags(n: int) -> int:
    """Newey-West rule-of-thumb truncation lag: floor(4*(n/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS of y on [1, X] with Newey-West (1987) HAC standard errors.

    Returns dict with ``coef`` (const first), ``t`` (HAC t-stats), ``r2``, ``n``, ``lags``.
    Bartlett kernel; the standard monthly-returns inference when residuals are serially
    correlated.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if lags is None:
        lags = nw_lags(n)
    Z = np.column_stack([np.ones(n), X])
    k = Z.shape[1]
    XtX_inv = np.linalg.inv(Z.T @ Z)
    beta = XtX_inv @ (Z.T @ y)
    u = y - Z @ beta
    # Newey-West long-run covariance of the score
    S = (Z * u[:, None]).T @ (Z * u[:, None]) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = (Z[lag:] * u[lag:, None]).T @ (Z[:-lag] * u[:-lag, None]) / n
        S += w * (G + G.T)
    V = n * (XtX_inv @ S @ XtX_inv)          # cov(beta_hat)
    se = np.sqrt(np.diag(V))
    t = beta / se
    ss_res = float(u @ u)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"coef": beta, "t": t, "se": se, "r2": r2, "n": n, "lags": lags, "resid": u}


# --------------------------------------------------------------------------- #
# CAPM on the monthly frame
# --------------------------------------------------------------------------- #
def capm(df: pd.DataFrame, start: str | None = None, end: str | None = None,
         lags: int | None = None) -> dict:
    """Excess-vs-excess CAPM of BRK on the market over [start, end].

    alpha is annualised as 12 x the monthly intercept (arithmetic, the FKP convention);
    Sharpe ratios are excess-vs-excess (both legs over the same T-bill).
    """
    d = df.copy()
    if start is not None:
        d = d[d.index >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.index <= pd.Timestamp(end)]
    d = d.dropna(subset=["brk", "mkt", "rf"])
    yb = (d["brk"] - d["rf"]).to_numpy()
    xm = (d["mkt"] - d["rf"]).to_numpy()
    fit = ols_hac(yb, xm, lags=lags)
    a_m, b = fit["coef"][0], fit["coef"][1]
    resid_sig_ann = float(np.std(fit["resid"], ddof=2)) * np.sqrt(12.0)
    out = {
        "n": fit["n"], "lags": fit["lags"],
        "start": str(d.index.min().date()), "end": str(d.index.max().date()),
        "alpha_m": a_m, "alpha_ann_pct": a_m * 12.0 * 100.0,
        "t_alpha": float(fit["t"][0]),
        "beta": float(b), "t_beta": float(fit["t"][1]),
        "r2": float(fit["r2"]),
        "ir": (a_m * 12.0) / resid_sig_ann if resid_sig_ann > 0 else np.nan,
        "brk_exc_ann_pct": float(yb.mean() * 12.0 * 100.0),
        "mkt_exc_ann_pct": float(xm.mean() * 12.0 * 100.0),
        "brk_sharpe": float(yb.mean() / yb.std(ddof=1) * np.sqrt(12.0)),
        "mkt_sharpe": float(xm.mean() / xm.std(ddof=1) * np.sqrt(12.0)),
        "brk_vol_ann_pct": float(yb.std(ddof=1) * np.sqrt(12.0) * 100.0),
    }
    return out


def rolling_alpha(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """10-year rolling CAPM: annualised alpha + HAC t + beta, indexed by window END."""
    d = df.dropna(subset=["brk", "mkt", "rf"])
    yb = (d["brk"] - d["rf"]).to_numpy()
    xm = (d["mkt"] - d["rf"]).to_numpy()
    rows, idx = [], []
    for i in range(window, len(d) + 1):
        fit = ols_hac(yb[i - window:i], xm[i - window:i])
        rows.append({"alpha_ann_pct": fit["coef"][0] * 12.0 * 100.0,
                     "t_alpha": float(fit["t"][0]),
                     "beta": float(fit["coef"][1])})
        idx.append(d.index[i - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def decade_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-decade CAPM rows (1980s..2020s; the 2020s row is a partial decade, labeled)."""
    rows = []
    for dec in (1980, 1990, 2000, 2010, 2020):
        d = df[(df.index.year >= dec) & (df.index.year < dec + 10)]
        if len(d) < 24:
            continue
        c = capm(d)
        rows.append({"decade": f"{dec}s", "months": c["n"],
                     "alpha_ann_pct": c["alpha_ann_pct"], "t_alpha": c["t_alpha"],
                     "beta": c["beta"],
                     "brk_exc_ann_pct": c["brk_exc_ann_pct"],
                     "mkt_exc_ann_pct": c["mkt_exc_ann_pct"]})
    return pd.DataFrame(rows).set_index("decade")


def factor_lite(df: pd.DataFrame, lags: int | None = None) -> dict:
    """FKP attribution with investable proxies (ETF era, honestly labeled).

    On the common QUAL/USMV window: CAPM alpha vs alpha after adding QUAL and USMV
    *excess* returns. If the FKP story holds, the loadings on quality/min-vol should be
    positive and the alpha should shrink toward zero.
    """
    d = df.dropna(subset=["brk", "mkt", "rf", "qual", "usmv"]).copy()
    yb = (d["brk"] - d["rf"]).to_numpy()
    xm = (d["mkt"] - d["rf"]).to_numpy()
    xq = (d["qual"] - d["rf"]).to_numpy()
    xu = (d["usmv"] - d["rf"]).to_numpy()
    capm_fit = ols_hac(yb, xm, lags=lags)
    full_fit = ols_hac(yb, np.column_stack([xm, xq, xu]), lags=lags)
    return {
        "start": str(d.index.min().date()), "end": str(d.index.max().date()),
        "n": len(d),
        "capm_alpha_ann_pct": capm_fit["coef"][0] * 12.0 * 100.0,
        "capm_t_alpha": float(capm_fit["t"][0]),
        "capm_beta": float(capm_fit["coef"][1]),
        "full_alpha_ann_pct": full_fit["coef"][0] * 12.0 * 100.0,
        "full_t_alpha": float(full_fit["t"][0]),
        "b_mkt": float(full_fit["coef"][1]), "t_mkt": float(full_fit["t"][1]),
        "b_qual": float(full_fit["coef"][2]), "t_qual": float(full_fit["t"][2]),
        "b_usmv": float(full_fit["coef"][3]), "t_usmv": float(full_fit["t"][3]),
        "r2_capm": float(capm_fit["r2"]), "r2_full": float(full_fit["r2"]),
    }


def subperiod_fade(df: pd.DataFrame, split: str = "2010-12-31") -> dict:
    """Early-vs-late alpha contrast with a HAC t on the DIFFERENCE.

    Runs one regression of brk excess on [mkt excess, late-dummy, late-dummy x mkt excess]
    plus intercept — the intercept is the early alpha, the dummy coefficient is the alpha
    CHANGE, and its HAC t is the honest test of "has the alpha faded?" (a justified split:
    FKP's sample ends in 2011; 'the last 15 years' is the out-of-sample continuation).
    """
    d = df.dropna(subset=["brk", "mkt", "rf"]).copy()
    yb = (d["brk"] - d["rf"]).to_numpy()
    xm = (d["mkt"] - d["rf"]).to_numpy()
    late = np.asarray(d.index > pd.Timestamp(split), dtype=float)
    X = np.column_stack([xm, late, late * xm])
    fit = ols_hac(yb, X)
    return {
        "alpha_early_ann_pct": fit["coef"][0] * 12.0 * 100.0,
        "t_alpha_early": float(fit["t"][0]),
        "d_alpha_ann_pct": fit["coef"][2] * 12.0 * 100.0,
        "t_d_alpha": float(fit["t"][2]),
        "d_beta": float(fit["coef"][3]), "t_d_beta": float(fit["t"][3]),
        "n": fit["n"], "split": split,
    }
