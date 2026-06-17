"""Strategy and analysis layer for Study 278 (Sunshine-Effect).

The Hirshleifer-Shumway (2003) claim: more cloud cover over the exchange city
predicts a *lower* same-day index return (sunshine lifts mood, mood lifts
prices). We test it three honest ways on the de-seasonalised cloud anomaly:

  1. **Mean-difference (sunny vs cloudy days).** Split trading days at the median
     de-seasonalised cloud anomaly; compare mean returns; report a HAC
     (Newey-West) t-stat on the difference, since daily returns are mildly
     autocorrelated and heteroskedastic.
  2. **OLS regression** of daily return on cloud anomaly with a Newey-West HAC
     standard error on the slope. This is the coefficient Hirshleifer-Shumway
     report; a real effect has a *negative* slope with |t| >= 2.
  3. **A tradable sleeve.** Go long the index on below-median-cloud (sunny)
     mornings, flat otherwise; charge one-way costs on the implied daily
     turnover (the position flips most days). Net Sharpe vs buy-and-hold tells
     you whether the (small) gross edge survives.

A HAC |t| >= 2 on the REAL tape is the bar for a Real signal stamp. Literature
support alone is, by the desk rubric, only Weak. We compute the real-tape t
honestly and let it decide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# HAC (Newey-West) helpers
# ---------------------------------------------------------------------------
def _nw_lags(n: int) -> int:
    """Newey-West rule-of-thumb lag count: floor(4*(n/100)^(2/9))."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def hac_mean_tstat(x: np.ndarray, lags: int | None = None) -> dict:
    """Newey-West HAC t-stat for the sample mean of ``x`` (a return series).

    Returns ``{'mean', 'se', 'tstat', 'lags', 'n'}``. Mirrors
    quantlab.analytics.mean_tstat_hac but kept local so the study is
    self-contained and the tests need no quantlab import.
    """
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return {"mean": float("nan"), "se": float("nan"),
                "tstat": float("nan"), "lags": 0, "n": n}
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = _nw_lags(n)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(e[k:] @ e[:-k]) / n
        lrv += 2.0 * w * gamma_k
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean": float(mu),
        "se": float(se),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "lags": int(lags),
        "n": int(n),
    }


def hac_ols_slope(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of ``y`` on ``x`` (with intercept), Newey-West HAC SE on the slope.

    Returns ``{'slope', 'intercept', 'se_slope', 'tstat', 'lags', 'n'}``. The
    slope is in return units per unit of ``x``; multiply by 1e4 for bps/octa.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = y.size
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "se_slope": float("nan"), "tstat": float("nan"),
                "lags": 0, "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    if lags is None:
        lags = _nw_lags(n)
    # Newey-West meat: S = sum_k w_k (Gamma_k + Gamma_k')
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Xk = X[k:] * resid[k:, None]
        Xk0 = X[:-k] * resid[:-k, None]
        G = Xk.T @ Xk0
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_slope = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "se_slope": se_slope,
        "tstat": float(beta[1] / se_slope) if se_slope > 0 else float("nan"),
        "lags": int(lags),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Sunny vs cloudy split
# ---------------------------------------------------------------------------
def sunny_cloudy_split(df: pd.DataFrame, anom_col: str = "cloud_anom",
                       ret_col: str = "ret") -> dict:
    """Split days at the median cloud anomaly and compare mean returns.

    'Sunny' days have a below-median (more negative) de-seasonalised cloud
    anomaly; 'cloudy' days are at-or-above the median. Returns a dict with the
    two group means (in bps), the difference (sunny - cloudy, in bps), and a HAC
    t-stat on the difference computed via a signed contrast series so the
    autocorrelation correction is honest.
    """
    anom = df[anom_col].to_numpy(dtype=float)
    ret = df[ret_col].to_numpy(dtype=float)
    med = np.median(anom)
    sunny = anom < med
    cloudy = ~sunny

    mean_sunny = float(ret[sunny].mean())
    mean_cloudy = float(ret[cloudy].mean())

    # Two-sample HAC difference: regress ret on a sunny dummy. The slope equals
    # (mean_sunny - mean_cloudy) and its Newey-West SE accounts for daily
    # autocorrelation, so the t-stat is honest.
    dummy = sunny.astype(float)
    reg = hac_ols_slope(ret, dummy)
    # slope = mean_sunny - mean_cloudy
    diff = float(reg["slope"])
    return {
        "n": int(len(df)),
        "n_sunny": int(sunny.sum()),
        "n_cloudy": int(cloudy.sum()),
        "mean_sunny_bps": round(mean_sunny * 1e4, 3),
        "mean_cloudy_bps": round(mean_cloudy * 1e4, 3),
        "diff_bps": round(diff * 1e4, 3),
        "diff_tstat": round(float(reg["tstat"]), 3),
        "hac_lags": int(reg["lags"]),
    }


# ---------------------------------------------------------------------------
# Full inference summary
# ---------------------------------------------------------------------------
def cloud_return_stats(df: pd.DataFrame, anom_col: str = "cloud_anom",
                       ret_col: str = "ret") -> dict:
    """Headline inference: OLS slope (bps/octa) with HAC t, plus sunny-cloudy split.

    The slope sign convention: returns regressed on the cloud anomaly. A genuine
    Hirshleifer-Shumway effect gives a *negative* slope (more cloud -> lower
    return). We report the slope in bps/octa and its HAC t-stat; the headline
    significance number for the Signal axis is ``slope_tstat``.
    """
    y = df[ret_col].to_numpy(dtype=float)
    x = df[anom_col].to_numpy(dtype=float)
    reg = hac_ols_slope(y, x)
    split = sunny_cloudy_split(df, anom_col=anom_col, ret_col=ret_col)
    return {
        "n": int(len(df)),
        "slope_bps_per_octa": round(reg["slope"] * 1e4, 3),
        "slope_se_bps": round(reg["se_slope"] * 1e4, 3),
        "slope_tstat": round(float(reg["tstat"]), 3),
        "hac_lags": int(reg["lags"]),
        "mean_sunny_bps": split["mean_sunny_bps"],
        "mean_cloudy_bps": split["mean_cloudy_bps"],
        "diff_bps": split["diff_bps"],
        "diff_tstat": split["diff_tstat"],
        "n_sunny": split["n_sunny"],
        "n_cloudy": split["n_cloudy"],
    }


# ---------------------------------------------------------------------------
# Tradable sleeve: long-on-sunny, with costs
# ---------------------------------------------------------------------------
def sunshine_sleeve(
    df: pd.DataFrame,
    anom_col: str = "cloud_anom",
    ret_col: str = "ret",
    cost_bps: float = 1.0,
    periods_per_year: int = 252,
) -> dict:
    """Long the index on sunny mornings (below-median cloud), flat otherwise.

    The position is set at the open on the morning sky (execution lag: signal at
    open, return earned open-to-close, here proxied by the same-day close-to-close
    return). Turnover is charged at ``cost_bps`` one-way per change in exposure
    (x NAV). We report gross and net annualised return, net Sharpe, and the
    buy-and-hold benchmark over the same days.

    Returns a dict of headline tradability numbers.
    """
    anom = df[anom_col].to_numpy(dtype=float)
    ret = df[ret_col].to_numpy(dtype=float)
    med = np.median(anom)
    pos = (anom < med).astype(float)  # 1 = long on sunny days, 0 = flat

    # One-way cost on the change in exposure each day (position flips often).
    turn = np.abs(np.diff(pos, prepend=0.0))
    cost = turn * (cost_bps * 1e-4)

    gross = pos * ret
    net = gross - cost

    n = len(ret)
    ann = periods_per_year
    gross_ann = float(gross.mean() * ann)
    net_ann = float(net.mean() * ann)
    bah_ann = float(ret.mean() * ann)
    net_sd = float(net.std(ddof=1))
    net_sharpe = float((net.mean() / net_sd) * np.sqrt(ann)) if net_sd > 0 else float("nan")
    bah_sd = float(ret.std(ddof=1))
    bah_sharpe = float((ret.mean() / bah_sd) * np.sqrt(ann)) if bah_sd > 0 else float("nan")

    return {
        "n": int(n),
        "avg_exposure": round(float(pos.mean()), 3),
        "avg_daily_turnover": round(float(turn.mean()), 3),
        "gross_ann_pct": round(gross_ann * 100, 2),
        "net_ann_pct": round(net_ann * 100, 2),
        "bah_ann_pct": round(bah_ann * 100, 2),
        "net_sharpe": round(net_sharpe, 3),
        "bah_sharpe": round(bah_sharpe, 3),
        "cost_bps": cost_bps,
    }


# ---------------------------------------------------------------------------
# Compact summary for results.md / notebooks
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, cost_bps: float = 1.0) -> dict:
    """One-stop summary dict mirroring docs/results.md and the notebooks' R dict."""
    cs = cloud_return_stats(df)
    sl = sunshine_sleeve(df, cost_bps=cost_bps)
    return {
        "n": cs["n"],
        "slope_bps_per_octa": cs["slope_bps_per_octa"],
        "slope_tstat": cs["slope_tstat"],
        "hac_lags": cs["hac_lags"],
        "mean_sunny_bps": cs["mean_sunny_bps"],
        "mean_cloudy_bps": cs["mean_cloudy_bps"],
        "diff_bps": cs["diff_bps"],
        "diff_tstat": cs["diff_tstat"],
        "gross_ann_pct": sl["gross_ann_pct"],
        "net_ann_pct": sl["net_ann_pct"],
        "bah_ann_pct": sl["bah_ann_pct"],
        "net_sharpe": sl["net_sharpe"],
        "bah_sharpe": sl["bah_sharpe"],
        "avg_daily_turnover": sl["avg_daily_turnover"],
    }
