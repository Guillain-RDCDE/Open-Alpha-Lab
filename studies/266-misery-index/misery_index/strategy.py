"""Strategy and analysis layer for Study 266 (Misery-Index).

The misery index (Okun): ``misery = CPI_YoY + unemployment``.  We test whether
the December misery level (or its year-on-year change) predicts the *next*
calendar-year S&P 500 return.  Two competing stories:

  - **Leading-bear**: high misery -> low forward returns (negative beta).
  - **Contrarian/maximum-pessimism**: high misery -> high forward returns
    (positive beta), because misery peaks at the market bottom.

The honest tests:

  - **HAC regression.** Regress the forward annual return on the standardized
    misery level (and, separately, on the misery change).  The slope's
    significance is judged by a Newey-West (HAC) t-stat, since overlapping
    macro states induce serial correlation.  REAL needs |t_HAC| >= 2.
  - **Tercile sort.** Split years into low/mid/high-misery terciles; compare the
    mean forward return in the top vs bottom tercile (a contrarian long/short).
  - **Hit-rate vs the base rate.** The S&P rises in ~70%+ of years
    unconditionally; any "buy when X" rule inherits that base rate for free.
  - **Positive control.** A synthetic tape with a planted misery->return beta to
    confirm the machinery detects a real effect when one exists.

The killer here is the small, serially-correlated sample (~76 annual
observations, with multi-year inflation regimes) and the fact that the macro
print itself is published with a lag in practice.  We pin every claim to a HAC
t-stat, not an OLS one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# HAC (Newey-West) OLS for a single predictor
# ---------------------------------------------------------------------------
def hac_regression(
    y: np.ndarray,
    x: np.ndarray,
    n_lags: int = 3,
) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey-West (HAC) standard errors.

    Returns a dict with ``alpha``, ``beta`` (slope on x), ``t_ols``, ``t_hac``,
    ``r2``, and ``n``.  ``n_lags`` is the Newey-West bandwidth (Bartlett kernel).
    Pure numpy -- no statsmodels dependency.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_hat
    k = X.shape[1]

    # Classical OLS covariance
    sigma2 = (resid @ resid) / (n - k)
    cov_ols = sigma2 * XtX_inv

    # Newey-West HAC covariance (Bartlett kernel)
    S = (X * resid[:, None]).T @ (X * resid[:, None])  # lag 0
    for L in range(1, n_lags + 1):
        w = 1.0 - L / (n_lags + 1.0)
        Xr = X * resid[:, None]
        gamma = Xr[L:].T @ Xr[:-L]
        S += w * (gamma + gamma.T)
    cov_hac = XtX_inv @ S @ XtX_inv

    se_ols = np.sqrt(np.diag(cov_ols))
    se_hac = np.sqrt(np.diag(cov_hac))

    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha": float(beta_hat[0]),
        "beta": float(beta_hat[1]),
        "t_ols": float(beta_hat[1] / se_ols[1]) if se_ols[1] > 0 else float("nan"),
        "t_hac": float(beta_hat[1] / se_hac[1]) if se_hac[1] > 0 else float("nan"),
        "r2": float(r2),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Predictive regressions on the misery level / change
# ---------------------------------------------------------------------------
def predictive_stats(
    df: pd.DataFrame,
    ret_col: str = "fwd_return",
    n_lags: int = 3,
) -> dict:
    """Regress the forward return on standardized misery level and misery change.

    The slope is reported per 1-SD of the predictor (the predictors are
    z-scored).  ``t_hac`` is the headline statistic for the verdict.

    Returns a dict with ``level_*`` and ``chg_*`` blocks plus the sample size.
    """
    y = df[ret_col].to_numpy(dtype=float)

    mis = df["misery"].to_numpy(dtype=float)
    mis_z = (mis - mis.mean()) / mis.std(ddof=0)
    lev = hac_regression(y, mis_z, n_lags=n_lags)

    if "misery_chg" in df.columns:
        chg = df["misery_chg"].to_numpy(dtype=float)
        m2 = np.isfinite(chg)
        chg_z = np.full_like(chg, np.nan)
        chg_z[m2] = (chg[m2] - chg[m2].mean()) / chg[m2].std(ddof=0)
        chgreg = hac_regression(y, chg_z, n_lags=n_lags)
    else:
        chgreg = {"alpha": float("nan"), "beta": float("nan"),
                  "t_ols": float("nan"), "t_hac": float("nan"),
                  "r2": float("nan"), "n": 0}

    return {
        "n": int(len(y)),
        "level_beta": round(lev["beta"], 4),
        "level_t_ols": round(lev["t_ols"], 3),
        "level_t_hac": round(lev["t_hac"], 3),
        "level_r2": round(lev["r2"], 4),
        "chg_beta": round(chgreg["beta"], 4),
        "chg_t_ols": round(chgreg["t_ols"], 3),
        "chg_t_hac": round(chgreg["t_hac"], 3),
        "chg_r2": round(chgreg["r2"], 4),
    }


# ---------------------------------------------------------------------------
# Tercile sort -- contrarian long/short on misery level
# ---------------------------------------------------------------------------
def tercile_sort(
    df: pd.DataFrame,
    ret_col: str = "fwd_return",
) -> dict:
    """Split years into low/mid/high-misery terciles; compare forward returns.

    The contrarian spread is (high-misery mean forward return) minus
    (low-misery mean forward return).  A Welch t-test compares the two extreme
    terciles.  Also reports the unconditional up-rate and the up-rate in each
    extreme tercile (the base-rate trap).
    """
    d = df.dropna(subset=["misery", ret_col]).copy()
    n = len(d)
    q1, q2 = d["misery"].quantile([1 / 3, 2 / 3])
    low = d[d["misery"] <= q1]
    high = d[d["misery"] >= q2]
    mid = d[(d["misery"] > q1) & (d["misery"] < q2)]

    low_ret = low[ret_col].to_numpy(dtype=float)
    high_ret = high[ret_col].to_numpy(dtype=float)

    spread = float(high_ret.mean() - low_ret.mean())
    if len(low_ret) >= 2 and len(high_ret) >= 2:
        wt, wp = scipy_stats.ttest_ind(high_ret, low_ret, equal_var=False)
    else:
        wt, wp = float("nan"), float("nan")

    uncond_up = float((d[ret_col] > 0).mean())
    low_up = float((low[ret_col] > 0).mean()) if len(low) else float("nan")
    high_up = float((high[ret_col] > 0).mean()) if len(high) else float("nan")

    return {
        "n": int(n),
        "n_low": int(len(low)),
        "n_mid": int(len(mid)),
        "n_high": int(len(high)),
        "low_mean": round(float(low_ret.mean()), 4),
        "mid_mean": round(float(mid[ret_col].mean()), 4) if len(mid) else float("nan"),
        "high_mean": round(float(high_ret.mean()), 4),
        "spread_high_minus_low": round(spread, 4),
        "welch_t": round(float(wt), 3),
        "welch_p": round(float(wp), 4),
        "uncond_up_rate": round(uncond_up, 4),
        "low_up_rate": round(low_up, 4),
        "high_up_rate": round(high_up, 4),
    }


# ---------------------------------------------------------------------------
# Contrarian timing strategy -- long S&P only when misery is high
# ---------------------------------------------------------------------------
def contrarian_backtest(
    df: pd.DataFrame,
    ret_col: str = "fwd_return",
    cost_bps: float = 10.0,
) -> dict:
    """Long S&P in the year after a top-tercile (high-misery) print, else flat.

    Compares the timing rule (gross and net of a one-way ``cost_bps`` per
    rebalance applied to NAV on each in/out switch) to passive buy-and-hold.
    Returns annualized CAGRs and the realized turnover.
    """
    d = df.dropna(subset=["misery", ret_col]).copy().reset_index(drop=True)
    q2 = d["misery"].quantile(2 / 3)
    in_mkt = (d["misery"] >= q2).astype(int).to_numpy()
    rets = d[ret_col].to_numpy(dtype=float)

    # Turnover: each switch (in<->out) trades 100% of NAV once.
    switches = int(np.abs(np.diff(np.concatenate([[0], in_mkt, [0]]))).sum())
    n_yrs = len(d)
    cost = cost_bps / 1e4

    gross = in_mkt * rets
    # apply cost on each year we change state (open or close the position)
    state = np.concatenate([[0], in_mkt])
    trade = np.abs(np.diff(state))  # 1 when we open/close
    net = gross - trade * cost

    bah_cagr = float((1 + rets).prod() ** (1 / n_yrs) - 1)
    gross_cagr = float((1 + gross).prod() ** (1 / n_yrs) - 1)
    net_cagr = float((1 + net).prod() ** (1 / n_yrs) - 1)
    frac_in = float(in_mkt.mean())

    return {
        "n_years": int(n_yrs),
        "frac_in_market": round(frac_in, 4),
        "n_switches": switches,
        "turnover_per_yr": round(switches / n_yrs, 4),
        "bah_cagr": round(bah_cagr, 4),
        "gross_cagr": round(gross_cagr, 4),
        "net_cagr": round(net_cagr, 4),
        "net_minus_bah": round(net_cagr - bah_cagr, 4),
    }


# ---------------------------------------------------------------------------
# Summarize -- compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_lags: int = 3, cost_bps: float = 10.0) -> dict:
    """One-stop summary dict for the merged misery/return DataFrame.

    Mirrors the R = dict(...) in build_notebooks.py and docs/results.md.
    """
    pred = predictive_stats(df, n_lags=n_lags)
    terc = tercile_sort(df)
    bt = contrarian_backtest(df, cost_bps=cost_bps)

    return {
        "n": pred["n"],
        # predictive regression (per 1-SD)
        "level_beta_pp": round(pred["level_beta"] * 100, 2),
        "level_t_ols": pred["level_t_ols"],
        "level_t_hac": pred["level_t_hac"],
        "level_r2": pred["level_r2"],
        "chg_beta_pp": round(pred["chg_beta"] * 100, 2),
        "chg_t_ols": pred["chg_t_ols"],
        "chg_t_hac": pred["chg_t_hac"],
        "chg_r2": pred["chg_r2"],
        # tercile sort
        "low_mean_pp": round(terc["low_mean"] * 100, 1),
        "mid_mean_pp": round(terc["mid_mean"] * 100, 1),
        "high_mean_pp": round(terc["high_mean"] * 100, 1),
        "spread_pp": round(terc["spread_high_minus_low"] * 100, 1),
        "terc_welch_t": terc["welch_t"],
        "terc_welch_p": terc["welch_p"],
        "uncond_up_pct": round(terc["uncond_up_rate"] * 100, 1),
        "low_up_pct": round(terc["low_up_rate"] * 100, 1),
        "high_up_pct": round(terc["high_up_rate"] * 100, 1),
        # backtest
        "frac_in_market": bt["frac_in_market"],
        "turnover_per_yr": bt["turnover_per_yr"],
        "bah_cagr_pct": round(bt["bah_cagr"] * 100, 1),
        "gross_cagr_pct": round(bt["gross_cagr"] * 100, 1),
        "net_cagr_pct": round(bt["net_cagr"] * 100, 1),
        "net_minus_bah_pp": round(bt["net_minus_bah"] * 100, 1),
    }
