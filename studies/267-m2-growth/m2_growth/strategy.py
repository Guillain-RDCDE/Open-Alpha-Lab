"""Strategy and analysis layer for Study 267 (M2-Growth).

Does money-supply (M2) growth predict the stock market? We test the monetarist
claim three ways, all look-ahead-free (the M2 reading dated month t predicts the
return earned in month t+1 or over the next 12 months):

  - **Predictive regression** of forward ^GSPC returns on lagged M2-YoY growth,
    with a **Newey-West (HAC) t-stat**. Overlapping forward windows induce
    serial correlation, so the naive OLS t-stat is badly inflated — the HAC
    correction is the honest bar. ``REAL`` requires |t_HAC| >= 2.

  - **Quantile sort**: bucket months by lagged M2 growth and compare the mean
    next-month return of the high-money vs low-money buckets.

  - **A tradable rule** with costs: go long ^GSPC when lagged M2 YoY growth is
    above its trailing median, else hold cash. We charge a one-way cost on every
    flip and compare net to passive buy-and-hold.

The killer subtlety: M2 growth is extremely persistent and *contemporaneously*
moves with the cycle (QE in recessions => high M2 growth + recovering stocks),
which makes naive overlapping correlations look strong. With the execution lag
imposed and a HAC t-stat, the forward-predictive content is thin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 267


# ---------------------------------------------------------------------------
# Newey-West (HAC) OLS for a single regressor + intercept
# ---------------------------------------------------------------------------
def nw_regression(y: np.ndarray, x: np.ndarray, lags: int = 12) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey-West HAC standard errors.

    Returns a dict with ``beta`` (slope), ``alpha`` (intercept), ``t_hac``
    (HAC t-stat on the slope), ``t_ols`` (naive OLS t-stat on the slope),
    ``r2``, and ``n``. ``lags`` is the Bartlett-kernel truncation lag (use the
    forward-window length for overlapping returns).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 5:
        return {"beta": np.nan, "alpha": np.nan, "t_hac": np.nan,
                "t_ols": np.nan, "r2": np.nan, "n": int(n)}

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef

    # Naive OLS SE
    sigma2 = (resid @ resid) / (n - 2)
    cov_ols = sigma2 * XtX_inv
    se_ols = np.sqrt(np.diag(cov_ols))

    # Newey-West HAC sandwich
    S = (X * resid[:, None]).T @ (X * resid[:, None])  # lag 0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = X * resid[:, None]
        Gamma = u[L:].T @ u[:-L]
        S += w * (Gamma + Gamma.T)
    cov_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.diag(cov_hac))

    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "beta": float(coef[1]),
        "alpha": float(coef[0]),
        "t_hac": float(coef[1] / se_hac[1]) if se_hac[1] > 0 else np.nan,
        "t_ols": float(coef[1] / se_ols[1]) if se_ols[1] > 0 else np.nan,
        "r2": float(r2),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Predictive regressions on the panel
# ---------------------------------------------------------------------------
def predictive_regression(
    df: pd.DataFrame,
    m2_col: str = "m2_yoy",
    target_col: str = "sp_fwd1",
    lags: int = 1,
) -> dict:
    """Regress a forward return on the *lagged* M2 reading with a HAC t-stat.

    The M2 reading dated month t (``m2_col``) predicts the forward return earned
    after t (``target_col``). For overlapping multi-month targets, pass
    ``lags`` >= the window length so the Newey-West correction covers the induced
    autocorrelation.
    """
    sub = df[[m2_col, target_col]].dropna()
    return nw_regression(sub[target_col].to_numpy(), sub[m2_col].to_numpy(), lags=lags)


# ---------------------------------------------------------------------------
# Quantile sort: high-money vs low-money months
# ---------------------------------------------------------------------------
def quantile_sort(
    df: pd.DataFrame,
    m2_col: str = "m2_yoy",
    target_col: str = "sp_fwd1",
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Bucket months by lagged M2 growth; report mean forward return per bucket.

    Returns a DataFrame indexed by bucket (1 = lowest M2 growth, ``n_buckets`` =
    highest) with columns ``mean_fwd_ret`` (percent), ``n``, ``mean_m2`` (percent).
    """
    sub = df[[m2_col, target_col]].dropna().copy()
    sub["bucket"] = pd.qcut(sub[m2_col], n_buckets, labels=False, duplicates="drop") + 1
    g = sub.groupby("bucket")
    out = pd.DataFrame({
        "mean_fwd_ret": g[target_col].mean() * 100.0,
        "n": g[target_col].size(),
        "mean_m2": g[m2_col].mean(),
    })
    return out


def high_minus_low(
    df: pd.DataFrame,
    m2_col: str = "m2_yoy",
    target_col: str = "sp_fwd1",
    n_buckets: int = 5,
) -> dict:
    """High-money minus low-money mean next-month return, with a t-stat.

    Computes the difference in mean forward returns between the top and bottom
    M2-growth buckets and a two-sample Welch t-stat. Returns a dict with
    ``hi_mean``, ``lo_mean`` (percent), ``spread`` (pp), ``t_stat``, ``n_hi``,
    ``n_lo``.
    """
    from scipy import stats as scipy_stats

    sub = df[[m2_col, target_col]].dropna().copy()
    sub["bucket"] = pd.qcut(sub[m2_col], n_buckets, labels=False, duplicates="drop") + 1
    top = int(sub["bucket"].max())
    bot = int(sub["bucket"].min())
    hi = sub.loc[sub["bucket"] == top, target_col].to_numpy()
    lo = sub.loc[sub["bucket"] == bot, target_col].to_numpy()
    t, _ = scipy_stats.ttest_ind(hi, lo, equal_var=False)
    return {
        "hi_mean": float(hi.mean() * 100.0),
        "lo_mean": float(lo.mean() * 100.0),
        "spread": float((hi.mean() - lo.mean()) * 100.0),
        "t_stat": float(t),
        "n_hi": int(len(hi)),
        "n_lo": int(len(lo)),
    }


# ---------------------------------------------------------------------------
# Tradable rule: long when lagged M2 growth > trailing median, else cash
# ---------------------------------------------------------------------------
def backtest_rule(
    df: pd.DataFrame,
    m2_col: str = "m2_yoy",
    ret_col: str = "sp_ret",
    lookback: int = 60,
    cost_bps: float = 5.0,
) -> dict:
    """Long ^GSPC when lagged M2 YoY > trailing median, else flat (cash at 0%).

    One-month execution lag: the position for month t uses the M2 reading from
    month t-1, compared to its trailing ``lookback``-month median (also through
    t-1). A one-way ``cost_bps`` is charged on NAV at every position flip
    (entering and exiting both cost; this rule is long/flat so there is no
    borrow). Returns a dict with annualized gross/net strategy return, buy-and-
    hold return, Sharpe ratios, turnover (flips/yr), and the equity curves.
    """
    sub = df[[m2_col, ret_col]].dropna().copy()
    m2_lag = sub[m2_col].shift(1)
    trailing_med = m2_lag.rolling(lookback, min_periods=12).median()
    pos = (m2_lag > trailing_med).astype(float)
    pos = pos.fillna(0.0)

    rets = sub[ret_col].to_numpy()
    pos_arr = pos.to_numpy()

    # Costs on flips
    flips = np.abs(np.diff(np.concatenate([[0.0], pos_arr])))
    cost = flips * (cost_bps / 1e4)

    gross = pos_arr * rets
    net = gross - cost

    n = len(rets)
    months_per_year = 12.0

    def _ann(series):
        gr = float(np.prod(1.0 + series) ** (months_per_year / n) - 1.0)
        return gr

    def _sharpe(series):
        mu = np.mean(series) * months_per_year
        sd = np.std(series, ddof=1) * np.sqrt(months_per_year)
        return float(mu / sd) if sd > 0 else float("nan")

    bah = rets
    n_flips = float(flips.sum())
    turnover_per_yr = n_flips / (n / months_per_year)

    eq_net = np.cumprod(1.0 + net)
    eq_bah = np.cumprod(1.0 + bah)

    return {
        "ann_gross": _ann(gross),
        "ann_net": _ann(net),
        "ann_bah": _ann(bah),
        "sharpe_net": _sharpe(net),
        "sharpe_bah": _sharpe(bah),
        "time_in_market": float(pos_arr.mean()),
        "n_flips": int(n_flips),
        "turnover_per_yr": float(turnover_per_yr),
        "n_months": int(n),
        "eq_net": eq_net,
        "eq_bah": eq_bah,
        "dates": sub.index,
    }


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, lookback: int = 60, cost_bps: float = 5.0) -> dict:
    """One-stop summary dict for the merged (m2_yoy + ^GSPC) DataFrame.

    Mirrors the frozen ``R`` dict in build_notebooks.py and docs/results.md.
    """
    reg1 = predictive_regression(df, target_col="sp_fwd1", lags=3)
    reg12 = predictive_regression(df, target_col="sp_fwd12", lags=18)
    hml = high_minus_low(df, target_col="sp_fwd1")
    bt = backtest_rule(df, lookback=lookback, cost_bps=cost_bps)

    return {
        "n_months": int(df[["m2_yoy", "sp_fwd1"]].dropna().shape[0]),
        # 1-month forward predictive regression
        "beta_fwd1": round(reg1["beta"], 5),
        "t_hac_fwd1": round(reg1["t_hac"], 3),
        "t_ols_fwd1": round(reg1["t_ols"], 3),
        "r2_fwd1": round(reg1["r2"], 5),
        # 12-month forward predictive regression
        "beta_fwd12": round(reg12["beta"], 5),
        "t_hac_fwd12": round(reg12["t_hac"], 3),
        "t_ols_fwd12": round(reg12["t_ols"], 3),
        "r2_fwd12": round(reg12["r2"], 4),
        # Quantile sort
        "hi_mean": round(hml["hi_mean"], 3),
        "lo_mean": round(hml["lo_mean"], 3),
        "spread": round(hml["spread"], 3),
        "spread_t": round(hml["t_stat"], 3),
        # Tradable rule
        "ann_gross": round(bt["ann_gross"] * 100, 2),
        "ann_net": round(bt["ann_net"] * 100, 2),
        "ann_bah": round(bt["ann_bah"] * 100, 2),
        "sharpe_net": round(bt["sharpe_net"], 3),
        "sharpe_bah": round(bt["sharpe_bah"], 3),
        "time_in_market": round(bt["time_in_market"], 3),
        "turnover_per_yr": round(bt["turnover_per_yr"], 2),
    }
