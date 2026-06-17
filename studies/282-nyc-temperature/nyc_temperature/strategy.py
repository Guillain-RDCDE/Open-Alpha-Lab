"""Strategy and analysis layer for Study 282 (NYC-Temperature).

The weather-mood claim (Saunders 1993; Hirshleifer & Shumway 2003): the local
temperature at the exchange tilts that day's equity returns through trader mood.
We operationalize "the temperature on Wall Street" as the NYC daily
temperature *anomaly* (today's temperature minus its seasonal normal), and pin
it against same-day ^GSPC daily returns with:

  - A **cold vs warm day sort**: split days into the coldest tercile (anomaly in
    the bottom third) and the warmest tercile, and compare mean daily returns.
  - A **Newey-West HAC t-stat** on the cold-minus-warm daily return spread (the
    bar for a REAL signal is |t| >= 2 on the real tape).
  - An **OLS slope** of daily return on the standardized anomaly (the
    "sensitivity" coefficient), with its HAC t-stat.
  - A **permutation test**: shuffle the temperature labels and see where the
    real cold-minus-warm spread lands in the null distribution.
  - A **tradable overlay**: lag the signal one full day (yesterday's anomaly ->
    today's position), apply one-way costs x NAV, and compare gross vs net.

The honest baseline is that daily equity returns are ~zero-mean with ~1%/day
vol; any cold-minus-warm spread of a basis point or two is dwarfed by sampling
noise even over decades of daily data, because temperature anomalies are highly
autocorrelated (a cold *spell* is one effective observation, not twenty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Newey-West HAC t-stat on a daily series mean
# ---------------------------------------------------------------------------
def nw_tstat(series, lags: int | None = None) -> dict:
    """Mean, vol, and Newey-West HAC t-stat for a daily series' mean.

    Returns a dict with ``mean``, ``vol``, ``tstat``, ``se``, ``n``, ``lags``.
    The lag length defaults to the Newey-West rule
    ``floor(4 * (n/100)^(2/9))`` -- appropriate for autocorrelated daily data.
    """
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {"mean": np.nan, "vol": np.nan, "tstat": np.nan,
                "se": np.nan, "n": n, "lags": 0}
    mu = float(r.mean())
    std = float(r.std(ddof=1))
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    e = r.to_numpy() - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")
    return {"mean": mu, "vol": std, "tstat": tstat, "se": float(se),
            "n": n, "lags": int(lags)}


# ---------------------------------------------------------------------------
# Cold vs warm day sort
# ---------------------------------------------------------------------------
def cold_warm_sort(
    df: pd.DataFrame,
    anomaly_col: str = "temp_anomaly_f",
    ret_col: str = "ret",
    quantile: float = 1.0 / 3.0,
) -> dict:
    """Split days by temperature anomaly into cold / warm terciles and compare.

    "Cold" = anomaly in the bottom ``quantile`` of the sample; "warm" = top
    ``quantile``. Returns mean daily returns (bps) for each bucket, the
    cold-minus-warm spread, and the HAC t-stat of the *paired* spread series
    (cold-day returns demeaned vs warm-day returns demeaned is not paired by day,
    so we test the spread via the difference in HAC-robust means).

    Returns a dict:
      ``n``, ``n_cold``, ``n_warm``, ``mean_cold_bps``, ``mean_warm_bps``,
      ``spread_bps``, ``tstat`` (HAC t of cold-minus-warm), ``cold_thresh``,
      ``warm_thresh``.
    """
    d = df[[anomaly_col, ret_col]].dropna()
    anom = d[anomaly_col].to_numpy()
    ret = d[ret_col].to_numpy()
    lo = np.quantile(anom, quantile)
    hi = np.quantile(anom, 1.0 - quantile)

    cold_mask = anom <= lo
    warm_mask = anom >= hi
    cold_ret = ret[cold_mask]
    warm_ret = ret[warm_mask]

    mean_cold = float(cold_ret.mean()) if cold_ret.size else float("nan")
    mean_warm = float(warm_ret.mean()) if warm_ret.size else float("nan")
    spread = mean_cold - mean_warm

    # HAC t-stat for the difference in means: build a sign-coded daily series
    # over the union of cold/warm days (cold = +ret, warm = -ret) centered so
    # its mean equals the spread, then HAC the difference directly via a
    # two-sample HAC-robust SE (independent buckets, autocorr within each).
    nc = nw_tstat(cold_ret)
    nw = nw_tstat(warm_ret)
    se_diff = np.sqrt(nc["se"] ** 2 + nw["se"] ** 2) if (
        np.isfinite(nc["se"]) and np.isfinite(nw["se"])) else float("nan")
    tstat = float(spread / se_diff) if (se_diff and se_diff > 0) else float("nan")

    return {
        "n": int(d.shape[0]),
        "n_cold": int(cold_mask.sum()),
        "n_warm": int(warm_mask.sum()),
        "mean_cold_bps": round(mean_cold * 1e4, 3),
        "mean_warm_bps": round(mean_warm * 1e4, 3),
        "spread_bps": round(spread * 1e4, 3),
        "tstat": round(tstat, 3),
        "cold_thresh": round(float(lo), 3),
        "warm_thresh": round(float(hi), 3),
    }


# ---------------------------------------------------------------------------
# OLS slope of return on standardized anomaly
# ---------------------------------------------------------------------------
def ols_slope(
    df: pd.DataFrame,
    anomaly_col: str = "temp_anomaly_f",
    ret_col: str = "ret",
) -> dict:
    """OLS of daily return (bps) on the standardized temperature anomaly.

    Returns ``slope_bps`` (bps of daily return per +1 std anomaly), the HAC
    t-stat of the slope, ``r_squared``, and ``n``. A positive slope means warmer
    days earn more (the "sunshine" sign); the cold-day folklore predicts a
    negative slope.
    """
    d = df[[anomaly_col, ret_col]].dropna()
    x = d[anomaly_col].to_numpy()
    y = d[ret_col].to_numpy() * 1e4  # bps
    z = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    n = z.size
    X = np.column_stack([np.ones(n), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # HAC (Newey-West) covariance of OLS beta.
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Xe = X * resid[:, None]
        G = Xe[k:].T @ Xe[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_slope = float(np.sqrt(max(cov[1, 1], 0.0)))
    slope = float(beta[1])
    tstat = slope / se_slope if se_slope > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope_bps": round(slope, 4),
        "tstat": round(float(tstat), 3),
        "r_squared": round(float(r2), 6),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Permutation test on the cold-minus-warm spread
# ---------------------------------------------------------------------------
def permutation_spread(
    df: pd.DataFrame,
    anomaly_col: str = "temp_anomaly_f",
    ret_col: str = "ret",
    quantile: float = 1.0 / 3.0,
    n_permutations: int = 2000,
    seed: int = 282,
) -> dict:
    """Shuffle the temperature labels and recompute the cold-minus-warm spread.

    Returns ``obs_spread_bps``, ``perm_p`` (two-sided fraction of |perm spread|
    >= |observed|), and the full ``perm_spreads`` array (bps) for plotting.
    """
    d = df[[anomaly_col, ret_col]].dropna()
    anom = d[anomaly_col].to_numpy()
    ret = d[ret_col].to_numpy()
    lo = np.quantile(anom, quantile)
    hi = np.quantile(anom, 1.0 - quantile)
    obs = (ret[anom <= lo].mean() - ret[anom >= hi].mean()) * 1e4

    rng = np.random.default_rng(seed)
    n = ret.size
    perms = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_anom = rng.permutation(anom)
        plo = np.quantile(perm_anom, quantile)
        phi = np.quantile(perm_anom, 1.0 - quantile)
        perms[i] = (ret[perm_anom <= plo].mean() - ret[perm_anom >= phi].mean()) * 1e4
    perm_p = float((np.abs(perms) >= abs(obs)).mean())
    return {
        "obs_spread_bps": round(float(obs), 3),
        "perm_p": round(perm_p, 4),
        "perm_spreads": perms,
    }


# ---------------------------------------------------------------------------
# Tradable overlay -- lag one day, costs x NAV
# ---------------------------------------------------------------------------
def backtest_overlay(
    df: pd.DataFrame,
    anomaly_col: str = "temp_anomaly_f",
    ret_col: str = "ret",
    quantile: float = 1.0 / 3.0,
    cost_bps: float = 1.0,
    lag: int = 1,
) -> dict:
    """A daily long/short overlay: long S&P on cold days, short on warm days.

    Honest execution: the position for day t is set by the anomaly observed at
    t-``lag`` (default one full trading day of lag). Cold (bottom tercile) ->
    long (+1), warm (top tercile) -> short (-1), else flat. One-way transaction
    cost ``cost_bps`` is charged x NAV on every change in position
    (a short pays the same one-way cost; borrow is folded into ``cost_bps``).

    Returns gross/net annualized return and Sharpe, daily HAC t-stat of the net
    strategy return, turnover (avg |dposition| per day), and ``n``.
    """
    d = df[[anomaly_col, ret_col]].dropna().copy()
    anom = d[anomaly_col].to_numpy()
    ret = d[ret_col].to_numpy()
    lo = np.quantile(anom, quantile)
    hi = np.quantile(anom, 1.0 - quantile)
    pos_raw = np.where(anom <= lo, 1.0, np.where(anom >= hi, -1.0, 0.0))

    # Lag the position by `lag` days (no look-ahead).
    pos = np.zeros_like(pos_raw)
    if lag < len(pos_raw):
        pos[lag:] = pos_raw[:-lag]

    gross = pos * ret
    dpos = np.abs(np.diff(pos, prepend=0.0))
    cost = dpos * (cost_bps / 1e4)
    net = gross - cost

    ann = 252.0
    def _ann(x):
        mu = x.mean() * ann
        vol = x.std(ddof=1) * np.sqrt(ann)
        sr = mu / vol if vol > 0 else float("nan")
        return float(mu), float(sr)

    gross_ret, gross_sr = _ann(gross)
    net_ret, net_sr = _ann(net)
    nt = nw_tstat(net)
    turnover = float(dpos.mean())

    return {
        "n": int(d.shape[0]),
        "gross_ann_ret": round(gross_ret, 4),
        "gross_sharpe": round(gross_sr, 3),
        "net_ann_ret": round(net_ret, 4),
        "net_sharpe": round(net_sr, 3),
        "net_tstat": round(nt["tstat"], 3),
        "turnover": round(turnover, 4),
        "cost_bps": cost_bps,
        "lag": lag,
    }


# ---------------------------------------------------------------------------
# One-stop summary -- compact dict mirroring docs/results.md
# ---------------------------------------------------------------------------
def summarize(
    df: pd.DataFrame,
    n_permutations: int = 2000,
    seed: int = 282,
) -> dict:
    """One-stop summary dict for a (return, temperature) daily frame.

    Mirrors the frozen ``R`` dict in build_notebooks.py and docs/results.md.
    """
    cw = cold_warm_sort(df)
    sl = ols_slope(df)
    pm = permutation_spread(df, n_permutations=n_permutations, seed=seed)
    bt = backtest_overlay(df)
    return {
        "n": cw["n"],
        "n_cold": cw["n_cold"],
        "n_warm": cw["n_warm"],
        "mean_cold_bps": cw["mean_cold_bps"],
        "mean_warm_bps": cw["mean_warm_bps"],
        "spread_bps": cw["spread_bps"],
        "spread_tstat": cw["tstat"],
        "slope_bps": sl["slope_bps"],
        "slope_tstat": sl["tstat"],
        "r_squared": sl["r_squared"],
        "perm_p": pm["perm_p"],
        "net_ann_ret": bt["net_ann_ret"],
        "net_sharpe": bt["net_sharpe"],
        "net_tstat": bt["net_tstat"],
        "gross_sharpe": bt["gross_sharpe"],
        "turnover": bt["turnover"],
    }
