"""Strategy and honest controls -- Study 265 (IPO-Volume).

The froth thesis (Ibbotson-Jaffe 1975; Ritter 1991; Baker-Wurgler 2006): a flood
of IPOs marks investor euphoria and a market top, so high IPO volume should
*predict low* forward equity returns. We test it three ways:

1. **Predictive regression** -- regress next-year S&P 500 return on the
   standardized (expanding, past-only) IPO-count froth signal, with HAC
   (Newey-West) standard errors. The folklore predicts a *negative* slope.

2. **Long/flat timing rule** -- a tradable wrapper: be long the S&P next year
   when this year's froth z-score is below a threshold (the IPO window is not
   wide open), flat otherwise. Compared against buy-and-hold.

3. **Positive control** -- the synthetic generator plants a known negative slope;
   the estimator must recover it (and read ~zero on the null).

**No look-ahead**: the froth signal at the end of year Y uses the IPO count for
year Y standardized against years <= Y only (expanding mean/std). It predicts the
S&P return earned during year Y+1. The minimum burn-in is ``min_obs`` prior years.

**Execution lag**: documented and explicit -- the position for year Y+1 is set at
the end of year Y from data fully observed by then (one calendar-year lag).

**Costs**: the timing rule trades the index in/out at most once per year; a
one-way cost in bps is applied to NAV on each switch (long-only, so no borrow).

**Tape labels**: S&P returns are *price returns* (Shiller index level; dividends
excluded). The honesty axis is the tiny annual sample (n ~ 44 forecastable years).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def froth_signal(
    df: pd.DataFrame,
    count_col: str = "ipo_count",
    min_obs: int = 5,
) -> pd.Series:
    """Expanding-window standardized IPO-count froth z-score (past-only).

    For each year Y, the score is::

        z[Y] = (count[Y] - mean(count[<=Y])) / std(count[<=Y])

    using only observations up to and including year Y (an expanding window),
    so no future IPO statistics leak into the score. High z = a flood of IPOs
    (wide-open window = froth); low z = a frozen IPO window.

    The first ``min_obs`` years are NaN (insufficient history to standardize).

    Returns a Series aligned to ``df`` index, named ``froth_z``.
    """
    counts = pd.Series(df[count_col].to_numpy(dtype=float), index=df.index)
    out = pd.Series(np.nan, index=df.index, name="froth_z")
    for i in range(len(counts)):
        hist = counts.iloc[: i + 1]
        if len(hist) < min_obs:
            continue
        mu = hist.mean()
        sd = hist.std(ddof=0)
        if sd > 0:
            out.iloc[i] = (counts.iloc[i] - mu) / sd
    return out


# ---------------------------------------------------------------------------
# Predictive regression with HAC errors
# ---------------------------------------------------------------------------
def predictive_regression(
    z: pd.Series,
    fwd_return: pd.Series,
) -> dict:
    """OLS of next-year return on the froth z-score, with Newey-West HAC t-stat.

    Model: ``fwd_return = alpha + beta * z + eps``.

    The folklore predicts ``beta < 0`` (more IPOs -> lower forward return).
    Returns a dict with ``alpha``, ``beta``, ``beta_t`` (HAC), ``r2``, ``n``,
    and ``corr`` (Pearson correlation of z and fwd_return).
    """
    pair = pd.concat([z.rename("z"), fwd_return.rename("y")], axis=1).dropna()
    n = int(len(pair))
    if n < 5:
        return {k: np.nan for k in ("alpha", "beta", "beta_t", "r2", "corr")} | {"n": n}

    x = pair["z"].to_numpy(dtype=float)
    y = pair["y"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    alpha, beta = float(coef[0]), float(coef[1])

    resid = y - X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Newey-West HAC covariance of the OLS coefficients.
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = np.zeros((2, 2))
    u = resid
    for t in range(n):
        xt = X[t : t + 1].T  # 2x1
        S += (u[t] ** 2) * (xt @ xt.T)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        for t in range(k, n):
            xt = X[t : t + 1].T
            xtk = X[t - k : t - k + 1].T
            gamma = u[t] * u[t - k] * (xt @ xtk.T + xtk @ xt.T)
            S += w * gamma
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    beta_t = beta / se_beta if se_beta > 0 else float("nan")

    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")

    return {
        "alpha": alpha,
        "beta": beta,
        "beta_t": float(beta_t),
        "r2": float(r2),
        "corr": corr,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Long/flat timing rule
# ---------------------------------------------------------------------------
def timing_returns(
    z: pd.Series,
    fwd_return: pd.Series,
    threshold: float = 0.0,
    one_way_bps: float = 5.0,
) -> pd.DataFrame:
    """Long-the-index-next-year-when-froth-is-low timing rule vs buy-and-hold.

    Position for year Y+1: long (1.0) if ``z[Y] < threshold`` (IPO window not
    wide open), else flat (0.0). The position is applied to ``fwd_return`` (the
    return earned in Y+1) -- a clean one-year execution lag.

    A one-way transaction cost (bps of NAV) is charged whenever the position
    changes from one year to the next (long-only, so no borrow cost on the short
    side -- there is no short side).

    Returns a DataFrame indexed like the aligned (z, fwd_return) pair with:
    ``z``, ``fwd_return``, ``position``, ``strat_gross``, ``strat_net``,
    ``buy_hold``.
    """
    pair = pd.concat([z.rename("z"), fwd_return.rename("fwd_return")], axis=1).dropna()
    pos = (pair["z"] < threshold).astype(float)

    gross = pos * pair["fwd_return"]
    # Cost on switches (including the initial entry from flat).
    prev = pos.shift(1).fillna(0.0)
    switched = (pos != prev).astype(float)
    cost = switched * one_way_bps * 1e-4
    net = gross - cost

    out = pd.DataFrame(
        {
            "z": pair["z"],
            "fwd_return": pair["fwd_return"],
            "position": pos,
            "strat_gross": gross,
            "strat_net": net,
            "buy_hold": pair["fwd_return"],
        }
    )
    return out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(series: pd.Series, periods: int = 1) -> dict:
    """Headline statistics for an annual return series (Newey-West HAC t-stat).

    ``periods`` is the number of periods per year (1 for annual). Sharpe is
    reported per-period; annualised externally if needed.
    """
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown")} | {"n": n}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": mu,
        "vol": std,
        "sharpe": float(sr),
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }


def annualise(stats: dict, periods: int = 1) -> dict:
    """Convert per-period stats to annualised equivalents (annual data => identity)."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if "mean_ann" in out and "vol_ann" in out and out["vol_ann"] > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out
