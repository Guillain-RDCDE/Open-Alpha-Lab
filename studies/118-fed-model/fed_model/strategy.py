"""Signal analysis and honest controls — Study 118 (Fed-Model).

The Fed Model (Yardeni 1997): when the equity earnings yield (E/P) exceeds the
10-year Treasury yield, stocks are "cheap" relative to bonds and forward returns
should be above average. We formalise this as a predictive regression of forward
real returns on the Fed-Model spread and pin it against the **unconditional mean**
(the buy-and-hold baseline that uses no signal at all).

Three tests share one engine:

- **In-sample OLS regression** — slope, R², and a Newey–West HAC t-stat that
  corrects for the severe serial overlap in 10-year forward returns.
- **Out-of-sample R² (Campbell & Thompson 2008)** — train on 1871–1969, forecast
  on 1970–2023. OOS R² < 0 means the unconditional mean beats the signal model.
- **Timing strategy** — go long the index when ``fed_spread > median``, hold cash
  otherwise; compare to unconditional buy-and-hold on *the same monthly periods*.

The Asness (2003) critique: E/P is a *real* quantity; comparing it to the *nominal*
Treasury yield is an apples-to-oranges comparison once inflation is non-trivial. We
separately test E/P vs the real rate (yield minus trailing CPI inflation) to see
whether that version improves predictability.

No look-ahead: all signals are formed from data available at month *t*; forward
returns run from *t* onward. Expanding-window training in the OOS test ensures no
future data leaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OLS with Newey–West HAC standard errors
# ---------------------------------------------------------------------------
def ols_hac(
    x: np.ndarray,
    y: np.ndarray,
    lags: int | None = None,
) -> dict:
    """OLS slope on y ~ x with a Newey–West HAC standard error on the slope.

    The forward 10-year returns overlap by ~119 months each, making adjacent
    observations highly correlated. A naive OLS t-stat massively overstates
    significance. Newey–West (1987) corrects for this by down-weighting distant
    residual autocovariances; the lag choice ``lags=None`` applies the standard
    rule of thumb ``floor(4*(n/100)^(2/9))``, but for 10-year overlapping returns
    callers should pass a larger value (e.g., 120) to capture the full overlap.

    Returns a dict with keys: ``slope``, ``intercept``, ``r2``, ``tstat``, ``se``,
    ``n``, ``lags``.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        nan = float("nan")
        return {"slope": nan, "intercept": nan, "r2": nan, "tstat": nan, "se": nan, "n": n, "lags": 0}

    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(xm @ xm)
    slope = float(xm @ ym) / denom if denom > 0 else float("nan")
    intercept = y.mean() - slope * x.mean()
    resid = y - (intercept + slope * x)
    ss_res = float(resid @ resid)
    ss_tot = float(ym @ ym)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # HAC SE via the sandwich: score = resid * xm / denom
    score = resid * xm  # numerator of the score; divide by denom² below
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(score @ score) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(score[k:] @ score[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n) / (denom / n) if denom > 0 else float("nan")
    tstat = slope / se if (se and np.isfinite(se) and se > 0) else float("nan")

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "tstat": tstat,
        "se": se,
        "n": n,
        "lags": lags,
    }


# ---------------------------------------------------------------------------
# Out-of-sample R² (Campbell & Thompson 2008)
# ---------------------------------------------------------------------------
def oos_r2(
    panel: pd.DataFrame,
    signal_col: str,
    target_col: str,
    train_end_year: int = 1969,
    lags: int | None = None,
) -> dict:
    """Expanding-window out-of-sample R² for one signal/target pair.

    Trains an OLS on data up to ``train_end_year`` (inclusive), then predicts each
    subsequent month using the slope estimated on all data up to but not including
    that month (expanding). OOS R² is:

        R²_OOS = 1 - SS_pred / SS_hist

    where SS_pred is the sum of squared prediction errors and SS_hist is the sum of
    squared errors from the historical mean (the unconditional baseline). R²_OOS < 0
    means the unconditional mean beats the regression model out of sample — the
    model adds no practical value. Campbell & Thompson (2008) show that many
    predictors with positive in-sample R² deliver negative OOS R².
    """
    sub = panel[[signal_col, target_col]].dropna()
    train = sub[sub.index.year <= train_end_year]
    test = sub[sub.index.year > train_end_year]

    if len(train) < 50 or len(test) < 30:
        return {"oos_r2": float("nan"), "n_train": len(train), "n_test": len(test)}

    x_tr = train[signal_col].values
    y_tr = train[target_col].values
    x_te = test[signal_col].values
    y_te = test[target_col].values

    # Fit OLS on training set
    xm_tr = x_tr - x_tr.mean()
    ym_tr = y_tr - y_tr.mean()
    denom = float(xm_tr @ xm_tr)
    slope = float(xm_tr @ ym_tr) / denom if denom > 0 else 0.0
    intercept = y_tr.mean() - slope * x_tr.mean()

    pred = intercept + slope * x_te
    hist_mean = y_tr.mean()  # unconditional baseline (expanding mean from training set)

    ss_pred = float(((y_te - pred) ** 2).sum())
    ss_hist = float(((y_te - hist_mean) ** 2).sum())
    r2_oos = 1.0 - ss_pred / ss_hist if ss_hist > 0 else float("nan")

    return {
        "oos_r2": r2_oos,
        "n_train": len(train),
        "n_test": len(test),
        "slope": slope,
        "intercept": intercept,
        "hist_mean": hist_mean,
    }


# ---------------------------------------------------------------------------
# Timing strategy: long when spread > median, else hold cash
# ---------------------------------------------------------------------------
def timing_returns(
    panel: pd.DataFrame,
    signal_col: str,
    return_col: str = "fwd1",
    threshold: float | None = None,
    cost_pa: float = 0.0,
) -> pd.DataFrame:
    """Monthly-rebalance market-timing strategy based on a signal threshold.

    At each month *t*, the signal formed from data at *t* is compared to ``threshold``
    (default: expanding-window median — no look-ahead). If the signal exceeds the
    threshold the strategy is ``long`` (position = +1); otherwise ``flat`` (position = 0).
    The 1-year forward return is taken as the return over the next year.

    Returns a DataFrame with columns:
        signal      — the signal value
        position    — 1 (long) or 0 (flat)
        ret_strat   — strategy monthly return (position * next-month-return / 12)
        ret_bh      — buy-and-hold monthly return
        cum_strat   — cumulative strategy growth
        cum_bh      — cumulative buy-and-hold growth

    ``cost_pa`` is an annual drag subtracted from strategy returns when positioned long
    (a crude proxy for transaction costs / slippage on annual rebalancing).
    """
    sub = panel[[signal_col, return_col]].dropna().copy()
    sub = sub.rename(columns={signal_col: "signal", return_col: "ret_fwd"})

    if threshold is None:
        # Expanding-window median — uses only past data
        sub["threshold"] = sub["signal"].expanding(min_periods=60).median()
    else:
        sub["threshold"] = threshold

    sub["position"] = (sub["signal"] > sub["threshold"]).astype(int)

    # Forward returns are already annualised; monthly proxy = fwd1 / 12
    sub["ret_strat"] = sub["position"] * sub["ret_fwd"] / 12.0 - cost_pa / 12.0 * sub["position"]
    sub["ret_bh"] = sub["ret_fwd"] / 12.0

    sub["cum_strat"] = (1.0 + sub["ret_strat"]).cumprod()
    sub["cum_bh"] = (1.0 + sub["ret_bh"]).cumprod()

    return sub[["signal", "position", "ret_strat", "ret_bh", "cum_strat", "cum_bh"]]


# ---------------------------------------------------------------------------
# Summary statistics for one timing-return series
# ---------------------------------------------------------------------------
def summarize(
    timing: pd.DataFrame,
    col: str = "ret_strat",
    baseline_col: str = "ret_bh",
    periods_per_year: int = 12,
) -> dict:
    """Headline statistics for a monthly return series and its baseline.

    Returns mean annual return, Sharpe, and HAC t-stat (Newey–West) for both the
    strategy and the buy-and-hold baseline, plus the active return (strategy minus
    baseline) and its t-stat.
    """
    def _stats(s: np.ndarray, ppy: int) -> dict:
        s = s[np.isfinite(s)]
        n = len(s)
        if n < 5:
            return {"mean_pa": float("nan"), "sharpe": float("nan"), "tstat": float("nan"), "n": n}
        mu = s.mean()
        sd = s.std(ddof=1)
        e = s - mu
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lag + 1):
            w = 1.0 - k / (lag + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        return {
            "mean_pa": mu * ppy,
            "sharpe": mu / sd * np.sqrt(ppy) if sd > 0 else float("nan"),
            "tstat": mu / se if se > 0 else float("nan"),
            "n": n,
        }

    s = timing[col].to_numpy(dtype=float)
    b = timing[baseline_col].to_numpy(dtype=float)
    active = s - b

    strat_stats = _stats(s, periods_per_year)
    base_stats = _stats(b, periods_per_year)
    active_stats = _stats(active, periods_per_year)

    return {
        "strat_mean_pa": strat_stats["mean_pa"],
        "strat_sharpe": strat_stats["sharpe"],
        "strat_tstat": strat_stats["tstat"],
        "bh_mean_pa": base_stats["mean_pa"],
        "bh_sharpe": base_stats["sharpe"],
        "active_mean_pa": active_stats["mean_pa"],
        "active_tstat": active_stats["tstat"],
        "n": strat_stats["n"],
    }
