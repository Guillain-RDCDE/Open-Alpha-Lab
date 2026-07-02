"""The engine and its honest controls — Study 556 (Electricity-Demand).

The claim, at full strength: aggregate electricity demand is the *pulse of the real
economy* — factories, data centres, offices — printed monthly and impossible to spin.
So **electricity-demand growth** should be a leading nowcast of real activity, and a
hot power economy should precede stronger equity (or utility) returns. We test the
tradable version of that folklore.

The pipeline, honestly, on what a retail stack can reach:

1. **The signal — demand-growth momentum.** Year-over-year % change of monthly U.S.
   net generation. The YoY transform removes the enormous summer/winter seasonal
   (comparing each month to the same month a year earlier), leaving the business-cycle
   pulse. "Rising" = YoY growth above its own trailing median.

2. **The predictive regression.** OLS of the *forward* h-month equity return on the
   (lagged, public) demand-growth signal. The believers' sign is **positive** (more
   demand growth -> higher forward returns). HAC (Newey-West) standard errors handle
   the overlap when h > 1.

3. **The conditional split.** Forward returns in high-demand-growth months vs
   low-demand-growth months, with a Welch two-sample t and a label-shuffle placebo
   null — the same test the desk uses for every macro-nowcast folklore study.

4. **A tradable overlay.** Hold the equity only when demand growth is hot, else sit in
   cash; compare to buy-and-hold, gross and net of a round-trip cost per switch.

5. **A robustness sweep.** Horizons (1/3/6/12m), both tapes (SPY and XLU), and an
   ex-COVID cut — does the sign survive?

6. **A synthetic positive control.** A deterministic world with a *planted* demand
   ->returns link (``edge > 0``); the engine must recover a positive slope and stay
   flat at the null, averaged over >= 20 seeds (house rule).

Execution convention — ONE documented lag. The month-*t* net-generation figure is only
*published* by the EIA well into month *t+1* (Electric Power Monthly lands ~8 weeks
after the reference month). So the signal available to trade at the close of month *m*
is built from demand data through month *m-1* at the latest; we form the signal from the
YoY growth as-of month *m-1* and earn the return of month *m+1..m+h*. In code this is a
single ``shift(2)`` on the raw demand growth (one month for publication lag, one for the
standard signal->return convention), applied once. No future data enters the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# The signal — demand-growth momentum (seasonal-free), publication-lagged
# --------------------------------------------------------------------------- #
def demand_growth(netgen: pd.Series) -> pd.Series:
    """Year-over-year % change of monthly net generation (removes the seasonal)."""
    return netgen.pct_change(12).rename("demand_growth")


def signal_lagged(netgen: pd.Series, pub_lag: int = 2) -> pd.Series:
    """The *tradable* demand-growth signal: YoY growth shifted by ``pub_lag`` months.

    ``pub_lag=2`` = one month for the EIA publication lag + one for the standard
    signal-known-at-close-of-t -> earn-t+1 convention. So the value at month *m* uses
    only demand data public strictly before *m*. This is the single execution lag,
    applied once.
    """
    return demand_growth(netgen).shift(pub_lag).rename("signal")


def rising_flag(signal: pd.Series, lookback: int = 24) -> pd.Series:
    """Boolean: is demand growth 'hot' — above its trailing ``lookback``-month median?"""
    med = signal.rolling(lookback, min_periods=6).median()
    return (signal > med).rename("rising")


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(price: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward simple return over the next ``horizon`` months (log-safe simple)."""
    fwd = price.shift(-horizon) / price - 1.0
    return fwd.rename(f"fwd_{horizon}m")


# --------------------------------------------------------------------------- #
# The predictive regression — HAC (Newey-West) t on the slope
# --------------------------------------------------------------------------- #
def _newey_west_se(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance for OLS coefficients."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    return xtx_inv @ S @ xtx_inv


def predictive_regression(signal: pd.Series, price: pd.Series, horizon: int = 1) -> dict:
    """OLS of forward h-month return on the demand-growth signal, HAC t on the slope.

    Believers' sign is POSITIVE (hot demand -> higher forward returns). Returns slope,
    HAC slope-t, R^2 and n. HAC lags = horizon (overlap correction for h>1).
    """
    y = forward_returns(price, horizon)
    df = pd.DataFrame({"x": signal, "y": y}).dropna()
    if len(df) < 12:
        return {"slope": float("nan"), "slope_t": float("nan"), "r2": float("nan"), "n": len(df)}
    x = df["x"].to_numpy(dtype=float)
    yv = df["y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    cov = _newey_west_se(X, resid, lags=max(horizon, 1))
    se_slope = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope and se_slope > 0 else float("nan"),
        "r2": r2,
        "n": int(len(df)),
    }


# --------------------------------------------------------------------------- #
# The conditional split — Welch two-sample t + placebo null
# --------------------------------------------------------------------------- #
def conditional_split(signal: pd.Series, price: pd.Series, horizon: int = 1,
                      lookback: int = 24) -> dict:
    """Forward returns in hot- vs cold-demand-growth months; Welch two-sample t.

    Believers' sign: hot > cold (positive spread). Returns bucket means, spread, Welch
    t, and n in each bucket.
    """
    rising = rising_flag(signal, lookback)
    y = forward_returns(price, horizon)
    df = pd.DataFrame({"rising": rising, "y": y}).dropna()
    hot = df.loc[df["rising"], "y"].to_numpy(dtype=float)
    cold = df.loc[~df["rising"], "y"].to_numpy(dtype=float)
    if len(hot) < 3 or len(cold) < 3:
        return {"hot_mean": float("nan"), "cold_mean": float("nan"), "spread": float("nan"),
                "t": float("nan"), "n_hot": len(hot), "n_cold": len(cold)}
    va, vb = hot.var(ddof=1), cold.var(ddof=1)
    se = np.sqrt(va / len(hot) + vb / len(cold))
    t = (hot.mean() - cold.mean()) / se if se > 0 else float("nan")
    return {
        "hot_mean": float(hot.mean()),
        "cold_mean": float(cold.mean()),
        "spread": float(hot.mean() - cold.mean()),
        "t": float(t),
        "n_hot": int(len(hot)),
        "n_cold": int(len(cold)),
    }


def placebo_pvalue(signal: pd.Series, price: pd.Series, horizon: int = 1,
                   lookback: int = 24, n_perm: int = 2000, seed: int = 556) -> float:
    """Label-shuffle placebo p-value for the hot-minus-cold spread.

    Shuffle the rising labels against forward returns; the p-value is the fraction of
    shuffles whose |spread| >= the observed |spread|. A real conditional signal sits in
    the tail; noise does not.
    """
    base = conditional_split(signal, price, horizon, lookback)
    obs = abs(base["spread"])
    if not np.isfinite(obs):
        return float("nan")
    rising = rising_flag(signal, lookback)
    y = forward_returns(price, horizon)
    df = pd.DataFrame({"rising": rising.astype(float), "y": y}).dropna()
    r = df["rising"].to_numpy(dtype=bool)
    yv = df["y"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(yv)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        rp = r[perm]
        if rp.sum() < 1 or (~rp).sum() < 1:
            continue
        spread = yv[rp].mean() - yv[~rp].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# Tradable overlay — hold-when-hot vs buy-and-hold
# --------------------------------------------------------------------------- #
def overlay_backtest(signal: pd.Series, price: pd.Series, lookback: int = 24,
                     cost_bps: float = 5.0) -> dict:
    """'Hold equity when demand growth is hot, else cash' vs buy-and-hold.

    Monthly rebalance; a round-trip one-way cost of ``cost_bps`` is charged on every
    switch (in or out). Returns annualised gross/net overlay return, buy-and-hold
    return, both Sharpes, and the number of switches.
    """
    rising = rising_flag(signal, lookback)
    mret = price.pct_change()
    df = pd.DataFrame({"pos": rising.astype(float), "r": mret}).dropna()
    if len(df) < 12:
        return {}
    pos = df["pos"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    switches = int(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    strat_gross = pos * r
    # cost on each change of position
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat_net = strat_gross - dpos * cost_bps * 1e-4
    ann = 12.0

    def _ann_ret(x):
        return float((1 + x).prod() ** (ann / len(x)) - 1)

    def _sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")

    return {
        "bh_ann": _ann_ret(r),
        "overlay_gross_ann": _ann_ret(strat_gross),
        "overlay_net_ann": _ann_ret(strat_net),
        "bh_sharpe": _sharpe(r),
        "overlay_sharpe": _sharpe(strat_net),
        "switches": switches,
        "n": int(len(df)),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control — seed-robust (house rule)
# --------------------------------------------------------------------------- #
def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 25, base_seed: int = 556,
                     horizon: int = 1) -> float:
    """Average the predictive-regression slope-t over ``n_seeds`` synthetic worlds.

    Any synthetic-dependent claim averages the stat over >= 20 seeds so no single lucky
    RNG seed can manufacture significance. Returns the mean HAC slope-t across seeds for
    a planted ``edge``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        df = data_mod.synthetic_series(edge=edge, seed=s)
        sig = signal_lagged(df["netgen"])
        reg = predictive_regression(sig, df["equity"], horizon=horizon)
        ts.append(reg["slope_t"])
    return float(np.nanmean(ts))
