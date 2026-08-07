"""Strategy + inference for Study 824 — the Cochrane-Piazzesi factor.

The claim (Cochrane & Piazzesi 2005): a **single** linear combination of forward
rates ``CP_t = gamma' f_t`` forecasts the one-year-ahead excess return of Treasury
bonds of *every* maturity. Method here, on the coarse constant-maturity grid yfinance
exposes:

* **Forwards.** Treat each constant-maturity yield ``y(n)`` (``IRX`` 0.25y, ``FVX`` 5y,
  ``TNX`` 10y, ``TYX`` 30y, in decimals) as a continuously-compounded zero yield with
  log price ``p(n) = -n·y(n)``. The implied forward between adjacent nodes is
  ``f(n1→n2) = (n2·y2 − n1·y1)/(n2 − n1)``. The forward vector is
  ``[y_short, f(0.25→5), f(5→10), f(10→30)]`` — the coarse analogue of CP's five
  one-year forwards.
* **Left-hand side.** Each bond ETF's realised **one-year (252-day) total return**
  minus the one-year risk-free (the ``IRX`` bill yield known at ``t``); averaged across
  ``SHY / IEF / TLT`` into the average excess return ``avg_rx_{t→t+252}``.
* **The regression.** OLS of ``avg_rx_{t+252}`` on ``[1, forward vector_t]`` gives the
  loadings ``gamma`` and the fitted **CP factor**; the in-sample predictive R^2 is the
  headline, and a Newey-West (HAC) *t* on the single-factor predictive slope — computed
  with a long lag to absorb the 252-day overlap — is the honest significance bar.
* **Honesty rails.** The 252-day forward return windows overlap massively, so a plain
  *t* would be wildly overstated; the HAC *t* uses ~1.5×horizon lags. An out-of-sample
  (expanding-window) R^2 and a block placebo grade whether the in-sample fit is real. A
  costed duration-timing overlay asks whether any of it is a paycheck. A seeded
  synthetic control proves the machinery is unbiased.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TRADING_DAYS = 252

YIELD_COLS = ["IRX", "FVX", "TNX", "TYX"]
ETF_COLS = ["SHY", "IEF", "TLT"]
MATS = np.array([0.25, 5.0, 10.0, 30.0])


# --------------------------------------------------------------------------- #
# Forward-rate construction
# --------------------------------------------------------------------------- #
def forward_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Implied forward-rate vector from the four constant-maturity yields.

    Yields (percent) -> decimals; ``p(n) = -n·y(n)``; forward between adjacent nodes
    ``f = (n2·y2 − n1·y1)/(n2 − n1)``. Columns: ``y_short`` (the 0.25y bill, the CP
    "y1"), ``f_1`` (0.25→5y), ``f_2`` (5→10y), ``f_3`` (10→30y). Uses only same-day
    information, so the value on row ``t`` is public at the close of ``t``.
    """
    y = df[YIELD_COLS].to_numpy(dtype=float) / 100.0
    y_short = y[:, 0]
    f1 = (MATS[1] * y[:, 1] - MATS[0] * y[:, 0]) / (MATS[1] - MATS[0])
    f2 = (MATS[2] * y[:, 2] - MATS[1] * y[:, 1]) / (MATS[2] - MATS[1])
    f3 = (MATS[3] * y[:, 3] - MATS[2] * y[:, 2]) / (MATS[3] - MATS[2])
    out = pd.DataFrame(
        {"y_short": y_short, "f_1": f1, "f_2": f2, "f_3": f3}, index=df.index
    )
    return out


def avg_excess_return(df: pd.DataFrame, horizon: int = TRADING_DAYS) -> pd.Series:
    """Average (across SHY/IEF/TLT) one-year-ahead excess return, aligned to signal day.

    For each ETF the realised total return from close ``t`` to close ``t+horizon`` minus
    the one-year risk-free (``IRX`` bill yield at ``t``, in decimals). Averaged across
    the three ETFs. The value on row ``t`` is a *forward* return (known only at
    ``t+horizon``) aligned to the day-``t`` forward-rate signal — so the regression
    forecasts the future from information at ``t``. Rows in the last ``horizon`` days
    (no future price yet) are NaN.
    """
    prices = df[ETF_COLS].to_numpy(dtype=float)
    n = len(df)
    rf_ann = df["IRX"].to_numpy(dtype=float) / 100.0  # 1y risk-free proxy at t
    fwd = np.full((n, len(ETF_COLS)), np.nan)
    if n > horizon:
        fwd[:n - horizon] = prices[horizon:] / prices[:n - horizon] - 1.0
    excess = fwd - rf_ann[:, None]
    avg = np.full(n, np.nan)
    ok = ~np.isnan(excess).all(axis=1)  # rows with at least one ETF forward return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        avg[ok] = np.nanmean(excess[ok], axis=1)
    return pd.Series(avg, index=df.index, name="avg_rx")


# --------------------------------------------------------------------------- #
# Inference primitives (shared house style, mirrors study 803)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# HAC (Newey-West) OLS — the predictive regression with overlap-robust SEs
# --------------------------------------------------------------------------- #
def _ols_hac(X: np.ndarray, y: np.ndarray, lags: int) -> dict:
    """OLS of ``y`` on ``X`` (X already includes an intercept column) with Newey-West
    HAC standard errors (Bartlett kernel, ``lags``). Returns coefficients, HAC t-stats,
    fitted values, and R^2. Overlapping multi-period returns make the residuals strongly
    autocorrelated, so HAC SEs are mandatory for honest inference."""
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # HAC meat, vectorised via the score matrix u_t = resid_t * X_t (n x k):
    #   Gamma_l = sum_t u_t u_{t-l}' = A[l:].T @ A[:-l]
    #   S = w_0 Gamma_0 + sum_{l>=1} w_l (Gamma_l + Gamma_l')
    A = resid[:, None] * X
    S = A.T @ A  # l = 0 term (w_0 = 1)
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = A[l:].T @ A[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstat = np.where(se > 0, beta / se, np.nan)
    fitted = X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": beta, "se": se, "t": tstat, "fitted": fitted, "r2": r2, "n": n}


def cp_regression(df: pd.DataFrame, horizon: int = TRADING_DAYS,
                  nw_lags: int | None = None) -> dict:
    """Fit the Cochrane-Piazzesi return-forecasting regression on a cached tape.

    Regress the average one-year-ahead excess return on ``[1, y_short, f_1, f_2, f_3]``
    (the forward vector). Returns the loadings ``gamma`` and their HAC *t*-stats, the
    in-sample predictive R^2, the fitted **CP factor** (a Series aligned to the signal
    dates), and — the single headline significance number — the Newey-West *t* of the
    predictive slope in the univariate regression ``avg_rx = a + b·CP + e`` (the standard
    CP single-factor summary). ``nw_lags`` defaults to ``round(1.5*horizon)`` to absorb
    the overlap of the 252-day returns.
    """
    if nw_lags is None:
        nw_lags = int(round(1.5 * horizon))
    fwd = forward_rates(df)
    rx = avg_excess_return(df, horizon)
    d = pd.concat([fwd, rx], axis=1).dropna()
    y = d["avg_rx"].to_numpy(dtype=float)
    Xcols = ["y_short", "f_1", "f_2", "f_3"]
    X = np.column_stack([np.ones(len(d)), d[Xcols].to_numpy(dtype=float)])
    lags = min(nw_lags, len(d) - 2)
    reg = _ols_hac(X, y, lags)
    cp_fitted = reg["fitted"]
    cp_series = pd.Series(cp_fitted, index=d.index, name="cp_factor")
    # single-factor summary: regress realised avg_rx on the fitted CP factor
    Xs = np.column_stack([np.ones(len(d)), cp_fitted])
    single = _ols_hac(Xs, y, lags)
    return {
        "loadings": dict(zip(["const"] + Xcols, reg["beta"])),
        "loading_t": dict(zip(["const"] + Xcols, reg["t"])),
        "r2": reg["r2"],
        "n": reg["n"],
        "nw_lags": lags,
        "cp_factor": cp_series,
        "cp_slope": float(single["beta"][1]),
        "cp_slope_t": float(single["t"][1]),
        "dates": d.index,
        "y": y,
        "avg_rx_mean": float(y.mean()),
        "avg_rx_bps": float(y.mean() * 1e4),
    }


# --------------------------------------------------------------------------- #
# Out-of-sample R^2 (expanding window) — the honest predictability check
# --------------------------------------------------------------------------- #
def oos_r2(df: pd.DataFrame, horizon: int = TRADING_DAYS,
           min_train: int = 1000, step: int = 21) -> dict:
    """Campbell-Thompson out-of-sample R^2 of the CP forecast vs the prevailing mean.

    Expanding window: at each evaluation date (every ``step`` days after ``min_train``
    training rows) fit the forward-vector regression on data whose *forward return is
    already realised* (i.e. through the row ``horizon`` days before the fit date), then
    forecast the next observation's excess return. OOS R^2 = 1 − SSE(model)/SSE(mean).
    A positive value means the CP factor beats the naive prevailing-mean forecast out of
    sample — the bar that separates a real forecaster from an in-sample curve fit.
    """
    fwd = forward_rates(df)
    rx = avg_excess_return(df, horizon)
    d = pd.concat([fwd, rx], axis=1).dropna().reset_index(drop=True)
    Xcols = ["y_short", "f_1", "f_2", "f_3"]
    Xall = np.column_stack([np.ones(len(d)), d[Xcols].to_numpy(dtype=float)])
    yall = d["avg_rx"].to_numpy(dtype=float)
    n = len(d)
    sse_m, sse_c, preds = 0.0, 0.0, 0
    for i in range(min_train, n, step):
        # only rows whose forward window closed before the *signal* at i:
        train_end = i - horizon
        if train_end < 100:
            continue
        Xtr, ytr = Xall[:train_end], yall[:train_end]
        beta = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ ytr)
        fc = float(Xall[i] @ beta)
        mean_fc = float(ytr.mean())
        actual = yall[i]
        sse_c += (actual - fc) ** 2
        sse_m += (actual - mean_fc) ** 2
        preds += 1
    r2 = 1.0 - sse_c / sse_m if sse_m > 0 else float("nan")
    return {"oos_r2": r2, "n_preds": preds}


# --------------------------------------------------------------------------- #
# Block placebo — is the in-sample R^2 a lucky alignment?
# --------------------------------------------------------------------------- #
def placebo_r2(df: pd.DataFrame, horizon: int = TRADING_DAYS,
               n_perm: int = 1000, block: int = 252, seed: int = 824) -> dict:
    """Block-rotation placebo for the in-sample R^2.

    Circularly rotate the forward-return target against the forward-rate matrix in
    ``n_perm`` draws (block rotation preserves the heavy autocorrelation of overlapping
    returns, so the null is honest). p = share of rotated worlds whose R^2 >= observed.
    """
    fwd = forward_rates(df)
    rx = avg_excess_return(df, horizon)
    d = pd.concat([fwd, rx], axis=1).dropna().reset_index(drop=True)
    Xcols = ["y_short", "f_1", "f_2", "f_3"]
    X = np.column_stack([np.ones(len(d)), d[Xcols].to_numpy(dtype=float)])
    y = d["avg_rx"].to_numpy(dtype=float)
    n = len(y)

    def _r2(yv):
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ yv)
        resid = yv - X @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    obs = _r2(y)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(block, n - block))
        draws[i] = _r2(np.roll(y, shift))
    return {
        "obs_r2": obs,
        "placebo_mean_r2": float(draws.mean()),
        "placebo_sd_r2": float(draws.std(ddof=1)),
        "p_value": float((draws >= obs).mean()),
        "n_perm": n_perm,
        "draws": draws,
    }


# --------------------------------------------------------------------------- #
# Costed duration-timing overlay — can you get paid for it?
# --------------------------------------------------------------------------- #
def timer_stats(df: pd.DataFrame, horizon: int = TRADING_DAYS,
                cost_bps: float = 2.0, lookback: int = 1000) -> dict:
    """Own TLT (duration) when the *out-of-sample* CP forecast is above its rolling
    median, else cash; net of one-way costs at each switch.

    The CP factor is refit on an expanding window (no look-ahead); the position is set
    from the forecast known at the close of ``t−1`` and held on day ``t`` (one documented
    lag). Costs are one-way ``cost_bps`` × traded NAV per switch. Reports the timer's net
    annualised Sharpe vs buy-and-hold TLT and the switch frequency.
    """
    fwd = forward_rates(df)
    rx = avg_excess_return(df, horizon)
    Xcols = ["y_short", "f_1", "f_2", "f_3"]
    base = pd.concat([fwd, rx], axis=1)
    tlt = df["TLT"].astype(float)
    tlt_ret = np.log(tlt).diff()

    d = base.dropna().reset_index()
    dates = d["date"].to_numpy()
    X = np.column_stack([np.ones(len(d)), d[Xcols].to_numpy(dtype=float)])
    y = d["avg_rx"].to_numpy(dtype=float)

    # expanding-window forecast, using only rows whose forward window closed
    fc = np.full(len(d), np.nan)
    for i in range(lookback, len(d)):
        train_end = i - horizon
        if train_end < 100:
            continue
        Xtr, ytr = X[:train_end], y[:train_end]
        beta = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ ytr)
        fc[i] = float(X[i] @ beta)
    sig = pd.Series(fc, index=pd.DatetimeIndex(dates))
    med = sig.expanding(min_periods=lookback).median()
    want = (sig > med).astype(float)

    # align position (known at t-1) to daily TLT returns held at t
    pos = want.shift(1).reindex(tlt_ret.index).ffill().fillna(0.0)
    switches = pos.diff().abs().fillna(0.0)
    cost = switches * cost_bps * 1e-4
    timer = pos * tlt_ret - cost
    valid = ~tlt_ret.isna() & (pos.index >= sig.dropna().index.min())
    timer = timer[valid]
    bh = tlt_ret[valid]

    def _sharpe(x):
        x = x.dropna().to_numpy()
        s = x.std(ddof=1)
        return float(x.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else float("nan")

    return {
        "timer_sharpe": _sharpe(timer),
        "bh_sharpe": _sharpe(bh),
        "switches_per_yr": float(switches[valid].sum() / max(len(timer), 1) * TRADING_DAYS),
        "days_invested_frac": float(pos[valid].mean()),
        "net_mean_bps": float(timer.dropna().mean() * 1e4),
        "n": int(valid.sum()),
    }


# --------------------------------------------------------------------------- #
# Headline bundle + synthetic detector
# --------------------------------------------------------------------------- #
def headline(df: pd.DataFrame, horizon: int = TRADING_DAYS) -> dict:
    """Everything the results table quotes, in one call."""
    reg = cp_regression(df, horizon)
    return {
        "n": reg["n"],
        "r2": reg["r2"],
        "cp_slope_t": reg["cp_slope_t"],
        "avg_rx_bps": reg["avg_rx_bps"],
        "loadings": reg["loadings"],
        "loading_t": reg["loading_t"],
        "nw_lags": reg["nw_lags"],
    }


def synthetic_detect(df: pd.DataFrame, horizon: int = TRADING_DAYS) -> dict:
    """Run the CP regression on a synthetic tape (machinery proof)."""
    reg = cp_regression(df, horizon)
    return {"r2": reg["r2"], "cp_slope_t": reg["cp_slope_t"], "n": reg["n"]}
