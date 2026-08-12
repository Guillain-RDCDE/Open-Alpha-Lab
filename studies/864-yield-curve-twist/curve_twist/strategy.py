"""Strategy + inference for Study 864 (Yield-Curve Twist / Butterfly).

The claim under test: beyond the curve's **level** and **slope**, its **curvature** —
the butterfly ``fly = 2*y10 - y5 - y30`` (belly vs wings) and its change ``dfly`` (a
"twist") — carries information about forward Treasury (IEF/TLT) and equity (SPY)
returns that is **distinct** from the 2s10s slope (studies 66/132) and the roll-down
carry (study 380).

Five complementary tests share one HAC inference engine:

1. **Predictive regression (the headline).** Regress the forward log return of a target
   ETF over horizon ``h`` on the *lagged, standardised* butterfly level (and, in the
   incremental variant, jointly with the slope control). Newey-West (HAC) *t* on the
   butterfly loading. The **incremental** regression is the dedup test: does the
   butterfly's *t* survive after the slope is partialled out?

2. **Quintile-sorted forward returns.** Bucket days by the out-of-sample rolling
   butterfly rank into quintiles; the Q5−Q1 spread of forward returns, HAC *t*.

3. **The twist (change) signal.** The same regression / sort on ``dfly`` instead of the
   level — does a *steepening* of the curvature predict anything the level does not?

4. **A costed bond-timing overlay.** Own the target ETF when the lagged butterfly rank
   is favourable, cash otherwise; charge a one-way cost per switch on the traded NAV,
   and compare net Sharpe to buy-and-hold.

5. **A permutation placebo + a seeded synthetic positive control.** The placebo breaks
   the signal → outcome link; the synthetic control proves the machinery recovers a
   *planted* twist edge and stays silent on the null.

No look-ahead: the butterfly at the close of day *t* forms the signal; the forward
return starts at the close of day *t+1* (one documented execution lag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (canonical desk set, mirroring study 803)
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


def nw_regression(y: np.ndarray, X: np.ndarray, lags: int = 21) -> dict:
    """OLS of ``y`` on ``X`` (a design matrix WITHOUT an intercept column — one is
    added) with Newey-West (HAC, Bartlett) standard errors.

    Overlapping forward returns are serially correlated, so a plain OLS *t* overstates
    significance; the HAC covariance corrects it. Returns coefficients, HAC *t*-stats
    (intercept first, then the columns of ``X`` in order), R², and n.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    n = len(y)
    Z = np.column_stack([np.ones(n), X])
    k = Z.shape[1]
    if n <= k + 2:
        return {"beta": np.full(k, np.nan), "t": np.full(k, np.nan), "r2": np.nan, "n": n}
    XtX_inv = np.linalg.pinv(Z.T @ Z)
    beta = XtX_inv @ (Z.T @ y)
    resid = y - Z @ beta
    # Newey-West HAC meat: sum_{l} w_l (S_l + S_l')
    S = np.zeros((k, k))
    u = Z * resid[:, None]
    S += u.T @ u
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": beta, "t": t, "r2": r2, "n": n, "se": se}


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def forward_return(close: pd.Series, horizon: int = 21) -> pd.Series:
    """Log forward return from close *t* to close *t+horizon*, aligned at day *t*."""
    lc = np.log(close)
    return lc.shift(-horizon) - lc


def rolling_rank(sig: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Causal rolling percentile rank in [0,1] (value at *t* uses history through *t*)."""
    return sig.rolling(window, min_periods=min_periods).rank(pct=True)


def quintile(sig: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Rolling quintile label 1..5 (1 = lowest signal, 5 = highest), out-of-sample."""
    pct = rolling_rank(sig, window, min_periods)
    return np.ceil(pct * 5).clip(1, 5).astype("Int64")


def zscore(sig: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Causal rolling z-score of the signal (mean/std over trailing ``window``)."""
    m = sig.rolling(window, min_periods=min_periods).mean()
    s = sig.rolling(window, min_periods=min_periods).std(ddof=0)
    return (sig - m) / s.where(s > 0)


# --------------------------------------------------------------------------- #
# 1. Predictive regression — headline + incremental (the dedup test)
# --------------------------------------------------------------------------- #
def predictive_regression(
    df: pd.DataFrame,
    signal_col: str = "fly",
    target: str = "IEF",
    horizon: int = 21,
    controls: list[str] | None = None,
    lags: int | None = None,
    window: int = 252,
) -> dict:
    """Regress the forward log return of ``target`` (over ``horizon`` days) on the
    *lagged, rolling-z-scored* ``signal_col`` (and any ``controls``, each z-scored the
    same way). One execution lag: the signal is known at close *t-1* (``.shift(1)``),
    the forward return runs close *t* → close *t+horizon*.

    Returns beta (bps of forward return per +1σ of signal), HAC *t*, R², n, and — when
    ``controls`` are given — the same for each control. ``beta_bps`` and ``t`` are the
    headline butterfly loading; ``controls_t`` maps each control name to its HAC *t*.
    """
    close = df[f"{target}_close"]
    fwd = forward_return(close, horizon).to_numpy(dtype=float)

    sig = zscore(df[signal_col], window).shift(1)
    cols = [sig.to_numpy(dtype=float)]
    names = [signal_col]
    controls = controls or []
    for c in controls:
        cols.append(zscore(df[c], window).shift(1).to_numpy(dtype=float))
        names.append(c)
    X = np.column_stack(cols)

    reg = nw_regression(fwd, X, lags=lags if lags is not None else horizon)
    beta = reg["beta"]; t = reg["t"]
    out = {
        "target": target, "signal": signal_col, "horizon": horizon,
        "n": reg["n"], "r2": reg["r2"],
        "beta_bps": float(beta[1] * 1e4) if len(beta) > 1 else float("nan"),
        "t": float(t[1]) if len(t) > 1 else float("nan"),
        "controls_t": {},
        "controls_beta_bps": {},
    }
    for j, c in enumerate(controls, start=2):
        out["controls_t"][c] = float(t[j])
        out["controls_beta_bps"][c] = float(beta[j] * 1e4)
    return out


# --------------------------------------------------------------------------- #
# 2. Quintile-sorted forward returns
# --------------------------------------------------------------------------- #
def quintile_spread(
    df: pd.DataFrame,
    signal_col: str = "fly",
    target: str = "IEF",
    horizon: int = 21,
    window: int = 252,
    lags: int | None = None,
) -> dict:
    """Q5 (highest signal) − Q1 (lowest signal) mean forward-return spread, HAC *t*.

    The signal is lagged one day (known at close *t-1*); the quintile label buckets the
    day; the forward return runs close *t* → *t+horizon*.
    """
    q = quintile(df[signal_col], window).shift(1)
    fwd = forward_return(df[f"{target}_close"], horizon)
    qv = q.to_numpy(dtype=float)
    fv = fwd.to_numpy(dtype=float)
    ok = np.isfinite(fv)
    r5 = fv[(qv == 5) & ok]
    r1 = fv[(qv == 1) & ok]
    L = lags if lags is not None else horizon
    return {
        "target": target, "signal": signal_col, "horizon": horizon,
        "n5": int(r5.size), "n1": int(r1.size),
        "q5_bps": float(r5.mean() * 1e4) if r5.size else float("nan"),
        "q1_bps": float(r1.mean() * 1e4) if r1.size else float("nan"),
        "spread_bps": float((r5.mean() - r1.mean()) * 1e4) if (r5.size and r1.size) else float("nan"),
        "t_spread": newey_west_t(np.concatenate([r5, -r1]), L) if (r5.size and r1.size) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# 3. Costed bond-timing overlay
# --------------------------------------------------------------------------- #
def timing_overlay(
    df: pd.DataFrame,
    signal_col: str = "fly",
    target: str = "IEF",
    threshold: float = 0.5,
    cost_bps: float = 2.0,
    window: int = 252,
) -> dict:
    """Own ``target`` when the lagged butterfly rank exceeds ``threshold``, else cash.

    A one-way ``cost_bps`` is charged on the traded NAV at each regime switch. Returns
    active/passive/spread mean daily returns, HAC *t* on the spread, annualised net
    Sharpe of the timer vs buy-and-hold, and the switch count.
    """
    ret = df[f"{target}_ret"]
    rank = rolling_rank(df[signal_col], window)
    signal = (rank > threshold).shift(1)  # known at close t-1
    sig = signal.astype("boolean")
    switch = sig.ne(sig.shift(1)) & sig.notna() & sig.shift(1).notna()

    r_passive = ret
    r_active = ret.where(sig.fillna(False), 0.0) - switch.astype(float) * cost_bps * 1e-4
    out = pd.DataFrame({
        "r_passive": r_passive, "r_active": r_active,
        "r_spread": r_active - r_passive, "switched": switch,
    }).dropna(subset=["r_passive", "r_active"])

    sp = out["r_spread"].to_numpy(dtype=float)
    ra = out["r_active"].to_numpy(dtype=float)
    rp = out["r_passive"].to_numpy(dtype=float)

    def _sharpe(r):
        s = r.std(ddof=1)
        return float(r.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else float("nan")

    yrs = len(out) / TRADING_DAYS
    return {
        "target": target, "signal": signal_col,
        "n": int(len(out)),
        "active_bps": float(ra.mean() * 1e4),
        "passive_bps": float(rp.mean() * 1e4),
        "spread_bps": float(sp.mean() * 1e4),
        "t_spread": one_sample_t(sp),
        "sharpe_active": _sharpe(ra),
        "sharpe_passive": _sharpe(rp),
        "switches": int(out["switched"].sum()),
        "switches_per_yr": float(out["switched"].sum() / yrs) if yrs > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# 4. Permutation placebo — is the regression t a lucky alignment?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    df: pd.DataFrame,
    signal_col: str = "fly",
    target: str = "IEF",
    horizon: int = 21,
    window: int = 252,
    n_perm: int = 500,
    base_seed: int = 864,
) -> dict:
    """Shuffle the (lagged) signal against forward returns and re-fit the univariate
    HAC regression each time. p = share of permuted |t| >= observed |t| (two-sided)."""
    close = df[f"{target}_close"]
    fwd = forward_return(close, horizon).to_numpy(dtype=float)
    sig = zscore(df[signal_col], window).shift(1).to_numpy(dtype=float)
    mask = np.isfinite(fwd) & np.isfinite(sig)
    y = fwd[mask]; s = sig[mask]
    obs = nw_regression(y, s, lags=horizon)
    obs_t = float(obs["t"][1])

    rng = np.random.default_rng(base_seed)
    ts = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(s))
        ts[i] = nw_regression(y, s[perm], lags=horizon)["t"][1]
    p = float((np.abs(ts) >= abs(obs_t)).mean())
    return {
        "obs_t": obs_t, "perm_mean_t": float(ts.mean()), "perm_sd_t": float(ts.std(ddof=1)),
        "p_value": p, "n_perm": n_perm,
    }


# --------------------------------------------------------------------------- #
# 5. Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame, horizon: int = 21) -> dict:
    """Run the headline butterfly regression + Q5-Q1 sort on a synthetic tape."""
    reg = predictive_regression(df, "fly", "IEF", horizon)
    qs = quintile_spread(df, "fly", "IEF", horizon)
    return {
        "beta_bps": reg["beta_bps"], "t": reg["t"], "r2": reg["r2"],
        "spread_bps": qs["spread_bps"], "t_spread": qs["t_spread"],
        "n": reg["n"],
    }
