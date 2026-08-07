"""Strategy + inference for Study 823 — Variance-Risk-Premium Return Predictor.

The claim (Bollerslev, Tauchen & Zhou 2009): the **variance risk premium**

    VRP_t = IV_t − RV_t   (implied variance − realized variance, monthly units)

**predicts the aggregate market's forward excess return** in a time-series predictive
regression — positive VRP → higher forward return — with the R² peaking near the
quarterly (3-month) horizon.

This is a *time-series predictive regression* on the index, distinct from:

* [130-vol-risk-premium](../../130-vol-risk-premium/) — whether the VRP *exists* / is
  positive on average (its level, harvested by shorting variance), not whether it
  *predicts the market's direction*;
* [111-vix-term-structure](../../111-vix-term-structure/) — the *shape* of the VIX
  futures curve (contango/backwardation), not the implied-minus-realized gap;
* [3-fear-gauge](../../3-fear-gauge/) — the VIX *level* as a contrarian dip signal, not
  the variance *risk premium*.

Method:

* **Monthly build.** From daily SPY (total-return) and ^VIX closes, resample to
  month-end. ``IV_t = (VIX_t/100)² / 12`` (monthly implied variance). ``RV_t`` =
  trailing-21-trading-day sum of daily squared log returns as of month-end t (monthly
  realized variance). ``VRP_t = IV_t − RV_t``.
* **Forward excess return.** ``fwd_h`` = the SPY log return over the next ``h`` months
  (t → t+h). The risk-free leg is proxied at zero and named on the Signal axis — a
  (near-constant) monthly bill shifts the regression *intercept*, not the *slope* that
  carries the t-stat.
* **Predictive regression.** OLS of ``fwd_h`` on ``VRP_t`` with a **Newey-West (HAC)**
  standard error on the slope (Bartlett kernel, lags scaled to the overlap ``h``); the
  claim predicts a positive slope. Report slope, HAC *t*, and R².
* **Placebo, robustness, and a costed timer** cross-check the headline; a seeded
  synthetic positive control proves the machinery is unbiased.

All heavy paths are vectorised numpy — no per-row pandas ``.loc`` loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS_YEAR = 12


# --------------------------------------------------------------------------- #
# Monthly build: IV, RV, VRP, forward returns
# --------------------------------------------------------------------------- #
def build_monthly(spy: pd.Series, vix: pd.Series, rv_window: int = 21) -> pd.DataFrame:
    """Assemble the month-end predictive frame from daily SPY + VIX closes.

    Columns (index = month-end date):

    * ``spy``  — month-end SPY close.
    * ``iv``   — implied variance ``(VIX/100)² / 12`` (monthly units).
    * ``rv``   — realized variance: trailing ``rv_window`` daily squared **log** returns,
      summed, as of month-end (monthly units).
    * ``vrp``  — ``iv − rv`` (the variance risk premium, monthly variance units).
    * ``ret``  — the month's own SPY log return (t−1 → t), for building forward returns.

    Everything on row ``t`` is known at the close of month-end ``t`` (the RV window and
    the VIX print are both trailing) — the forward returns are formed downstream.
    """
    spy = spy.sort_index().astype(float)
    vix = vix.sort_index().astype(float)
    logret = np.log(spy).diff()
    sq = logret ** 2
    # Trailing rv_window-day realized variance, evaluated on every day, then sampled M-E.
    rv_daily = sq.rolling(rv_window, min_periods=rv_window).sum()

    # Month-end sampling (label = last calendar day of month; values = last obs).
    me_spy = spy.resample("ME").last()
    me_vix = vix.resample("ME").last()
    me_rv = rv_daily.resample("ME").last()

    df = pd.DataFrame({"spy": me_spy, "vix": me_vix, "rv": me_rv}).dropna()
    df["iv"] = (df["vix"] / 100.0) ** 2 / MONTHS_YEAR
    df["vrp"] = df["iv"] - df["rv"]
    df["ret"] = np.log(df["spy"]).diff()
    return df.drop(columns="vix")


def forward_return(monthly: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Forward ``horizon``-month SPY log return aligned to the signal month ``t``.

    ``fwd_h[t]`` is the cumulative log return from month-end ``t`` to month-end
    ``t+horizon`` — i.e. the sum of the *next* ``horizon`` monthly returns. Uses only
    returns dated after ``t`` (the single one-month execution lag: the VRP is known at
    the close of month ``t``; the position is held over ``t → t+h``). No look-ahead.
    """
    r = monthly["ret"]
    # sum of ret[t+1..t+h], indexed at t
    fwd = r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return fwd.rename(f"fwd_{horizon}")


# --------------------------------------------------------------------------- #
# Inference primitives (shared house kit)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
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
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The predictive regression with a Newey-West slope t
# --------------------------------------------------------------------------- #
def predictive_regression(x: np.ndarray, y: np.ndarray, lags: int | None = None) -> dict:
    """OLS of ``y`` on ``x`` (with intercept) and a HAC (Newey-West) *t* on the slope.

    Vectorised: builds the 2-column design ``[1, x]``, solves least squares, then forms
    the Newey-West sandwich ``(X'X)⁻¹ · S · (X'X)⁻¹`` with
    ``S = Σ g g' + Σ_l w_l (Γ_l + Γ_l')`` and score ``g_t = X_t · u_t`` — the standard
    HAC covariance for a regression slope with an overlapping (serially correlated)
    dependent variable. Returns ``slope``, ``alpha``, ``t_nw`` (HAC t on the slope),
    ``t_ols`` (homoskedastic t), ``r2`` and ``n``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 10:
        return {"slope": float("nan"), "alpha": float("nan"), "t_nw": float("nan"),
                "t_ols": float("nan"), "r2": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float((resid ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")

    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West 1994 rule
        lags = max(lags, 1)
    g = X * resid[:, None]                      # score, n×2
    S = g.T @ g
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        Gl = g[l:].T @ g[:-l]
        S += w * (Gl + Gl.T)
    V = XtX_inv @ S @ XtX_inv                    # HAC covariance of beta
    se_nw = float(np.sqrt(max(V[1, 1], 0.0)))
    # Homoskedastic (OLS) slope se, for cross-reference.
    sigma2 = ssr / (n - 2)
    se_ols = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    return {
        "slope": float(beta[1]),
        "alpha": float(beta[0]),
        "t_nw": float(beta[1] / se_nw) if se_nw > 0 else float("nan"),
        "t_ols": float(beta[1] / se_ols) if se_ols > 0 else float("nan"),
        "r2": r2,
        "n": n,
    }


def _nw_lags_for_horizon(horizon: int) -> int:
    """HAC lag length for an ``horizon``-month overlapping forward return (⌈1.5·h⌉)."""
    return max(int(np.ceil(1.5 * horizon)), 1)


def headline(monthly: pd.DataFrame, horizon: int = 1) -> dict:
    """The predictive regression of the forward ``horizon``-month return on VRP_t.

    Returns the slope (per unit of monthly variance and rescaled per volatility-point),
    the HAC *t*, R², the in-sample forward-return spread between the top and bottom VRP
    terciles (a model-free read of the same relation), and n.
    """
    fwd = forward_return(monthly, horizon)
    d = pd.DataFrame({"vrp": monthly["vrp"], "fwd": fwd}).dropna()
    x = d["vrp"].to_numpy()
    y = d["fwd"].to_numpy()
    reg = predictive_regression(x, y, lags=_nw_lags_for_horizon(horizon))

    # Model-free tercile read: mean forward return of the high-VRP third minus the low.
    order = np.argsort(x, kind="stable")
    k = max(1, len(x) // 3)
    lo_mean = float(np.mean(y[order[:k]]))
    hi_mean = float(np.mean(y[order[-k:]]))
    return {
        "horizon": horizon,
        "slope": reg["slope"],
        "t_nw": reg["t_nw"],
        "t_ols": reg["t_ols"],
        "r2": reg["r2"],
        "n": reg["n"],
        "hi_tercile_pct": hi_mean * 100.0,
        "lo_tercile_pct": lo_mean * 100.0,
        "tercile_spread_pct": (hi_mean - lo_mean) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Placebo — is the slope real, or a lucky alignment?
# --------------------------------------------------------------------------- #
def placebo_pvalue(monthly: pd.DataFrame, horizon: int = 1, n_perm: int = 2000,
                   block: int = 6, seed: int = 823) -> dict:
    """Block-bootstrap placebo for the predictive slope.

    Circularly rotate the forward-return series against the VRP in blocks of ``block``
    months ``n_perm`` times (block rotation preserves the overlap-induced
    autocorrelation of the forward return, so the null is honest). ``p`` = the share of
    rotations whose slope is ≥ the observed slope (a right-tail test — the claim is a
    *positive* slope). Fully vectorised: one design matrix, ``n_perm`` rolled y-vectors.
    """
    fwd = forward_return(monthly, horizon)
    d = pd.DataFrame({"vrp": monthly["vrp"], "fwd": fwd}).dropna()
    x = d["vrp"].to_numpy(dtype=float)
    y = d["fwd"].to_numpy(dtype=float)
    n = len(x)
    if n < 30:
        return {"obs_slope": float("nan"), "p_value": float("nan"), "n_perm": 0,
                "placebo_mean": float("nan"), "placebo_sd": float("nan")}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    proj = XtX_inv @ X.T                          # 2×n; slope = proj[1] @ y
    slope_row = proj[1]
    obs = float(slope_row @ y)

    rng = np.random.default_rng(seed)
    shifts = rng.integers(block, n - block, size=n_perm)
    # Build rolled y-matrix (n_perm × n) via index gather, then one matmul.
    base = np.arange(n)
    idx = (base[None, :] - shifts[:, None]) % n   # np.roll(y, shift) == y[(i-shift)%n]
    Y = y[idx]                                     # n_perm × n
    slopes = Y @ slope_row                         # n_perm
    return {
        "obs_slope": obs,
        "placebo_mean": float(slopes.mean()),
        "placebo_sd": float(slopes.std(ddof=1)),
        "p_value": float((slopes >= obs).mean()),
        "n_perm": int(n_perm),
    }


# --------------------------------------------------------------------------- #
# Robustness — horizon sweep and two-era cut
# --------------------------------------------------------------------------- #
def horizon_sweep(monthly: pd.DataFrame, horizons=(1, 3, 6, 12)) -> pd.DataFrame:
    """Predictive slope / HAC t / R² across forward horizons (months)."""
    rows = []
    for h in horizons:
        s = headline(monthly, horizon=h)
        rows.append((f"{h}m", s["slope"], s["t_nw"], s["r2"], s["n"]))
    return pd.DataFrame(rows, columns=["horizon", "slope", "t_nw", "r2", "n"]).set_index("horizon")


def era_cut(monthly: pd.DataFrame, split: str = "2010-01-01", horizon: int = 3) -> pd.DataFrame:
    """Predictive regression within two eras split at ``split``."""
    rows = []
    for lab, lo, hi in [("early", "1900-01-01", split), ("late", split, "2100-01-01")]:
        sub = monthly[(monthly.index >= lo) & (monthly.index < hi)]
        s = headline(sub, horizon=horizon)
        rows.append((lab, s["slope"], s["t_nw"], s["r2"], s["n"]))
    return pd.DataFrame(rows, columns=["era", "slope", "t_nw", "r2", "n"]).set_index("era")


# --------------------------------------------------------------------------- #
# The costed timer — can you get paid for it?
# --------------------------------------------------------------------------- #
def timer_stats(monthly: pd.DataFrame, horizon: int = 1, cost_bps: float = 5.0) -> dict:
    """Long-SPY / flat market-timer conditioned on the VRP sign vs its median.

    Each month, hold SPY over ``t → t+1`` when VRP_t is above its expanding-window median
    (fat premium ⇒ own the market), else sit in cash; charge ``cost_bps`` one-way × NAV
    on each switch. Compares the timer's net annualised Sharpe and mean return to plain
    buy-and-hold SPY. (Long-only: the VRP signal is a directional market-timer, not a
    long-short book, so there is no borrow leg.)
    """
    vrp = monthly["vrp"]
    ret = monthly["ret"]
    med = vrp.expanding(min_periods=12).median()
    pos = (vrp.shift(1) > med.shift(1)).astype(float)   # signal known at t−1, hold t
    d = pd.DataFrame({"pos": pos, "ret": ret}).dropna()
    if len(d) < 24:
        return {}
    switches = d["pos"].diff().abs().fillna(0.0)
    cost = switches * cost_bps * 1e-4
    timer = d["pos"] * d["ret"] - cost
    bh = d["ret"]

    def sharpe(x):
        s = x.std(ddof=1)
        return float(x.mean() / s * np.sqrt(MONTHS_YEAR)) if s > 0 else float("nan")

    diff = (timer - bh).to_numpy(dtype=float)
    return {
        "timer_sharpe": sharpe(timer),
        "bh_sharpe": sharpe(bh),
        "switches_per_yr": float(switches.sum() / len(d) * MONTHS_YEAR),
        "invested_frac": float(d["pos"].mean()),
        "spread_pct_mo": float(diff.mean() * 100.0),
        "spread_t": newey_west_t(diff, lags=6),
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (seed-robust; the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, horizon: int = 3) -> dict:
    """Run the headline predictive regression on a synthetic SPY+VIX panel."""
    monthly = build_monthly(panel["SPY"], panel["VIX"])
    return headline(monthly, horizon=horizon)


def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 20, base_seed: int = 823,
                     horizon: int = 3) -> dict:
    """Average the predictive slope / HAC t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the statistic over ≥ 20 seeds so
    no single lucky RNG seed can manufacture significance.
    """
    slopes, ts, r2s = [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel = data_mod.synthetic_daily(edge=edge, seed=s)
        out = synthetic_detect(panel, horizon=horizon)
        slopes.append(out["slope"]); ts.append(out["t_nw"]); r2s.append(out["r2"])
    return {
        "edge": edge,
        "mean_slope": float(np.nanmean(slopes)),
        "mean_t": float(np.nanmean(ts)),
        "mean_r2": float(np.nanmean(r2s)),
        "fire_frac": float(np.mean(np.abs(ts) >= 2.0)),
        "n_seeds": n_seeds,
    }
