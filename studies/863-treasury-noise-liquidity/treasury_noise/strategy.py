"""Strategy + inference for Study 863 — Treasury Noise Liquidity.

The claim (Hu, Pan & Wang 2013): the Treasury yield curve's **cross-maturity roughness**
is a market-wide illiquidity gauge. Build a daily **noise** = RMS deviation of the four
CMT yields (13w / 5y / 10y / 30y) from a smooth (quadratic-in-maturity) fit; high noise
flags funding stress and is said to **precede lower equity returns and wider credit
spreads**. We test that as a *time-series predictive regression* of the forward SPY and
forward HYG−IEF (credit-excess) returns on the noise level.

This is distinct from the desk's other rates-stress studies:

* [112-move-index](../../112-move-index/) — **MOVE**, the option-implied *volatility* of
  Treasuries (a forward-looking vol gauge), not the realized cross-maturity *roughness* of
  the yield levels themselves;
* [383-sofr-repo-stress](../../383-sofr-repo-stress/) — discrete **repo/SOFR funding
  spikes** (a short-rate money-market dislocation), not a whole-curve fitting residual;
* [386-nfci-conditions](../../386-nfci-conditions/) — a broad **financial-conditions
  index** proxy blending vol/credit/dollar, not a Treasury-only curve residual;
* [581-term-premium](../../581-term-premium/) — the **level** of the term premium (the
  compensation embedded in the curve's *shape*), not the *deviation from* a smooth shape.

Method:

* **Noise (roughness).** With four CMT yields at fixed maturities, fit
  ``y ≈ a + b·m + c·m²`` by OLS each day and take ``noise_t = RMS(residual)``. Because the
  maturities are fixed, the residual-projection matrix ``P⊥ = I − M(MᵀM)⁻¹Mᵀ`` is
  constant, so the whole series is one matmul (no per-day loop).
* **Targets.** Forward ``h``-day cumulative log return of **SPY** (equity) and of
  **HYG − IEF** (daily high-yield-minus-Treasury credit-excess return, cumulated) — the
  two assets the noise is said to warn.
* **Predictive regression.** OLS of the forward return on the *standardised* noise known
  at the close of day ``t`` (held ``t → t+h``, one lag, zero look-ahead), with a
  **Newey-West (HAC)** slope *t* (lags scaled to the overlap ``h``). The claim predicts a
  **negative** slope (high noise → lower forward return / wider credit).
* **Placebo, era cut, and a costed regime timer** cross-check the headline; a seeded
  synthetic positive control proves the machinery is unbiased.

All heavy paths are vectorised numpy — no per-row pandas ``.loc`` loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MATURITIES, YIELD_COLS

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The noise (roughness) construction — one constant projection matrix
# --------------------------------------------------------------------------- #
def _perp_matrix(maturities: np.ndarray = MATURITIES) -> np.ndarray:
    """Residual-projection matrix ``P⊥ = I − M(MᵀM)⁻¹Mᵀ`` for a quadratic-in-maturity fit.

    ``M`` is the ``(k, 3)`` design ``[1, m, m²]``. ``P⊥`` is symmetric idempotent; for any
    yield vector ``y``, ``P⊥ y`` is the vector of residuals from the least-squares quadratic
    fit. Fixed maturities ⇒ ``P⊥`` is constant ⇒ the whole daily series is one matmul.
    """
    m = np.asarray(maturities, dtype=float)
    M = np.column_stack([np.ones_like(m), m, m ** 2])
    P = M @ np.linalg.inv(M.T @ M) @ M.T
    return np.eye(len(m)) - P


def noise_series(yields: pd.DataFrame) -> pd.Series:
    """Daily roughness = RMS deviation of the four yields from a quadratic fit.

    ``yields`` has the four CMT columns (``YIELD_COLS``) in maturity order. Returns the
    ``noise_t`` series in the yields' units (percentage points). Vectorised: residuals
    ``R = Y · P⊥`` (P⊥ symmetric), ``noise = sqrt(mean(R², axis=1))``.
    """
    Y = yields[YIELD_COLS].to_numpy(dtype=float)
    resid = Y @ _perp_matrix()                       # (T, k) residuals
    rms = np.sqrt(np.mean(resid ** 2, axis=1))
    return pd.Series(rms, index=yields.index, name="noise")


def build_daily(panel: pd.DataFrame) -> pd.DataFrame:
    """Assemble the daily predictive frame from the cached/synthetic panel.

    Columns (index = trading day):

    * ``noise``      — curve roughness (RMS residual of the quadratic fit), % points.
    * ``noise_z``    — full-sample z-score of ``noise`` (for a per-1σ slope reading).
    * ``ret_spy``    — daily SPY log return.
    * ``ret_credit`` — daily HYG − IEF log-return difference (credit-excess return).

    Everything on row ``t`` is known at the close of day ``t``; the forward returns are
    formed downstream by :func:`forward_return` with a strict one-day-plus lag.
    """
    df = panel.sort_index()
    noise = noise_series(df)
    ret_spy = np.log(df["SPY"].astype(float)).diff()
    ret_credit = np.log(df["HYG"].astype(float)).diff() - np.log(df["IEF"].astype(float)).diff()
    out = pd.DataFrame(
        {"noise": noise, "ret_spy": ret_spy, "ret_credit": ret_credit}
    ).dropna()
    mu, sd = out["noise"].mean(), out["noise"].std(ddof=0)
    out["noise_z"] = (out["noise"] - mu) / sd if sd > 0 else 0.0
    return out


def forward_return(daily: pd.DataFrame, target: str = "ret_spy", horizon: int = 21) -> pd.Series:
    """Forward ``horizon``-day cumulative log return of ``target`` aligned to signal day ``t``.

    ``fwd[t]`` sums ``target`` over days ``t+1 … t+horizon`` (strictly future): the noise is
    known at the close of ``t``, the position is held ``t → t+h``. One documented lag, no
    look-ahead. Uses ``ret_spy`` or ``ret_credit``.
    """
    r = daily[target]
    fwd = r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return fwd.rename(f"fwd_{target}_{horizon}")


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

    Builds the 2-column design ``[1, x]``, solves least squares, then forms the
    Newey-West sandwich ``(XᵀX)⁻¹ · S · (XᵀX)⁻¹`` with score ``g_t = X_t · u_t`` — the
    standard HAC covariance for a slope with an overlapping (serially correlated) dependent
    variable. Returns ``slope``, ``alpha``, ``t_nw`` (HAC t on the slope), ``t_ols``
    (homoskedastic t), ``r2`` and ``n``.
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
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float((resid ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")

    if lags is None:
        lags = max(int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))), 1)  # Newey-West 1994
    g = X * resid[:, None]
    S = g.T @ g
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        Gl = g[l:].T @ g[:-l]
        S += w * (Gl + Gl.T)
    V = XtX_inv @ S @ XtX_inv
    se_nw = float(np.sqrt(max(V[1, 1], 0.0)))
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
    """HAC lag length for an ``horizon``-day overlapping forward return (⌈1.5·h⌉)."""
    return max(int(np.ceil(1.5 * horizon)), 1)


# --------------------------------------------------------------------------- #
# Headline: predictive regression of forward return on noise
# --------------------------------------------------------------------------- #
def headline(daily: pd.DataFrame, target: str = "ret_spy", horizon: int = 21) -> dict:
    """Predict the forward ``horizon``-day ``target`` return from the noise level.

    Regresses the forward return on the standardised noise (``noise_z``, per-1σ slope) with
    a HAC *t*; also reports a model-free tercile read (mean forward return of the
    **high-noise** third minus the **low-noise** third). The claim predicts a **negative**
    slope and a negative high−low spread. Slopes/spreads are reported in percent.
    """
    fwd = forward_return(daily, target, horizon)
    d = pd.DataFrame({"noise_z": daily["noise_z"], "fwd": fwd}).dropna()
    x = d["noise_z"].to_numpy()
    y = d["fwd"].to_numpy()
    reg = predictive_regression(x, y, lags=_nw_lags_for_horizon(horizon))

    order = np.argsort(x, kind="stable")
    k = max(1, len(x) // 3)
    lo_mean = float(np.mean(y[order[:k]]))     # low-noise third
    hi_mean = float(np.mean(y[order[-k:]]))    # high-noise third
    return {
        "target": target,
        "horizon": horizon,
        "slope_pct": reg["slope"] * 100.0,
        "t_nw": reg["t_nw"],
        "t_ols": reg["t_ols"],
        "r2": reg["r2"],
        "n": reg["n"],
        "hi_noise_pct": hi_mean * 100.0,
        "lo_noise_pct": lo_mean * 100.0,
        "hi_minus_lo_pct": (hi_mean - lo_mean) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Placebo — is the slope real, or a lucky alignment?
# --------------------------------------------------------------------------- #
def placebo_pvalue(daily: pd.DataFrame, target: str = "ret_spy", horizon: int = 21,
                   n_perm: int = 2000, block: int = 21, seed: int = 863) -> dict:
    """Block-rotation placebo for the predictive slope.

    Circularly rotate the forward-return series against the noise in blocks of ``block``
    days ``n_perm`` times (block rotation preserves the overlap-induced autocorrelation of
    the forward return, so the null is honest). Because the claim is a **negative** slope,
    ``p`` = the share of rotations whose slope is **≤** the observed slope (left-tail test).
    Fully vectorised: one design matrix, ``n_perm`` rolled y-vectors.
    """
    fwd = forward_return(daily, target, horizon)
    d = pd.DataFrame({"noise_z": daily["noise_z"], "fwd": fwd}).dropna()
    x = d["noise_z"].to_numpy(dtype=float)
    y = d["fwd"].to_numpy(dtype=float)
    n = len(x)
    if n < 30:
        return {"obs_slope": float("nan"), "p_value": float("nan"), "n_perm": 0,
                "placebo_mean": float("nan"), "placebo_sd": float("nan")}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    slope_row = (XtX_inv @ X.T)[1]                # slope = slope_row @ y
    obs = float(slope_row @ y)

    rng = np.random.default_rng(seed)
    shifts = rng.integers(block, n - block, size=n_perm)
    base = np.arange(n)
    idx = (base[None, :] - shifts[:, None]) % n
    slopes = y[idx] @ slope_row
    return {
        "obs_slope": obs,
        "placebo_mean": float(slopes.mean()),
        "placebo_sd": float(slopes.std(ddof=1)),
        "p_value": float((slopes <= obs).mean()),   # left tail (claim: negative slope)
        "n_perm": int(n_perm),
    }


# --------------------------------------------------------------------------- #
# Robustness — era cut
# --------------------------------------------------------------------------- #
def era_cut(daily: pd.DataFrame, target: str = "ret_spy", horizon: int = 21,
            split: str = "2016-01-01") -> pd.DataFrame:
    """Predictive regression within two eras split at ``split``."""
    rows = []
    for lab, lo, hi in [("early", "1900-01-01", split), ("late", split, "2100-01-01")]:
        sub = daily[(daily.index >= lo) & (daily.index < hi)]
        s = headline(sub, target=target, horizon=horizon)
        rows.append((lab, s["slope_pct"], s["t_nw"], s["r2"], s["n"]))
    return pd.DataFrame(rows, columns=["era", "slope_pct", "t_nw", "r2", "n"]).set_index("era")


def horizon_sweep(daily: pd.DataFrame, target: str = "ret_spy",
                  horizons=(5, 21, 63)) -> pd.DataFrame:
    """Predictive slope / HAC t / R² across forward horizons (trading days)."""
    rows = []
    for h in horizons:
        s = headline(daily, target=target, horizon=h)
        rows.append((f"{h}d", s["slope_pct"], s["t_nw"], s["r2"],
                     s["hi_minus_lo_pct"], s["n"]))
    return pd.DataFrame(
        rows, columns=["horizon", "slope_pct", "t_nw", "r2", "hi_minus_lo_pct", "n"]
    ).set_index("horizon")


# --------------------------------------------------------------------------- #
# The costed timer — can you get paid for it?
# --------------------------------------------------------------------------- #
def timer_stats(daily: pd.DataFrame, cost_bps: float = 3.0) -> dict:
    """Long-SPY / flat regime timer conditioned on the noise level vs its expanding median.

    Each day, hold SPY over ``t → t+1`` when the noise known at ``t−1`` is **below** its
    expanding-window median (calm curve ⇒ own the market; the claim says high noise
    precedes weakness), else sit in cash; charge ``cost_bps`` one-way × NAV on each switch.
    Compares the timer's net annualised Sharpe and mean return to plain buy-and-hold SPY.
    Long-only directional timer — no borrow leg.
    """
    noise = daily["noise"]
    ret = daily["ret_spy"]
    med = noise.expanding(min_periods=60).median()
    pos = (noise.shift(1) < med.shift(1)).astype(float)     # signal at t−1, hold t
    d = pd.DataFrame({"pos": pos, "ret": ret}).dropna()
    if len(d) < 60:
        return {}
    switches = d["pos"].diff().abs().fillna(0.0)
    cost = switches * cost_bps * 1e-4
    timer = d["pos"] * d["ret"] - cost
    bh = d["ret"]

    def sharpe(v):
        s = v.std(ddof=1)
        return float(v.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else float("nan")

    diff = (timer - bh).to_numpy(dtype=float)
    return {
        "timer_sharpe": sharpe(timer),
        "bh_sharpe": sharpe(bh),
        "switches_per_yr": float(switches.sum() / len(d) * TRADING_DAYS),
        "invested_frac": float(d["pos"].mean()),
        "spread_bps_day": float(diff.mean() * 1e4),
        "spread_t": newey_west_t(diff, lags=10),
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (seed-robust; the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, target: str = "ret_spy", horizon: int = 21) -> dict:
    """Run the headline predictive regression on a synthetic panel."""
    daily = build_daily(panel)
    return headline(daily, target=target, horizon=horizon)


def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 20, base_seed: int = 863,
                     target: str = "ret_spy", horizon: int = 21) -> dict:
    """Average the predictive slope / HAC t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the statistic over ≥ 20 seeds so no
    single lucky RNG seed can manufacture significance.
    """
    slopes, ts, r2s = [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel = data_mod.synthetic_daily(edge=edge, seed=s)
        out = synthetic_detect(panel, target=target, horizon=horizon)
        slopes.append(out["slope_pct"]); ts.append(out["t_nw"]); r2s.append(out["r2"])
    ts = np.asarray(ts)
    return {
        "edge": edge,
        "mean_slope_pct": float(np.nanmean(slopes)),
        "mean_t": float(np.nanmean(ts)),
        "mean_r2": float(np.nanmean(r2s)),
        "fire_frac": float(np.mean(np.abs(ts) >= 2.0)),
        "n_seeds": n_seeds,
    }
