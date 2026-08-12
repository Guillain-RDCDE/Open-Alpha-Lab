"""Strategy + inference for Study 880 — Aggregate Short Interest.

The claim (Rapach, Ringgenberg & Zhou 2016): the **market-wide** short-interest
index predicts the aggregate equity return with a **negative** slope — a high,
detrended aggregate short-interest reading forecasts **lower** forward market
returns, "arguably the strongest known predictor" of the market.

Pipeline:

* **The index.** ``si_index`` is the equal-weight cross-sectional mean
  days-to-cover across the panel on each bi-monthly settlement date (built in
  ``data.aggregate_index``).
* **Detrend (the RRZ step).** Raw aggregate short interest carries a strong
  low-frequency trend; the predictor is the **detrended log index** — the
  residual of ``log(si_index)`` on a linear time trend (:func:`detrend_log`) —
  standardised. High ``sii`` = unusually crowded shorts *relative to trend*.
* **Forward return, one documented lag.** A settlement-date-``t`` short-interest
  print is published ~8 business days later, so it is acted on at the **next**
  settlement (``lag=1`` period); the forward SPY return is measured over the next
  ``horizon`` settlement periods starting from ``t+lag`` (no look-ahead).
* **Predictive regression.** OLS of the forward return on ``sii`` with
  **Newey-West (HAC)** standard errors on the slope (overlapping horizons are
  serially correlated), the R^2, an era cut, a sign-permutation placebo, and a
  costed timing overlay. Honest stamp from the real slope sign & significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Index construction — detrended log short-interest ("SII")
# --------------------------------------------------------------------------- #
def detrend_log(index: pd.Series) -> pd.Series:
    """Residual of ``log(index)`` on a linear time trend, standardised (RRZ detrend).

    Returns a z-scored series: the detrended, unit-variance aggregate
    short-interest index. High => crowded shorts relative to the secular trend.
    """
    y = np.log(index.to_numpy(dtype=float))
    n = len(y)
    t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sd = resid.std(ddof=0)
    # Guard: a (near-)perfectly-linear log index has ~0 residual variance; dividing
    # would amplify float noise, so return a flat zero series in that degenerate case.
    scale = max(abs(y).mean(), 1.0)
    z = resid / sd if sd > 1e-10 * scale else np.zeros_like(resid)
    return pd.Series(z, index=index.index, name="sii")


# --------------------------------------------------------------------------- #
# Align the index to SPY forward returns on the settlement grid
# --------------------------------------------------------------------------- #
def _spy_on_grid(spy: pd.Series, grid) -> pd.Series:
    """SPY close on the nearest trading day on/before each settlement date."""
    s = spy.sort_index()
    pos = s.index.searchsorted(pd.DatetimeIndex(grid), side="right") - 1
    pos = np.clip(pos, 0, len(s) - 1)
    return pd.Series(s.to_numpy()[pos], index=pd.DatetimeIndex(grid))


def build_frame(real: dict, horizon: int = 1, lag: int = 1,
                min_names: int = 20) -> pd.DataFrame:
    """Regression frame: ``sii`` (detrended index at ``t``) vs forward SPY return.

    The forward log return runs from the settlement ``t+lag`` to ``t+lag+horizon``
    (``lag`` = the publication/execution lag in settlement periods). Rows where the
    horizon overruns the tape are dropped.
    """
    idx = real["index"]
    idx = idx[idx["n"] >= min_names]
    si = idx["si_index"].dropna()
    sii = detrend_log(si)

    px = _spy_on_grid(real["spy"], si.index)
    logp = np.log(px.to_numpy(dtype=float))
    n = len(si)
    fwd = np.full(n, np.nan)
    for t in range(n):
        a = t + lag
        b = t + lag + horizon
        if b < n:
            fwd[t] = logp[b] - logp[a]
    out = pd.DataFrame({"sii": sii.to_numpy(), "fwd": fwd, "si_index": si.to_numpy()},
                       index=si.index).dropna(subset=["fwd"])
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (house canon — see study 803)
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
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The predictive regression — forward return on the detrended index, NW slope t
# --------------------------------------------------------------------------- #
def _nw_cov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficient vector."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((k, k))
    Xu = X * u[:, None]
    S += Xu.T @ Xu
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv


def predictive_regression(frame: pd.DataFrame, nw_lags: int = 6) -> dict:
    """OLS of forward return on the detrended index, with a Newey-West slope t.

    Returns the slope (beta), its NW t, the R^2, the intercept, n, and the sign
    predicted by the claim (RRZ => beta < 0).
    """
    x = frame["sii"].to_numpy(dtype=float)
    y = frame["fwd"].to_numpy(dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return {"n": n, "beta": float("nan"), "t_nw": float("nan"),
                "r2": float("nan"), "alpha": float("nan")}
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    cov = _nw_cov(X, resid, nw_lags)
    se_slope = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    t_nw = beta[1] / se_slope if se_slope and se_slope > 0 else float("nan")
    return {"n": n, "alpha": float(beta[0]), "beta": float(beta[1]),
            "t_nw": float(t_nw), "r2": float(r2),
            "fwd_mean_bps": float(y.mean() * 1e4)}


def era_split(frame: pd.DataFrame, cut: str, nw_lags: int = 6) -> dict:
    """The predictive regression on the two eras split at ``cut`` (a date string)."""
    c = pd.Timestamp(cut)
    early = frame[frame.index < c]
    late = frame[frame.index >= c]
    return {"cut": cut,
            "early": predictive_regression(early, nw_lags),
            "late": predictive_regression(late, nw_lags)}


# --------------------------------------------------------------------------- #
# Placebo — is the slope a lucky alignment? Sign-flip / permutation null
# --------------------------------------------------------------------------- #
def placebo_pvalue(frame: pd.DataFrame, n_draws: int = 5000, seed: int = 880,
                   nw_lags: int = 6) -> dict:
    """Permute the forward-return vector against the index; how often is a permuted
    slope at least as **negative** as observed? (Left-tail test on RRZ's negative
    slope.) p = P[permuted beta <= observed beta]."""
    x = frame["sii"].to_numpy(dtype=float)
    y = frame["fwd"].to_numpy(dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return {"obs_beta": float("nan"), "p_value": float("nan"), "n_draws": 0}
    xc = x - x.mean()
    denom = float(xc @ xc)
    obs = float((xc @ (y - y.mean())) / denom)
    rng = np.random.default_rng(seed)
    betas = np.empty(n_draws)
    for i in range(n_draws):
        yp = y[rng.permutation(n)]
        betas[i] = float((xc @ (yp - yp.mean())) / denom)
    return {"obs_beta": obs,
            "placebo_mean": float(betas.mean()),
            "placebo_sd": float(betas.std(ddof=1)),
            "p_value": float((betas <= obs).mean()),
            "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Tradability — a market-timing overlay driven by the index
# --------------------------------------------------------------------------- #
def timer_stats(frame: pd.DataFrame, cost_bps: float = 5.0,
                thresh: float = 0.0) -> dict:
    """De-risk when shorts are crowded: hold SPY when ``sii <= thresh`` (shorts not
    crowded), sit in cash when ``sii > thresh``. One-way ``cost_bps`` per switch,
    charged on NAV. Positions use the same publication lag as ``build_frame`` (the
    forward return in ``frame`` already starts at ``t+lag``). Reports the overlay's
    per-period mean net of costs, its annualised figures, and the switch count."""
    x = frame["sii"].to_numpy(dtype=float)
    r = frame["fwd"].to_numpy(dtype=float)   # forward one-period return (if horizon=1)
    pos = (x <= thresh).astype(float)        # 1 = invested, 0 = cash
    switches = np.abs(np.diff(np.concatenate([[0.0], pos])))
    c = cost_bps / 1e4
    gross = pos * r
    net = gross - switches * c
    n = len(r)
    periods_per_yr = 24.0                     # bi-monthly settlement grid
    mu = float(net.mean()); sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    bh_mu = float(r.mean())
    return {
        "n": n, "n_switches": int((switches > 0).sum()),
        "bh_bps": bh_mu * 1e4,
        "overlay_gross_bps": float(gross.mean()) * 1e4,
        "overlay_net_bps": mu * 1e4,
        "overlay_ann_pct": mu * periods_per_yr * 100,
        "bh_ann_pct": bh_mu * periods_per_yr * 100,
        "overlay_sharpe": (mu / sd * np.sqrt(periods_per_yr)) if sd and sd > 0 else float("nan"),
        "cost_bps": cost_bps,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(real: dict, horizon: int = 1, lag: int = 1) -> dict:
    """Run the headline predictive regression on a synthetic frame."""
    fr = build_frame(real, horizon=horizon, lag=lag, min_names=1)
    reg = predictive_regression(fr)
    return {"beta": reg["beta"], "t_nw": reg["t_nw"], "r2": reg["r2"], "n": reg["n"]}
