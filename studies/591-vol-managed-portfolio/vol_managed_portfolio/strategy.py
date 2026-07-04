"""Strategy + stats for Study 591 — Vol-Managed Portfolio (Moreira & Muir, JF 2017).

The tested rule, exactly one execution lag:

    at the close of the LAST trading day of month m, compute RV_m = sum of squared
    daily returns of month m; hold, for the WHOLE of month m+1, a weight

        w_{m+1} = min( cap,  c_m / RV_m ),        cap = 1.5

    where c_m is the EXPANDING mean of monthly RV up to and including month m (a
    past-only normaliser that targets an average weight near 1 — no full-sample
    look-ahead, unlike the paper's ex-post variance-matching constant).

Everything is excess-vs-excess: the managed leg earns ``w * excess`` (leverage above
1 is financed at the same T-bill rate; a retail borrow spread is charged separately in
the cost sweep). Inference is a Newey-West (HAC) regression of managed-on-unmanaged
monthly excess returns — the Moreira-Muir alpha — plus the appraisal ratio, an
excess-vs-excess Sharpe race, crash-window drawdowns, and a shuffled-signal placebo
(>= 20 seeds, averaged — single-seed baselines are banned on this desk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12
CAP = 1.5
BURN_IN = 12          # months of RV history before the first weight is formed


# --------------------------------------------------------------------------- #
# Monthly aggregation (works on plain arrays -> shared by real tape & synthetic)
# --------------------------------------------------------------------------- #
def monthly_from_daily(exc: np.ndarray, raw: np.ndarray,
                       month_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compound monthly excess returns + monthly realized variance (sum of squared
    RAW daily returns), ordered by month. ``month_id`` must be sorted."""
    months, idx = np.unique(month_id, return_index=True)
    order = np.argsort(idx)
    months = months[order]
    exc_m = np.empty(len(months))
    rv_m = np.empty(len(months))
    for i, m in enumerate(months):
        sel = month_id == m
        exc_m[i] = float(np.prod(1.0 + exc[sel]) - 1.0)
        rv_m[i] = float(np.sum(raw[sel] ** 2))
    return exc_m, rv_m


def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Real-tape monthly table from a daily frame with columns ret/rf/exc/month.

    Columns: ``exc`` (compounded monthly excess), ``rf`` (compounded monthly rf),
    ``rv`` (sum of squared raw daily returns), indexed by Period[M].
    """
    g = df.groupby("month")
    out = pd.DataFrame({
        "exc": g["exc"].apply(lambda s: float(np.prod(1.0 + s) - 1.0)),
        "rf": g["rf"].apply(lambda s: float(np.prod(1.0 + s) - 1.0)),
        "rv": g["ret"].apply(lambda s: float(np.sum(np.asarray(s) ** 2))),
    })
    return out.sort_index()


# --------------------------------------------------------------------------- #
# The vol-managed weight rule (past-only, one clean lag)
# --------------------------------------------------------------------------- #
def weights_from_rv(rv: np.ndarray, cap: float = CAP,
                    burn_in: int = BURN_IN) -> np.ndarray:
    """w[t] applies to month t and uses ONLY months <= t-1 (RV of t-1, expanding mean
    of RV up to t-1). First ``burn_in`` + 1 entries are NaN (no history)."""
    rv = np.asarray(rv, dtype=float)
    n = len(rv)
    w = np.full(n, np.nan)
    csum = np.cumsum(rv)
    for t in range(burn_in + 1, n):
        c = csum[t - 1] / t                    # expanding mean of rv[0..t-1]
        w[t] = min(cap, c / rv[t - 1])
    return w


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_alpha(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS ``y = a + b x`` with Newey-West (Bartlett) HAC standard errors.

    Returns annualised alpha (x12, in %), HAC t of alpha, beta, and the appraisal
    ratio ``alpha / sd(resid) * sqrt(12)``.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if lags is None:
        lags = nw_lags(n)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    # HAC covariance: (X'X)^-1 [ sum_k w_k Gamma_k ] (X'X)^-1
    Xu = X * e[:, None]
    S = Xu.T @ Xu
    for k in range(1, lags + 1):
        wgt = 1.0 - k / (lags + 1.0)
        G = Xu[k:].T @ Xu[:-k]
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se_a = float(np.sqrt(max(V[0, 0], 0.0)))
    resid_sd = float(e.std(ddof=2))
    alpha_m = float(b[0])
    return {
        "alpha_m": alpha_m,
        "alpha_ann_pct": alpha_m * MONTHS * 100.0,
        "t_alpha": alpha_m / se_a if se_a > 0 else np.nan,
        "beta": float(b[1]),
        "appraisal": alpha_m / resid_sd * np.sqrt(MONTHS) if resid_sd > 0 else np.nan,
        "resid_vol_ann_pct": resid_sd * np.sqrt(MONTHS) * 100.0,
        "n": n, "lags": lags,
    }


def sharpe_stats(exc_m: np.ndarray) -> dict:
    """Annualised Sharpe of monthly EXCESS returns + HAC t of the mean."""
    r = np.asarray(exc_m, float)
    r = r[np.isfinite(r)]
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    lags = nw_lags(n)
    e = r - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * (float(e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "sharpe_ann": mu / sd * np.sqrt(MONTHS) if sd > 0 else np.nan,
        "mean_ann_pct": mu * MONTHS * 100.0,
        "vol_ann_pct": sd * np.sqrt(MONTHS) * 100.0,
        "t_mean_hac": mu / se if se > 0 else np.nan,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# The race (gross + net), on any monthly table
# --------------------------------------------------------------------------- #
def run_overlay(exc_m: np.ndarray, rv_m: np.ndarray, cap: float = CAP,
                cost_bps: float = 0.0, borrow_spread_ann: float = 0.0) -> dict:
    """Managed vs unmanaged excess returns under the vol-managed rule.

    Costs: one-way ``cost_bps`` x |w_t - w_{t-1}| x NAV per rebalance, plus a retail
    ``borrow_spread_ann`` charged on the levered fraction max(w-1, 0) (leverage at the
    T-bill rate is already implicit in ``w * excess``; the spread is what a retail
    margin account pays ABOVE that).
    """
    exc_m = np.asarray(exc_m, float)
    w = weights_from_rv(rv_m, cap=cap)
    ok = np.isfinite(w)
    wv, ev = w[ok], exc_m[ok]
    dw = np.abs(np.diff(np.concatenate([[1.0], wv])))     # start from fully invested
    cost = cost_bps / 1e4 * dw
    borrow = borrow_spread_ann / MONTHS * np.maximum(wv - 1.0, 0.0)
    managed = wv * ev - cost - borrow
    return {
        "managed": managed, "bh": ev, "w": wv, "ok": ok,
        "avg_w": float(wv.mean()), "share_capped": float((wv >= cap - 1e-12).mean()),
        "avg_turnover": float(dw.mean()),
    }


def race_summary(exc_m: np.ndarray, rv_m: np.ndarray, cap: float = CAP,
                 cost_bps: float = 0.0, borrow_spread_ann: float = 0.0) -> dict:
    ov = run_overlay(exc_m, rv_m, cap=cap, cost_bps=cost_bps,
                     borrow_spread_ann=borrow_spread_ann)
    reg = hac_alpha(ov["managed"], ov["bh"])
    return {
        **{f"reg_{k}": v for k, v in reg.items()},
        "sharpe_managed": sharpe_stats(ov["managed"])["sharpe_ann"],
        "sharpe_bh": sharpe_stats(ov["bh"])["sharpe_ann"],
        "avg_w": ov["avg_w"], "share_capped": ov["share_capped"],
        "avg_turnover": ov["avg_turnover"], "n_months": len(ov["managed"]),
    }


# --------------------------------------------------------------------------- #
# Placebo — shuffle the vol signal (>= 20 seeds, averaged)
# --------------------------------------------------------------------------- #
def placebo_shuffle(exc_m: np.ndarray, rv_m: np.ndarray, n_seeds: int = 200,
                    cap: float = CAP, base_seed: int = 591) -> dict:
    """Destroy the vol-timing information by permuting the monthly RV series across
    months (same marginal distribution, no persistence/alignment), rebuild the whole
    rule per seed, and collect the HAC alpha t. Reports the mean/sd of the shuffled t
    across seeds and ``p = Pr[shuffled alpha >= observed alpha]``."""
    obs = race_summary(exc_m, rv_m, cap=cap)
    ts, alphas = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        rv_sh = rng.permutation(rv_m)
        r = race_summary(exc_m, rv_sh, cap=cap)
        ts.append(r["reg_t_alpha"])
        alphas.append(r["reg_alpha_ann_pct"])
    ts, alphas = np.array(ts), np.array(alphas)
    return {
        "obs_alpha_ann_pct": obs["reg_alpha_ann_pct"], "obs_t": obs["reg_t_alpha"],
        "placebo_mean_t": float(ts.mean()), "placebo_sd_t": float(ts.std(ddof=1)),
        "placebo_mean_alpha": float(alphas.mean()),
        "p_alpha": float((alphas >= obs["reg_alpha_ann_pct"]).mean()),
        "n_seeds": n_seeds,
    }


# --------------------------------------------------------------------------- #
# Crash-window drawdowns (real tape, monthly TOTAL-return NAVs)
# --------------------------------------------------------------------------- #
CRASH_WINDOWS = {
    "GFC 2008-09": ("2007-10", "2009-06"),
    "COVID 2020": ("2020-01", "2020-12"),
    "2022 bear": ("2022-01", "2022-12"),
}


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def crash_table(tbl: pd.DataFrame, cap: float = CAP) -> dict:
    """Max drawdown of the managed vs buy-and-hold TOTAL-return NAV (monthly
    granularity) inside each named crash window + full sample.

    ``tbl`` is the real-tape monthly table (exc / rf / rv, Period[M] index)."""
    w = weights_from_rv(tbl["rv"].values, cap=cap)
    ok = np.isfinite(w)
    sub = tbl.iloc[ok].copy()
    sub["w"] = w[ok]
    r_man = sub["rf"] + sub["w"] * sub["exc"]
    r_bh = sub["rf"] + sub["exc"]
    nav_man = (1.0 + r_man).cumprod()
    nav_bh = (1.0 + r_bh).cumprod()
    out = {}
    for name, (a, b) in CRASH_WINDOWS.items():
        m = (sub.index >= pd.Period(a, "M")) & (sub.index <= pd.Period(b, "M"))
        if m.sum() < 3:
            continue
        out[name] = {"managed": max_drawdown(nav_man[m]), "bh": max_drawdown(nav_bh[m])}
    out["full sample"] = {"managed": max_drawdown(nav_man), "bh": max_drawdown(nav_bh)}
    return out


# --------------------------------------------------------------------------- #
# Synthetic harness — averaged over seeds
# --------------------------------------------------------------------------- #
def synthetic_check(disconnect: float, n_seeds: int = 20, n_months: int = 360,
                    cap: float = CAP, base_seed: int = 591) -> dict:
    """Run the full rule on ``n_seeds`` independent synthetic worlds and average the
    HAC alpha t (the desk rule: random baselines are never single-seed)."""
    from . import data as d
    ts, alphas = [], []
    for s in range(n_seeds):
        r, mid, _ = d.synthetic_world(n_months=n_months, disconnect=disconnect,
                                      seed=base_seed + 1000 * s)
        exc_m, rv_m = monthly_from_daily(r, r, mid)   # rf = 0 in the synthetic world
        res = race_summary(exc_m, rv_m, cap=cap)
        ts.append(res["reg_t_alpha"])
        alphas.append(res["reg_alpha_ann_pct"])
    ts, alphas = np.array(ts), np.array(alphas)
    return {
        "mean_t": float(ts.mean()), "sd_t": float(ts.std(ddof=1)),
        "mean_alpha_ann_pct": float(alphas.mean()),
        "share_t_ge_2": float((ts >= 2.0).mean()), "n_seeds": n_seeds,
    }
