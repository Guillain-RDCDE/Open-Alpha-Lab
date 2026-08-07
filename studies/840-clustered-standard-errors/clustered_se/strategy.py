"""Inference primitives + the clustered-SE experiments — Study 840.

The claim under test (Petersen 2009; Fama & MacBeth 1973; Moulton 1986; Cameron-Gelbach-Miller
2011): in a panel with a **common time effect** (a shock that hits every firm in a period at
once), the residuals of a pooled regression are **correlated across firms within a period**.
The ordinary i.i.d. OLS standard error — and one-way **firm** clustering — ignore that
dependence and are far too small, so the *t*-statistic **overstates significance**.
**Fama-MacBeth** (a cross-sectional regression each period, averaged over time) and **time /
two-way clustering** restore calibration.

The point of the demonstration is that all four estimators share (essentially) the *same
point estimate*; what differs is the **standard error** they attach to it. We put a number on
the pitfall four ways:

1. **Calibration.** The Monte-Carlo standard deviation of the pooled slope across replications
   is the *truth*. Each estimator's mean SE is compared to it: naive OLS and firm-clustering
   land at ~1/3 of the truth (an SE **ratio** far below 1); Fama-MacBeth / time-clustering land
   at ~1.
2. **False-positive rate.** Under the null (``beta = 0``) the naive-*t* and firm-clustered-*t*
   reject the (true) null far more than 5% of the time; the Fama-MacBeth-*t* and time-clustered
   -*t* stay near 5%.
3. **The inflation factor.** The naive-*t* SD under the null equals the closed-form Moulton
   factor ``sqrt(1 + (N-1)*rho_x*rho_e)``; we recover it empirically and match it, and trace it
   as ``rho_e`` and ``N`` grow.

The **positive control** (a planted non-zero ``beta``) confirms Fama-MacBeth still *fires*
when there genuinely is an effect — it is unbiased, not merely conservative.

Everything is pure numpy + stdlib, deterministic, vectorised over the Monte-Carlo replications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Scalar inference primitives (single panel / single series) — the readable core
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """Naive i.i.d. one-sample *t* that mean(x) differs from 0 (no dependence assumed)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch two-sample *t* (unequal variances) — a cross-check primitive."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (used on the FP rate)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def _norm_cdf(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def nominal_rate(crit: float = 1.96) -> float:
    """The two-sided rejection rate a calibrated test should deliver at ``crit``."""
    return 2.0 * (1.0 - _norm_cdf(crit))


# --------------------------------------------------------------------------- #
# The four standard errors for the SAME pooled slope — vectorised over reps
# --------------------------------------------------------------------------- #
def panel_inference(X: np.ndarray, Y: np.ndarray) -> dict:
    """Estimate ``y = a + b*x`` per replication and attach FOUR standard errors to ``b``.

    ``X``, ``Y`` are ``(R, T, N)`` (reps × periods × firms). Returns per-rep arrays:

    * ``b``        — the pooled OLS slope (grand-demeaned; identical for OLS / firm / time SE).
    * ``se_ols``   — naive i.i.d. OLS SE (``s^2 / Sxx``).
    * ``se_firm``  — one-way cluster-robust SE, clustering by **firm** (the *wrong* dimension
                     for a time effect — it does not help).
    * ``se_time``  — one-way cluster-robust SE, clustering by **time** (the right dimension;
                     handles the cross-sectional dependence).
    * ``b_fm`` / ``se_fm`` — the **Fama-MacBeth** slope and SE (a cross-sectional regression
                     each period, then the mean and SE of the T per-period slopes).
    * ``t_ols`` / ``t_firm`` / ``t_time`` / ``t_fm`` — the corresponding *t*-statistics.

    Cluster SEs use the standard finite-sample scaling ``G/(G-1) * (NT-1)/(NT-2)`` with ``G``
    the number of clusters. The demeaned-univariate score for the slope is ``x_dm * u``.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    R, T, N = X.shape
    NT = T * N

    Xf = X.reshape(R, NT); Yf = Y.reshape(R, NT)
    xbar = Xf.mean(axis=1, keepdims=True); ybar = Yf.mean(axis=1, keepdims=True)
    xd = Xf - xbar; yd = Yf - ybar
    Sxx = (xd * xd).sum(axis=1)                          # (R,)
    b = (xd * yd).sum(axis=1) / Sxx                      # pooled OLS slope (R,)
    a = ybar[:, 0] - b * xbar[:, 0]
    u = Yf - a[:, None] - b[:, None] * Xf               # residuals (R, NT)

    # --- naive i.i.d. OLS SE ---
    s2 = (u * u).sum(axis=1) / (NT - 2)
    se_ols = np.sqrt(s2 / Sxx)

    # --- score for the slope, reshaped to (R, T, N) ---
    score = (xd * u).reshape(R, T, N)                    # x_dm * residual

    # --- one-way firm clustering (group = firm = axis N): the WRONG dimension ---
    firm_meat = (score.sum(axis=1) ** 2).sum(axis=1)     # sum over t, square, sum over firms
    c_firm = (N / (N - 1)) * (NT - 1) / (NT - 2)
    se_firm = np.sqrt(firm_meat * c_firm) / Sxx

    # --- one-way time clustering (group = time = axis T): the RIGHT dimension ---
    time_meat = (score.sum(axis=2) ** 2).sum(axis=1)     # sum over firms, square, sum over t
    c_time = (T / (T - 1)) * (NT - 1) / (NT - 2)
    se_time = np.sqrt(time_meat * c_time) / Sxx

    # --- Fama-MacBeth: a cross-sectional regression each period, then average ---
    xbt = X.mean(axis=2, keepdims=True); ybt = Y.mean(axis=2, keepdims=True)
    xdt = X - xbt; ydt = Y - ybt
    Sxx_t = (xdt * xdt).sum(axis=2)                      # (R, T)
    b_t = (xdt * ydt).sum(axis=2) / Sxx_t               # per-period slopes (R, T)
    b_fm = b_t.mean(axis=1)
    se_fm = b_t.std(axis=1, ddof=1) / np.sqrt(T)

    def _t(num, den):
        out = np.full(R, np.nan)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out

    return {
        "b": b, "b_fm": b_fm,
        "se_ols": se_ols, "se_firm": se_firm, "se_time": se_time, "se_fm": se_fm,
        "t_ols": _t(b, se_ols), "t_firm": _t(b, se_firm),
        "t_time": _t(b, se_time), "t_fm": _t(b_fm, se_fm),
        "b_t": b_t,
    }


# --------------------------------------------------------------------------- #
# Experiment 1 + 2 — calibration and the false-positive rate under the null
# --------------------------------------------------------------------------- #
def calibration(X: np.ndarray, Y: np.ndarray, crit: float = 1.96) -> dict:
    """Per-estimator SE calibration and rejection rate over the replications of ``(X, Y)``.

    On a null (``beta = 0``) panel the rejection rate is the **false-positive rate**; it
    should sit near ``nominal_rate(crit)`` (~5%). The *truth* the SEs are graded against is the
    Monte-Carlo SD of the pooled slope across reps (``true_sd``); a calibrated estimator's mean
    SE equals it (SE ratio ~ 1). Also returns the naive-*t* SD (~ the Moulton inflation factor).
    """
    R = X.shape[0]
    inf = panel_inference(X, Y)
    true_sd = float(np.std(inf["b"], ddof=1))            # the TRUTH for the pooled slope
    true_sd_fm = float(np.std(inf["b_fm"], ddof=1))      # the truth for the FM slope
    nominal = nominal_rate(crit)
    out = {
        "n_reps": R, "crit": crit, "nominal": nominal,
        "true_sd": true_sd, "true_sd_fm": true_sd_fm,
        "b_mean": float(np.mean(inf["b"])), "b_fm_mean": float(np.mean(inf["b_fm"])),
    }
    labels = {"ols": true_sd, "firm": true_sd, "time": true_sd, "fm": true_sd_fm}
    for key, truth in labels.items():
        t = inf[f"t_{key}"]; se = inf[f"se_{key}"]
        good = np.isfinite(t)
        k = int(np.sum(np.abs(t[good]) > crit))
        n = int(good.sum())
        out[f"{key}_fp"] = k / n if n else float("nan")
        out[f"{key}_fp_ci"] = wilson_interval(k, n)
        out[f"{key}_se_mean"] = float(np.mean(se[np.isfinite(se)]))
        out[f"{key}_se_ratio"] = float(np.mean(se[np.isfinite(se)]) / truth) if truth > 0 else float("nan")
        out[f"{key}_t_sd"] = float(np.nanstd(t, ddof=1))
    out["_inf"] = inf
    return out


# --------------------------------------------------------------------------- #
# Experiment 3 — the inflation curve vs the amount of cross-sectional dependence
# --------------------------------------------------------------------------- #
def inflation_curve_rho(
    rhos, n_reps: int, n_periods: int, n_firms: int, rho_x: float,
    seed: int, crit: float = 1.96,
) -> pd.DataFrame:
    """Naive / firm / time / FM false-positive rate and naive-*t* SD across a grid of the
    residual intra-period correlation ``rho_e``, matched to the Moulton closed form."""
    from . import data as d
    rows = []
    for rho_e in rhos:
        X, Y = d.panel(n_reps, n_periods, n_firms, rho_x=rho_x, rho_e=rho_e, beta=0.0, seed=seed)
        c = calibration(X, Y, crit=crit)
        rows.append({
            "rho_e": rho_e,
            "ols_fp": c["ols_fp"], "firm_fp": c["firm_fp"],
            "time_fp": c["time_fp"], "fm_fp": c["fm_fp"],
            "naive_t_sd": c["ols_t_sd"],
            "theory_moulton": d.theoretical_moulton(n_firms, rho_x, rho_e),
        })
    return pd.DataFrame(rows)


def inflation_curve_nfirms(
    n_firms_grid, n_reps: int, n_periods: int, rho_x: float, rho_e: float,
    seed: int, crit: float = 1.96,
) -> pd.DataFrame:
    """Same experiment across a grid of cross-section sizes ``N`` (the cluster size) — the
    inflation grows with ``N`` exactly as ``sqrt(1 + (N-1)*rho_x*rho_e)``."""
    from . import data as d
    rows = []
    for N in n_firms_grid:
        X, Y = d.panel(n_reps, n_periods, N, rho_x=rho_x, rho_e=rho_e, beta=0.0, seed=seed)
        c = calibration(X, Y, crit=crit)
        rows.append({
            "n_firms": N,
            "ols_fp": c["ols_fp"], "firm_fp": c["firm_fp"],
            "time_fp": c["time_fp"], "fm_fp": c["fm_fp"],
            "naive_t_sd": c["ols_t_sd"],
            "theory_moulton": d.theoretical_moulton(N, rho_x, rho_e),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Experiment 4 — the no-dependence control (the honest placebo)
# --------------------------------------------------------------------------- #
def iid_control(n_reps: int, n_periods: int, n_firms: int, seed: int,
                crit: float = 1.96) -> dict:
    """Remove the pitfall's cause and it vanishes. With ``rho_x = rho_e = 0`` there is **no**
    common time factor — the panel is i.i.d. across both dimensions — so *all four* estimators
    are calibrated (FP ~ nominal). Same generator, same estimators, only the cross-sectional
    dependence switched off: it isolates the shared time shock as *the* cause of the naive
    inflation."""
    from . import data as d
    X, Y = d.panel(n_reps, n_periods, n_firms, rho_x=0.0, rho_e=0.0, beta=0.0, seed=seed)
    c = calibration(X, Y, crit=crit)
    return {
        "ols_fp": c["ols_fp"], "firm_fp": c["firm_fp"],
        "time_fp": c["time_fp"], "fm_fp": c["fm_fp"],
        "naive_t_sd": c["ols_t_sd"], "nominal": c["nominal"],
        "ols_fp_ci": c["ols_fp_ci"], "ols_se_ratio": c["ols_se_ratio"],
    }


# --------------------------------------------------------------------------- #
# Positive control — Fama-MacBeth still fires on a planted effect
# --------------------------------------------------------------------------- #
def power_check(
    n_reps: int, n_periods: int, n_firms: int, rho_x: float, rho_e: float,
    beta: float, seed: int, crit: float = 1.96,
) -> dict:
    """Plant a genuine slope ``beta`` into the panel; report the Fama-MacBeth rejection
    (power) rate, the mean/sign of the FM *t*, and — for contrast — the naive rejection rate
    (already inflated under the null, so uninformative as 'power'). A correct test must reject
    the null most of the time here — proving Fama-MacBeth is unbiased, not merely
    conservative."""
    from . import data as d
    X, Y = d.panel(n_reps, n_periods, n_firms, rho_x=rho_x, rho_e=rho_e, beta=beta, seed=seed)
    inf = panel_inference(X, Y)
    t_fm = inf["t_fm"]; t_ols = inf["t_ols"]
    return {
        "planted_beta": beta,
        "fm_power": float(np.mean(np.abs(t_fm) > crit)),
        "naive_power": float(np.mean(np.abs(t_ols) > crit)),
        "fm_t_mean": float(np.mean(t_fm)),
        "fm_t_positive_share": float(np.mean(t_fm > 0)),
        "b_fm_mean": float(np.mean(inf["b_fm"])),
    }


# --------------------------------------------------------------------------- #
# The costed timer — there is, by construction, nothing to trade
# --------------------------------------------------------------------------- #
def timer_stats(X_one: np.ndarray, Y_one: np.ndarray, ret_scale: float = 0.01,
                cost_bps: float = 5.0, borrow_bps_yr: float = 50.0) -> dict:
    """Cost a notional dollar-neutral long-short whose 'signal' is the null predictor ``x``.

    Each period build a dollar-neutral book with weights ``w_it = (x_it - xbar_t) /
    sum_i|x_it - xbar_t|`` (gross = 1, long the high-x names, short the low-x) and record the
    period return ``sum_i w_it * (ret_scale * y_it)``. Since ``beta = 0`` the gross mean is
    ~0; charging the round-trip turnover plus borrow on the short leg pushes the net firmly
    negative — a Mirage, by construction. Kept for parity with the desk's other timers."""
    X_one = np.asarray(X_one, dtype=float); Y_one = np.asarray(Y_one, dtype=float)
    T, N = X_one.shape
    xd = X_one - X_one.mean(axis=1, keepdims=True)
    gross_w = np.abs(xd).sum(axis=1, keepdims=True)
    w = np.divide(xd, gross_w, out=np.zeros_like(xd), where=gross_w > 0)   # (T,N), sum|w|=1
    ret = (w * (ret_scale * Y_one)).sum(axis=1)                            # period returns
    # turnover: full rebalance each period (weights redraw); charge |w_t - w_{t-1}|
    dw = np.abs(np.diff(w, axis=0, prepend=w[:1])).sum(axis=1)             # per-period turnover
    round_trip = dw * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0 * 0.5                     # short leg ~ half book
    net = ret - round_trip - borrow_daily
    n = len(ret)
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net.mean() / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_periods": n,
        "gross_bps": float(ret.mean() * 1e4),
        "net_bps": float(net.mean() * 1e4),
        "cost_bps_per_period": float((round_trip.mean() + borrow_daily) * 1e4),
        "ann_net_pct": float(net.mean() * TRADING_DAYS * 100),
        "sharpe_net": float(sharpe),
        "t_net_naive": one_sample_t(net),
    }
