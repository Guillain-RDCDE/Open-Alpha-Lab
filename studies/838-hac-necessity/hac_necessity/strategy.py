"""Inference primitives + the HAC-necessity experiments — Study 838.

The claim under test (Newey & West 1987; Hansen & Hodrick 1980): when a return series is
**serially correlated** — as any daily strategy with an overlapping formation window is —
the ordinary i.i.d. OLS standard error is *wrong*, and the naive *t*-statistic it produces
**overstates significance**. The heteroskedasticity- and autocorrelation-consistent (HAC)
Newey-West standard error corrects it.

We put a number on the pitfall three ways:

1. **False-positive rate.** Monte-Carlo many mean-zero-but-autocorrelated null paths; the
   naive-*t* rejects the (true) null far more than 5% of the time, the Newey-West-*t* stays
   near 5%.
2. **Inflation factor.** The naive *t* under the null has SD ``sqrt(LRV/gamma0)`` instead of
   1; we recover that factor empirically and match it to the closed form
   (``sqrt(window)`` for an overlap, ``sqrt((1+rho)/(1-rho))`` for AR(1)).
3. **The control.** Switch the autocorrelation *off* at the source (a ``window=1`` overlap
   is plain i.i.d. noise); both the naive and the Newey-West false-positive rates sit at
   ~5% — proving the inflation is *caused by* the serial correlation, and nothing else.

A positive control (a planted non-zero mean) confirms the HAC machinery still *fires* when
there genuinely is an effect — it is unbiased, not merely conservative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Bandwidth
# --------------------------------------------------------------------------- #
def nw_auto_lags(n: int) -> int:
    """Newey-West automatic bandwidth floor(4*(n/100)^(2/9)), at least 1."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


# --------------------------------------------------------------------------- #
# Inference primitives (single series)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """Naive i.i.d. OLS *t* that mean(x) differs from 0 (assumes no autocorrelation)."""
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


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0.

    ``lags=None`` uses the automatic bandwidth :func:`nw_auto_lags`. For an overlapping
    ``window``-day signal the correct bandwidth is at least ``window-1`` (Hansen-Hodrick).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    L = nw_auto_lags(n) if lags is None else int(lags)
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def autocorr(x: np.ndarray, lags: int = 30) -> np.ndarray:
    """Sample autocorrelation function rho_hat(1..lags) of a series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    u = x - x.mean()
    g0 = float(u @ u) / n
    if g0 <= 0:
        return np.full(lags, np.nan)
    return np.array([float(u[l:] @ u[:-l]) / n / g0 for l in range(1, lags + 1)])


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (used on the FP rate)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Vectorised inference across a Monte-Carlo matrix (n_reps x n_days)
# --------------------------------------------------------------------------- #
def one_sample_t_matrix(X: np.ndarray) -> np.ndarray:
    """Naive i.i.d. *t* for every row of ``X`` (vectorised)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    mu = X.mean(axis=1)
    sd = X.std(axis=1, ddof=1)
    se = sd / np.sqrt(n)
    out = np.full(X.shape[0], np.nan)
    ok = se > 0
    out[ok] = mu[ok] / se[ok]
    return out


def newey_west_t_matrix(X: np.ndarray, lags: int) -> np.ndarray:
    """HAC (Newey-West) *t* for every row of ``X`` with a fixed Bartlett bandwidth ``lags``."""
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    L = int(lags)
    mu = X.mean(axis=1)
    U = X - mu[:, None]
    var = (U * U).sum(axis=1) / n           # gamma0 per row
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        cov = (U[:, l:] * U[:, :-l]).sum(axis=1) / n
        var += 2.0 * w * cov
    se = np.sqrt(np.maximum(var, 0.0) / n)
    out = np.full(X.shape[0], np.nan)
    ok = se > 0
    out[ok] = mu[ok] / se[ok]
    return out


# --------------------------------------------------------------------------- #
# Experiment 1 — the false-positive rate
# --------------------------------------------------------------------------- #
def false_positive_rate(X: np.ndarray, lags: int, crit: float = 1.96) -> dict:
    """Rejection rate of H0: mean=0 under the naive vs Newey-West *t*, over the rows of ``X``.

    On a mean-zero (null) matrix this is the **false-positive rate**; it should sit near the
    nominal ``P(|Z|>crit)`` (~5% at crit=1.96) for a calibrated test. The naive test, blind
    to the serial correlation baked into ``X``, rejects far more often.
    """
    t_naive = one_sample_t_matrix(X)
    t_nw = newey_west_t_matrix(X, lags)
    n = X.shape[0]
    k_naive = int(np.sum(np.abs(t_naive) > crit))
    k_nw = int(np.sum(np.abs(t_nw) > crit))
    nominal = 2.0 * (1.0 - _norm_cdf(crit))
    return {
        "n_reps": n,
        "crit": crit,
        "nominal": nominal,
        "naive_fp": k_naive / n,
        "nw_fp": k_nw / n,
        "naive_fp_ci": wilson_interval(k_naive, n),
        "nw_fp_ci": wilson_interval(k_nw, n),
        "naive_t_sd": float(np.nanstd(t_naive, ddof=1)),   # ~ inflation factor under the null
        "nw_t_sd": float(np.nanstd(t_nw, ddof=1)),         # ~ 1 for a calibrated test
        "t_naive": t_naive,
        "t_nw": t_nw,
    }


def _norm_cdf(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Experiment 2 — the inflation curve vs the amount of autocorrelation
# --------------------------------------------------------------------------- #
def inflation_curve_overlap(
    windows, n_reps: int, n_days: int, seed: int, daily_vol: float = 0.01,
    crit: float = 1.96,
) -> pd.DataFrame:
    """Naive/NW false-positive rate and naive-*t* SD across a grid of overlap ``windows``.

    NW bandwidth is set to ``2*window`` (a generous Bartlett span: the triangular kernel
    downweights the far lags, so a bandwidth at the overlap length alone slightly
    under-covers). The empirical naive-*t* SD is matched against the closed-form
    ``sqrt(window)``.
    """
    from . import data as d
    rows = []
    for w in windows:
        X = d.overlap_matrix(n_reps, n_days, window=w, seed=seed, daily_vol=daily_vol, mean=0.0)
        L = max(1, 2 * w)
        fp = false_positive_rate(X, lags=L, crit=crit)
        rows.append({
            "window": w,
            "naive_fp": fp["naive_fp"],
            "nw_fp": fp["nw_fp"],
            "naive_t_sd": fp["naive_t_sd"],
            "theory_inflation": d.theoretical_inflation_overlap(w),
        })
    return pd.DataFrame(rows)


def inflation_curve_ar1(
    rhos, n_reps: int, n_days: int, seed: int, daily_vol: float = 0.01,
    lags: int | None = None, crit: float = 1.96,
) -> pd.DataFrame:
    """Naive/NW false-positive rate and naive-*t* SD across a grid of AR(1) ``rhos``.

    NW bandwidth defaults to the automatic :func:`nw_auto_lags`. The empirical naive-*t* SD
    is matched against the closed-form ``sqrt((1+rho)/(1-rho))``.
    """
    from . import data as d
    rows = []
    for rho in rhos:
        X = d.ar1_matrix(n_reps, n_days, rho=rho, seed=seed, daily_vol=daily_vol, mean=0.0)
        L = nw_auto_lags(n_days) if lags is None else lags
        fp = false_positive_rate(X, lags=L, crit=crit)
        rows.append({
            "rho": rho,
            "naive_fp": fp["naive_fp"],
            "nw_fp": fp["nw_fp"],
            "naive_t_sd": fp["naive_t_sd"],
            "theory_inflation": d.theoretical_inflation_ar1(rho),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Experiment 3 — the no-autocorrelation control (the honest placebo)
# --------------------------------------------------------------------------- #
def iid_control(n_reps: int, n_days: int, seed: int, daily_vol: float = 0.01,
                lags: int = 42, crit: float = 1.96) -> dict:
    """Remove the pitfall's cause and the pitfall vanishes. A ``window=1`` overlap is just
    i.i.d. noise — no serial correlation — so BOTH the naive and the Newey-West *t* are
    calibrated (FP ~ nominal). This isolates autocorrelation as *the* cause of the naive
    inflation: it is the same generator, the same estimators, only the serial correlation
    switched off.

    (Note: row-shuffling an *already-drawn* autocorrelated path is NOT a valid placebo — the
    sample mean and SD are permutation-invariant, so the naive *t* is unchanged. The cause
    must be removed at generation, which is exactly what ``window=1`` does.)
    """
    from . import data as d
    X = d.overlap_matrix(n_reps, n_days, window=1, seed=seed, daily_vol=daily_vol, mean=0.0)
    fp = false_positive_rate(X, lags=lags, crit=crit)
    return {
        "naive_fp": fp["naive_fp"],
        "nw_fp": fp["nw_fp"],
        "naive_t_sd": fp["naive_t_sd"],
        "nominal": fp["nominal"],
        "naive_fp_ci": fp["naive_fp_ci"],
    }


# --------------------------------------------------------------------------- #
# Positive control — the HAC machinery still fires on a planted effect
# --------------------------------------------------------------------------- #
def power_check(
    n_reps: int, n_days: int, window: int, mean: float, seed: int,
    daily_vol: float = 0.01, lags: int | None = None, crit: float = 1.96,
) -> dict:
    """Plant a genuine positive mean into the overlapping series; report the Newey-West
    rejection (power) rate and the mean/sign of the NW *t*. A correct test must reject the
    null most of the time here — proving it is unbiased, not merely conservative."""
    from . import data as d
    X = d.overlap_matrix(n_reps, n_days, window=window, seed=seed, daily_vol=daily_vol, mean=mean)
    L = max(1, window) if lags is None else lags
    t_nw = newey_west_t_matrix(X, L)
    t_naive = one_sample_t_matrix(X)
    return {
        "planted_mean": mean,
        "nw_power": float(np.mean(np.abs(t_nw) > crit)),
        "naive_power": float(np.mean(np.abs(t_naive) > crit)),
        "nw_t_mean": float(np.mean(t_nw)),
        "nw_t_positive_share": float(np.mean(t_nw > 0)),
    }


# --------------------------------------------------------------------------- #
# The costed timer — there is, by construction, nothing to trade
# --------------------------------------------------------------------------- #
def timer_stats(x: np.ndarray, cost_bps: float = 5.0, borrow_bps_yr: float = 50.0) -> dict:
    """Cost a notional daily long-short whose 'edge' is the null series ``x``.

    Since the series is a mean-zero null, the gross mean is ~0; charging even one round-trip
    plus borrow pushes the net firmly negative — a Mirage, by construction. Kept for parity
    with the desk's other timers."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    round_trip = 2.0 * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0
    net = x - round_trip - borrow_daily
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net.mean() / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": float(x.mean() * 1e4),
        "net_bps": float(net.mean() * 1e4),
        "cost_bps_per_day": (round_trip + borrow_daily) * 1e4,
        "ann_net_pct": float(net.mean() * TRADING_DAYS * 100),
        "sharpe_net": float(sharpe),
        "t_net_naive": one_sample_t(net),
    }
