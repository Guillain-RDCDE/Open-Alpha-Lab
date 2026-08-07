"""Strategy + inference for Study 837 — Look-Ahead Standardization.

The two ways to turn a raw feature panel ``X[t, i]`` into a standardised signal, and the machinery
that measures the damage the wrong one does:

* **The honest way — expanding / point-in-time.** :func:`expanding_standardize` z-scores each name
  using only the mean & std of its **own history up to and including t** — never a value the
  researcher could not have seen at the close of ``t``. This is the correct preprocessing for a
  backtest.

* **The leaky way — full sample.** :func:`full_standardize` z-scores each name with the mean & std of
  its **entire** series, ``t = 0 .. T`` — including the future. It is the single most common
  accidental look-ahead in a feature pipeline (``df = (df - df.mean()) / df.std()`` run once, before
  the train/test split).

We score each standardised signal against the forward return with a cross-sectional Information
Coefficient (rank IC) per day, a Newey-West (HAC) *t* on the daily IC series, a long-short fractile
book and its annualised Sharpe, and a costed timer. On a null the honest signal reads ~0; the leaky
one manufactures a large IC and a gorgeous fake Sharpe. A planted-edge control proves the honest
machinery still fires on a *real* effect (so its silence on the nulls means something).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The two standardisations — honest (expanding) vs leaky (full-sample)
# --------------------------------------------------------------------------- #
def full_standardize(X: np.ndarray) -> np.ndarray:
    """LEAKY per-name z-score using the **full-sample** (future-inclusive) mean & std.

    ``Z[t, i] = (X[t, i] - mean_i) / std_i`` where ``mean_i``/``std_i`` use the whole column
    ``t = 0 .. T`` — including data from *after* ``t``. Population std (``ddof=0``); columns with zero
    variance become NaN.
    """
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, np.nan)
    return (X - mu) / sd


def expanding_standardize(X: np.ndarray, min_periods: int = 60) -> np.ndarray:
    """HONEST per-name z-score using an **expanding** (past-inclusive, point-in-time) window.

    ``Z[t, i] = (X[t, i] - mean(X[:t+1, i])) / std(X[:t+1, i])`` — the mean & std use only rows
    ``0 .. t`` (data known at the close of ``t``; using ``X[t]`` itself is fine, it is observed).
    Population std (``ddof=0``). Rows before ``min_periods`` (burn-in) and zero-variance windows are
    NaN. Vectorised via cumulative sums.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    n = np.arange(1, T + 1, dtype=float)[:, None]
    csum = np.cumsum(X, axis=0)
    csum2 = np.cumsum(X * X, axis=0)
    mean = csum / n
    var = csum2 / n - mean * mean
    var = np.where(var > 0, var, np.nan)
    Z = (X - mean) / np.sqrt(var)
    if min_periods > 0:
        Z[: min_periods] = np.nan
    return Z


# --------------------------------------------------------------------------- #
# Cross-sectional Information Coefficient (rank IC) per day
# --------------------------------------------------------------------------- #
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two 1-D arrays (finite entries only)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a, kind="stable"))
    rb = np.argsort(np.argsort(b, kind="stable"))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else np.nan


def cross_sectional_ic(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Daily cross-sectional rank IC between the signal ``Z[t]`` and forward return ``R[t]``.

    Returns a 1-D array (length T) of per-day Spearman ICs; days with fewer than 3 jointly-finite
    names are NaN.
    """
    Z = np.asarray(Z, dtype=float)
    R = np.asarray(R, dtype=float)
    T = Z.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        out[t] = _spearman(Z[t], R[t])
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (the desk's shared toolkit)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
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
# The long-short fractile book (what the naive backtester would "trade")
# --------------------------------------------------------------------------- #
def long_short_spread(Z: np.ndarray, R: np.ndarray, frac: float = 0.2,
                      min_names: int = 10) -> np.ndarray:
    """Daily equal-weight top-minus-bottom fractile spread of the signal ``Z`` on forward return ``R``.

    On each day rank names by ``Z[t]``; ``spread = mean(R of top frac) - mean(R of bottom frac)``
    (long high-signal, short low-signal). Days with fewer than ``min_names`` names are NaN.
    """
    Z = np.asarray(Z, dtype=float); R = np.asarray(R, dtype=float)
    T = Z.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        z, r = Z[t], R[t]
        m = np.isfinite(z) & np.isfinite(r)
        if m.sum() < min_names:
            continue
        zz, rr = z[m], r[m]
        k = max(1, int(np.floor(len(zz) * frac)))
        order = np.argsort(zz, kind="stable")
        lo = rr[order[:k]].mean()
        hi = rr[order[-k:]].mean()
        out[t] = hi - lo
    return out


def book_stats(spread: np.ndarray, nw_lags: int = 10) -> dict:
    """Headline stats of a daily long-short spread series.

    The naive backtester trades in the *direction its own in-sample spread points*, so the harvested
    Sharpe is the **absolute** annualised Sharpe (``|mean|/std * sqrt(252)``); ``spread_mean`` keeps
    the sign so the mechanism (reversion toward the peeked mean) is visible.
    """
    sp = np.asarray(spread, dtype=float)
    sp = sp[np.isfinite(sp)]
    n = len(sp)
    if n < 3:
        return {"n": n, "spread_mean": float("nan"), "sharpe_abs": float("nan"),
                "t_nw": float("nan"), "t_1s": float("nan")}
    sd = sp.std(ddof=1)
    sharpe = abs(sp.mean()) / sd * np.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    return {
        "n": n,
        "spread_mean": float(sp.mean()),
        "sharpe_abs": float(sharpe),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
    }


def timer_stats(spread: np.ndarray, cost_bps: float = 5.0,
                borrow_bps_yr: float = 50.0, turnover: float = 1.0) -> dict:
    """Cost the long-short book: one-way cost x NAV per rebalance leg + borrow on the short leg.

    A daily-rebalanced 2x-NAV long-short book pays ``2 * cost_bps`` round-trip on the traded fraction
    (``turnover``, defaulting to a full daily rotation) plus a daily slice of ``borrow_bps_yr`` on the
    short. Even a fake edge cannot survive being charged to trade — this is what stamps the Mirage.
    """
    sp = np.asarray(spread, dtype=float)
    sp = sp[np.isfinite(sp)]
    n = len(sp)
    round_trip = turnover * 2.0 * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0
    # trade in the in-sample-implied direction (harvest the magnitude), then pay costs
    signed = np.sign(sp.mean()) * sp if n else sp
    net = signed - round_trip - borrow_daily
    sd = net.std(ddof=1) if n > 1 else float("nan")
    sharpe = net.mean() / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n": n,
        "gross_bps": float(abs(sp.mean()) * 1e4) if n else float("nan"),
        "net_bps": float(net.mean() * 1e4) if n else float("nan"),
        "cost_bps_per_day": (round_trip + borrow_daily) * 1e4,
        "sharpe_net": float(sharpe),
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# The headline — full-sample vs expanding on one world
# --------------------------------------------------------------------------- #
def leak_report(X: np.ndarray, R: np.ndarray, frac: float = 0.2,
                min_periods: int = 60, nw_lags: int = 10) -> dict:
    """Full-sample vs expanding-window standardisation on one panel, scored the same way.

    Returns the mean IC, the Newey-West *t* of the daily IC series, the fake long-short annualised
    Sharpe, and the spread NW *t*, for **each** standardisation, plus the IC and Sharpe **gap**
    (full minus expanding). A big positive gap = the leak manufactured performance.
    """
    Zf = full_standardize(X)
    Ze = expanding_standardize(X, min_periods)
    ic_f = cross_sectional_ic(Zf, R)
    ic_e = cross_sectional_ic(Ze, R)
    bf = book_stats(long_short_spread(Zf, R, frac))
    be = book_stats(long_short_spread(Ze, R, frac))
    full_ic = float(np.nanmean(ic_f))
    exp_ic = float(np.nanmean(ic_e))
    return {
        "full_ic": full_ic,
        "exp_ic": exp_ic,
        "full_ic_t": newey_west_t(ic_f, nw_lags),
        "exp_ic_t": newey_west_t(ic_e, nw_lags),
        "full_sharpe": bf["sharpe_abs"],
        "exp_sharpe": be["sharpe_abs"],
        "full_spread_t": bf["t_nw"],
        "exp_spread_t": be["t_nw"],
        "ic_gap": full_ic - exp_ic,
        "abs_ic_gap": abs(full_ic) - abs(exp_ic),
        "sharpe_gap": bf["sharpe_abs"] - be["sharpe_abs"],
        "n_days_ic": int(np.isfinite(ic_e).sum()),
    }


# --------------------------------------------------------------------------- #
# Seed-robust aggregation (house rule: >= 20 seeds for any synthetic claim)
# --------------------------------------------------------------------------- #
def seed_robust(world_fn, n_seeds: int = 20, base_seed: int = 837,
                frac: float = 0.2, min_periods: int = 60, **world_kw) -> dict:
    """Average :func:`leak_report` over ``n_seeds`` independent synthetic worlds.

    Reports the across-seed mean of each metric and, crucially, how many seeds each method flags as
    "significant" (|IC NW t| >= 2): the leak (full-sample) should light up on (almost) every seed of
    the non-stationary null, the honest (expanding) method on (almost) none — the machinery-proof
    that expanding is unbiased and full is contaminated.
    """
    keys = ["full_ic", "exp_ic", "full_ic_t", "exp_ic_t", "full_sharpe", "exp_sharpe",
            "ic_gap", "abs_ic_gap", "sharpe_gap"]
    acc = {k: [] for k in keys}
    full_hits = exp_hits = 0
    for s in range(n_seeds):
        X, R = world_fn(seed=base_seed + s, **world_kw)
        rep = leak_report(X, R, frac=frac, min_periods=min_periods)
        for k in keys:
            acc[k].append(rep[k])
        if np.isfinite(rep["full_ic_t"]) and abs(rep["full_ic_t"]) >= 2:
            full_hits += 1
        if np.isfinite(rep["exp_ic_t"]) and abs(rep["exp_ic_t"]) >= 2:
            exp_hits += 1
    out = {k: float(np.nanmean(acc[k])) for k in keys}
    out["n_seeds"] = n_seeds
    out["full_sig_seeds"] = full_hits
    out["exp_sig_seeds"] = exp_hits
    return out


# --------------------------------------------------------------------------- #
# Robustness sweeps — how the leak scales with horizon and sample length
# --------------------------------------------------------------------------- #
def horizon_sweep(horizons=(1, 5, 10, 20, 40), n_seeds: int = 10, base_seed: int = 837,
                  n_names: int = 60, n_days: int = 1000, world_fn=None) -> pd.DataFrame:
    """Leak vs forward-return horizon on the non-stationary null: the gap grows with the horizon."""
    from . import data as _data
    world_fn = world_fn or _data.null_nonstationary
    rows = []
    for h in horizons:
        r = seed_robust(world_fn, n_seeds=n_seeds, base_seed=base_seed,
                        n_names=n_names, n_days=n_days, horizon=h)
        rows.append({"horizon": h, "full_ic": r["full_ic"], "exp_ic": r["exp_ic"],
                     "full_sharpe": r["full_sharpe"], "exp_sharpe": r["exp_sharpe"]})
    return pd.DataFrame(rows).set_index("horizon")


def length_sweep(lengths=(250, 500, 1000, 2000), n_seeds: int = 10, base_seed: int = 837,
                 n_names: int = 60, horizon: int = 10, world_fn=None) -> pd.DataFrame:
    """Leak vs sample length on the non-stationary null: the leak dilutes as the sample grows.

    A longer sample makes each future observation a smaller share of the full-sample mean, so the
    look-ahead contamination shrinks — the tell-tale signature that this is a finite-sample leak, not
    a real effect."""
    from . import data as _data
    world_fn = world_fn or _data.null_nonstationary
    rows = []
    for T in lengths:
        r = seed_robust(world_fn, n_seeds=n_seeds, base_seed=base_seed,
                        n_names=n_names, n_days=T, horizon=horizon)
        rows.append({"n_days": T, "full_ic": r["full_ic"], "exp_ic": r["exp_ic"],
                     "full_sharpe": r["full_sharpe"], "exp_sharpe": r["exp_sharpe"]})
    return pd.DataFrame(rows).set_index("n_days")
