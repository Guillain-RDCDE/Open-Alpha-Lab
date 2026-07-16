"""Strategy + inference for Study 782 — CEO-Name-Length.

The claim: **surname length predicts returns.** Shape B — a cross-sectional characteristic
sort. Standardise the characteristic across the universe, sort into terciles, and hold a
dollar-neutral **long longest-surname tercile / short shortest-surname tercile** book,
equal-weighted within each leg, rebalanced monthly. A positive long/short mean would say
"longer surnames earn more"; a negative one, "shorter surnames earn more"; zero is the
honest prior.

Because each month's long/short return is the natural unit, the primary statistic is a
**one-sample t** of the monthly LS return series. A **label-shuffle placebo** (randomly
re-assigning surname lengths across tickers and recomputing the LS mean) checks whether the
observed spread is anything but the luck of which names happened to land in which tercile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

Q = 1.0 / 3.0       # tercile breakpoints
COST_BPS = 5.0      # one-way, per leg, per monthly rebalance


# --------------------------------------------------------------------------- #
# The sort: characteristic -> monthly long/short return series
# --------------------------------------------------------------------------- #
def leg_masks(chars: pd.Series, q: float = Q) -> tuple[pd.Index, pd.Index]:
    """(long_names, short_names): long = top-q surname length, short = bottom-q.

    Ties at the breakpoint are resolved by rank so both legs are non-empty and roughly
    balanced. Long the LONGEST surnames, short the SHORTEST — so a positive LS return means
    'longer surnames earn more'."""
    r = chars.rank(method="first")
    n = len(chars)
    k = max(1, int(round(n * q)))
    order = r.sort_values()
    short_names = order.index[:k]
    long_names = order.index[-k:]
    return long_names, short_names


def long_short_series(rets: pd.DataFrame, chars: pd.Series, q: float = Q,
                      cost_bps: float = 0.0) -> pd.Series:
    """Monthly dollar-neutral LS return: mean(long leg) - mean(short leg), equal weight.

    ``cost_bps`` (one-way, per leg) is charged as a flat per-month round-trip drag
    ``2 * cost_bps/1e4`` — a deliberately CONSERVATIVE upper bound (it assumes the whole
    book turns over every month, whereas a static-membership equal-weight book only drifts).
    """
    long_names, short_names = leg_masks(chars.reindex(rets.columns).dropna(), q=q)
    ls = rets[long_names].mean(axis=1) - rets[short_names].mean(axis=1)
    if cost_bps:
        ls = ls - 2.0 * cost_bps / 1e4
    return ls


def tercile_means(rets: pd.DataFrame, chars: pd.Series, q: float = Q) -> dict:
    """Mean monthly return of the short (shortest-surname) and long (longest) legs."""
    long_names, short_names = leg_masks(chars.reindex(rets.columns).dropna(), q=q)
    return {
        "short_leg_mean": float(rets[short_names].mean(axis=1).mean()),
        "long_leg_mean": float(rets[long_names].mean(axis=1).mean()),
        "n_long": len(long_names), "n_short": len(short_names),
    }


# --------------------------------------------------------------------------- #
# Inference primitives (shared shape with the template)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- here the unit is the monthly LS return."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hit_rate(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def sharpe(x: np.ndarray, periods: int = 12) -> float:
    """Annualised Sharpe of a monthly return series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))


# --------------------------------------------------------------------------- #
# Label-shuffle placebo: is the spread anything but tercile luck?
# --------------------------------------------------------------------------- #
def placebo_pvalue(rets: pd.DataFrame, chars: pd.Series, q: float = Q,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 798,
                   tail: str = "two") -> dict:
    """Randomly PERMUTE the surname-length labels across tickers, recompute the LS mean,
    repeat n_seeds x n_draws_per_seed times, and locate the observed LS mean in that null.

    ``tail``: "two" (a claim of any nonzero spread -> p = share of |null| >= |observed|),
    "right" (positive spread) or "left" (negative spread).
    """
    chars = chars.reindex(rets.columns).dropna()
    rets = rets[chars.index]
    obs = float(long_short_series(rets, chars, q=q).mean())
    vals = chars.values
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            perm = pd.Series(rng.permutation(vals), index=chars.index)
            means.append(float(long_short_series(rets, perm, q=q).mean()))
    means = np.asarray(means)
    if tail == "right":
        p = float((means >= obs).mean())
    elif tail == "left":
        p = float((means <= obs).mean())
    else:
        p = float((np.abs(means) >= abs(obs)).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Jackknife: is the spread one name, or broad?
# --------------------------------------------------------------------------- #
def jackknife_t(rets: pd.DataFrame, chars: pd.Series, q: float = Q) -> dict:
    """Leave-one-ticker-out t-stat range of the monthly LS series."""
    chars = chars.reindex(rets.columns).dropna()
    rets = rets[chars.index]
    ts = []
    for drop in chars.index:
        keep = [c for c in chars.index if c != drop]
        ls = long_short_series(rets[keep], chars[keep], q=q)
        ts.append(one_sample_t(ls.values)["t"])
    return {"lo": float(np.nanmin(ts)), "hi": float(np.nanmax(ts)), "n": len(ts)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, q: float = Q) -> dict:
    """Run the LS-sort + one-sample-t detector on a synthetic world with a planted
    characteristic->return slope. Positive ``bump`` should lift the t monotonically."""
    rets, chars, _ = dt.synthetic_world(bump=bump, seed=seed)
    ls = long_short_series(rets, chars, q=q)
    return one_sample_t(ls.values)
