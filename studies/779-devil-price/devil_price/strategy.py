"""Strategy + inference for Study 779 — Devil-Price.

The claim: **a stock sitting on a "666" price level then underperforms.** Rendered as a
weekly **cross-sectional characteristic sort** (shape B) on a real S&P 100 universe:

* **The characteristic.** Each name's ``devil_distance`` — the circular log-price distance
  from its price to the nearest 666 level (6.66 / 66.6 / 666 / …), in ``[0, 0.5]``. Small
  = "on the beast."

* **The sort.** At each weekly rebalance, split the cross-section into quintiles by devil
  distance. The **devil** leg is the nearest quintile (smallest distance); the **clean**
  leg is the farthest quintile. The dollar-neutral long/short spread is
  ``LS = clean − devil``. Cross-sectionally demeaned returns make the whole thing
  market-neutral; ``SPY`` is used only for the "does the devil quintile lag the *market*?"
  cross-check.

* **The forward horizons.** Forward simple return over the next FWD_SHORT (1 week) and
  FWD_LONG (1 month) sessions. If the superstition bit, the devil quintile's demeaned
  forward return is negative and the LS spread is positive.

The primary statistic is a **one-sample t** of the per-rebalance spread across independent
weekly rebalances (the rebalances do overlap at the 1-month horizon, which we flag). A
**random-bucket placebo** (drawing random quintiles instead of the devil sort) checks
whether the devil characteristic explains any of the spread. A **leave-one-year-out
jackknife** checks that no single year drives it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

FWD_SHORT = 5     # ~1 week
FWD_LONG = 21     # ~1 month
SPACING = 5       # weekly rebalance
MIN_NAMES = 30    # need a real cross-section to sort into quintiles
QFRAC = 0.20      # quintile buckets
COST_BPS = 5.0    # one-way, per leg


# --------------------------------------------------------------------------- #
# Panel -> per-rebalance cross-sections (devil distance + forward returns)
# --------------------------------------------------------------------------- #
def build_panel_events(panel: pd.DataFrame, spy: pd.Series | None = None,
                       fwd_s: int = FWD_SHORT, fwd_l: int = FWD_LONG,
                       spacing: int = SPACING, min_names: int = MIN_NAMES) -> list[dict]:
    """One entry per weekly rebalance: the valid cross-section's devil distances and
    forward returns.

    A name is INCLUDED at a rebalance only if it has a price on that date and on both
    forward horizons; a rebalance is kept only if at least ``min_names`` names qualify.
    """
    idx = panel.index
    vals = panel.values  # dates x names
    spy_vals = spy.reindex(idx).values if spy is not None else None
    n = len(idx)
    events = []
    for p in range(spacing, n - fwd_l - 1, spacing):
        row0 = vals[p]
        row_s = vals[p + fwd_s]
        row_l = vals[p + fwd_l]
        ok = np.isfinite(row0) & np.isfinite(row_s) & np.isfinite(row_l) & (row0 > 0)
        if ok.sum() < min_names:
            continue
        pr = row0[ok]
        dist = dt.devil_distance(pr)
        fs = row_s[ok] / pr - 1.0
        fl = row_l[ok] / pr - 1.0
        ev = dict(date=idx[p], n=int(ok.sum()), dist=dist, fwd_s=fs, fwd_l=fl)
        if spy_vals is not None and np.isfinite(spy_vals[p]) and spy_vals[p] > 0:
            ev["spy_s"] = float(spy_vals[p + fwd_s] / spy_vals[p] - 1.0)
            ev["spy_l"] = float(spy_vals[p + fwd_l] / spy_vals[p] - 1.0)
        else:
            ev["spy_s"] = float("nan")
            ev["spy_l"] = float("nan")
        events.append(ev)
    return events


def _bucket_means(dist: np.ndarray, vals: np.ndarray, frac: float = QFRAC):
    """(near_mean, far_mean): mean of ``vals`` over the smallest-distance and
    largest-distance ``frac`` slices of the cross-section."""
    order = np.argsort(dist, kind="stable")
    k = max(1, int(round(len(dist) * frac)))
    near = vals[order[:k]].mean()
    far = vals[order[-k:]].mean()
    return float(near), float(far)


def build_sort_table(events: list[dict], frac: float = QFRAC) -> pd.DataFrame:
    """Per-rebalance devil/clean bucket returns, demeaned spread, and vs-SPY abnormal."""
    rows = []
    for ev in events:
        r = dict(date=ev["date"], n=ev["n"])
        for h in ("s", "l"):
            fwd = ev[f"fwd_{h}"]
            um = float(fwd.mean())
            near, far = _bucket_means(ev["dist"], fwd, frac)
            r[f"um_{h}"] = um
            r[f"devil_{h}"] = near
            r[f"clean_{h}"] = far
            r[f"devil_{h}_dm"] = near - um       # devil quintile abnormal (vs peers)
            r[f"clean_{h}_dm"] = far - um
            r[f"ls_{h}"] = far - near            # clean minus devil (folklore: > 0)
            r[f"devil_{h}_abn"] = near - ev.get(f"spy_{h}", float("nan"))  # vs market
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives (shared with the template)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0."""
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


# --------------------------------------------------------------------------- #
# Random-bucket placebo: does the devil characteristic explain the spread?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: list[dict], horizon: str = "s", frac: float = QFRAC,
                   n_seeds: int = 20, n_draws_per_seed: int = 100, base_seed: int = 789,
                   tail: str = "right") -> dict:
    """Replace the devil sort with RANDOM quintiles and recompute the clean−devil spread.

    For each rebalance we draw two disjoint random buckets of the same size the real sort
    uses and average ``random_far − random_near`` across rebalances; repeated
    ``n_seeds × n_draws_per_seed`` times. If the real spread sits in the tail of this null,
    the devil characteristic — not bucket size — carried it. ``tail='right'`` for the
    folklore's positive clean−devil spread.
    """
    obs = float(np.mean([_bucket_means(e["dist"], e[f"fwd_{horizon}"], frac)[1]
                         - _bucket_means(e["dist"], e[f"fwd_{horizon}"], frac)[0]
                         for e in events]))
    fwds = [e[f"fwd_{horizon}"] for e in events]
    sizes = [max(1, int(round(len(f) * frac))) for f in fwds]
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            spreads = []
            for f, k in zip(fwds, sizes):
                perm = rng.permutation(len(f))
                near = f[perm[:k]].mean()
                far = f[perm[k:2 * k]].mean()
                spreads.append(far - near)
            means.append(np.mean(spreads))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Forward return by devil-distance decile (the anatomy figure)
# --------------------------------------------------------------------------- #
def distance_profile(events: list[dict], horizon: str = "s", n_bins: int = 10) -> pd.Series:
    """Mean demeaned forward return in each devil-distance decile, pooled across
    rebalances. Bin 0 = nearest the beast; bin ``n_bins-1`` = farthest."""
    acc = [[] for _ in range(n_bins)]
    for ev in events:
        dist = ev["dist"]
        fwd = ev[f"fwd_{horizon}"]
        dm = fwd - fwd.mean()
        order = np.argsort(dist, kind="stable")
        m = len(order)
        for j, i in enumerate(order):
            b = min(n_bins - 1, j * n_bins // m)
            acc[b].append(dm[i])
    return pd.Series([float(np.mean(a)) if a else float("nan") for a in acc],
                     index=range(n_bins))


# --------------------------------------------------------------------------- #
# Leave-one-year-out jackknife of the spread t-stat
# --------------------------------------------------------------------------- #
def jackknife_years(table: pd.DataFrame, col: str = "ls_s") -> dict:
    """Leave-one-calendar-year-out t of ``col`` — is the spread broad across years or one
    lucky year?"""
    yrs = pd.DatetimeIndex(table["date"]).year
    full = one_sample_t(table[col].values)["t"]
    ts = []
    for y in sorted(set(yrs)):
        sub = table[yrs != y]
        ts.append(one_sample_t(sub[col].values)["t"])
    return {"full_t": full, "lo": float(min(ts)), "hi": float(max(ts)), "n_years": len(ts)}


# --------------------------------------------------------------------------- #
# Synthetic positive control (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, horizon: str = "s",
                     frac: float = QFRAC, side: str = "devil") -> dict:
    """Run the sort detector on a synthetic world with a planted devil-quintile drag.

    ``side='devil'`` returns the one-sample t of the devil quintile's demeaned forward
    return (should go NEGATIVE as ``bump`` rises — "touching 666 then underperforms");
    ``side='ls'`` returns the clean−devil spread t (should go POSITIVE)."""
    prices, _ = dt.synthetic_world(bump=bump, seed=seed)
    events = build_panel_events(prices, spy=None, fwd_s=FWD_SHORT, fwd_l=FWD_SHORT,
                                spacing=SPACING, min_names=10)
    table = build_sort_table(events, frac=frac)
    col = f"devil_{horizon}_dm" if side == "devil" else f"ls_{horizon}"
    return one_sample_t(table[col].values)
