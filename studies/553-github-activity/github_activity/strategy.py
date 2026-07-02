"""Engine + honest controls — Study 553 (GitHub-Activity).

The believers' rule: each rebalance, rank the tech field by **public GitHub velocity** (the blended
commit / merged-PR / new-star run-rate of a firm's open-source org, z-scored across the field). The
firms whose engineering output is *accelerating* are shipping faster than the market realises, so
go **long the top velocity decile / short the bottom** and hold to the next rebalance. The claimed
alpha is the innovation the crowd hasn't priced yet.

We measure it, honestly, on the deterministic synthetic panel (a real point-in-time GitHub->ticker
tape does not exist on a retail stack — see ``data.py``):

1. **Per-period cross-sectional IC.** Each period, the Spearman rank correlation between velocity
   and *forward* return. The headline is the mean IC across periods and its *t*-stat
   (mean / (std / sqrt(#periods))) — the honest test of a cross-sectional nowcast. `WEAK` needs a
   plausibly-real IC; but with no real tape the ceiling is `WEAK` regardless.

2. **A label-shuffle placebo null.** Shuffle velocity against forward returns *within each period*
   many times; the p-value is the fraction of shuffles whose |mean IC| ≥ the observed. A real
   cross-sectional signal sits in the tail; noise does not.

3. **The decile long-short book.** Long top-velocity names, short bottom, equal weight, rebalanced
   each period. Report the gross spread, its *t*, and the **net** after one-way costs on both legs
   plus an annual **borrow** on the short leg (you pay to be short the laggards).

4. **A robustness sweep.** Vary the tail fraction (deciles vs terciles vs halves) and the sub-sample
   (first half / second half of the panel) — a real IC is sign-stable; a fragile one is not.

5. **A seed-robust synthetic positive control.** Average the mean-IC *t* over >= 20 synthetic worlds
   for each planted ``ic``: the engine must stay flat at the null (``ic = 0``) and clear the bar as
   the planted IC grows. No single lucky seed can manufacture significance.

Execution convention (one documented lag): velocity is a *trailing* z-score known at the rebalance
close; it is scored against the *forward* (next-period) return. No future data enters the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 4  # quarterly rebalance


# --------------------------------------------------------------------------- #
# Spearman rank correlation (no scipy dependency)
# --------------------------------------------------------------------------- #
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank of ``x`` (ties get the mean rank), matching scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # resolve ties to average rank
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or b.size < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- #
# Per-period cross-sectional IC and the headline mean-IC t
# --------------------------------------------------------------------------- #
def period_ics(panel: pd.DataFrame) -> pd.Series:
    """Spearman IC (velocity vs forward return) for each period."""
    out = {}
    for t, g in panel.groupby("period"):
        out[t] = spearman(g["gh_vel"].to_numpy(), g["fwd_ret"].to_numpy())
    return pd.Series(out, name="ic").sort_index()


def ic_summary(panel: pd.DataFrame) -> dict:
    """Mean IC across periods, its std, t-stat and n (the headline nowcast test).

    t = mean_IC / (std_IC / sqrt(#periods)); H0: mean IC = 0.
    """
    ics = period_ics(panel).dropna().to_numpy(dtype=float)
    n = ics.size
    if n < 3:
        return {"mean_ic": float("nan"), "std_ic": float("nan"),
                "t": float("nan"), "n": n, "hit": float("nan")}
    mean = float(ics.mean())
    sd = float(ics.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    hit = float((ics > 0).mean())  # fraction of periods the IC is positive
    return {"mean_ic": mean, "std_ic": sd, "t": float(t), "n": n, "hit": hit}


# --------------------------------------------------------------------------- #
# Placebo / label-shuffle null
# --------------------------------------------------------------------------- #
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 553) -> float:
    """Within-period label-shuffle p-value for the mean IC.

    Each permutation shuffles velocity against forward returns *inside each period* and recomputes
    the mean IC; p = fraction of shuffles with |mean IC| >= observed. Preserves the per-period
    cross-sectional structure while breaking the velocity->return link.
    """
    obs = abs(ic_summary(panel)["mean_ic"])
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    groups = [(g["gh_vel"].to_numpy(dtype=float), g["fwd_ret"].to_numpy(dtype=float))
              for _, g in panel.groupby("period")]
    count = 0
    for _ in range(n_perm):
        ics = []
        for vel, ret in groups:
            perm = rng.permutation(len(vel))
            ics.append(spearman(vel[perm], ret))
        m = np.nanmean(ics)
        if abs(m) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# The decile long-short book
# --------------------------------------------------------------------------- #
def long_short_book(panel: pd.DataFrame, frac: float = 0.2) -> dict:
    """Long top-``frac`` velocity / short bottom-``frac``, equal weight, per-period spread.

    Returns per-period spread series, the mean annualised spread, its t-stat and n.
    """
    spreads = {}
    for t, g in panel.groupby("period"):
        g = g.sort_values("gh_vel")
        k = max(1, int(round(len(g) * frac)))
        low = g.head(k)["fwd_ret"].mean()
        high = g.tail(k)["fwd_ret"].mean()
        spreads[t] = float(high - low)  # long high-velocity, short low-velocity
    s = pd.Series(spreads, name="spread").sort_index()
    x = s.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return {"spread": s, "mean_ann": float("nan"), "t": float("nan"), "n": n}
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    return {"spread": s, "mean_per": mean, "mean_ann": mean * PERIODS_PER_YEAR,
            "t": float(t), "n": n}


def net_spread(book: dict, cost_bps: float = 10.0, borrow_ann_bps: float = 100.0) -> dict:
    """Net the long-short book for one-way costs (both legs, each rebalance) + short-leg borrow.

    Rebalanced every period (quarter): charge ``cost_bps`` one-way on each leg's full turnover per
    rebalance (assume full turnover — velocity ranks churn), i.e. 2 legs x cost per period; plus an
    annual ``borrow_ann_bps`` on the short leg, pro-rated per period.
    """
    if "mean_per" not in book:
        return {"gross_ann": float("nan"), "net_ann": float("nan")}
    cost_per = 2.0 * cost_bps * 1e-4
    borrow_per = borrow_ann_bps * 1e-4 / PERIODS_PER_YEAR
    net_per = book["mean_per"] - cost_per - borrow_per
    return {"gross_ann": book["mean_ann"], "net_ann": net_per * PERIODS_PER_YEAR}


# --------------------------------------------------------------------------- #
# Robustness sweep
# --------------------------------------------------------------------------- #
def robustness(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean IC / t / long-short spread across tail fractions and sub-samples."""
    periods = sorted(panel["period"].unique())
    mid = periods[len(periods) // 2]
    subs = {
        "full": panel,
        "first half": panel[panel["period"] < mid],
        "second half": panel[panel["period"] >= mid],
    }
    rows = []
    for sname, sub in subs.items():
        ics = ic_summary(sub)
        for frac, flabel in [(0.1, "deciles"), (0.2, "quintiles"), (1 / 3, "terciles")]:
            bk = long_short_book(sub, frac=frac)
            rows.append({
                "sample": sname, "split": flabel,
                "mean_ic": ics["mean_ic"], "ic_t": ics["t"],
                "ls_ann": bk.get("mean_ann", float("nan")), "ls_t": bk["t"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Seed-robust synthetic positive control (house rule: >= 20 seeds)
# --------------------------------------------------------------------------- #
def synthetic_mean_t(data_mod, ic: float, n_seeds: int = 25, base_seed: int = 553) -> float:
    """Average the mean-IC t-stat over ``n_seeds`` synthetic worlds for a planted ``ic``.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single lucky
    RNG seed can manufacture significance. Returns the mean IC-t across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(ic=ic, seed=s)
        ts.append(ic_summary(panel)["t"])
    return float(np.nanmean(ts))
