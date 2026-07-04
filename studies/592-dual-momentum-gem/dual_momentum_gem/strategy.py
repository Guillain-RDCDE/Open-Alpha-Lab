"""Strategy + statistics for Study 592 — Dual Momentum (GEM).

The tested rules are Antonacci's official GEM decision tree, computed on month-end data:

1. **Absolute momentum gate.** At the end of month *m*, compare SPY's trailing ``lookback``-month
   total return against the cumulated T-bill return over the same window.
2. **Relative momentum.** If SPY clears the T-bill hurdle, hold whichever of SPY / EFA has the
   higher trailing return; otherwise hold the bond leg (AGG, IEF-spliced early).
3. **Execution lag — exactly one, documented.** The signal uses data through the close of month
   *m*; the position earns month *m+1*'s return. No same-bar fills anywhere.

Costs: a switch sells one ETF and buys another — **two one-way legs x 100% NAV** each charged at
``cost_bps`` (the desk convention: costs are one-way x NAV). The initial entry pays one leg.
Long-only, no shorts, so no borrow.

Statistics: Newey-West (HAC) *t* on mean monthly active return (serial correlation), Welch *t*
for group splits, and a **random-switching baseline averaged over >= 20 seeds** (single-seed
baselines are banned): each seed permutes GEM's own monthly holdings — same assets, same
time-in-market, random timing — so the test isolates *when* GEM switches, not *what* it holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
LOOKBACK = 12          # Antonacci's canonical 12-month formation window


# --------------------------------------------------------------------------- #
# GEM engine
# --------------------------------------------------------------------------- #
def gem_positions(panel: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    """Asset held during each month, decided from data through the PRIOR month-end.

    ``panel`` has monthly-return columns SPY / EFA / BOND / RF. For each month *t* we form the
    trailing ``lookback``-month cumulative returns of SPY, EFA and the T-bill using months
    ``t-lookback .. t-1`` (i.e. the signal known at the end of month *t-1*), then apply the
    Antonacci tree: SPY-vs-T-bill absolute gate, SPY-vs-EFA relative pick, bonds otherwise.
    Months without a full formation window (or without a bond return) are dropped.
    """
    cum = {}
    for col in ("SPY", "EFA", "RF"):
        cum[col] = (1.0 + panel[col]).rolling(lookback).apply(np.prod, raw=True) - 1.0
    mom_spy = cum["SPY"].shift(1)      # known at end of month t-1
    mom_efa = cum["EFA"].shift(1)
    hurdle = cum["RF"].shift(1)

    ok = mom_spy.notna() & mom_efa.notna() & hurdle.notna() & panel["BOND"].notna()
    pos = pd.Series(index=panel.index, dtype=object)
    abs_pass = mom_spy > hurdle
    rel_spy = mom_spy >= mom_efa
    pos[ok & abs_pass & rel_spy] = "SPY"
    pos[ok & abs_pass & ~rel_spy] = "EFA"
    pos[ok & ~abs_pass] = "BOND"
    return pos.dropna()


def run_gem(panel: pd.DataFrame, lookback: int = LOOKBACK,
            cost_bps: float = 0.0) -> pd.DataFrame:
    """Monthly GEM returns (gross and net) plus the position and switch flags.

    Net charges ``cost_bps`` (one-way, x NAV) per traded leg: 2 legs per switch (sell + buy),
    1 leg for the initial entry.
    """
    pos = gem_positions(panel, lookback=lookback)
    rows = []
    prev = None
    for t, asset in pos.items():
        gross = float(panel.loc[t, asset])
        legs = 1 if prev is None else (2 if asset != prev else 0)
        net = gross - legs * cost_bps * 1e-4
        rows.append({"date": t, "asset": asset, "gross": gross, "net": net,
                     "legs": legs, "rf": float(panel.loc[t, "RF"])})
        prev = asset
    return pd.DataFrame(rows).set_index("date")


def benchmark_returns(panel: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """SPY buy-and-hold and a monthly-rebalanced 60/40 (SPY/BOND), aligned to GEM's months."""
    bm = pd.DataFrame(index=index)
    bm["SPY"] = panel.loc[index, "SPY"]
    bm["B6040"] = 0.6 * panel.loc[index, "SPY"] + 0.4 * panel.loc[index, "BOND"]
    bm["RF"] = panel.loc[index, "RF"]
    return bm


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series."""
    x = pd.Series(r).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    denom = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else float("nan")


def perf(r: pd.Series, rf: pd.Series) -> dict:
    """CAGR, ann. vol, Sharpe (EXCESS of T-bill — excess-vs-excess races), max drawdown."""
    r = pd.Series(r).astype(float).dropna()
    rf = pd.Series(rf).reindex(r.index).fillna(0.0)
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("cagr", "vol", "sharpe", "max_dd", "n")}
    eq = (1.0 + r).cumprod()
    yrs = n / MONTHS_PER_YEAR
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    ex = r - rf
    sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) \
        if ex.std(ddof=1) > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd, "n": int(n)}


def active_stats(strat: pd.Series, bench: pd.Series) -> dict:
    """Mean active return (bps/mo and %/yr) + HAC t of the monthly active series."""
    act = (pd.Series(strat) - pd.Series(bench)).dropna()
    return {"mean_bps": float(act.mean() * 1e4),
            "ann_pct": float(act.mean() * MONTHS_PER_YEAR * 100.0),
            "hac_t": hac_tstat(act), "n": int(len(act))}


# --------------------------------------------------------------------------- #
# Random-switching baseline (>= 20 seeds, averaged Welch t)
# --------------------------------------------------------------------------- #
def random_switch_baseline(panel: pd.DataFrame, gem: pd.DataFrame,
                           n_seeds: int = 40, base_seed: int = 592) -> dict:
    """Permute GEM's own monthly holdings at random — same assets, same unconditional
    allocation, random TIMING — one permutation per seed.

    Returns the across-seed mean Welch t (GEM gross vs the shuffled strategy's months), the mean
    baseline CAGR, and the share of seeds GEM's CAGR beats. Averaged over ``n_seeds`` >= 20
    (single-seed baselines are banned).
    """
    assert n_seeds >= 20
    idx = gem.index
    labels = gem["asset"].to_numpy()
    gem_r = gem["gross"].to_numpy(dtype=float)
    rf = panel.loc[idx, "RF"]
    tstats, cagrs, sharpes = [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        shuffled = labels[rng.permutation(len(labels))]
        rr = np.array([panel.loc[t, a] for t, a in zip(idx, shuffled)], dtype=float)
        tstats.append(welch_t(gem_r, rr))
        p = perf(pd.Series(rr, index=idx), rf)
        cagrs.append(p["cagr"])
        sharpes.append(p["sharpe"])
    gem_cagr = perf(gem["gross"], rf)["cagr"]
    return {"n_seeds": n_seeds,
            "welch_t_mean": float(np.mean(tstats)),
            "base_cagr_mean": float(np.mean(cagrs)),
            "base_sharpe_mean": float(np.mean(sharpes)),
            "beat_share": float(np.mean([gem_cagr > c for c in cagrs]))}


# --------------------------------------------------------------------------- #
# Grids and splits
# --------------------------------------------------------------------------- #
def lookback_grid(panel: pd.DataFrame, lookbacks=(3, 6, 9, 12)) -> list[dict]:
    """The placebo grid: is 12 months special, or lucky? Same engine, only the window moves."""
    out = []
    for lb in lookbacks:
        g = run_gem(panel, lookback=lb)
        # align every lookback to the common (shortest-history = lb=12) start for fairness
        out.append({"lookback": lb, "gem": g})
    # common start across the grid so windows race on identical months
    start = max(o["gem"].index.min() for o in out)
    rows = []
    for o in out:
        g = o["gem"][o["gem"].index >= start]
        bm = benchmark_returns(panel, g.index)
        p = perf(g["gross"], bm["RF"])
        a = active_stats(g["gross"], bm["SPY"])
        rows.append({"lookback": o["lookback"], "cagr": p["cagr"], "sharpe": p["sharpe"],
                     "max_dd": p["max_dd"], "act_ann_pct": a["ann_pct"], "hac_t": a["hac_t"],
                     "n": p["n"]})
    return rows


def subperiod_stats(panel: pd.DataFrame, gem: pd.DataFrame,
                    start=None, end=None, drop=None) -> dict:
    """GEM vs SPY on a sub-window (optionally with a dropped interval, e.g. the GFC)."""
    g = gem.copy()
    if start is not None:
        g = g[g.index >= pd.Timestamp(start)]
    if end is not None:
        g = g[g.index <= pd.Timestamp(end)]
    if drop is not None:
        d0, d1 = pd.Timestamp(drop[0]), pd.Timestamp(drop[1])
        g = g[(g.index < d0) | (g.index > d1)]
    bm = benchmark_returns(panel, g.index)
    pg = perf(g["gross"], bm["RF"])
    ps = perf(bm["SPY"], bm["RF"])
    a = active_stats(g["gross"], bm["SPY"])
    return {"gem": pg, "spy": ps, "active": a,
            "span": (str(g.index.min().date()), str(g.index.max().date()))}
