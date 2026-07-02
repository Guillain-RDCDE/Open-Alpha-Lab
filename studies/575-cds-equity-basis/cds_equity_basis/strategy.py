"""The engine and its honest controls — Study 575 (CDS-Equity-Basis).

The claim: the **CDS-equity basis** (single-name CDS spread minus the equity-implied structural
spread) predicts the firm's forward equity return — a cross-asset *convergence* trade. A *wide*
basis (CDS pricing more distress than the stock) is folklore-supposed to forecast a *falling*
equity (the stock catches down to credit), i.e. a **negative** basis→return relation; a *tight/
negative* basis forecasts a rising equity. The tradable expression sorts names by their basis each
month, longs the low-basis (credit-complacent) names and shorts the high-basis (credit-worried)
ones, and holds one month.

This module measures that, honestly, on a synthetic tape (no free real CDS feed exists — the data
gap is on the SIGNAL axis):

1. **The basis signal.** Cross-sectionally z-score each month's basis, so the signal is
   dollar-neutral and comparable across months.

2. **The panel (pooled) regression.** Forward equity return on the (lagged) basis across all
   name-months — the *sign and t-stat* of the slope is the claim. A **cluster-robust** standard
   error clustered by month guards against the cross-sectional correlation an OLS SE would ignore.

3. **The decile long-short.** Each month, sort by basis; long the lowest-basis decile, short the
   highest. Its mean monthly return, an IID *t*, a **month-label-shuffle placebo** null, and costs
   + a short borrow.

4. **A robustness sweep.** Re-estimate the pooled slope-*t* over rolling sub-samples of the panel
   (early / mid / late thirds) — a real cross-asset lead-lag should keep its sign.

5. **A synthetic positive control.** A deterministic world where the effect is planted
   (``convergence_beta < 0``); the engine must recover a negative slope / positive long-short and
   stay flat at the null — averaged over >= 20 seeds (house rule) so no lucky seed fakes it.

Execution convention: the basis is measured at the close of month *t* (public then); the position
is entered at that close and the forward return is realised over month *t+1*. That one-month lag is
the single documented execution lag — the basis never sees the return it predicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 575


# ---------------------------------------------------------------------------
# The basis signal
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def add_basis_z(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a per-month cross-sectionally z-scored basis column ``basis_z``.

    Z-scoring within each month makes the signal dollar-neutral and comparable across months
    (removing the common credit level so only the *relative* dislocation drives the sort).
    """
    if panel.empty:
        return panel.assign(basis_z=pd.Series(dtype=float))
    out = panel.copy()
    out["basis_z"] = out.groupby("month")["basis_bp"].transform(_z)
    return out


# ---------------------------------------------------------------------------
# The pooled (panel) regression — the sign IS the claim
# ---------------------------------------------------------------------------
def pooled_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward equity return on the z-scored basis, clustered by month.

    Slope < 0 = the convergence claim (a wider basis predicts a lower forward equity return). The
    standard error is **clustered by month** (Liang-Zeger / Cameron-Miller): within a month all
    names share market-wide shocks, so a naive OLS SE would badly overstate significance. Returns
    the slope, its cluster-robust *t*, the pooled correlation, and n.
    """
    if panel.empty or len(panel) < 20:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    df = add_basis_z(panel)
    x = df["basis_z"].to_numpy(dtype=float)
    y = df["forward_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)

    # Cluster-robust (by month) meat matrix.
    months = df["month"].to_numpy()
    meat = np.zeros((2, 2))
    for m in np.unique(months):
        idx = months == m
        Xg = X[idx]
        ug = resid[idx]
        sg = Xg.T @ ug
        meat += np.outer(sg, sg)
    cov = xtx_inv @ meat @ xtx_inv
    # small-sample cluster correction
    G = len(np.unique(months))
    n, k = X.shape
    if G > 1:
        cov *= (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se_slope = float(np.sqrt(max(cov[1, 1], 0.0)))
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(beta[1]),
        "slope_t": float(beta[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# The decile long-short
# ---------------------------------------------------------------------------
def decile_long_short(panel: pd.DataFrame, frac: float = 0.2) -> dict:
    """Monthly long-low-basis / short-high-basis decile spread and its IID t.

    Each month, sort names by basis; long the lowest-``frac`` (credit-complacent, folklore says
    equity rises) and short the highest-``frac`` (credit-worried, equity falls). The convergence
    claim predicts a **positive** long-short return. Returns the per-month spread series, its mean,
    an IID *t* on the monthly spreads, and n_months.
    """
    if panel.empty:
        return {"monthly": pd.Series(dtype=float), "mean": float("nan"), "t": float("nan"), "n": 0}
    spreads = {}
    for m, g in panel.groupby("month"):
        if len(g) < 5:
            continue
        gg = g.sort_values("basis_bp")
        k = max(1, int(round(len(gg) * frac)))
        lo = gg.head(k)["forward_ret"]     # low basis -> long (equity expected up)
        hi = gg.tail(k)["forward_ret"]     # high basis -> short (equity expected down)
        spreads[m] = float(lo.mean() - hi.mean())
    ms = pd.Series(spreads).sort_index()
    if len(ms) < 2:
        return {"monthly": ms, "mean": float("nan"), "t": float("nan"), "n": len(ms)}
    sd = ms.std(ddof=1)
    t = float(ms.mean() / (sd / np.sqrt(len(ms)))) if sd > 0 else float("nan")
    return {"monthly": ms, "mean": float(ms.mean()), "t": t, "n": int(len(ms))}


# ---------------------------------------------------------------------------
# Placebo / month-label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = SEED, frac: float = 0.2) -> float:
    """Month-label-shuffle placebo p-value for the long-short mean.

    Within each month, shuffle the basis labels against forward returns, recompute the decile
    long-short mean, and read the fraction of shuffles whose |mean| >= the observed |mean|. A real
    cross-sectional signal sits in the tail; noise does not.
    """
    if panel.empty:
        return float("nan")
    obs = abs(decile_long_short(panel, frac)["mean"])
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    groups = [(m, g["basis_bp"].to_numpy(), g["forward_ret"].to_numpy())
              for m, g in panel.groupby("month") if len(g) >= 5]
    count = 0
    for _ in range(n_perm):
        spr = []
        for _m, b, y in groups:
            perm = rng.permutation(len(y))
            order = np.argsort(b[perm])       # sort returns by shuffled basis
            ys = y[order]
            k = max(1, int(round(len(ys) * frac)))
            spr.append(ys[:k].mean() - ys[-k:].mean())
        if abs(np.mean(spr)) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_long_short(ls: dict, cost_bps: float = 10.0, borrow_ann_bps: float = 150.0) -> dict:
    """Long-short mean monthly return net of round-trip costs and a short borrow.

    Rebalanced monthly, both legs turn over each month: charge a round-trip one-way cost on each
    leg (entry + exit, both legs) every month, plus a monthly slice of an annual borrow on the
    short leg. The short leg (credit-worried names) is exactly the distressed tail that is
    expensive to borrow, so the borrow is deliberately punitive. Returns gross/net mean monthly
    and the annualised net.
    """
    gross = ls.get("mean", float("nan"))
    if not np.isfinite(gross):
        return {"gross_m": float("nan"), "net_m": float("nan"), "net_ann": float("nan")}
    cost = 4.0 * cost_bps * 1e-4               # entry+exit, two legs
    borrow = borrow_ann_bps * 1e-4 / 12.0      # monthly slice on the short
    net_m = gross - cost - borrow
    return {"gross_m": float(gross), "net_m": float(net_m), "net_ann": float(net_m * 12.0)}


# ---------------------------------------------------------------------------
# Robustness — pooled slope-t over sub-samples
# ---------------------------------------------------------------------------
def robustness_thirds(panel: pd.DataFrame) -> list[dict]:
    """Re-estimate the pooled slope-t over early/mid/late thirds of the month index.

    A genuine cross-asset lead-lag keeps its sign across sub-samples; a spurious one wanders.
    """
    if panel.empty:
        return []
    months = np.sort(panel["month"].unique())
    thirds = np.array_split(months, 3)
    labels = ["early third", "mid third", "late third"]
    out = []
    for lab, ms in zip(labels, thirds):
        sub = panel[panel["month"].isin(ms)]
        reg = pooled_regression(sub)
        out.append({"window": lab, "slope_t": reg["slope_t"], "slope": reg["slope"], "n": reg["n"]})
    return out


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, convergence_beta: float, n_seeds: int = 25,
                     base_seed: int = 575) -> float:
    """Average the pooled (cluster-robust) slope-t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean slope-t across seeds for a
    planted ``convergence_beta``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(convergence_beta=convergence_beta, seed=s)
        ts.append(pooled_regression(panel)["slope_t"])
    return float(np.nanmean(ts))


def synthetic_mean_ls_t(data_mod, convergence_beta: float, n_seeds: int = 25,
                        base_seed: int = 575) -> float:
    """Average the decile long-short IID t over ``n_seeds`` synthetic worlds."""
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(convergence_beta=convergence_beta, seed=s)
        ts.append(decile_long_short(panel)["t"])
    return float(np.nanmean(ts))
