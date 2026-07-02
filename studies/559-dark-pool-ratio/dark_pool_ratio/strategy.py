"""The engine and its honest controls — Study 559 (Dark-Pool-Ratio).

The claim, at full strength: a stock's **dark-pool ratio** (share of volume printed off-exchange)
signals *informed accumulation* — institutions quietly building positions in the dark — so
high-DPR names should earn positive forward returns. The tradable expression is a cross-sectional
sort: long the highest-DPR names, short the lowest, rebalanced each period.

This module measures that, honestly, on the synthetic panel (a real daily DPR tape is not reachable
on a free stack — see ``data.py``; the study is machinery-only and capped below `REAL`):

1. **The information coefficient (IC).** The Spearman rank correlation between the dark-pool ratio
   and the forward return across the cross-section — the standard "does this factor rank stocks?"
   statistic. Its t-stat (Fisher-z) is the headline inference.

2. **The quintile long-short.** Sort by DPR, go long the top ``frac`` (darkest) names and short the
   bottom ``frac`` (most-lit), and compare forward returns — the folklore's tradable spread, with a
   two-sample (Welch) t on top-minus-bottom.

3. **A label-shuffle placebo null.** Shuffle the DPR labels against forward returns and read the
   IC's tail probability — a real cross-sectional signal sits in the tail; a confound-driven or
   noisy one does not.

4. **Costs + borrow.** The long-short is charged a round-trip one-way cost per leg and an annual
   short borrow on the *short* (most-lit) leg.

5. **A multi-window robustness sweep.** The IC re-measured as the planted effect is dialled from
   null up to strong — the sign and significance must move monotonically with the truth.

6. **A seed-robust synthetic positive control (>= 20 seeds).** The engine must recover a positive
   IC-t when a DPR effect is planted and stay flat (t ~ 0) at the null, averaged over many seeds so
   no single lucky RNG seed manufactures significance (house rule).

Execution convention: DPR is measured over the *formation* window (trailing off-exchange share,
fully public at the formation close) and the position is entered at that close and held over the
*forward* window — one documented execution lag, no look-ahead. On the synthetic panel the
formation/forward split is baked into the generator (``forward_ret`` is strictly post-formation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The information coefficient — the headline statistic
# ---------------------------------------------------------------------------
def information_coefficient(panel: pd.DataFrame) -> dict:
    """Spearman rank IC between the dark-pool ratio and forward return, with a Fisher-z t-stat.

    IC > 0 means darker names rank higher in forward return (the folklore). The t-stat uses the
    Fisher z-transform, t = z * sqrt(n - 3), the standard significance test for a rank correlation.
    Returns the IC, its t-stat, and n.
    """
    if panel.empty or len(panel) < 6:
        return {"ic": float("nan"), "ic_t": float("nan"), "n": int(len(panel))}
    x = panel["dark_pool_ratio"].to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    ic = float(stats.spearmanr(x, y).correlation)
    n = len(x)
    if not np.isfinite(ic) or abs(ic) >= 1.0:
        return {"ic": ic, "ic_t": float("nan"), "n": n}
    z = np.arctanh(ic)                    # Fisher z-transform
    t = float(z * np.sqrt(max(n - 3, 1)))
    return {"ic": ic, "ic_t": t, "n": n}


# ---------------------------------------------------------------------------
# The quintile long-short
# ---------------------------------------------------------------------------
def bucket_returns(panel: pd.DataFrame, frac: float = 0.2) -> dict:
    """Sort by dark-pool ratio; long the darkest ``frac``, short the most-lit ``frac``.

    ``frac`` is the tail fraction in each leg (0.2 = quintiles). The folklore predicts
    darkest > most-lit (a positive top-minus-bottom spread). Returns bucket means, the spread, and
    the per-name forward returns of each leg (for the two-sample t).
    """
    if panel.empty:
        return {}
    df = panel.sort_values("dark_pool_ratio")
    k = max(1, int(round(len(df) * frac)))
    lit = df.head(k)          # lowest dark-pool ratio (most lit)
    dark = df.tail(k)         # highest dark-pool ratio (darkest)
    lit_ret = lit["forward_ret"]
    dark_ret = dark["forward_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "dark_mean": float(dark_ret.mean()),
        "lit_mean": float(lit_ret.mean()),
        "spread": float(dark_ret.mean() - lit_ret.mean()),   # long dark, short lit
        "dark_ret": dark_ret,
        "lit_ret": lit_ret,
    }


def long_short_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) t on dark − lit forward returns (long dark, short lit)."""
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["dark_ret"].to_numpy(dtype=float)
    b = buckets["lit_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# The firm-level relation — a plain OLS slope, robustness to the confound
# ---------------------------------------------------------------------------
def cross_section_regression(panel: pd.DataFrame, control_size: bool = False) -> dict:
    """OLS of forward return on the dark-pool ratio (optionally controlling for size).

    Slope > 0 = the folklore (darker names earn more). With ``control_size=True`` the size proxy is
    added as a regressor, so the DPR slope is the *marginal* information beyond the size confound —
    the honest test of whether DPR carries anything the confounder doesn't. Returns the DPR slope,
    its t-stat, and n.
    """
    if panel.empty or len(panel) < 6:
        return {"slope": float("nan"), "slope_t": float("nan"), "n": int(len(panel))}
    d = panel["dark_pool_ratio"].to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    cols = [np.ones_like(d), d]
    if control_size and "size_z" in panel.columns:
        cols.append(panel["size_z"].to_numpy(dtype=float))
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 559) -> float:
    """Label-shuffle placebo p-value for the IC.

    Shuffle the dark-pool-ratio labels against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |IC| >= the observed |IC|. A real cross-sectional signal sits in the
    tail; noise (or a signal the shuffle also carries) does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(information_coefficient(panel)["ic"])
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    x = panel["dark_pool_ratio"].to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    xr = stats.rankdata(x)
    n = len(y)
    count = 0
    for _ in range(n_perm):
        yr = stats.rankdata(y[rng.permutation(n)])
        ic = np.corrcoef(xr, yr)[0, 1]
        if abs(ic) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 50.0,
               holding_years: float = 1.0) -> float:
    """Long-dark / short-lit spread net of costs and short borrow.

    The book is built and held ``holding_years``: charge a round-trip one-way cost on each leg
    (entry + exit, both legs = 4 crossings) plus a short borrow on the *lit* (short) leg. Lit names
    are large and easy to borrow, so the borrow here is modest (50 bps/yr) — the point is honesty,
    not a punitive number.
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — the IC as the planted effect is dialled up
# ---------------------------------------------------------------------------
def effect_sweep(data_mod, alphas, n_seeds: int = 25, base_seed: int = 559) -> list:
    """Mean IC-t over ``n_seeds`` synthetic worlds for each planted ``dpr_alpha``.

    Returns a list of ``(alpha, mean_ic, mean_ic_t)``. The sign and magnitude must move
    monotonically with the planted truth — flat at the null, rising as the effect grows.
    """
    out = []
    for a in alphas:
        ics, ts = [], []
        for s in range(base_seed, base_seed + n_seeds):
            panel, _ = data_mod.synthetic_panel(dpr_alpha=a, seed=s)
            r = information_coefficient(panel)
            ics.append(r["ic"])
            ts.append(r["ic_t"])
        out.append((float(a), float(np.nanmean(ics)), float(np.nanmean(ts))))
    return out


def synthetic_mean_t(data_mod, dpr_alpha: float, n_seeds: int = 25, base_seed: int = 559) -> float:
    """Average the IC-t over ``n_seeds`` synthetic worlds for a planted ``dpr_alpha``.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single lucky
    RNG seed can manufacture significance. Returns the mean IC-t across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(dpr_alpha=dpr_alpha, seed=s)
        ts.append(information_coefficient(panel)["ic_t"])
    return float(np.nanmean(ts))
