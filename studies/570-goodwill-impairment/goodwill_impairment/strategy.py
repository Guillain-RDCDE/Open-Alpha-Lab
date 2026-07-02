"""The engine and its honest controls — Study 570 (Goodwill-Impairment).

The claim, at full strength: a high **goodwill / total-assets** ratio flags a firm that overpaid
for acquisitions; that overpayment eventually surfaces as a **goodwill impairment** (write-down),
and the market's slowness to price it leaves high-goodwill firms with (a) a higher **impairment
rate** and (b) **lower forward returns** — a long-low-goodwill / short-high-goodwill spread.

This module measures that, honestly, on the deterministic synthetic panel:

1. **The signal.** ``goodwill_ta`` (goodwill / total assets), the single observable the sort uses.
2. **The quintile sort.** Rank the cross-section, form a *low-goodwill* bucket and a
   *high-goodwill* bucket, and compare their forward returns (and impairment rates).
3. **The long-short.** Long low-goodwill, short high-goodwill. Report the spread, a two-sample
   (Welch) *t* on the bucket forward-return difference, a **label-shuffle placebo** null, and
   costs + a short borrow on the high-goodwill leg.
4. **The impairment link.** Does high goodwill actually predict the write-down? A rank-sorted
   impairment-rate gap and its two-proportion *z*.
5. **The firm-level relation.** OLS of forward return on ``goodwill_ta`` — the *sign* of the slope
   is the puzzle (negative = high goodwill underperforms).
6. **A synthetic positive control.** A deterministic world with a planted effect
   (``ret_alpha < 0``); the engine must recover it and stay flat at the null, averaged over ≥ 20
   seeds so no single lucky RNG seed manufactures significance.

Execution convention: the signal (``goodwill_ta``) is the *year-t* balance, public at the scoring
date; ``impaired`` and ``forward_ret`` are strictly *forward* (year t+1..t+H). No future data
enters the signal (no look-ahead). The single documented lag is: score at year-t close, hold the
low-minus-high book over the forward window. The robustness sweep and the synthetic control use
the identical convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The signal and the sort
# ---------------------------------------------------------------------------
def goodwill_signal(panel: pd.DataFrame) -> pd.Series:
    """The sort key: goodwill / total assets (higher = more bloated = more overpaid)."""
    if panel.empty:
        return pd.Series(dtype=float)
    return panel["goodwill_ta"].rename("goodwill_ta")


def bucket_returns(panel: pd.DataFrame, frac: float = 0.2) -> dict:
    """Sort by goodwill/assets; return low / high bucket forward-return means and the spread.

    ``frac`` is the tail fraction in each bucket (0.2 ≈ quintiles). The claim predicts
    low-goodwill > high-goodwill (a *positive* long-low/short-high spread) and a higher impairment
    rate in the high bucket.

    Returns a dict with bucket means, the spread (low − high), bucket sizes, per-name forward
    returns of each bucket (for the two-sample t), and each bucket's impairment rate.
    """
    if panel.empty:
        return {}
    s = goodwill_signal(panel)
    df = panel.assign(_gw=s).sort_values("_gw")
    k = max(1, int(round(len(df) * frac)))
    low = df.head(k)             # lowest goodwill/assets
    high = df.tail(k)            # highest goodwill/assets
    low_ret = low["forward_ret"]
    high_ret = high["forward_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "low_mean": float(low_ret.mean()),
        "high_mean": float(high_ret.mean()),
        "spread": float(low_ret.mean() - high_ret.mean()),
        "low_ret": low_ret,
        "high_ret": high_ret,
        "low_imp_rate": float(low["impaired"].mean()),
        "high_imp_rate": float(high["impaired"].mean()),
    }


def long_short_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) t on low − high forward returns.

    The claim's tradable spread is a cross-sectional difference in means, so the honest test is a
    Welch t between the low- and high-goodwill buckets' forward returns. Returns spread, t, n.
    """
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["low_ret"].to_numpy(dtype=float)
    b = buckets["high_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# The impairment link — does high goodwill predict the write-down?
# ---------------------------------------------------------------------------
def impairment_gap(buckets: dict) -> dict:
    """High − low bucket impairment-rate gap with a two-proportion z-test.

    The overpaid-M&A story's first leg: the high-goodwill bucket should impair *more*. Returns the
    two rates, the gap (high − low), and a two-proportion z (positive = high impairs more).
    """
    if not buckets:
        return {"gap": float("nan"), "z": float("nan")}
    p_hi = buckets["high_imp_rate"]
    p_lo = buckets["low_imp_rate"]
    k = buckets["k"]
    n1 = n2 = k
    x1, x2 = p_hi * n1, p_lo * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p_hi - p_lo) / se if se > 0 else float("nan")
    return {"high_rate": float(p_hi), "low_rate": float(p_lo),
            "gap": float(p_hi - p_lo), "z": float(z)}


# ---------------------------------------------------------------------------
# The firm-level relation — the sign IS the puzzle
# ---------------------------------------------------------------------------
def cross_section_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on goodwill/assets across all firms.

    Slope < 0 = the overpaid-acquisition puzzle (more goodwill → less return). Returns the slope,
    its t-stat, and the Pearson correlation between goodwill/assets and forward return.
    """
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    x = goodwill_signal(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, frac: float = 0.2, n_perm: int = 2000,
                   seed: int = 570) -> float:
    """Label-shuffle placebo p-value for the long-short spread.

    Shuffle the goodwill *labels* against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| ≥ the observed |spread|. A real cross-sectional signal
    sits in the tail; noise does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_returns(panel, frac)["spread"])
    rng = np.random.default_rng(seed)
    x = goodwill_signal(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * frac)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(x[perm])           # sort returns by shuffled goodwill
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 150.0,
               holding_years: float = 1.0) -> float:
    """Long-low-goodwill / short-high-goodwill spread net of costs and short borrow.

    The book is built once and held ``holding_years``: charge a one-way cost on each leg at both
    entry and exit (4 crossings of ``cost_bps``) plus a short borrow on the high-goodwill leg
    (annual, pro-rated). High-goodwill / recent-acquirer names are exactly the ones prone to
    corporate actions and harder-to-borrow float, so the borrow is deliberately punitive.
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4              # entry + exit, two legs
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — sweep the bucket fraction (quintile / quartile / decile)
# ---------------------------------------------------------------------------
def robustness_sweep(panel: pd.DataFrame, fracs=(0.1, 0.2, 0.25, 0.3)) -> list[dict]:
    """Re-run the sort at several tail fractions; report spread + Welch t + firm slope-t.

    A real cross-sectional effect should hold its sign across sensible bucket definitions; a
    fragile one flips. The firm-level slope-t is fraction-independent (reported once per row for
    context).
    """
    reg = cross_section_regression(panel)
    rows = []
    for f in fracs:
        b = bucket_returns(panel, f)
        ls = long_short_tstat(b)
        rows.append({
            "frac": f,
            "spread": ls["spread"],
            "t": ls["t"],
            "slope_t": reg["slope_t"],
        })
    return rows


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, ret_alpha: float, n_seeds: int = 25,
                     base_seed: int = 570) -> dict:
    """Average the firm-level slope-t (and long-short Welch t) over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages a Welch-style stat over ≥ 20 seeds so
    no single lucky RNG seed can manufacture significance. Returns the mean firm slope-t and mean
    long-short t across seeds for a planted ``ret_alpha``.
    """
    slope_ts, ls_ts = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(ret_alpha=ret_alpha, seed=s)
        slope_ts.append(cross_section_regression(panel)["slope_t"])
        ls_ts.append(long_short_tstat(bucket_returns(panel, 0.2))["t"])
    return {"slope_t": float(np.nanmean(slope_ts)), "ls_t": float(np.nanmean(ls_ts))}
