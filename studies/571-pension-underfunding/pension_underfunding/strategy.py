"""The engine and its honest controls — Study 571 (Pension-Underfunding).

The claim (Franzoni & Marin 2006, *"Pension Plan Funding and Stock Market Efficiency,"* JF): the
firms whose defined-benefit pension plans are most **underfunded** — the biggest pension hole
relative to equity — go on to earn the **lowest** subsequent returns, because the market
under-reacts to this senior, off-balance-sheet liability. It is a *negative*-return anomaly: a
large hole predicts underperformance, so the tradable expression is **long well-funded, short
underfunded**.

This module measures that on the synthetic cross-section (the only tape available for free — see
``data.py``):

1. **The signal.** The pension hole scaled by equity, ``funding_gap = (assets - PBO)/mktcap``
   (negative = underfunded). We z-score its *depth* across the panel so higher = deeper hole.

2. **The quintile sort.** Rank by hole depth; form a *well-funded* bucket (smallest hole) and an
   *underfunded* bucket (deepest hole) and compare forward returns. The puzzle predicts
   well-funded > underfunded.

3. **The long-short.** Report the spread, a two-sample (Welch) *t* on the bucket forward returns,
   a **label-shuffle placebo** null, and costs + a short borrow on the underfunded leg.

4. **The firm-level IC / slope.** The information coefficient — Pearson corr(hole_depth, forward
   return) — and an OLS slope; the *sign* is the puzzle (negative = deeper hole -> lower return).

5. **A robustness sweep.** Re-run with quintile / decile / tercile tails and a market-cap-weighted
   variant, so the sign must survive design choices.

6. **A synthetic positive control.** A deterministic world with a planted effect
   (``underfunding_alpha < 0``); the engine must recover a negative slope / positive long-short and
   stay flat at the null — seed-robust over >= 20 seeds (house rule).

Execution convention: the funding measure is scored from the *last reported* pension footnote and
the position is entered *after* that report is public and held over the *forward* window. In the
synthetic panel this is baked in — ``forward_ret`` is strictly the return *after* the funding
measure is observed, so there is no look-ahead. The single execution lag (report -> entry) is the
documented convention; a one-period-later entry only shrinks the magnitude, it does not flip the
sign in the control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The signal — depth of the pension hole, z-scored
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def hole_depth(panel: pd.DataFrame) -> pd.Series:
    """Depth of the pension hole, z-scored across the panel; higher = more underfunded.

    depth = z( -funding_gap ) = z( (PBO - assets) / mktcap )

    A firm with a big obligation and few plan assets relative to its equity has a deep hole and a
    high depth score — exactly the firms Franzoni-Marin say earn low returns.
    """
    if panel.empty:
        return pd.Series(dtype=float)
    return _z(-panel["funding_gap"]).rename("hole_depth")


# ---------------------------------------------------------------------------
# The sort and the long-short
# ---------------------------------------------------------------------------
def bucket_returns(panel: pd.DataFrame, frac: float = 0.2, weight: str = "equal") -> dict:
    """Sort by hole depth; return well-funded / underfunded bucket forward-return means + spread.

    ``frac`` is the tail fraction in each bucket (0.2 = quintiles). ``weight`` is ``"equal"`` or
    ``"cap"`` (market-cap-weighted bucket means). The puzzle predicts well-funded > underfunded, so
    a *positive* spread (well-funded − underfunded) is the tradable long-short.
    """
    if panel.empty:
        return {}
    d = hole_depth(panel)
    df = panel.assign(hole_depth=d).sort_values("hole_depth")
    k = max(1, int(round(len(df) * frac)))
    well = df.head(k)             # smallest hole (well-funded)
    under = df.tail(k)            # deepest hole (underfunded)

    def _mean(bucket: pd.DataFrame) -> float:
        r = bucket["forward_ret"]
        if weight == "cap":
            w = bucket["mktcap"]
            return float(np.average(r, weights=w))
        return float(r.mean())

    well_m, under_m = _mean(well), _mean(under)
    return {
        "n": int(len(df)),
        "k": int(k),
        "well_mean": well_m,
        "under_mean": under_m,
        "spread": float(well_m - under_m),
        "well_ret": well["forward_ret"],
        "under_ret": under["forward_ret"],
    }


def long_short_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) *t* on well-funded − underfunded forward returns.

    The puzzle's tradable spread is a cross-sectional difference in means, so the honest test is a
    Welch two-sample *t* between the two buckets' forward returns. Returns spread, *t*, n.
    """
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["well_ret"].to_numpy(dtype=float)
    b = buckets["under_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# The firm-level relation — the sign IS the puzzle
# ---------------------------------------------------------------------------
def cross_section_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on hole depth across all names + the information coefficient.

    Slope < 0 (and IC < 0) = the Franzoni-Marin puzzle (deeper hole -> lower return). Returns the
    slope, its *t*-stat, and the Pearson IC between hole depth and forward return.
    """
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "ic": float("nan"), "n": 0}
    d = hole_depth(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(d), d])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    ic = float(np.corrcoef(d, y)[0, 1]) if d.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "ic": ic,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, frac: float = 0.2,
                   seed: int = 571) -> float:
    """Label-shuffle placebo p-value for the long-short spread.

    Shuffle the hole-depth *labels* against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose \\|spread\\| >= the observed \\|spread\\|. A real cross-sectional
    signal sits in the tail; noise does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_returns(panel, frac=frac)["spread"])
    rng = np.random.default_rng(seed)
    d = hole_depth(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * frac)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(d[perm])           # sort returns by shuffled depth
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
    """Long well-funded / short underfunded spread net of costs and short borrow.

    The book is built once and held ``holding_years``: charge a round-trip one-way cost on each leg
    (entry + exit, both legs -> 4 crossings) plus a short borrow on the underfunded leg (annual,
    pro-rated). Deeply underfunded firms are often stressed old-economy names — deliberately
    punitive borrow.
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness sweep — sign must survive design choices
# ---------------------------------------------------------------------------
def robustness_sweep(panel: pd.DataFrame) -> list[dict]:
    """Re-run the long-short across bucket definitions and weighting; report spread + slope-t.

    Designs: quintile/decile/tercile tails, equal- and cap-weighted. A signal worth its name keeps
    its sign across these choices.
    """
    if panel.empty:
        return []
    reg = cross_section_regression(panel)
    designs = [
        ("quintile (20%) eq", 0.2, "equal"),
        ("decile   (10%) eq", 0.1, "equal"),
        ("tercile  (33%) eq", 0.3333, "equal"),
        ("quintile (20%) cap", 0.2, "cap"),
    ]
    out = []
    for label, frac, weight in designs:
        b = bucket_returns(panel, frac=frac, weight=weight)
        ls = long_short_tstat(b)
        out.append({
            "design": label,
            "spread": b["spread"],
            "t": ls["t"],
            "slope_t": reg["slope_t"],
        })
    return out


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_stats(data_mod, underfunding_alpha: float, n_seeds: int = 25,
                         base_seed: int = 571) -> dict:
    """Average the control statistics over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns mean slope-t, mean long-short spread, mean
    long-short t and mean IC across seeds for a planted ``underfunding_alpha``.
    """
    slope_ts, spreads, ls_ts, ics = [], [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(underfunding_alpha=underfunding_alpha, seed=s)
        reg = cross_section_regression(panel)
        b = bucket_returns(panel, frac=0.2)
        ls = long_short_tstat(b)
        slope_ts.append(reg["slope_t"])
        ics.append(reg["ic"])
        spreads.append(b["spread"])
        ls_ts.append(ls["t"])
    return {
        "alpha": underfunding_alpha,
        "mean_slope_t": float(np.nanmean(slope_ts)),
        "mean_ic": float(np.nanmean(ics)),
        "mean_spread": float(np.nanmean(spreads)),
        "mean_ls_t": float(np.nanmean(ls_ts)),
        "n_seeds": n_seeds,
    }
