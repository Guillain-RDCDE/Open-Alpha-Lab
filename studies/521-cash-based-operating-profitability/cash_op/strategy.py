"""The cash-based operating-profitability long-short, its placebo null, and costs.

Ball, Gerakos, Linnainmaa & Nikolaev (2016): sort firms each year on *cash-based*
operating profitability (accounting operating profit with the accrual block stripped
out); go long the top quintile, short the bottom quintile, annual rebalance. BGLN show
this cash measure predicts the cross-section *better* than the accrual-laden
gross/operating profitability of Novy-Marx (2013, our Study 122). We test the head-to-head
on a large-cap survivor basket.

Engine kept drop-in compatible with the desk's annual-sort factor studies (122, 52, 65):
``quintile_hedge`` + ``summary`` share the same schema.

Inference bar (house): a one-sample t on the annual hedge series must clear |t| >= 2 AND
survive a label-shuffle placebo to earn a REAL stamp. Literature support alone = WEAK.

No look-ahead: the signal at year *y* is the year-*y* fiscal ratio; the strategy acts on
the next-12-month forward return entered one trading day after the fiscal-year-end is
public (lag baked into :func:`data.fetch_panel`).

Survivorship-biased basket (names still trading 2026): named on the Signal axis; positive
results are upper bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# One-way transaction cost (bps of NAV traded), plus annual borrow on the short leg.
ONE_WAY_BPS = 10.0
BORROW_BPS_ANNUAL = 50.0


# ---------------------------------------------------------------------------
# Core sorter
# ---------------------------------------------------------------------------
def quintile_hedge(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    min_names: int = 10,
) -> pd.DataFrame:
    """Annual top-q / bottom-q long-short on a (year x ticker) signal frame.

    Long the top-``q`` quintile, short the bottom-``q`` quintile; equal weight within each
    leg. The hedge for year *y* is ``mean(fwd | top-q) - mean(fwd | bot-q)``.

    Returns a DataFrame indexed by year with columns ``high``, ``low``, ``hedge``,
    ``market`` (equal-weight universe), ``n`` (names available), ``turnover``
    (one-sided name turnover of the long leg vs the prior year).
    """
    rows: dict[int, dict] = {}
    prev_hi: set | None = None
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        common = s.index.intersection(nxt.index)
        if len(common) < min_names:
            continue
        sc = s.reindex(common)
        nc = nxt.reindex(common)
        hi_idx = sc[sc >= sc.quantile(1.0 - q)].index
        lo_idx = sc[sc <= sc.quantile(q)].index
        r_hi = nc.reindex(hi_idx).mean()
        r_lo = nc.reindex(lo_idx).mean()
        if not np.isfinite(r_hi) or not np.isfinite(r_lo):
            continue
        hi_set = set(hi_idx)
        if prev_hi is None:
            turn = np.nan
        else:
            turn = 1.0 - len(hi_set & prev_hi) / max(len(hi_set), 1)
        prev_hi = hi_set
        rows[int(y)] = {
            "high": float(r_hi),
            "low": float(r_lo),
            "hedge": float(r_hi - r_lo),
            "market": float(nc.mean()),
            "n": int(len(common)),
            "turnover": float(turn) if np.isfinite(turn) else np.nan,
        }
    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------
# One-sample inference on the annual hedge series
# ---------------------------------------------------------------------------
def summary(annual_returns: pd.Series) -> dict:
    """Annualised statistics for a yearly return series.

    Returns mean, vol, Sharpe, a one-sample t-stat (mean / (std/sqrt(n)) -- the
    inference-bar number for a short annual series), hit-rate, max drawdown, and *n*.
    """
    r = pd.Series(annual_returns).astype(float).dropna()
    n = len(r)
    keys = ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")
    if n < 2:
        return {k: (np.nan if k != "n" else n) for k in keys}
    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    sr = mean / vol if vol > 0 else np.nan
    se = vol / np.sqrt(n) if vol > 0 else np.nan
    tstat = float(mean / se) if se and se > 0 else np.nan
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean,
        "vol": vol,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Label-shuffle placebo null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_shuffles: int = 500,
    seed: int = 521,
) -> dict:
    """Permutation p-value: shuffle the signal labels *within each year*, recompute the
    hedge mean, and ask how often the shuffled mean beats the real one.

    Destroys the signal-to-return alignment while preserving each year's marginal return
    and signal distributions. ``p`` = fraction of shuffles with |mean| >= |real mean|.
    """
    real = quintile_hedge(signal, fwd_ret, q=q)
    real_mean = float(real["hedge"].mean())
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shuf = signal.copy()
        for y in shuf.index:
            vals = shuf.loc[y].to_numpy()
            mask = np.isfinite(vals)
            perm = vals.copy()
            perm[mask] = rng.permutation(vals[mask])
            shuf.loc[y] = perm
        h = quintile_hedge(shuf, fwd_ret, q=q)
        null_means[i] = float(h["hedge"].mean()) if not h.empty else np.nan
    null_means = null_means[np.isfinite(null_means)]
    p = float((np.abs(null_means) >= abs(real_mean)).mean()) if len(null_means) else np.nan
    return {
        "real_mean": real_mean,
        "null_mean": float(np.mean(null_means)) if len(null_means) else np.nan,
        "null_std": float(np.std(null_means)) if len(null_means) else np.nan,
        "p_value": p,
        "n_shuffles": int(len(null_means)),
    }


# ---------------------------------------------------------------------------
# Costs: one-way bps x NAV x turnover, plus borrow on the short leg
# ---------------------------------------------------------------------------
def apply_costs(
    hedge: pd.DataFrame,
    one_way_bps: float = ONE_WAY_BPS,
    borrow_bps: float = BORROW_BPS_ANNUAL,
) -> pd.DataFrame:
    """Charge realistic costs against the gross annual hedge series.

    Each rebalance trades both legs: turnover *t* implies ``2 * t`` of NAV traded per leg
    transition (sell old names, buy new), and the long-short doubles it across both legs.
    We charge ``4 * one_way_bps * turnover`` per year (two legs, in and out), plus a flat
    annual ``borrow_bps`` on the short leg. Returns the frame with a ``net`` column.
    """
    h = hedge.copy()
    turn = h["turnover"].fillna(h["turnover"].mean())
    trade_cost = 4.0 * (one_way_bps / 1e4) * turn
    borrow_cost = borrow_bps / 1e4
    h["cost"] = trade_cost + borrow_cost
    h["net"] = h["hedge"] - h["cost"]
    return h


# ---------------------------------------------------------------------------
# Deterministic synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_control(premiums=(-0.08, -0.04, 0.0, 0.04, 0.08, 0.12), seed: int = 521) -> pd.DataFrame:
    """Sweep a planted cash-OP premium and confirm the hedge mean is monotone in it.

    Proves the engine is a faithful detector: a strong positive hedge when (and only when)
    a premium is planted, noise at zero. NEVER cited to support a real-tape stamp.
    """
    from . import data

    rows = []
    for prem in premiums:
        cop, _gpa, fwd, _ = data.synthetic_panel(
            n_firms=300, n_years=18, cop_premium=prem, seed=seed
        )
        h = quintile_hedge(cop, fwd)
        s = summary(h["hedge"])
        rows.append(
            {
                "premium": prem,
                "hedge_mean_%": s["mean"] * 100,
                "tstat": s["tstat"],
                "hit_%": s["hit_rate"] * 100,
            }
        )
    return pd.DataFrame(rows)
