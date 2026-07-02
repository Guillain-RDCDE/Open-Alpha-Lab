"""The engine and its honest controls — Study 573 (Customer-Concentration).

The claim, at full strength (Patatoukas 2012; Dhaliwal et al. 2016; Hertzel et al. 2008): customer
concentration is a *fundamental risk* factor. A firm that depends on a handful of big customers is
fragile — a lost customer can gut cash flows — so it should show (a) higher forward **volatility**
(the robust part) and (b) either a return **premium** (if the fragility is a priced systematic
risk) or a **discount** (if investors under-price the tail or concentration signals efficiency).

This module measures both legs on the synthetic tape:

1. **The concentration score.** The as-of Herfindahl-style customer-concentration index (higher =
   more concentrated / fragile). In the real world this is a z-scored 10-K disclosure; here it is
   the ``concentration`` column of the synthetic panel.

2. **The risk leg (H_vol).** Sort into a *diversified* tercile (low concentration) and a
   *concentrated* tercile (high). Compare their realised forward volatility. The claim predicts
   concentrated > diversified. A two-sample (Welch) *t* on the two buckets' forward vols.

3. **The return leg (H_ret).** The tradable expression of the *return* claim is a long-short:
   long-concentrated / short-diversified if you believe the premium, the reverse if you believe
   the discount. We report the spread, a two-sample *t*, a **label-shuffle placebo** null, and
   costs + a short borrow (the short leg pays borrow).

4. **The firm-level relations.** OLS of forward vol on concentration (slope > 0 = the risk story)
   and of forward return on concentration (the sign is the premium/discount question).

5. **A synthetic positive control.** Deterministic worlds where the effects are planted; the engine
   must recover them and stay flat at the null — averaged over >= 20 seeds (house rule).

Execution convention: concentration is an *as-of* characteristic (the last reported 10-K), and
forward vol / forward return are measured over the *subsequent* window — a single, documented
execution lag. In the synthetic panel these are separate columns, so no future data enters the
signal (no look-ahead). The robustness sweep and synthetic control use the identical convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The concentration score
# ---------------------------------------------------------------------------
def concentration_score(panel: pd.DataFrame) -> pd.Series:
    """The as-of customer-concentration score; higher = more concentrated / fragile.

    On the synthetic tape this is simply the ``concentration`` column (a Herfindahl-style index).
    Kept as a function so the sort/regression code reads the same as the real-tape desk studies.
    """
    if panel.empty:
        return pd.Series(dtype=float)
    return panel["concentration"].rename("concentration")


# ---------------------------------------------------------------------------
# The sort — buckets on concentration
# ---------------------------------------------------------------------------
def bucket_stats(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Sort by concentration; return diversified / concentrated bucket stats and spreads.

    ``frac`` is the tail fraction in each bucket (0.3 ~ terciles). Returns bucket means for both
    forward vol (the RISK leg) and forward return (the RETURN leg), the spreads, sizes, and the
    per-name series (for the two-sample t-stats).
    """
    if panel.empty:
        return {}
    d = concentration_score(panel)
    df = panel.assign(_c=d).sort_values("_c")
    k = max(1, int(round(len(df) * frac)))
    diversified = df.head(k)   # lowest concentration
    concentrated = df.tail(k)  # highest concentration
    return {
        "n": int(len(df)),
        "k": int(k),
        # RISK leg
        "div_vol": float(diversified["forward_vol"].mean()),
        "conc_vol": float(concentrated["forward_vol"].mean()),
        "vol_spread": float(concentrated["forward_vol"].mean() - diversified["forward_vol"].mean()),
        "div_vol_ser": diversified["forward_vol"],
        "conc_vol_ser": concentrated["forward_vol"],
        # RETURN leg
        "div_ret": float(diversified["forward_ret"].mean()),
        "conc_ret": float(concentrated["forward_ret"].mean()),
        "ret_spread": float(concentrated["forward_ret"].mean() - diversified["forward_ret"].mean()),
        "div_ret_ser": diversified["forward_ret"],
        "conc_ret_ser": concentrated["forward_ret"],
    }


def _welch_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch two-sample t for mean(a) - mean(b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"diff": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"diff": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


def vol_tstat(buckets: dict) -> dict:
    """Two-sample t on concentrated - diversified forward VOLATILITY (the risk story)."""
    if not buckets:
        return {"diff": float("nan"), "t": float("nan"), "n": 0}
    return _welch_t(
        buckets["conc_vol_ser"].to_numpy(dtype=float),
        buckets["div_vol_ser"].to_numpy(dtype=float),
    )


def return_tstat(buckets: dict) -> dict:
    """Two-sample t on concentrated - diversified forward RETURN (the premium/discount)."""
    if not buckets:
        return {"diff": float("nan"), "t": float("nan"), "n": 0}
    return _welch_t(
        buckets["conc_ret_ser"].to_numpy(dtype=float),
        buckets["div_ret_ser"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Firm-level relations — the sign IS the claim
# ---------------------------------------------------------------------------
def _ols_slope_t(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope of y on x with its t-stat and Pearson correlation."""
    if len(x) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": len(x)}
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


def vol_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward VOL on concentration. Slope > 0 = the risk story."""
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    x = concentration_score(panel).to_numpy(dtype=float)
    y = panel["forward_vol"].to_numpy(dtype=float)
    return _ols_slope_t(x, y)


def return_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward RETURN on concentration. Slope > 0 = premium, < 0 = discount."""
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    x = concentration_score(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    return _ols_slope_t(x, y)


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null (for the return long-short)
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 573) -> float:
    """Label-shuffle placebo p-value for the concentrated - diversified RETURN spread.

    Shuffle the concentration labels against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| >= the observed |spread|. A real cross-sectional return
    signal sits in the tail; noise does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_stats(panel)["ret_spread"])
    rng = np.random.default_rng(seed)
    c = concentration_score(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * 0.3)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(c[perm])          # sort returns by shuffled concentration
        ys = y[order]
        spread = ys[-k:].mean() - ys[:k].mean()  # concentrated (top) - diversified (bottom)
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow (for the return long-short)
# ---------------------------------------------------------------------------
def net_return_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 100.0,
                      holding_years: float = 1.0) -> float:
    """Long-concentrated / short-diversified return spread net of costs and short borrow.

    The book is built once and held ``holding_years``: charge a round-trip one-way cost on each leg
    (entry + exit, both legs = 4 crossings) plus a short borrow on the shorted (diversified) leg.
    Reported so the RETURN claim is judged gross AND net.
    """
    if not buckets:
        return float("nan")
    gross = buckets["ret_spread"]
    cost = 4.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — sweep the tail fraction (decile / quintile / tercile / half)
# ---------------------------------------------------------------------------
def robustness_sweep(panel: pd.DataFrame, fracs=(0.1, 0.2, 0.3, 0.4)) -> list:
    """Re-run the sort at several tail fractions; report (frac, vol_t, ret_t) each.

    A robust risk story keeps vol_t positive and large across cut points; a robust return story
    keeps its sign. Returns a list of dicts.
    """
    out = []
    for f in fracs:
        b = bucket_stats(panel, frac=f)
        out.append({
            "frac": float(f),
            "vol_spread": b.get("vol_spread", float("nan")),
            "vol_t": vol_tstat(b)["t"],
            "ret_spread": b.get("ret_spread", float("nan")),
            "ret_t": return_tstat(b)["t"],
        })
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic controls (house rule: >= 20 seeds)
# ---------------------------------------------------------------------------
def synthetic_vol_mean_t(data_mod, vol_beta: float, ret_alpha: float = 0.0,
                         n_seeds: int = 25, base_seed: int = 573) -> float:
    """Average the forward-vol-on-concentration slope t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky seed can manufacture significance. Returns the mean vol-regression slope-t across seeds
    for a planted ``vol_beta`` (the risk leg).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(vol_beta=vol_beta, ret_alpha=ret_alpha, seed=s)
        ts.append(vol_regression(panel)["slope_t"])
    return float(np.nanmean(ts))


def synthetic_ret_mean_t(data_mod, ret_alpha: float, vol_beta: float = 0.0,
                         n_seeds: int = 25, base_seed: int = 573) -> float:
    """Average the forward-return-on-concentration slope t over ``n_seeds`` synthetic worlds.

    Returns the mean return-regression slope-t across seeds for a planted ``ret_alpha`` (the return
    leg). Positive planted alpha -> positive mean t; zero -> flat.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(ret_alpha=ret_alpha, vol_beta=vol_beta, seed=s)
        ts.append(return_regression(panel)["slope_t"])
    return float(np.nanmean(ts))
