"""The engine and its honest controls — Study 548 (Happiness-Index-Country).

The claim, stated as a trade: sort the world's investable equity markets by their **World Happiness
Report** rank, go **long the happiest** countries and **short the gloomiest**, and collect a
spread. It is a cross-country alt-data folklore — the kind of story that survives on a scatterplot
and dies on a standard error, because the investable cross-section is tiny (n ~ 24).

This module measures that, honestly:

1. **The happiness rank.** WHR rank (1 = happiest), converted to a 0..1 *happiness score* so the
   slope has a natural sign (positive slope = happy markets earn more).

2. **The rank-sorted spread.** Sort countries by happiness, form a *happy* bucket (best ranks) and
   a *gloomy* bucket (worst ranks), and compare their forward returns — a long-happy / short-gloomy
   spread with a two-sample (Welch) *t*.

3. **The rank correlation.** Spearman rank correlation between happiness rank and forward return —
   the single number the folklore lives or dies on.

4. **The firm-(country-)level relation.** OLS of forward return on the happiness score; the sign
   of the slope *is* the folklore (positive = happier markets win).

5. **A placebo / label-shuffle null.** Shuffle the happiness labels against forward returns; a real
   cross-sectional signal sits in the tail, noise does not — the spurious-correlation antidote.

6. **Costs + borrow.** The long-happy / short-gloomy book pays a round-trip cost per leg and a
   short borrow on the gloomy leg.

7. **A synthetic positive control.** A deterministic world where the effect is planted
   (``rank_alpha > 0``); the engine must recover a positive slope / positive spread and stay flat
   at the null — averaged over >= 20 seeds so no lucky seed manufactures significance.

Execution convention: happiness rank is the *published* WHR value for the edition preceding the
holding window, so the signal is fully public at entry; the position is entered at that same close
and held over the forward window. No future data enters the signal (no look-ahead). The same-close
fill is a documented convention; on a multi-year hold a one-bar-later entry moves the headline
spread trivially and changes no stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The rank-sorted spread
# ---------------------------------------------------------------------------
def bucket_returns(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Sort by happiness, return happy / gloomy bucket forward-return means and the spread.

    ``frac`` is the tail fraction in each bucket (0.3 ≈ terciles, appropriate for ~24 countries).
    The folklore predicts happy > gloomy. Returns bucket means, the spread (happy − gloomy), bucket
    sizes, and the per-country forward returns of each bucket (for the two-sample t).
    """
    if panel.empty:
        return {}
    df = panel.sort_values("rank")   # rank 1 = happiest first
    k = max(1, int(round(len(df) * frac)))
    happy = df.head(k)               # best ranks (happiest)
    gloomy = df.tail(k)              # worst ranks (gloomiest)
    happy_ret = happy["forward_ret"]
    gloomy_ret = gloomy["forward_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "happy_mean": float(happy_ret.mean()),
        "gloomy_mean": float(gloomy_ret.mean()),
        "spread": float(happy_ret.mean() - gloomy_ret.mean()),
        "happy_ret": happy_ret,
        "gloomy_ret": gloomy_ret,
    }


def long_short_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) t on happy − gloomy forward returns.

    The folklore's tradable spread is a cross-sectional difference in means, so the honest test is a
    Welch two-sample t between the happy and gloomy buckets. Returns the spread, the t-stat and n.
    """
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["happy_ret"].to_numpy(dtype=float)
    b = buckets["gloomy_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# The rank correlation — the number the folklore lives on
# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (stdlib-only Spearman helper)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    means = sums / counts
    return means[inv]


def rank_correlation(panel: pd.DataFrame) -> dict:
    """Spearman rank correlation between *happiness* (score) and forward return, with a t.

    Positive = happier markets earn more (the folklore). Returns rho, its large-sample t
    (rho*sqrt((n-2)/(1-rho^2))) and n.
    """
    if panel.empty or len(panel) < 4:
        return {"rho": float("nan"), "t": float("nan"), "n": 0}
    rh = _rankdata(panel["happiness"].to_numpy(dtype=float))
    ry = _rankdata(panel["forward_ret"].to_numpy(dtype=float))
    if rh.std() == 0 or ry.std() == 0:
        return {"rho": float("nan"), "t": float("nan"), "n": int(len(panel))}
    rho = float(np.corrcoef(rh, ry)[0, 1])
    n = len(panel)
    denom = 1.0 - rho * rho
    t = rho * np.sqrt((n - 2) / denom) if denom > 1e-12 else float("nan")
    return {"rho": rho, "t": float(t), "n": int(n)}


# ---------------------------------------------------------------------------
# The country-level relation — the sign IS the folklore
# ---------------------------------------------------------------------------
def cross_section_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on the happiness score across all countries.

    Slope > 0 = the folklore (happier markets earn more). Returns the slope, its t-stat, and the
    Pearson correlation between happiness and forward return.
    """
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    x = panel["happiness"].to_numpy(dtype=float)
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
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 548) -> float:
    """Label-shuffle placebo p-value for the long-happy / short-gloomy spread.

    Shuffle the happiness *labels* against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| ≥ the observed |spread|. A real cross-sectional signal sits
    in the tail; a spurious one does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_returns(panel)["spread"])
    rng = np.random.default_rng(seed)
    happ = panel["happiness"].to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * 0.3)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(-happ[perm])       # sort returns by shuffled happiness (best first)
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 60.0,
               holding_years: float = 1.0) -> float:
    """Long-happy / short-gloomy spread net of costs and short borrow.

    The book is built once and held ``holding_years``: charge a round-trip one-way cost on each leg
    (entry + exit, both legs) plus a short borrow on the gloomy leg (annual, pro-rated). Country
    ETFs are cheap to borrow, so the borrow is modest (a footnote unless the spread is small).
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4          # entry + exit, two legs
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — seed-robust synthetic positive control (house rule)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, rank_alpha: float, n_seeds: int = 25,
                     base_seed: int = 548) -> float:
    """Average the country-level regression slope t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages a stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean slope-t across seeds for a planted
    ``rank_alpha``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(rank_alpha=rank_alpha, seed=s)
        ts.append(cross_section_regression(panel)["slope_t"])
    return float(np.nanmean(ts))


def synthetic_mean_spread_t(data_mod, rank_alpha: float, n_seeds: int = 25,
                            base_seed: int = 548) -> float:
    """Average the long-happy / short-gloomy Welch t over ``n_seeds`` synthetic worlds."""
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(rank_alpha=rank_alpha, seed=s)
        ts.append(long_short_tstat(bucket_returns(panel))["t"])
    return float(np.nanmean(ts))
