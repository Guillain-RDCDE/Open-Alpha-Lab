"""The engine and its honest controls — Study 543 (Western-Zodiac-CEO).

The claim, at folklore strength: a company run by a CEO of a given western sun sign (an
"entrepreneurial Aries", a "visionary Aquarius") outperforms — the horoscope applied to the
C-suite, and the western cousin of the Chinese-zodiac-year folklore ([Study 165]).

There is no mechanism (a birthday is independent of how a firm trades), so the honest test is a
*disproof* framed generously: we give the folklore its best shot and show it still finds nothing.

The machinery:

1. **Sort by sign.** Group the cross-section by the CEO's sun sign; compute each sign's mean
   forward return and the grand mean.

2. **ANOVA F.** A one-way ANOVA across the twelve signs — the natural omnibus test of "do the
   signs differ at all?" We report F and its analytic p, but the honest p is the *permutation* one
   (below), because the per-sign cells are tiny.

3. **Best-vs-rest spread.** The folklore's tradable expression: go long the *best* sign, short the
   rest. This is a **selected** contrast (we picked the best sign after the fact), so its naive t
   is upward-biased — we correct it with a label-shuffle placebo that re-selects the best sign on
   every shuffle (a max-statistic / multiple-comparisons null).

4. **Label-shuffle placebo.** Shuffle the sign labels against forward returns; on each shuffle
   recompute the ANOVA F (and the max best-vs-rest spread). The p-value is the fraction of
   shuffles as extreme as observed. This is the null that matters for a 12-way test on ~30 names.

5. **Costs + borrow.** The best-vs-rest book pays a round-trip one-way cost on each leg plus a
   borrow on the short (rest) leg.

6. **A synthetic positive control.** A deterministic world where one sign is given a genuine alpha
   bump (``sign_alpha > 0``); the engine must light up the ANOVA and stay flat at the null —
   averaged over ≥ 20 seeds (house rule) so no single lucky seed manufactures significance.

Execution convention: the sun sign is fixed at birth (decades before any price), so the signal is
trivially public in advance — there is no look-ahead to worry about. The one documented lag is the
single as-of scoring date and a same-close entry held over the forward window; on a multi-year
hold a one-bar-later entry changes nothing material.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SIGNS


# ---------------------------------------------------------------------------
# The sign sort
# ---------------------------------------------------------------------------
def sign_means(panel: pd.DataFrame) -> pd.Series:
    """Mean forward return per sun sign (only signs present in the panel)."""
    if panel.empty:
        return pd.Series(dtype=float)
    return panel.groupby("sign")["forward_ret"].mean()


def sign_counts(panel: pd.DataFrame) -> pd.Series:
    if panel.empty:
        return pd.Series(dtype=int)
    return panel.groupby("sign")["forward_ret"].size()


# ---------------------------------------------------------------------------
# One-way ANOVA across signs — the omnibus statistic
# ---------------------------------------------------------------------------
def anova_f(panel: pd.DataFrame) -> dict:
    """One-way ANOVA F across the sun signs on forward return.

    Returns F, its analytic p (F-distribution), the between/within df, the number of groups with
    >= 1 member, and n. A large F means the signs differ more than chance — but on tiny cells the
    honest p is the permutation one (see ``placebo_pvalue``).
    """
    if panel.empty:
        return {"F": float("nan"), "p": float("nan"), "df_between": 0, "df_within": 0,
                "groups": 0, "n": 0}
    groups = [g["forward_ret"].to_numpy(dtype=float) for _, g in panel.groupby("sign")]
    groups = [g for g in groups if len(g) >= 1]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return {"F": float("nan"), "p": float("nan"), "df_between": max(k - 1, 0),
                "df_within": max(n - k, 0), "groups": k, "n": n}
    grand = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = k - 1, n - k
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w if df_w > 0 else np.nan
    F = ms_b / ms_w if ms_w and ms_w > 0 else float("nan")
    # analytic p from the F survival function (SciPy if available; else NaN — permutation is king)
    p = float("nan")
    try:
        from scipy import stats  # optional
        p = float(stats.f.sf(F, df_b, df_w))
    except Exception:  # noqa: BLE001
        p = float("nan")
    return {"F": float(F), "p": p, "df_between": int(df_b), "df_within": int(df_w),
            "groups": int(k), "n": int(n)}


# ---------------------------------------------------------------------------
# Best-vs-rest spread (the folklore's tradable expression) — a SELECTED contrast
# ---------------------------------------------------------------------------
def best_vs_rest(panel: pd.DataFrame, min_count: int = 2) -> dict:
    """Long the best-performing sign, short the rest.

    Only signs with >= ``min_count`` members are eligible to be "the best" (a single-name sign is
    not a strategy). Returns the winning sign, its mean, the rest mean, the spread, a Welch
    two-sample t (best-sign names vs rest), and n. The spread's naive t is upward-biased because
    the sign was *selected* to be the best — corrected by the max-statistic placebo below.
    """
    if panel.empty:
        return {"sign": None, "best_mean": float("nan"), "rest_mean": float("nan"),
                "spread": float("nan"), "t": float("nan"), "n": 0}
    counts = sign_counts(panel)
    eligible = counts[counts >= min_count].index.tolist()
    if not eligible:
        eligible = counts.index.tolist()
    means = sign_means(panel)
    best = means.loc[eligible].idxmax()
    a = panel.loc[panel["sign"] == best, "forward_ret"].to_numpy(dtype=float)
    b = panel.loc[panel["sign"] != best, "forward_ret"].to_numpy(dtype=float)
    spread = a.mean() - b.mean()
    if len(a) >= 2 and len(b) >= 2:
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = spread / se if se > 0 else float("nan")
    else:
        t = float("nan")
    return {"sign": str(best), "best_mean": float(a.mean()), "rest_mean": float(b.mean()),
            "spread": float(spread), "t": float(t), "n": int(len(a) + len(b))}


# ---------------------------------------------------------------------------
# Label-shuffle placebo — the multiple-comparisons null
# ---------------------------------------------------------------------------
def _fast_F_and_maxspread(codes: np.ndarray, y: np.ndarray, k_groups: int,
                          eligible_mask: np.ndarray) -> tuple[float, float]:
    """Fast numpy ANOVA F and max best-vs-rest spread for integer-coded group labels.

    ``codes`` are 0..k_groups-1 group codes aligned to ``y``; ``eligible_mask`` marks which group
    codes may be selected as "the best" (>= min_count members). Avoids pandas in the hot loop.
    """
    n = len(y)
    counts = np.bincount(codes, minlength=k_groups).astype(float)
    sums = np.bincount(codes, weights=y, minlength=k_groups)
    present = counts > 0
    means = np.where(present, sums / np.where(counts == 0, 1, counts), np.nan)
    grand = y.mean()
    ss_between = np.nansum(counts * (means - grand) ** 2)
    ss_within = np.sum((y - means[codes]) ** 2)
    k = int(present.sum())
    df_b, df_w = k - 1, n - k
    if df_b <= 0 or df_w <= 0 or ss_within <= 0:
        F = float("nan")
    else:
        F = (ss_between / df_b) / (ss_within / df_w)
    # max best-vs-rest spread over eligible groups
    elig = eligible_mask & present
    if not elig.any():
        elig = present
    best_code = np.where(elig, means, -np.inf).argmax()
    a_mask = codes == best_code
    a_mean = y[a_mask].mean()
    b_mean = y[~a_mask].mean() if (~a_mask).any() else np.nan
    spread = a_mean - b_mean
    return float(F), float(spread)


def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 5000, seed: int = 543,
                   min_count: int = 2) -> dict:
    """Label-shuffle placebo p-values for BOTH the ANOVA F and the max best-vs-rest spread.

    Shuffle the sign labels against forward returns ``n_perm`` times. For each shuffle recompute
    (a) the ANOVA F and (b) the *max* best-vs-rest spread (re-selecting the best eligible sign on
    every shuffle — the honest max-statistic correction for having picked the best sign).

    Returns ``{"p_F": ..., "p_spread": ...}`` = the fraction of shuffles at least as extreme as
    observed. A real 12-way effect sits in the tail of BOTH; folklore does not.
    """
    if panel.empty or len(panel) < 6:
        return {"p_F": float("nan"), "p_spread": float("nan")}
    obs_F = anova_f(panel)["F"]
    obs_spread = best_vs_rest(panel, min_count)["spread"]
    y = panel["forward_ret"].to_numpy(dtype=float)
    labels = panel["sign"].to_numpy()
    uniq, codes = np.unique(labels, return_inverse=True)
    k_groups = len(uniq)
    base_counts = np.bincount(codes, minlength=k_groups)
    eligible_mask = base_counts >= min_count  # eligibility is by the count PATTERN of the shuffle
    rng = np.random.default_rng(seed)
    n = len(y)
    cnt_F = cnt_s = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        pc = codes[perm]
        # recompute eligibility on the shuffled counts (they are a permutation, so counts are
        # identical to base_counts — eligibility is stable — but recompute for correctness)
        f, s = _fast_F_and_maxspread(pc, y, k_groups, eligible_mask)
        if np.isfinite(f) and np.isfinite(obs_F) and f >= obs_F - 1e-12:
            cnt_F += 1
        if np.isfinite(s) and np.isfinite(obs_spread) and s >= obs_spread - 1e-12:
            cnt_s += 1
    return {"p_F": (cnt_F + 1) / (n_perm + 1), "p_spread": (cnt_s + 1) / (n_perm + 1)}


# ---------------------------------------------------------------------------
# Costs + borrow on the best-vs-rest book
# ---------------------------------------------------------------------------
def net_spread(bvr: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 100.0,
               holding_years: float = 1.0) -> float:
    """Best-vs-rest spread net of round-trip costs and a borrow on the short (rest) leg."""
    if not bvr or not np.isfinite(bvr.get("spread", float("nan"))):
        return float("nan")
    gross = bvr["spread"]
    cost = 4.0 * cost_bps * 1e-4          # entry + exit, two legs
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — the same sort over several forward windows
# ---------------------------------------------------------------------------
def window_sweep(data_mod, prices: pd.DataFrame, spans: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Run the ANOVA + best-vs-rest over several forward windows on the real tape.

    ``spans`` is a list of ``(label, start, end)``. Returns a frame indexed by label with the
    ANOVA F, its permutation p, the winning sign and the best-vs-rest spread — to check whether any
    apparent effect (or the winning sign) is stable across windows. Empty windows are skipped.
    """
    rows = {}
    for label, sp, en in spans:
        pn = data_mod.build_panel(prices, sp, en)
        if pn.empty or len(pn) < 6:
            continue
        f = anova_f(pn)
        pl = placebo_pvalue(pn, n_perm=2000, seed=543)
        bvr = best_vs_rest(pn)
        rows[label] = {
            "F": f["F"], "p_perm": pl["p_F"], "best_sign": bvr["sign"],
            "spread": bvr["spread"], "n": f["n"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule: >= 20 seeds)
# ---------------------------------------------------------------------------
def synthetic_mean_F(data_mod, sign_alpha: float, n_seeds: int = 25,
                     base_seed: int = 543, favoured_sign: str = "Aries",
                     n_stocks: int = 120) -> dict:
    """Average the ANOVA F and the placebo p over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages its statistic over >= 20 seeds so no
    single lucky RNG seed can manufacture significance. Returns the mean ANOVA F and the mean
    permutation p across seeds for a planted ``sign_alpha`` (0 = null).
    """
    Fs, ps = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(
            n_stocks=n_stocks, sign_alpha=sign_alpha, favoured_sign=favoured_sign, seed=s)
        Fs.append(anova_f(panel)["F"])
        ps.append(placebo_pvalue(panel, n_perm=800, seed=s)["p_F"])
    return {"mean_F": float(np.nanmean(Fs)), "mean_p": float(np.nanmean(ps))}
