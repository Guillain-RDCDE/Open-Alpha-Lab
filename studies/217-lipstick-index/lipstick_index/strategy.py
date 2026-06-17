"""Strategy and analysis layer for Study 217 (Lipstick Index).

The Lipstick Index (Leonard Lauder ~2001): cosmetics companies outperform
the broad market during recessions because consumers trade down to affordable
luxuries. We test whether the cosmetics basket (EL, ULTA, COTY, ELF) shows
a statistically significant positive *relative return* vs SPY during NBER
recession months.

Tests conducted:
  - Welch t-test: mean relative return (cosm - SPY) in recession months
    vs mean relative return in expansion months.
  - Permutation test: shuffle recession labels 10,000 times; see where the
    real mean-in-recession lands.
  - Annualised relative return (alpha) in recession vs expansion windows.

The n problem: NBER recessions since EL's 1995 IPO cover only ~57 months
(out of ~360 total) — power is very limited. We document this honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Core stat: mean relative return in recession vs expansion
# ---------------------------------------------------------------------------
def recession_alpha_stats(
    df: pd.DataFrame,
    rel_col: str = "rel_ret",
    rec_col: str = "in_recession",
    n_permutations: int = 10_000,
    seed: int = 217,
) -> dict:
    """Compute the key stats for the lipstick index claim.

    Tests whether the cosmetics basket has a higher mean *relative return*
    (vs SPY) in recession months than in expansion months.

    Parameters
    ----------
    df : DataFrame with ``rel_col`` and ``rec_col`` columns.
    rel_col : column of monthly relative returns (cosm - SPY).
    rec_col : boolean column, True in recession months.
    n_permutations : shuffle count for the permutation test.
    seed : RNG seed.

    Returns
    -------
    dict with:
      n_total, n_rec, n_exp,
      mean_rel_rec, mean_rel_exp, mean_rel_diff (percentage points / month),
      mean_rel_rec_ann, mean_rel_exp_ann (annualised bps),
      welch_t, welch_p,
      perm_p (fraction of shuffles with mean_rec >= observed),
      perm_mean_dists (array for plotting).
    """
    rels = df[rel_col].values
    rec = df[rec_col].values.astype(bool)
    exp = ~rec

    n_total = len(rels)
    n_rec = rec.sum()
    n_exp = exp.sum()

    rec_rets = rels[rec]
    exp_rets = rels[exp]

    mean_rec = float(rec_rets.mean()) if n_rec > 0 else float("nan")
    mean_exp = float(exp_rets.mean()) if n_exp > 0 else float("nan")
    mean_diff = mean_rec - mean_exp

    # Annualised (multiply monthly by 12)
    mean_rec_ann = mean_rec * 12
    mean_exp_ann = mean_exp * 12

    # Welch t-test on monthly relative returns: recession vs expansion
    if n_rec >= 2 and n_exp >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(rec_rets, exp_rets, equal_var=False)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    # Permutation test: shuffle recession labels, recompute mean difference
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled_rec = rng.permuted(rec)
        perm_diffs[i] = rels[shuffled_rec].mean() - rels[~shuffled_rec].mean()
    perm_p = float((perm_diffs >= mean_diff).mean())

    return {
        "n_total": int(n_total),
        "n_rec": int(n_rec),
        "n_exp": int(n_exp),
        "mean_rel_rec": round(float(mean_rec) * 100, 3),         # % / month
        "mean_rel_exp": round(float(mean_exp) * 100, 3),
        "mean_rel_diff": round(float(mean_diff) * 100, 3),       # pp / month
        "mean_rel_rec_ann_bps": round(float(mean_rec_ann) * 10_000, 0),
        "mean_rel_exp_ann_bps": round(float(mean_exp_ann) * 10_000, 0),
        "welch_t": round(float(welch_t), 3),
        "welch_p": round(float(welch_p), 4),
        "perm_p": round(float(perm_p), 4),
        "perm_diffs": perm_diffs,
    }


# ---------------------------------------------------------------------------
# Cumulative relative return inside vs outside recessions
# ---------------------------------------------------------------------------
def cumulative_relative(
    df: pd.DataFrame,
    rel_col: str = "rel_ret",
    rec_col: str = "in_recession",
) -> pd.DataFrame:
    """Cumulative relative return split by recession / expansion.

    Returns a DataFrame aligned to df's index with:
      ``cum_rel_rec``: cumulative product of (1 + rel_ret) for recession months
                       (expansion months contribute 0 return, i.e., factor=1).
      ``cum_rel_exp``: same for expansion months.
      ``cum_rel_all``: cumulative over all months.
    """
    rels = df[rel_col].fillna(0.0).values
    rec = df[rec_col].values.astype(bool)

    rec_only = np.where(rec, rels, 0.0)
    exp_only = np.where(~rec, rels, 0.0)

    cum_rec = (1 + rec_only).cumprod()
    cum_exp = (1 + exp_only).cumprod()
    cum_all = (1 + rels).cumprod()

    out = df[["month", rec_col]].copy()
    out["cum_rel_rec"] = cum_rec
    out["cum_rel_exp"] = cum_exp
    out["cum_rel_all"] = cum_all
    return out


# ---------------------------------------------------------------------------
# Full summary dict
# ---------------------------------------------------------------------------
def summarize(
    df: pd.DataFrame,
    rel_col: str = "rel_ret",
    rec_col: str = "in_recession",
    n_permutations: int = 10_000,
    seed: int = 217,
) -> dict:
    """One-stop summary dict mirroring build_notebooks.py ``R`` and docs/results.md.

    All numbers rounded to useful precision for display.
    """
    st = recession_alpha_stats(
        df, rel_col=rel_col, rec_col=rec_col,
        n_permutations=n_permutations, seed=seed,
    )
    n = st["n_total"]
    n_rec = st["n_rec"]
    n_exp = st["n_exp"]

    # Mean SPY return in recession/expansion (absolute, not relative)
    spy_rec = df.loc[df[rec_col], "SPY"].mean() * 100 if "SPY" in df.columns else float("nan")
    spy_exp = df.loc[~df[rec_col].astype(bool), "SPY"].mean() * 100 if "SPY" in df.columns else float("nan")
    cosm_rec = df.loc[df[rec_col], "cosm_ret"].mean() * 100 if "cosm_ret" in df.columns else float("nan")
    cosm_exp = df.loc[~df[rec_col].astype(bool), "cosm_ret"].mean() * 100 if "cosm_ret" in df.columns else float("nan")

    return {
        "n_months": n,
        "n_rec_months": n_rec,
        "n_exp_months": n_exp,
        "mean_rel_rec_pct_month": st["mean_rel_rec"],
        "mean_rel_exp_pct_month": st["mean_rel_exp"],
        "mean_rel_diff_pct_month": st["mean_rel_diff"],
        "mean_rel_rec_ann_bps": st["mean_rel_rec_ann_bps"],
        "welch_t": st["welch_t"],
        "welch_p": st["welch_p"],
        "perm_p": st["perm_p"],
        "spy_rec_pct_month": round(float(spy_rec), 3),
        "spy_exp_pct_month": round(float(spy_exp), 3),
        "cosm_rec_pct_month": round(float(cosm_rec), 3),
        "cosm_exp_pct_month": round(float(cosm_exp), 3),
    }
