"""The strategy and its honest controls — Study 194 (Turkey).

The folk claim: the Wednesday before Thanksgiving and the Friday after
(Black Friday, a short session) deliver reliably positive returns — a
pre/post-holiday effect. This is distinct from the pooled 'all pre-holidays'
finding in the literature: here we isolate Thanksgiving specifically.

We run four tests:

1. **Wednesday before Thanksgiving vs all other Wednesdays** (primary test A).
   HAC Newey-West t-stat on the per-day return series for each group, and a
   Welch t-test on the contrast. The placebo arm is the Tuesday before
   Thanksgiving (matched-window, same week, one day earlier).

2. **Black Friday (day after Thanksgiving) vs all other Fridays** (primary test B).
   Same methodology.

3. **Tuesday placebo vs other Tuesdays.**
   If Tuesday before Thanksgiving is as 'anomalous' as Wednesday, the signal
   is a weekly pattern, not a pre-holiday effect.

4. **Multiple-comparisons correction.**
   We run two main tests (Wednesday-before and Friday-after); applying
   Bonferroni k=2, the threshold for significance at alpha=0.05 is p < 0.025.

No look-ahead: the calendar date is known before the open; the return is the
close-to-close log-return of that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-statistic
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> dict:
    """Newey-West (HAC) t-statistic on the sample mean of ``r``."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return {
            "tstat": float("nan"),
            "mean": float(r.mean()) if n else float("nan"),
            "se": float("nan"),
            "n": n,
        }
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "mean": float(mu),
        "se": float(se),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Helper: weekday-matched comparison
# ---------------------------------------------------------------------------
def _compare_groups(target: np.ndarray, other: np.ndarray) -> dict:
    """Welch t-test + HAC t-stats for two return arrays.

    Returns a dict with n_target, mean_target_bps, tstat_hac_target,
    n_other, mean_other_bps, contrast_bps, tstat_welch, pval_welch.
    """
    from scipy.stats import ttest_ind

    target = target[np.isfinite(target)]
    other = other[np.isfinite(other)]

    h_t = _hac_tstat(target)
    h_o = _hac_tstat(other)

    if len(target) > 1 and len(other) > 1:
        t_welch, pval_welch = ttest_ind(target, other, equal_var=False)
    else:
        t_welch, pval_welch = float("nan"), float("nan")

    return {
        "n_target": int(len(target)),
        "mean_target_bps": float(target.mean() * 1e4) if len(target) else float("nan"),
        "tstat_hac": h_t["tstat"],
        "n_other": int(len(other)),
        "mean_other_bps": float(other.mean() * 1e4) if len(other) else float("nan"),
        "tstat_hac_other": h_o["tstat"],
        "contrast_bps": float((target.mean() - other.mean()) * 1e4)
        if len(target) and len(other)
        else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval_welch),
    }


# ---------------------------------------------------------------------------
# Primary test A: Wednesday before Thanksgiving vs other Wednesdays
# ---------------------------------------------------------------------------
def wednesday_comparison(df: pd.DataFrame) -> dict:
    """Compare the Wednesday-before-Thanksgiving return to all other-Wednesday returns.

    Parameters
    ----------
    df : DataFrame with columns ``ret, is_wed_before_tsg``.
         The index must have a ``weekday`` attribute (DatetimeIndex).

    Returns
    -------
    dict with keys: n_wed, mean_wed_bps, tstat_hac_wed, n_other_wed,
    mean_other_wed_bps, contrast_bps, tstat_welch, pval_welch.
    """
    # Mask all Wednesdays in the index.
    all_wed_mask = df.index.weekday == 2  # Monday=0, Wednesday=2
    wed_tsg = df.loc[df["is_wed_before_tsg"], "ret"].to_numpy(dtype=float)
    other_wed = df.loc[
        all_wed_mask & ~df["is_wed_before_tsg"].values, "ret"
    ].to_numpy(dtype=float)

    res = _compare_groups(wed_tsg, other_wed)
    return {
        "n_wed": res["n_target"],
        "mean_wed_bps": res["mean_target_bps"],
        "tstat_hac_wed": res["tstat_hac"],
        "n_other_wed": res["n_other"],
        "mean_other_wed_bps": res["mean_other_bps"],
        "contrast_bps": res["contrast_bps"],
        "tstat_welch": res["tstat_welch"],
        "pval_welch": res["pval_welch"],
    }


# ---------------------------------------------------------------------------
# Primary test B: Black Friday (day after Thanksgiving) vs other Fridays
# ---------------------------------------------------------------------------
def friday_comparison(df: pd.DataFrame) -> dict:
    """Compare the Black Friday return to all other-Friday returns.

    Parameters
    ----------
    df : DataFrame with columns ``ret, is_fri_after_tsg``.

    Returns
    -------
    dict with keys: n_fri, mean_fri_bps, tstat_hac_fri, n_other_fri,
    mean_other_fri_bps, contrast_bps, tstat_welch, pval_welch.
    """
    all_fri_mask = df.index.weekday == 4  # Friday=4
    fri_tsg = df.loc[df["is_fri_after_tsg"], "ret"].to_numpy(dtype=float)
    other_fri = df.loc[
        all_fri_mask & ~df["is_fri_after_tsg"].values, "ret"
    ].to_numpy(dtype=float)

    res = _compare_groups(fri_tsg, other_fri)
    return {
        "n_fri": res["n_target"],
        "mean_fri_bps": res["mean_target_bps"],
        "tstat_hac_fri": res["tstat_hac"],
        "n_other_fri": res["n_other"],
        "mean_other_fri_bps": res["mean_other_bps"],
        "contrast_bps": res["contrast_bps"],
        "tstat_welch": res["tstat_welch"],
        "pval_welch": res["pval_welch"],
    }


# ---------------------------------------------------------------------------
# Placebo: Tuesday before Thanksgiving vs other Tuesdays
# ---------------------------------------------------------------------------
def tuesday_placebo(df: pd.DataFrame) -> dict:
    """Compare the Tuesday-before-Thanksgiving return to all other-Tuesday returns.

    The Tuesday before Thanksgiving is same week, same seasonal pocket, but
    *not* the immediate pre-holiday day.  If Tuesday is as 'anomalous' as
    Wednesday the signal is just general late-November seasonality, not a
    pre-holiday effect.

    Parameters
    ----------
    df : DataFrame with columns ``ret, is_placebo_tue``.

    Returns
    -------
    dict with keys: n_tue, mean_tue_bps, tstat_hac_tue, n_other_tue,
    mean_other_tue_bps, contrast_bps, tstat_welch, pval_welch.
    """
    all_tue_mask = df.index.weekday == 1  # Tuesday=1
    tue_tsg = df.loc[df["is_placebo_tue"], "ret"].to_numpy(dtype=float)
    other_tue = df.loc[
        all_tue_mask & ~df["is_placebo_tue"].values, "ret"
    ].to_numpy(dtype=float)

    res = _compare_groups(tue_tsg, other_tue)
    return {
        "n_tue": res["n_target"],
        "mean_tue_bps": res["mean_target_bps"],
        "tstat_hac_tue": res["tstat_hac"],
        "n_other_tue": res["n_other"],
        "mean_other_tue_bps": res["mean_other_bps"],
        "contrast_bps": res["contrast_bps"],
        "tstat_welch": res["tstat_welch"],
        "pval_welch": res["pval_welch"],
    }


# ---------------------------------------------------------------------------
# Summarize — all tests in one call
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """All Thanksgiving tests combined.

    Returns a flat dict mirroring docs/results.md for use in notebooks and
    verify.py.  Also includes the Bonferroni-corrected p for each primary
    test (k=2: two primary hypotheses — Wednesday-before and Friday-after).
    """
    wed = wednesday_comparison(df)
    fri = friday_comparison(df)
    plac = tuesday_placebo(df)

    bonf_wed = min(wed["pval_welch"] * 2, 1.0) if np.isfinite(wed["pval_welch"]) else float("nan")
    bonf_fri = min(fri["pval_welch"] * 2, 1.0) if np.isfinite(fri["pval_welch"]) else float("nan")

    return {
        # Wednesday before Thanksgiving
        "n_wed": wed["n_wed"],
        "n_other_wed": wed["n_other_wed"],
        "mean_wed_bps": wed["mean_wed_bps"],
        "mean_other_wed_bps": wed["mean_other_wed_bps"],
        "contrast_wed_bps": wed["contrast_bps"],
        "tstat_welch_wed": wed["tstat_welch"],
        "pval_welch_wed": wed["pval_welch"],
        "tstat_hac_wed": wed["tstat_hac_wed"],
        "pval_bonferroni_wed": bonf_wed,
        # Friday after Thanksgiving (Black Friday)
        "n_fri": fri["n_fri"],
        "n_other_fri": fri["n_other_fri"],
        "mean_fri_bps": fri["mean_fri_bps"],
        "mean_other_fri_bps": fri["mean_other_fri_bps"],
        "contrast_fri_bps": fri["contrast_bps"],
        "tstat_welch_fri": fri["tstat_welch"],
        "pval_welch_fri": fri["pval_welch"],
        "tstat_hac_fri": fri["tstat_hac_fri"],
        "pval_bonferroni_fri": bonf_fri,
        # Placebo: Tuesday before Thanksgiving
        "n_tue": plac["n_tue"],
        "mean_tue_bps": plac["mean_tue_bps"],
        "contrast_tue_bps": plac["contrast_bps"],
        "pval_welch_tue": plac["pval_welch"],
    }
