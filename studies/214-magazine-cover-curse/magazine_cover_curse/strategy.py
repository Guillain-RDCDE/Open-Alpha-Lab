"""Strategy and analysis layer for Study 214 (Magazine-Cover-Curse).

The magazine-cover contrarian thesis: when a financial publication puts a
boom or bust on its cover, the move is already over — the cover marks a
sentiment extreme that tends to reverse.

We test this event-study style on the hardcoded cover table:

  - **Euphoric covers** → expect negative 6/12-month forward S&P returns
    (the thesis predicts a top had been reached).
  - **Doom covers** → expect positive 6/12-month forward S&P returns
    (the thesis predicts a bottom had been reached).
  - **Combined (contrarian signal)** → merge the two predictions and test
    whether forward returns are consistent with the thesis.

The critical problems:
  1. **Cherry-picking.** Every famous cover is famous *because* it was followed
     by a reversal. Covers that were followed by continuation are forgotten.
     We test the full table honestly, including the failures.
  2. **Tiny n.** With ~35–40 covers spanning 40+ years, there are fewer than
     20 events per group. Statistical power is essentially zero.
  3. **Base rate.** The S&P 500 rises about 75% of 12-month periods. Any
     doom-cover signal that predicts "up" gets this for free.

Inference: Welch t-test on forward returns by tone; permutation test;
mean forward return vs the unconditional base rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def contrarian_signal(df: pd.DataFrame) -> pd.Series:
    """Return the contrarian prediction for each cover.

    +1 for 'doom' covers (predict market rises — contrarian bullish).
    -1 for 'euphoric' covers (predict market falls — contrarian bearish).
    """
    return df["tone"].map({"doom": 1, "euphoric": -1}).rename("contrarian_signal")


# ---------------------------------------------------------------------------
# Event-study statistics
# ---------------------------------------------------------------------------
def event_study_stats(
    df: pd.DataFrame,
    ret_col: str = "fwd_12m",
    n_permutations: int = 10_000,
    seed: int = 214,
) -> dict:
    """Compute statistics for the magazine-cover contrarian event study.

    Parameters
    ----------
    df : DataFrame with ``tone`` column ('euphoric'/'doom') and a forward
         return column (``ret_col``).
    ret_col : which forward return column to analyse.
    n_permutations : shuffles for the permutation test.
    seed : RNG seed.

    Returns a dict with:
      - ``n_total``, ``n_doom``, ``n_euphoric``, ``n_valid``
      - ``mean_doom``, ``mean_euphoric``, ``mean_all``
      - ``uncond_up_rate`` : fraction of all events where forward return > 0
      - ``doom_up_rate``, ``euphoric_up_rate``
      - ``welch_t``, ``welch_p`` : Welch t on doom vs euphoric forward returns
      - ``contrarian_correct_rate`` : fraction where the contrarian prediction
        was right (doom AND mkt up, OR euphoric AND mkt down)
      - ``binom_p`` : binomial test (contrarian correct rate vs uncond_up_rate
        for doom covers, vs 1-uncond_up_rate for euphoric)
      - ``perm_p`` : permutation p-value (shuffle tone labels)
      - ``perm_diffs`` : array of permuted mean differences (doom - euphoric)
      - ``obs_diff`` : observed mean difference (doom - euphoric)
    """
    df_valid = df[df[ret_col].notna()].copy()
    n_valid = len(df_valid)

    doom_mask = df_valid["tone"] == "doom"
    euph_mask = df_valid["tone"] == "euphoric"

    n_doom = int(doom_mask.sum())
    n_euph = int(euph_mask.sum())
    n_total = len(df)

    rets_doom = df_valid.loc[doom_mask, ret_col].values
    rets_euph = df_valid.loc[euph_mask, ret_col].values
    rets_all = df_valid[ret_col].values

    mean_doom = float(rets_doom.mean()) if n_doom > 0 else float("nan")
    mean_euph = float(rets_euph.mean()) if n_euph > 0 else float("nan")
    mean_all = float(rets_all.mean()) if n_valid > 0 else float("nan")

    uncond_up_rate = float((rets_all > 0).mean()) if n_valid > 0 else float("nan")
    doom_up_rate = float((rets_doom > 0).mean()) if n_doom > 0 else float("nan")
    euph_up_rate = float((rets_euph > 0).mean()) if n_euph > 0 else float("nan")

    # Welch t-test: doom returns vs euphoric returns
    if n_doom >= 2 and n_euph >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(rets_doom, rets_euph, equal_var=False)
        welch_t, welch_p = float(welch_t), float(welch_p)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    # Contrarian correct rate: doom+up OR euphoric+down
    tones_v = df_valid["tone"].values
    mkt_up = rets_all > 0
    correct = np.where(tones_v == "doom", mkt_up.astype(int), (~mkt_up).astype(int))
    contrarian_correct_rate = float(correct.mean()) if n_valid > 0 else float("nan")

    # Binomial test vs the unconditional up-rate as the null
    # (doom covers predict 'up' — tested against the base rate)
    if n_doom >= 1 and not np.isnan(uncond_up_rate):
        binom_result = scipy_stats.binomtest(
            k=int((rets_doom > 0).sum()),
            n=n_doom,
            p=float(uncond_up_rate),
            alternative="greater",
        )
        binom_p = float(binom_result.pvalue)
    else:
        binom_p = float("nan")

    # Permutation test: shuffle tone labels, recompute mean difference (doom - euphoric)
    obs_diff = mean_doom - mean_euph if not (np.isnan(mean_doom) or np.isnan(mean_euph)) else float("nan")
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(n_permutations, dtype=float)
    tones_arr = df_valid["tone"].values.copy()
    for i in range(n_permutations):
        shuffled = rng.permuted(tones_arr)
        d_mask = shuffled == "doom"
        e_mask = shuffled == "euphoric"
        if d_mask.sum() >= 1 and e_mask.sum() >= 1:
            perm_diffs[i] = rets_all[d_mask].mean() - rets_all[e_mask].mean()
        else:
            perm_diffs[i] = 0.0

    if not np.isnan(obs_diff):
        perm_p = float((perm_diffs >= obs_diff).mean())
    else:
        perm_p = float("nan")

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_doom": n_doom,
        "n_euphoric": n_euph,
        "mean_doom": round(mean_doom * 100, 2) if not np.isnan(mean_doom) else float("nan"),
        "mean_euphoric": round(mean_euph * 100, 2) if not np.isnan(mean_euph) else float("nan"),
        "mean_all": round(mean_all * 100, 2) if not np.isnan(mean_all) else float("nan"),
        "uncond_up_rate": round(uncond_up_rate, 4) if not np.isnan(uncond_up_rate) else float("nan"),
        "doom_up_rate": round(doom_up_rate, 4) if not np.isnan(doom_up_rate) else float("nan"),
        "euphoric_up_rate": round(euph_up_rate, 4) if not np.isnan(euph_up_rate) else float("nan"),
        "contrarian_correct_rate": round(contrarian_correct_rate, 4),
        "welch_t": round(welch_t, 3) if not np.isnan(welch_t) else float("nan"),
        "welch_p": round(welch_p, 4) if not np.isnan(welch_p) else float("nan"),
        "binom_p": round(binom_p, 4) if not np.isnan(binom_p) else float("nan"),
        "perm_p": round(perm_p, 4) if not np.isnan(perm_p) else float("nan"),
        "obs_diff": round(obs_diff * 100, 2) if not np.isnan(obs_diff) else float("nan"),
        "perm_diffs": perm_diffs,
    }


# ---------------------------------------------------------------------------
# Summarize for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 214) -> dict:
    """One-stop summary dict for the real event-study DataFrame.

    Returns a flat dict mirroring the R = dict(...) in build_notebooks.py and
    docs/results.md. All numbers are rounded to useful precision.
    """
    s6 = event_study_stats(df, ret_col="fwd_6m",
                           n_permutations=n_permutations, seed=seed)
    s12 = event_study_stats(df, ret_col="fwd_12m",
                            n_permutations=n_permutations, seed=seed)
    return {
        "n_covers": s12["n_total"],
        "n_valid_12m": s12["n_valid"],
        "n_doom": s12["n_doom"],
        "n_euphoric": s12["n_euphoric"],
        # 12-month horizon (primary)
        "mean_doom_12m": s12["mean_doom"],
        "mean_euph_12m": s12["mean_euphoric"],
        "mean_all_12m": s12["mean_all"],
        "uncond_up_rate_12m": round(s12["uncond_up_rate"] * 100, 1),
        "doom_up_rate_12m": round(s12["doom_up_rate"] * 100, 1),
        "euph_up_rate_12m": round(s12["euphoric_up_rate"] * 100, 1),
        "contrarian_correct_12m": round(s12["contrarian_correct_rate"] * 100, 1),
        "welch_t_12m": s12["welch_t"],
        "welch_p_12m": s12["welch_p"],
        "binom_p_12m": s12["binom_p"],
        "perm_p_12m": s12["perm_p"],
        "obs_diff_12m": s12["obs_diff"],
        # 6-month horizon (secondary)
        "n_valid_6m": s6["n_valid"],
        "mean_doom_6m": s6["mean_doom"],
        "mean_euph_6m": s6["mean_euphoric"],
        "welch_t_6m": s6["welch_t"],
        "welch_p_6m": s6["welch_p"],
        "perm_p_6m": s6["perm_p"],
    }
