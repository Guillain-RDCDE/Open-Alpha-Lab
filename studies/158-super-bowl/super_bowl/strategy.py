"""Strategy and analysis layer for Study 158 (Super-Bowl).

The Super Bowl Indicator (Krueger & Kennedy 1990): if an NFC (or original-NFL)
team wins the Super Bowl, the S&P 500 supposedly goes up that year; if an AFC
(or AFL) team wins, it goes down. We implement this as a year-by-year signal and
pin it against:

  - The **unconditional base rate** (how often does the S&P go up at all?). This
    is the killer: the market rises in ~70% of years regardless, so any binary
    signal that predicts "up" gets credit for the base rate — not for insight.
  - A **binomial test**: given n = ~58 observations and p₀ = 0.70 unconditional
    up-rate, what hit-rate do we need to claim significance?
  - A **permutation test**: shuffle conference labels 10,000 times and see where
    the real hit-rate lands.
  - Explicit **multiple-comparisons** accounting: "NFC wins", "original-NFL wins",
    "game year" vs "subsequent year" are all tested — the textbook multiple-
    hypothesis problem that spawns false discoveries.

The tiny-n reckoning: n ≈ 58 Super Bowls in total, ~30–32 per conference.
A hit-rate of 70% on 32 NFC-year returns is *expected* if the S&P just goes up
70% of the time unconditionally. The binomial null must be p₀ = unconditional
up-rate, not 0.5 — a common mistake that flatters the indicator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------
def signal_nfc(df: pd.DataFrame) -> pd.Series:
    """Return +1 (bull) when NFC wins, -1 (bear) when AFC wins.

    This is the primary Krueger-Kennedy signal, using post-merger conference
    assignment.
    """
    return df["conference"].map({"NFC": 1, "AFC": -1}).rename("signal_nfc")


def signal_orig_nfl(df: pd.DataFrame) -> pd.Series:
    """Return +1 (bull) when an original-NFL team wins, -1 otherwise.

    The stricter Krueger-Kennedy version that ignores post-merger conference
    reassignments.
    """
    return df["orig_nfl"].map({True: 1, False: -1}).rename("signal_orig_nfl")


# ---------------------------------------------------------------------------
# Hit-rate and inference
# ---------------------------------------------------------------------------
def hit_rate_stats(
    df: pd.DataFrame,
    signal: pd.Series,
    ret_col: str = "sp500_return",
    n_permutations: int = 10_000,
    seed: int = 158,
) -> dict:
    """Compute hit-rate, unconditional base rate, binomial test, and permutation test.

    The 'hit' is defined as: the market went up in a 'bull-signal' year (signal=+1)
    OR the market went down in a 'bear-signal' year (signal=-1). We separately
    report:
      - ``hit_rate_all``: fraction of all years where the signal was correct
      - ``hit_rate_bull``: fraction of NFC/orig-NFL years where mkt went up
      - ``hit_rate_bear``: fraction of AFC years where mkt went down
      - ``uncond_up_rate``: unconditional fraction of up years (the honest baseline)
      - ``n_total``, ``n_bull_years``, ``n_bear_years``
      - ``binom_p``: p-value for the binomial test H₀: p_up_in_bull_years = uncond_up_rate
      - ``perm_p``: permutation p-value (fraction of shuffled samples with hit_rate ≥ observed)

    Parameters
    ----------
    df : DataFrame with ``sp500_return`` and ``mkt_up`` columns
    signal : Series aligned to df, values in {+1, -1}
    ret_col : column name for the annual return
    n_permutations : permutation draws
    seed : RNG seed for permutations
    """
    s = signal.values
    mkt_up = df["mkt_up"].values
    rets = df[ret_col].values
    n = len(s)

    bull_mask = s == 1
    bear_mask = s == -1

    n_bull = bull_mask.sum()
    n_bear = bear_mask.sum()

    # Hit: bull-signal AND mkt up, OR bear-signal AND mkt down.
    hits = np.where(s == 1, mkt_up.astype(int), (~mkt_up).astype(int))
    hit_rate_all = hits.mean()

    hit_rate_bull = mkt_up[bull_mask].mean() if n_bull > 0 else float("nan")
    hit_rate_bear = (~mkt_up[bear_mask]).mean() if n_bear > 0 else float("nan")

    uncond_up_rate = mkt_up.mean()

    # Mean return conditioned on signal.
    mean_bull = float(rets[bull_mask].mean()) if n_bull > 0 else float("nan")
    mean_bear = float(rets[bear_mask].mean()) if n_bear > 0 else float("nan")
    mean_all = float(rets.mean())

    # HAC t-stat on (return_t - unconditional_mean) for bull years vs bear years.
    # With only ~30 observations per group, this is a Welch t-test on the annual returns.
    if n_bull >= 2 and n_bear >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(
            rets[bull_mask], rets[bear_mask], equal_var=False
        )
    else:
        welch_t, welch_p = float("nan"), float("nan")

    # Binomial test: are up-years in 'bull signal' years more frequent than the
    # unconditional base rate?  H₀: p_up|bull = uncond_up_rate.
    binom_result = scipy_stats.binomtest(
        k=int(mkt_up[bull_mask].sum()),
        n=int(n_bull),
        p=float(uncond_up_rate),
        alternative="greater",
    )
    binom_p = float(binom_result.pvalue)

    # Permutation test: shuffle conference labels, recompute hit_rate_all.
    rng = np.random.default_rng(seed)
    perm_hits = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permuted(s)
        perm_hit = np.where(shuffled == 1, mkt_up.astype(int), (~mkt_up).astype(int)).mean()
        perm_hits[i] = perm_hit
    perm_p = float((perm_hits >= hit_rate_all).mean())

    return {
        "n_total": int(n),
        "n_bull_years": int(n_bull),
        "n_bear_years": int(n_bear),
        "hit_rate_all": round(float(hit_rate_all), 4),
        "hit_rate_bull": round(float(hit_rate_bull), 4),
        "hit_rate_bear": round(float(hit_rate_bear), 4),
        "uncond_up_rate": round(float(uncond_up_rate), 4),
        "mean_return_bull": round(float(mean_bull), 4),
        "mean_return_bear": round(float(mean_bear), 4),
        "mean_return_all": round(float(mean_all), 4),
        "welch_t": round(float(welch_t), 3),
        "welch_p": round(float(welch_p), 4),
        "binom_p": round(float(binom_p), 4),
        "perm_p": round(float(perm_p), 4),
        "perm_hits": perm_hits,  # full permutation distribution, for plots
    }


# ---------------------------------------------------------------------------
# Multiple-comparisons summary
# ---------------------------------------------------------------------------
def multiple_comparisons_summary(df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 158) -> pd.DataFrame:
    """Run both signal variants and report uncorrected and Bonferroni-corrected p-values.

    Tests:
      1. NFC win → bull (primary Krueger-Kennedy signal)
      2. Original-NFL win → bull (Krueger-Kennedy original spirit)

    Returns a DataFrame with columns: test, hit_rate, binom_p, perm_p,
    bonferroni_binom_p, bonferroni_perm_p.

    With only 2 hypotheses the Bonferroni factor is small, but the key point is
    that *both* tests share the same n ≈ 58 data points and the same 70%
    unconditional base rate — the Bonferroni correction is almost irrelevant
    compared to the tiny-n problem.
    """
    tests = [
        ("NFC win -> bull (conference)", "signal_nfc"),
        ("Orig-NFL win -> bull (orig_nfl)", "signal_orig_nfl"),
    ]
    from super_bowl.strategy import signal_nfc, signal_orig_nfl
    sig_funcs = {"signal_nfc": signal_nfc, "signal_orig_nfl": signal_orig_nfl}

    rows = []
    n_tests = len(tests)
    for label, sig_key in tests:
        sig = sig_funcs[sig_key](df)
        st = hit_rate_stats(df, sig, n_permutations=n_permutations, seed=seed)
        rows.append({
            "test": label,
            "n": st["n_total"],
            "n_bull": st["n_bull_years"],
            "hit_rate_all": st["hit_rate_all"],
            "uncond_up_rate": st["uncond_up_rate"],
            "binom_p": st["binom_p"],
            "perm_p": st["perm_p"],
            "bonferroni_binom_p": min(1.0, st["binom_p"] * n_tests),
            "bonferroni_perm_p": min(1.0, st["perm_p"] * n_tests),
            "welch_t": st["welch_t"],
            "welch_p": st["welch_p"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 158) -> dict:
    """One-stop summary dict for the real merged DataFrame.

    Returns a flat dict mirroring the R = dict(...) in build_notebooks.py and
    docs/results.md. All numbers are rounded to useful precision.
    """
    sig_nfc = signal_nfc(df)
    sig_orig = signal_orig_nfl(df)

    st_nfc = hit_rate_stats(df, sig_nfc, n_permutations=n_permutations, seed=seed)
    st_orig = hit_rate_stats(df, sig_orig, n_permutations=n_permutations, seed=seed)

    n = st_nfc["n_total"]
    uncond = st_nfc["uncond_up_rate"]

    return {
        # Sample size
        "n": n,
        "n_nfc_years": st_nfc["n_bull_years"],
        "n_afc_years": st_nfc["n_bear_years"],
        # Unconditional base rate
        "uncond_up_rate_pct": round(uncond * 100, 1),
        # NFC signal
        "nfc_hit_rate_pct": round(st_nfc["hit_rate_all"] * 100, 1),
        "nfc_hit_bull_pct": round(st_nfc["hit_rate_bull"] * 100, 1),
        "nfc_hit_bear_pct": round(st_nfc["hit_rate_bear"] * 100, 1),
        "nfc_mean_bull": round(st_nfc["mean_return_bull"] * 100, 1),
        "nfc_mean_bear": round(st_nfc["mean_return_bear"] * 100, 1),
        "nfc_mean_all": round(st_nfc["mean_return_all"] * 100, 1),
        "nfc_binom_p": st_nfc["binom_p"],
        "nfc_perm_p": st_nfc["perm_p"],
        "nfc_welch_t": st_nfc["welch_t"],
        "nfc_welch_p": st_nfc["welch_p"],
        # Orig-NFL signal
        "orig_hit_rate_pct": round(st_orig["hit_rate_all"] * 100, 1),
        "orig_binom_p": st_orig["binom_p"],
        "orig_perm_p": st_orig["perm_p"],
        "orig_welch_t": st_orig["welch_t"],
        # Bonferroni
        "bonf_nfc_binom_p": min(1.0, st_nfc["binom_p"] * 2),
        "bonf_orig_binom_p": min(1.0, st_orig["binom_p"] * 2),
    }
