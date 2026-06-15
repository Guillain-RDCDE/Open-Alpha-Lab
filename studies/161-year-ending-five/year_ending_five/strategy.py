"""The strategy and its honest controls — Study 161 (Year-Ending-Five).

The folk claim: years ending in 5 are the best for stocks; years ending in 0 are the
worst — a neat mnemonic for the "decennial cycle."  We measure this claim three ways:

1. **Mean return by last digit** — compute the mean calendar-year return grouped by
   each of the 10 possible last digits and report the full decennial cycle, not just
   the cherry-picked 5 and 0.

2. **HAC t-stat on the digit-5 contrast** — is the mean return of digit-5 years
   significantly above the grand mean, after Newey-West autocorrelation correction?
   With n ≈ 16, the power floor is brutal; a large t-stat could still be data-noise.

3. **Permutation / multiple-comparisons correction** — we are testing the best of 10
   digits post-hoc.  We permute year labels 50,000 times to find what fraction of
   random digit assignments produce a "best digit" at least as strong as the observed
   digit-5 mean.  This is the Bonferroni-by-simulation version of "did we just pick
   the winner after looking at all 10?"

The fair null: an unconditional buy-and-hold strategy (no timing) and a random-digit
control (randomly relabelling years to digits).

No look-ahead: the calendar year's last digit is known on 1 January of that year (the
year has already started, so "buy in years ending in 5" means buying in January of
the year — the digit is known before the year's return is realised).  For a long-only
investor this is trivially actionable once per decade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core grouping — mean return by last digit
# ---------------------------------------------------------------------------
def returns_by_digit(df: pd.DataFrame, col: str = "ret_pct") -> pd.DataFrame:
    """Mean, std and n of annual returns grouped by last calendar digit.

    Returns a DataFrame indexed 0–9 with columns ``mean``, ``std``, ``n``,
    sorted by digit.  The full decennial table — not just the digit-5 headline —
    is the honest way to present the claim.
    """
    return (
        df.groupby("last_digit")[col]
        .agg(["mean", "std", "count"])
        .rename(columns={"count": "n"})
        .sort_index()
    )


def digit5_contrast(df: pd.DataFrame, col: str = "ret_pct") -> dict:
    """Mean return of digit-5 years vs the grand mean and vs all other digits.

    Returns the digit-5 mean, grand mean, contrast (5 minus others), and the
    HAC t-stat on the contrast (null: contrast == 0).
    """
    fives = df[df["last_digit"] == 5][col].to_numpy(dtype=float)
    others = df[df["last_digit"] != 5][col].to_numpy(dtype=float)
    grand = df[col].to_numpy(dtype=float)

    mean5 = float(fives.mean()) if len(fives) else float("nan")
    mean_others = float(others.mean()) if len(others) else float("nan")
    grand_mean = float(grand.mean()) if len(grand) else float("nan")
    contrast = mean5 - mean_others

    # HAC t-stat on the fives sample vs the grand mean as the null.
    hac = _hac_tstat_vs_null(fives, grand_mean)

    return {
        "n5": int(len(fives)),
        "n_others": int(len(others)),
        "mean5_pct": mean5,
        "mean_others_pct": mean_others,
        "grand_mean_pct": grand_mean,
        "contrast_pct": contrast,
        "tstat_hac": hac["tstat"],
        "se": hac["se"],
        "std5": float(fives.std(ddof=1)) if len(fives) > 1 else float("nan"),
        "low_power_warning": len(fives) < 20,
    }


# ---------------------------------------------------------------------------
# Permutation / multiple-comparisons correction
# ---------------------------------------------------------------------------
def permutation_test(
    df: pd.DataFrame,
    col: str = "ret_pct",
    n_perm: int = 50_000,
    seed: int = 161,
) -> dict:
    """Permutation test for the decennial-cycle claim, with multiple-comparisons accounting.

    Two p-values are reported:

    - ``pval_digit5`` — what fraction of random permutations of digit labels assign
      digit 5 a mean >= the observed digit-5 mean?  (The naive single-hypothesis p.)
    - ``pval_best_of_10`` — what fraction of random permutations have *some* digit
      with a mean >= the observed digit-5 mean?  This is the multiple-comparisons
      corrected p (the "best-of-10" or data-snooping correction): we are implicitly
      claiming digit 5 is special only because it is the best of 10 digits; the null
      is that any digit could have been the best by chance.

    The ratio ``pval_best_of_10 / pval_digit5`` measures the multiple-comparisons
    inflation factor — how much harder the corrected bar is to clear.
    """
    rng = np.random.default_rng(seed)
    all_vals = df[col].to_numpy(dtype=float)
    all_digits = df["last_digit"].to_numpy(dtype=int)
    observed5 = float(all_vals[all_digits == 5].mean())
    observed_max = float(max(
        all_vals[all_digits == d].mean()
        for d in range(10)
        if (all_digits == d).sum() > 0
    ))

    count5 = 0
    count_max = 0
    for _ in range(n_perm):
        perm = rng.permutation(all_digits)
        # Digit-5 mean under permuted labels.
        mask5 = perm == 5
        if mask5.sum() > 0 and all_vals[mask5].mean() >= observed5:
            count5 += 1
        # Best-of-10 under permuted labels.
        digit_means = [
            all_vals[perm == d].mean()
            for d in range(10)
            if (perm == d).sum() > 0
        ]
        if max(digit_means) >= observed_max:
            count_max += 1

    pval5 = count5 / n_perm
    pval_max = count_max / n_perm
    return {
        "observed5_mean_pct": observed5,
        "observed_max_mean_pct": observed_max,
        "pval_digit5": pval5,
        "pval_best_of_10": pval_max,
        "inflation_factor": float(pval_max / pval5) if pval5 > 0 else float("nan"),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Decade-by-decade out-of-sample split
# ---------------------------------------------------------------------------
def pre_post_split(
    df: pd.DataFrame,
    split_year: int = 1945,
    col: str = "ret_pct",
) -> dict:
    """Split digit-5 mean return pre/post a split year (informal out-of-sample).

    If the effect exists only in the in-sample period used to coin the claim
    (often cited in post-WWII data), the post-period should show a weaker pattern.
    We report both halves so the reader can judge.
    """
    pre = df[df.index < split_year]
    post = df[df.index >= split_year]

    def _stats(sub):
        fives = sub[sub["last_digit"] == 5][col].to_numpy(dtype=float)
        others = sub[sub["last_digit"] != 5][col].to_numpy(dtype=float)
        return {
            "n5": int(len(fives)),
            "mean5": float(fives.mean()) if len(fives) else float("nan"),
            "mean_others": float(others.mean()) if len(others) else float("nan"),
            "contrast": float(fives.mean() - others.mean())
            if len(fives) and len(others)
            else float("nan"),
        }

    return {
        "split_year": split_year,
        "pre": _stats(pre),
        "post": _stats(post),
    }


# ---------------------------------------------------------------------------
# Could you trade it? — buy-and-hold vs digit-based timing
# ---------------------------------------------------------------------------
def decennial_timing_strategy(
    df: pd.DataFrame,
    good_digits: tuple[int, ...] = (5,),
    bad_digits: tuple[int, ...] = (0,),
    col: str = "ret_pct",
) -> dict:
    """Simulate a 'buy in good years, avoid bad years' timing strategy vs buy-and-hold.

    The strategy: hold S&P 500 when the calendar year's last digit is in ``good_digits``;
    sit in cash (0 % return) when it's in ``bad_digits``; hold S&P in all other years.
    Compare the compounded terminal wealth against unconditional buy-and-hold over the
    full sample.

    No transaction-cost adjustment is made (just one trade per decade at most, so costs
    are trivial).  The point of this section is to illustrate that even with a 'real'
    decennial signal, a long-only investor who is already in the market loses almost
    nothing by holding through 'bad' digit years, and the timing benefit is negligible
    on an annualised basis.
    """
    r = df[col].to_numpy(dtype=float) / 100.0  # fraction
    digits = df["last_digit"].to_numpy(dtype=int)
    n = len(r)

    # Strategy position: +1 (hold) or 0 (cash) each year.
    position = np.ones(n, dtype=float)  # default: hold
    for d in bad_digits:
        position[digits == d] = 0.0

    strat_ret = position * r
    bah_ret = r.copy()

    strat_cum = float(np.prod(1.0 + strat_ret))
    bah_cum = float(np.prod(1.0 + bah_ret))
    n_yrs = n

    strat_ann = float(strat_cum ** (1.0 / n_yrs) - 1.0) * 100.0
    bah_ann = float(bah_cum ** (1.0 / n_yrs) - 1.0) * 100.0

    return {
        "n_years": n_yrs,
        "n_avoided_years": int((position == 0).sum()),
        "strat_cum_total": float(strat_cum),
        "bah_cum_total": float(bah_cum),
        "strat_ann_pct": strat_ann,
        "bah_ann_pct": bah_ann,
        "ann_advantage_pct": strat_ann - bah_ann,
        "good_digits": list(good_digits),
        "bad_digits": list(bad_digits),
    }


# ---------------------------------------------------------------------------
# Omnibus summary
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_perm: int = 50_000, seed: int = 161) -> dict:
    """Headline inference for the decennial-cycle claim.

    Combines the digit-5 contrast, the full decennial table, the permutation
    double-correction test, and the pre/post split into a single result dict
    suitable for notebooks and verify.py.
    """
    con = digit5_contrast(df)
    perm = permutation_test(df, n_perm=n_perm, seed=seed)
    split = pre_post_split(df)
    timing = decennial_timing_strategy(df)
    tbl = returns_by_digit(df)

    return {
        # Core contrast
        "n_total": int(len(df)),
        "n5": con["n5"],
        "mean5_pct": con["mean5_pct"],
        "mean_others_pct": con["mean_others_pct"],
        "grand_mean_pct": con["grand_mean_pct"],
        "contrast_pct": con["contrast_pct"],
        "std5_pct": con["std5"],
        "tstat_hac": con["tstat_hac"],
        "low_power_warning": con["low_power_warning"],
        # Permutation
        "pval_digit5": perm["pval_digit5"],
        "pval_best_of_10": perm["pval_best_of_10"],
        "inflation_factor": perm["inflation_factor"],
        # Worst digit (0)
        "mean0_pct": float(df[df["last_digit"] == 0]["ret_pct"].mean()),
        "n0": int((df["last_digit"] == 0).sum()),
        # Pre/post
        "pre_n5": split["pre"]["n5"],
        "pre_mean5": split["pre"]["mean5"],
        "post_n5": split["post"]["n5"],
        "post_mean5": split["post"]["mean5"],
        # Timing
        "strat_ann_pct": timing["strat_ann_pct"],
        "bah_ann_pct": timing["bah_ann_pct"],
        "ann_advantage_pct": timing["ann_advantage_pct"],
        # Full decennial table (for display)
        "decennial_table": tbl,
    }


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-stat vs an external null mean
# ---------------------------------------------------------------------------
def _hac_tstat_vs_null(r: np.ndarray, null_mean: float) -> dict:
    """HAC t-statistic testing H0: mean(r) == null_mean.

    Uses Newey-West (Bartlett kernel) with the standard ``floor(4*(n/100)^(2/9))``
    lag rule.  With n < 10 the t-stat is computed but flagged as unreliable — at
    n ≈ 16 the power is low and each observation has an outsized influence.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mu = r.mean()
    e = r - mu
    if n < 4:
        return {"tstat": float("nan"), "mean": float(mu), "se": float("nan")}
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        if k < n:
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float((mu - null_mean) / se) if se > 0 else float("nan")
    return {"tstat": tstat, "mean": float(mu), "se": float(se)}
