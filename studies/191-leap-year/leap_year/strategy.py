"""The strategy and its honest controls — Study 191 (Leap-Year).

The folk claim: stocks return more (or less) in leap years.  Since leap years
coincide exactly with US presidential-election years in the modern era (both are
year % 4 == 0), there are three overlapping hypotheses a credulous researcher
could have tested:

1. **H₁ (leap-year premium)** — leap years earn a different mean return than
   non-leap years.
2. **H₂ (election-year premium)** — election years (year % 4 == 0) earn a
   different mean return than other years.  In the modern era H₁ and H₂ are
   identical; they diverge only for century years like 1900 (not a leap year,
   but still an election year).
3. **H₃ (presidential-cycle premium)** — the four-year presidential cycle
   (year-of-term 1–4) produces differentiated returns.  This is the Study 81
   (Four-Year-Itch) confound.

We report all three tests.  The key multiple-comparisons reckoning applies a
Bonferroni correction over the three related hypotheses.

**Control / baseline**: the unconditional mean annual return (the simplest
"always-in" benchmark).  A leap-year effect is only interesting if it is
distinguishable from the unconditional mean at the same statistical bar we demand
for any other claim.

No look-ahead: the year label (leap or not) is known before January 1st of that year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-statistic (local copy — no quantlab dependency)
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> dict:
    """Newey-West (HAC) t-statistic on the sample mean of ``r``.

    For annual return series the HAC lags are usually 1-2 given low effective n;
    it reverts to a standard t-stat when n < 8.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return {"tstat": float("nan"), "mean": float(r.mean()) if n else float("nan"),
                "se": float("nan"), "n": n}
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
# Primary test: leap year vs non-leap year
# ---------------------------------------------------------------------------
def leap_comparison(df: pd.DataFrame) -> dict:
    """Compare leap-year calendar returns to non-leap-year returns.

    Uses a Welch t-test (unequal variances) on the mean contrast and HAC
    t-stats on each group separately.

    Parameters
    ----------
    df : DataFrame with columns ``ret, is_leap, is_election, term_year``.

    Returns
    -------
    dict with keys: n_leap, n_non_leap, mean_leap_pct, mean_non_leap_pct,
    contrast_pct, tstat_welch, pval_welch, tstat_leap_hac, tstat_non_leap_hac.
    """
    from scipy.stats import ttest_ind

    leap = df.loc[df["is_leap"], "ret"].to_numpy(dtype=float)
    non_leap = df.loc[~df["is_leap"], "ret"].to_numpy(dtype=float)
    leap = leap[np.isfinite(leap)]
    non_leap = non_leap[np.isfinite(non_leap)]

    h_leap = _hac_tstat(leap)
    h_non = _hac_tstat(non_leap)

    if len(leap) > 1 and len(non_leap) > 1:
        t_w, p_w = ttest_ind(leap, non_leap, equal_var=False)
    else:
        t_w, p_w = float("nan"), float("nan")

    return {
        "n_leap": int(len(leap)),
        "n_non_leap": int(len(non_leap)),
        "mean_leap_pct": float(leap.mean() * 100) if len(leap) else float("nan"),
        "mean_non_leap_pct": float(non_leap.mean() * 100) if len(non_leap) else float("nan"),
        "contrast_pct": float((leap.mean() - non_leap.mean()) * 100)
            if len(leap) and len(non_leap) else float("nan"),
        "tstat_welch": float(t_w),
        "pval_welch": float(p_w),
        "tstat_leap_hac": h_leap["tstat"],
        "tstat_non_leap_hac": h_non["tstat"],
    }


# ---------------------------------------------------------------------------
# Presidential-cycle breakdown: year-of-term 1-4
# ---------------------------------------------------------------------------
def term_cycle_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Mean return and HAC t-stat for each year-of-presidential-term (1–4).

    Year 1 = election year (same as leap year in modern era).
    Year 4 = mid-term year (often cited as the best performer in the four-year itch).
    Returns a DataFrame sorted by term_year.
    """
    rows = []
    unconditional_mean = df["ret"].mean()
    for ty in [1, 2, 3, 4]:
        sub = df.loc[df["term_year"] == ty, "ret"].to_numpy(dtype=float)
        sub = sub[np.isfinite(sub)]
        h = _hac_tstat(sub)
        rows.append({
            "term_year": ty,
            "n": int(len(sub)),
            "mean_pct": float(sub.mean() * 100) if len(sub) else float("nan"),
            "std_pct": float(sub.std(ddof=1) * 100) if len(sub) > 1 else float("nan"),
            "contrast_vs_uncond_pct": float((sub.mean() - unconditional_mean) * 100)
                if len(sub) else float("nan"),
            "tstat_hac": h["tstat"],
        })
    return pd.DataFrame(rows).set_index("term_year")


# ---------------------------------------------------------------------------
# Multiple-comparisons sweep
# ---------------------------------------------------------------------------
def multiple_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Test the three overlapping quadrennial hypotheses with Bonferroni correction.

    The three tests a credulous researcher could run:
    1. Leap year (year % 4 == 0 and not century exception) vs rest.
    2. Election year (year % 4 == 0) vs rest — identical to leap in modern era.
    3. Best presidential-cycle year (year 4 = midterm) vs rest — the rival claim.

    Returns a DataFrame with raw and Bonferroni-corrected p-values.
    """
    from scipy.stats import ttest_ind

    n_tests = 3
    rows = []

    # H1: leap vs non-leap
    lc = leap_comparison(df)
    rows.append({
        "hypothesis": "Leap year premium",
        "n_group": lc["n_leap"],
        "mean_group_pct": lc["mean_leap_pct"],
        "contrast_pct": lc["contrast_pct"],
        "pval_raw": lc["pval_welch"],
        "pval_bonferroni": min(lc["pval_welch"] * n_tests, 1.0)
            if np.isfinite(lc["pval_welch"]) else float("nan"),
    })

    # H2: election year vs non-election (identical in modern era)
    elec = df.loc[df["is_election"], "ret"].to_numpy(dtype=float)
    non_elec = df.loc[~df["is_election"], "ret"].to_numpy(dtype=float)
    elec = elec[np.isfinite(elec)]
    non_elec = non_elec[np.isfinite(non_elec)]
    t2, p2 = ttest_ind(elec, non_elec, equal_var=False) if len(elec) > 1 and len(non_elec) > 1 \
        else (float("nan"), float("nan"))
    rows.append({
        "hypothesis": "Election year premium",
        "n_group": int(len(elec)),
        "mean_group_pct": float(elec.mean() * 100) if len(elec) else float("nan"),
        "contrast_pct": float((elec.mean() - non_elec.mean()) * 100)
            if len(elec) and len(non_elec) else float("nan"),
        "pval_raw": float(p2),
        "pval_bonferroni": float(min(p2 * n_tests, 1.0)) if np.isfinite(p2) else float("nan"),
    })

    # H3: mid-term year (year-of-term 4) vs rest — the rival "four-year itch" claim
    midterm = df.loc[df["term_year"] == 4, "ret"].to_numpy(dtype=float)
    non_midterm = df.loc[df["term_year"] != 4, "ret"].to_numpy(dtype=float)
    midterm = midterm[np.isfinite(midterm)]
    non_midterm = non_midterm[np.isfinite(non_midterm)]
    t3, p3 = ttest_ind(midterm, non_midterm, equal_var=False) if len(midterm) > 1 and len(non_midterm) > 1 \
        else (float("nan"), float("nan"))
    rows.append({
        "hypothesis": "Mid-term+1 year premium (yr-of-term 4)",
        "n_group": int(len(midterm)),
        "mean_group_pct": float(midterm.mean() * 100) if len(midterm) else float("nan"),
        "contrast_pct": float((midterm.mean() - non_midterm.mean()) * 100)
            if len(midterm) and len(non_midterm) else float("nan"),
        "pval_raw": float(p3),
        "pval_bonferroni": float(min(p3 * n_tests, 1.0)) if np.isfinite(p3) else float("nan"),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Control / unconditional baseline
# ---------------------------------------------------------------------------
def unconditional_stats(df: pd.DataFrame) -> dict:
    """Mean and HAC t-stat for the unconditional (all-years) return series."""
    r = df["ret"].dropna().to_numpy(dtype=float)
    h = _hac_tstat(r)
    return {
        "n": int(len(r)),
        "mean_pct": float(r.mean() * 100),
        "std_pct": float(r.std(ddof=1) * 100),
        "tstat_hac": h["tstat"],
    }


# ---------------------------------------------------------------------------
# Summarize — one flat dict for notebooks and verify.py
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """All headline numbers for the leap-year study in one flat dict.

    Mirrors the structure of docs/results.md so notebooks and verify.py can
    reference the same keys from ``R`` (frozen) or ``S`` (live).
    """
    lc = leap_comparison(df)
    tc = term_cycle_comparison(df)
    mc = multiple_comparisons(df)
    unc = unconditional_stats(df)

    return {
        # Core leap test
        "n_leap": lc["n_leap"],
        "n_non_leap": lc["n_non_leap"],
        "mean_leap_pct": lc["mean_leap_pct"],
        "mean_non_leap_pct": lc["mean_non_leap_pct"],
        "contrast_pct": lc["contrast_pct"],
        "tstat_welch": lc["tstat_welch"],
        "pval_welch": lc["pval_welch"],
        "tstat_leap_hac": lc["tstat_leap_hac"],
        # Unconditional baseline
        "n_total": unc["n"],
        "mean_all_pct": unc["mean_pct"],
        "std_all_pct": unc["std_pct"],
        # Term-year breakdown (as dict-of-dicts for readability)
        "term_y1_mean": float(tc.loc[1, "mean_pct"]),
        "term_y2_mean": float(tc.loc[2, "mean_pct"]),
        "term_y3_mean": float(tc.loc[3, "mean_pct"]),
        "term_y4_mean": float(tc.loc[4, "mean_pct"]),
        "term_y1_t": float(tc.loc[1, "tstat_hac"]),
        "term_y4_t": float(tc.loc[4, "tstat_hac"]),
        # Multiple comparisons
        "mc_table": mc,
        "pval_leap_bonf": float(mc.loc[0, "pval_bonferroni"]),
    }
