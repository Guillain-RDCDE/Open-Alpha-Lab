"""The strategy and its honest controls — Study 163 (Friday-13th).

The folk claim: stock returns on Friday the 13th are systematically *lower* than
on other Fridays (and lower than any average trading day). Kolb & Rodriguez (1987)
found a small negative anomaly in the Dow Jones in 1940–1987; most modern
replications find nothing.

We run three tests:

1. **Mean return on Friday-13th vs all other Fridays** (the primary test).
   HAC Newey-West t-stat on the per-day return series for each group, and a
   Welch t-test on the contrast.  The placebo arm is Friday-the-6th
   (same weekday, no superstition attached).

2. **Friday-13th vs all other trading days** — the broader baseline.
   Tests whether the anomaly would even pass the simpler bar of outperforming
   (or underperforming) the unconditional mean.

3. **Multiple-comparisons reckoning.**  We have *three* groups of "special
   Fridays" (the 6th, 13th, 20th, 27th) that could be tested; we apply a
   Bonferroni correction over the four day-of-month candidates.  This is the
   kill shot for spurious calendar findings: the 13th is not pre-registered
   *before* anyone looked — it was picked *because* of folklore, and a snooper
   testing all Fridays-of-each-month would find the most extreme one by chance.

No look-ahead: the date label is known before the open; the return is the
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
# Primary test: Friday-13th vs other Fridays
# ---------------------------------------------------------------------------
def friday_comparison(df: pd.DataFrame) -> dict:
    """Compare Friday-the-13th returns to all other-Friday returns.

    Uses a Welch t-test (unequal variances) on the mean contrast, and a
    HAC t-stat on each group separately.  The control arm is all Fridays
    *excluding* the 13th.

    Parameters
    ----------
    df : DataFrame with columns ``ret, is_f13, is_f6, is_friday``.

    Returns
    -------
    dict with keys: n_f13, mean_f13_bps, tstat_f13, n_other_fri,
    mean_other_bps, tstat_other, contrast_bps, tstat_welch, pval_welch.
    """
    from scipy.stats import ttest_ind

    f13 = df.loc[df["is_f13"], "ret"].to_numpy(dtype=float)
    other_fri = df.loc[df["is_friday"] & ~df["is_f13"], "ret"].to_numpy(dtype=float)
    f13 = f13[np.isfinite(f13)]
    other_fri = other_fri[np.isfinite(other_fri)]

    h13 = _hac_tstat(f13)
    hoth = _hac_tstat(other_fri)

    if len(f13) > 1 and len(other_fri) > 1:
        t_welch, pval_welch = ttest_ind(f13, other_fri, equal_var=False)
    else:
        t_welch, pval_welch = float("nan"), float("nan")

    return {
        "n_f13": int(len(f13)),
        "mean_f13_bps": float(f13.mean() * 1e4) if len(f13) else float("nan"),
        "tstat_f13": h13["tstat"],
        "n_other_fri": int(len(other_fri)),
        "mean_other_bps": float(other_fri.mean() * 1e4) if len(other_fri) else float("nan"),
        "tstat_other": hoth["tstat"],
        "contrast_bps": float((f13.mean() - other_fri.mean()) * 1e4)
        if len(f13) and len(other_fri) else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval_welch),
    }


# ---------------------------------------------------------------------------
# Placebo: Friday-6th vs other Fridays
# ---------------------------------------------------------------------------
def placebo_comparison(df: pd.DataFrame) -> dict:
    """Same test as ``friday_comparison`` but for Friday-the-6th (the placebo).

    If the 6th (or 20th, or 27th) looks just as anomalous, the 13th result is
    noise, not signal.
    """
    from scipy.stats import ttest_ind

    f6 = df.loc[df["is_f6"], "ret"].to_numpy(dtype=float)
    other_fri = df.loc[df["is_friday"] & ~df["is_f6"], "ret"].to_numpy(dtype=float)
    f6 = f6[np.isfinite(f6)]
    other_fri = other_fri[np.isfinite(other_fri)]

    h6 = _hac_tstat(f6)
    hoth = _hac_tstat(other_fri)

    if len(f6) > 1 and len(other_fri) > 1:
        t_welch, pval_welch = ttest_ind(f6, other_fri, equal_var=False)
    else:
        t_welch, pval_welch = float("nan"), float("nan")

    return {
        "n_f6": int(len(f6)),
        "mean_f6_bps": float(f6.mean() * 1e4) if len(f6) else float("nan"),
        "tstat_f6": h6["tstat"],
        "n_other_fri": int(len(other_fri)),
        "mean_other_bps": float(other_fri.mean() * 1e4) if len(other_fri) else float("nan"),
        "tstat_other": hoth["tstat"],
        "contrast_bps": float((f6.mean() - other_fri.mean()) * 1e4)
        if len(f6) and len(other_fri) else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval_welch),
    }


# ---------------------------------------------------------------------------
# Multiple-comparisons sweep: all four "Friday of the Nth" slots
# ---------------------------------------------------------------------------
def all_friday_dom_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Test every Friday day-of-month from 6–27 (the four non-overlapping slots).

    Bonferroni-corrects the 4 Welch p-values (these are the only "special Friday"
    day-of-month values that exist on a weekly grid: 6, 13, 20, 27).  Returns a
    DataFrame sorted by raw p-value so the most extreme result is at the top.

    This is the multiple-comparisons kill shot: if we had tested all four days
    first and then cherry-picked the most extreme one, the corrected alpha would
    be 0.05/4 = 0.0125, not 0.05.  A raw p of 0.04 on the 13th is not impressive
    against this bar — it would need to clear 0.0125.
    """
    from scipy.stats import ttest_ind

    days = [6, 13, 20, 27]
    other_fri = df.loc[df["is_friday"], "ret"].to_numpy(dtype=float)
    other_fri = other_fri[np.isfinite(other_fri)]
    n_tests = len(days)
    rows = []
    for d in days:
        mask = df["is_friday"] & (df.index.day == d)
        g = df.loc[mask, "ret"].to_numpy(dtype=float)
        g = g[np.isfinite(g)]
        rest = df.loc[df["is_friday"] & (df.index.day != d), "ret"].to_numpy(dtype=float)
        rest = rest[np.isfinite(rest)]
        if len(g) > 1 and len(rest) > 1:
            _, pval = ttest_ind(g, rest, equal_var=False)
        else:
            pval = float("nan")
        rows.append(
            {
                "day": int(d),
                "n": int(len(g)),
                "mean_bps": float(g.mean() * 1e4) if len(g) else float("nan"),
                "contrast_bps": float((g.mean() - rest.mean()) * 1e4)
                if len(g) and len(rest) else float("nan"),
                "pval_raw": float(pval),
                "pval_bonferroni": float(min(pval * n_tests, 1.0))
                if np.isfinite(pval) else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values("pval_raw")
    return out


# ---------------------------------------------------------------------------
# Broad baseline: Friday-13th vs all days
# ---------------------------------------------------------------------------
def vs_all_days(df: pd.DataFrame) -> dict:
    """Friday-13th vs the unconditional mean of all trading days.

    A simple check: even before worrying about 'other Fridays', does the 13th
    meaningfully depart from the baseline drift?
    """
    from scipy.stats import ttest_ind

    f13 = df.loc[df["is_f13"], "ret"].to_numpy(dtype=float)
    all_other = df.loc[~df["is_f13"], "ret"].to_numpy(dtype=float)
    f13 = f13[np.isfinite(f13)]
    all_other = all_other[np.isfinite(all_other)]

    if len(f13) > 1 and len(all_other) > 1:
        t_welch, pval_welch = ttest_ind(f13, all_other, equal_var=False)
    else:
        t_welch, pval_welch = float("nan"), float("nan")

    return {
        "n_f13": int(len(f13)),
        "mean_f13_bps": float(f13.mean() * 1e4) if len(f13) else float("nan"),
        "n_all_other": int(len(all_other)),
        "mean_all_bps": float(all_other.mean() * 1e4) if len(all_other) else float("nan"),
        "contrast_bps": float((f13.mean() - all_other.mean()) * 1e4)
        if len(f13) and len(all_other) else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval_welch),
    }


# ---------------------------------------------------------------------------
# Summarize — all three tests in one call
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers for Friday-13th: all three tests combined.

    Returns a flat dict mirroring docs/results.md for use in notebooks and
    verify.py.
    """
    pri = friday_comparison(df)
    plac = placebo_comparison(df)
    broad = vs_all_days(df)
    mc = all_friday_dom_comparison(df)

    # Extract the 13th row from the multiple-comparisons table.
    row13 = mc[mc["day"] == 13].iloc[0] if (mc["day"] == 13).any() else {}

    return {
        # Primary (Friday-13th vs other Fridays)
        "n_f13": pri["n_f13"],
        "n_other_fri": pri["n_other_fri"],
        "mean_f13_bps": pri["mean_f13_bps"],
        "mean_other_fri_bps": pri["mean_other_bps"],
        "contrast_bps": pri["contrast_bps"],
        "tstat_welch": pri["tstat_welch"],
        "pval_welch": pri["pval_welch"],
        "tstat_f13_hac": pri["tstat_f13"],
        # Placebo (Friday-6th vs other Fridays)
        "n_f6": plac["n_f6"],
        "mean_f6_bps": plac["mean_f6_bps"],
        "contrast_f6_bps": plac["contrast_bps"],
        "pval_f6": plac["pval_welch"],
        # Broad baseline (vs all days)
        "mean_all_bps": broad["mean_all_bps"],
        "pval_vs_all": broad["pval_welch"],
        # Bonferroni
        "pval_bonferroni_13": float(row13.get("pval_bonferroni", float("nan"))),
        # Multiple-comparisons table (full)
        "mc_table": mc,
    }
