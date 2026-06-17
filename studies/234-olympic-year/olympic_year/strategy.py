"""The strategy and its honest controls — Study 234 (Olympic-Year).

The folk claim: US stocks perform better in Summer Olympic years.  We measure this
claim three ways:

1. **Mean return comparison** — compute the mean calendar-year return in Olympic years
   vs non-Olympic years, reporting both the raw contrast and standard errors.

2. **HAC t-stat on the Olympic contrast** — is the mean return of Olympic years
   significantly above the grand mean, after Newey-West autocorrelation correction?
   With n ≈ 24 Olympic years, the power floor is severe.

3. **Permutation test** — we permute year labels 50,000 times to find what fraction
   of random assignments produce a contrast at least as large as the observed one.
   This bypasses distributional assumptions and handles the potential clustering
   of Olympic years every four years.

The fair null: an unconditional buy-and-hold strategy (no timing) and a random-label
control (randomly relabelling years as Olympic/non-Olympic).

No look-ahead: the Olympic year calendar is pre-announced well in advance by the IOC.
A long-only investor could "buy in Olympic years" from January of that year.  The
calendar is publicly known before the year's return is realised, so the strategy is
mechanically feasible — but that is the least important question given the n.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core grouping — mean return by Olympic vs non-Olympic
# ---------------------------------------------------------------------------
def returns_by_group(df: pd.DataFrame, col: str = "ret_pct") -> pd.DataFrame:
    """Mean, std and n of annual returns grouped by Olympic/non-Olympic.

    Returns a DataFrame indexed by bool (False = non-Olympic, True = Olympic)
    with columns ``mean``, ``std``, ``n``.
    """
    return (
        df.groupby("is_olympic")[col]
        .agg(["mean", "std", "count"])
        .rename(columns={"count": "n"})
        .sort_index()
    )


def olympic_contrast(df: pd.DataFrame, col: str = "ret_pct") -> dict:
    """Mean return of Olympic years vs non-Olympic years and vs the grand mean.

    Returns the Olympic mean, non-Olympic mean, grand mean, contrast, and the
    HAC t-stat on the contrast (null: contrast == 0).
    """
    olympic = df[df["is_olympic"]][col].to_numpy(dtype=float)
    non_olympic = df[~df["is_olympic"]][col].to_numpy(dtype=float)
    grand = df[col].to_numpy(dtype=float)

    mean_oly = float(olympic.mean()) if len(olympic) else float("nan")
    mean_non = float(non_olympic.mean()) if len(non_olympic) else float("nan")
    grand_mean = float(grand.mean()) if len(grand) else float("nan")
    contrast = mean_oly - mean_non

    # HAC t-stat on the Olympic sub-sample vs the grand mean as the null.
    hac = _hac_tstat_vs_null(olympic, grand_mean)

    return {
        "n_olympic": int(len(olympic)),
        "n_non_olympic": int(len(non_olympic)),
        "mean_olympic_pct": mean_oly,
        "mean_non_olympic_pct": mean_non,
        "grand_mean_pct": grand_mean,
        "contrast_pct": contrast,
        "tstat_hac": hac["tstat"],
        "se": hac["se"],
        "std_olympic": float(olympic.std(ddof=1)) if len(olympic) > 1 else float("nan"),
        "low_power_warning": len(olympic) < 30,
    }


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------
def permutation_test(
    df: pd.DataFrame,
    col: str = "ret_pct",
    n_perm: int = 50_000,
    seed: int = 234,
) -> dict:
    """Permutation test for the Olympic-year claim.

    ``pval_two_sided`` — what fraction of random permutations of the Olympic label
    produce an absolute contrast >= the observed absolute contrast?
    (Two-sided: we do not know a priori whether Olympic years should be better or worse.)

    ``pval_one_sided`` — one-sided test: what fraction of random permutations
    produce an Olympic mean >= the observed Olympic mean?
    (Favours finding a positive result; reported for completeness but the two-sided
    is the pre-specified test since the claim has no directional mechanism.)
    """
    rng = np.random.default_rng(seed)
    all_vals = df[col].to_numpy(dtype=float)
    is_oly = df["is_olympic"].to_numpy(dtype=bool)
    n_oly = int(is_oly.sum())
    observed_mean = float(all_vals[is_oly].mean())
    observed_mean_non = float(all_vals[~is_oly].mean())
    observed_contrast = observed_mean - observed_mean_non

    count_two = 0
    count_one = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(all_vals))
        perm_oly = perm[:n_oly]
        perm_non = perm[n_oly:]
        perm_mean = float(all_vals[perm_oly].mean())
        perm_mean_non = float(all_vals[perm_non].mean())
        perm_contrast = perm_mean - perm_mean_non
        if abs(perm_contrast) >= abs(observed_contrast):
            count_two += 1
        if perm_contrast >= observed_contrast:
            count_one += 1

    return {
        "observed_olympic_mean_pct": observed_mean,
        "observed_non_olympic_mean_pct": observed_mean_non,
        "observed_contrast_pct": observed_contrast,
        "pval_two_sided": count_two / n_perm,
        "pval_one_sided": count_one / n_perm,
        "n_perm": n_perm,
        "n_olympic": n_oly,
    }


# ---------------------------------------------------------------------------
# Pre/post Cold-War split
# ---------------------------------------------------------------------------
def pre_post_split(
    df: pd.DataFrame,
    split_year: int = 1972,
    col: str = "ret_pct",
) -> dict:
    """Split Olympic-year mean return pre/post a split year (informal out-of-sample).

    1972 is chosen as the midpoint of the modern era: it marks the end of the
    Bretton Woods era (gold standard ended 1971) and the beginning of the modern
    floating-rate, globalised financial era.
    """
    pre = df[df.index < split_year]
    post = df[df.index >= split_year]

    def _stats(sub):
        oly = sub[sub["is_olympic"]][col].to_numpy(dtype=float)
        non = sub[~sub["is_olympic"]][col].to_numpy(dtype=float)
        return {
            "n_olympic": int(len(oly)),
            "mean_olympic": float(oly.mean()) if len(oly) else float("nan"),
            "mean_non": float(non.mean()) if len(non) else float("nan"),
            "contrast": float(oly.mean() - non.mean()) if len(oly) and len(non) else float("nan"),
        }

    return {
        "split_year": split_year,
        "pre": _stats(pre),
        "post": _stats(post),
    }


# ---------------------------------------------------------------------------
# Could you trade it? — buy-and-hold vs Olympic-year timing
# ---------------------------------------------------------------------------
def olympic_timing_strategy(
    df: pd.DataFrame,
    col: str = "ret_pct",
) -> dict:
    """Simulate an 'hold only in Olympic years' timing strategy vs buy-and-hold.

    Strategy: hold S&P 500 in Olympic years; sit in cash (0% return) in non-Olympic years.
    Compare compounded terminal wealth against unconditional buy-and-hold over the
    full sample.  No transaction-cost adjustment (just one trade per 4 years max).
    """
    r = df[col].to_numpy(dtype=float) / 100.0
    is_oly = df["is_olympic"].to_numpy(dtype=bool)
    n = len(r)

    # Strategy position: +1 (hold) in Olympic years, 0 (cash) otherwise.
    position = is_oly.astype(float)

    strat_ret = position * r
    bah_ret = r.copy()

    strat_cum = float(np.prod(1.0 + strat_ret))
    bah_cum = float(np.prod(1.0 + bah_ret))
    n_yrs = n

    strat_ann = float(strat_cum ** (1.0 / n_yrs) - 1.0) * 100.0
    bah_ann = float(bah_cum ** (1.0 / n_yrs) - 1.0) * 100.0

    return {
        "n_years": n_yrs,
        "n_olympic_held": int(is_oly.sum()),
        "n_cash_years": int((~is_oly).sum()),
        "strat_cum_total": strat_cum,
        "bah_cum_total": bah_cum,
        "strat_ann_pct": strat_ann,
        "bah_ann_pct": bah_ann,
        "ann_advantage_pct": strat_ann - bah_ann,
    }


# ---------------------------------------------------------------------------
# Omnibus summary
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_perm: int = 50_000, seed: int = 234) -> dict:
    """Headline inference for the Olympic-year claim.

    Combines the Olympic contrast, the permutation test, and the pre/post split
    into a single result dict suitable for notebooks and verify.py.
    """
    con = olympic_contrast(df)
    perm = permutation_test(df, n_perm=n_perm, seed=seed)
    split = pre_post_split(df)
    timing = olympic_timing_strategy(df)
    tbl = returns_by_group(df)

    return {
        # Core contrast
        "n_total": int(len(df)),
        "n_olympic": con["n_olympic"],
        "n_non_olympic": con["n_non_olympic"],
        "mean_olympic_pct": con["mean_olympic_pct"],
        "mean_non_olympic_pct": con["mean_non_olympic_pct"],
        "grand_mean_pct": con["grand_mean_pct"],
        "contrast_pct": con["contrast_pct"],
        "std_olympic_pct": con["std_olympic"],
        "tstat_hac": con["tstat_hac"],
        "low_power_warning": con["low_power_warning"],
        # Permutation
        "pval_two_sided": perm["pval_two_sided"],
        "pval_one_sided": perm["pval_one_sided"],
        # Pre/post
        "pre_n_olympic": split["pre"]["n_olympic"],
        "pre_mean_olympic": split["pre"]["mean_olympic"],
        "pre_mean_non": split["pre"]["mean_non"],
        "post_n_olympic": split["post"]["n_olympic"],
        "post_mean_olympic": split["post"]["mean_olympic"],
        "post_mean_non": split["post"]["mean_non"],
        # Timing
        "strat_ann_pct": timing["strat_ann_pct"],
        "bah_ann_pct": timing["bah_ann_pct"],
        "ann_advantage_pct": timing["ann_advantage_pct"],
        # Group table
        "group_table": tbl,
    }


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-stat vs an external null mean
# ---------------------------------------------------------------------------
def _hac_tstat_vs_null(r: np.ndarray, null_mean: float) -> dict:
    """HAC t-statistic testing H0: mean(r) == null_mean.

    Uses Newey-West (Bartlett kernel) with the standard ``floor(4*(n/100)^(2/9))``
    lag rule.  With n < 10 the t-stat is computed but flagged as unreliable.
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
