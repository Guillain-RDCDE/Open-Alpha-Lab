"""The strategy and its honest controls — Study 166 (First-Five-Days).

The Stock Trader's Almanac "First-Five-Days Early Warning System": the sign of the
S&P 500's return over the first five trading days of January predicts the sign of the
full-year return.  We test this claim two ways:

1. **Hit-rate vs base rate**: does ``sign(ret_5d) == sign(ret_year)`` more often than
   the unconditional probability that any year is positive?  Equity markets drift up most
   years — roughly 73-74 % of years — so a naïve "always predict up" beats most calendar
   indicators on hit-rate.  We compare against the right null (the base rate) using a
   binomial test, and against a coin flip (50 %) for context.

2. **Mean full-year return by F5D sign**: do up-F5D years have meaningfully higher
   full-year returns than down-F5D years, controlling for equity drift?  We use a HAC
   t-stat (Newey-West) on the signed per-year returns as the primary inference number.

3. **Strategy vs buy-and-hold**: hold full-year after up F5D, stay flat after down F5D.
   We compare mean return and Sharpe against unconditional buy-and-hold.

The fair null: a random-label (shuffled F5D sign) control.  Effective n ≈ 76 years is
tiny — we say so loudly at every step.  A t = 4.9 on n = 76 *looks* impressive but the
desk's inference bar requires robustness across sub-periods and out-of-sample stability,
neither of which this effect delivers post-1985 (where the hit-rate drops to 60 % and
is not significant vs a coin).

No look-ahead: F5D sign is known after the close of the fifth trading day of January.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def f5d_sign(df: pd.DataFrame) -> pd.Series:
    """Return the sign of the first-five-days return (+1 / -1) for each year.

    Zero returns are theoretically possible but astronomically rare on the full index;
    they are excluded from hit-rate calculations.
    """
    return np.sign(df["ret_5d"]).rename("f5d_sign")


def base_rate(df: pd.DataFrame) -> float:
    """Unconditional probability that the full year is positive.

    The right null for any January-based indicator: equity markets drift upward most
    years regardless of what happens in the first few days.  A calendar rule only adds
    value if it beats this, not just a coin.
    """
    return float((df["ret_year"] > 0).mean())


# ---------------------------------------------------------------------------
# Hit-rate & binomial test
# ---------------------------------------------------------------------------
def hit_rate_analysis(df: pd.DataFrame) -> dict:
    """Full hit-rate breakdown and binomial significance tests.

    'Hit' = sign(ret_5d) == sign(ret_year) (both positive or both negative).

    We report:
    - overall hit-rate and a binomial p-value vs a coin (50 %).
    - binomial p-value vs the unconditional base rate (the correct null).
    - conditional hit-rates for up-F5D and down-F5D years.
    """
    from scipy.stats import binomtest

    s = f5d_sign(df)
    valid = df.loc[s != 0]
    sv = s[s != 0]
    hits = (np.sign(valid["ret_year"]) == sv).astype(int)

    n_total = int(len(hits))
    n_hits = int(hits.sum())
    hit_rate_val = float(hits.mean())

    up = sv == 1
    dn = sv == -1
    hr_up = float(hits[up].mean()) if up.any() else float("nan")
    hr_dn = float(hits[dn].mean()) if dn.any() else float("nan")
    n_up = int(up.sum())
    n_dn = int(dn.sum())

    # Binomial vs coin — the misleading test everyone reports.
    binom_coin = binomtest(n_hits, n_total, p=0.5, alternative="greater")
    pval_coin = float(binom_coin.pvalue)

    # Binomial vs base rate — the correct null for a drifting market.
    br = base_rate(df)
    binom_base = binomtest(n_hits, n_total, p=br, alternative="greater")
    pval_base = float(binom_base.pvalue)

    return {
        "n": n_total,
        "n_up_f5d": n_up,
        "n_dn_f5d": n_dn,
        "n_hits": n_hits,
        "hit_rate": hit_rate_val,
        "hit_rate_up_f5d": hr_up,
        "hit_rate_dn_f5d": hr_dn,
        "base_rate_yr_pos": br,
        "pval_vs_coin": pval_coin,
        "pval_vs_base_rate": pval_base,
    }


# ---------------------------------------------------------------------------
# Mean contrast & HAC t-stat
# ---------------------------------------------------------------------------
def mean_contrast(df: pd.DataFrame) -> dict:
    """Mean full-year return for up-F5D vs down-F5D years, and their contrast.

    The contrast = mean(ret_year | F5D > 0) - mean(ret_year | F5D < 0) is the primary
    test statistic.  A HAC t-stat (Newey-West) is computed on the signed per-year returns
    (sign(F5D) * ret_year), which centres the statistic on the contrast.  We also report
    a Welch t-test p-value for the contrast.
    """
    from scipy.stats import ttest_ind

    s = f5d_sign(df)
    up_mask = s == 1
    dn_mask = s == -1

    yr = df["ret_year"]
    up_rets = yr[up_mask].to_numpy(dtype=float)
    dn_rets = yr[dn_mask].to_numpy(dtype=float)

    mean_up = float(up_rets.mean()) if len(up_rets) else float("nan")
    mean_dn = float(dn_rets.mean()) if len(dn_rets) else float("nan")
    contrast = mean_up - mean_dn

    if len(up_rets) > 1 and len(dn_rets) > 1:
        _, pval_welsh = ttest_ind(up_rets, dn_rets, equal_var=False, alternative="greater")
    else:
        pval_welsh = float("nan")

    # HAC t-stat on signed per-year returns.
    s_clean = s[(up_mask | dn_mask)]
    r_clean = yr[(up_mask | dn_mask)]
    signed = (s_clean * r_clean).to_numpy(dtype=float)
    hac = _hac_tstat(signed)

    return {
        "mean_yr_up_f5d": mean_up,
        "mean_yr_dn_f5d": mean_dn,
        "contrast_pct": contrast * 100.0,
        "tstat_hac": hac["tstat"],
        "pval_welsh": pval_welsh,
        "n_up": int(up_mask.sum()),
        "n_dn": int(dn_mask.sum()),
    }


# ---------------------------------------------------------------------------
# F5D strategy vs buy-and-hold
# ---------------------------------------------------------------------------
def f5d_strategy(
    df: pd.DataFrame,
    cost_bps: float = 10.0,
) -> dict:
    """Simulate the F5D 'early warning' call: hold full-year after up F5D; flat after dn F5D.

    ``cost_bps`` is the one-way cost paid on each regime change (e.g. re-entering after a
    down-F5D year).  Compare to buy-and-hold of the full year every year.

    Returns per-year series plus summary statistics.
    """
    s = f5d_sign(df)
    yr = df["ret_year"]

    # Long after up-F5D, flat after down-F5D.
    position = (s > 0).astype(float)
    strat_ret = position * yr
    # Cost on posture changes.
    prev_pos = position.shift(1)
    changed = (position != prev_pos) & prev_pos.notna()
    cost_series = changed.astype(float) * (cost_bps * 1e-4)
    strat_net = strat_ret - cost_series

    bah = yr.copy()

    n = int(len(strat_net.dropna()))
    strat_summary = _hac_tstat(strat_net.dropna().to_numpy())
    bah_summary = _hac_tstat(bah.dropna().to_numpy())

    return {
        "n": n,
        "strat_mean_ann_pct": float(strat_net.mean() * 100),
        "strat_tstat": strat_summary["tstat"],
        "bah_mean_ann_pct": float(bah.mean() * 100),
        "bah_tstat": bah_summary["tstat"],
        "strat_sharpe": float(strat_net.mean() / strat_net.std(ddof=1))
        if strat_net.std() > 0 else float("nan"),
        "bah_sharpe": float(bah.mean() / bah.std(ddof=1))
        if bah.std() > 0 else float("nan"),
        "strat_returns": strat_net,
        "bah_returns": bah,
    }


# ---------------------------------------------------------------------------
# Random-label control (shuffled F5D signs)
# ---------------------------------------------------------------------------
def random_label_control(
    df: pd.DataFrame,
    n_shuffles: int = 5000,
    seed: int = 166,
) -> dict:
    """Compute the hit-rate distribution under randomly shuffled F5D signs.

    Shuffling breaks the F5D -> full-year link while preserving marginal distributions.
    The fraction of shuffles where the random hit-rate equals or exceeds the observed
    hit-rate is a permutation p-value.
    """
    rng = np.random.default_rng(seed)
    s = f5d_sign(df).to_numpy()
    yr_sign = np.sign(df["ret_year"].to_numpy())
    observed = float((s == yr_sign).mean())

    shuffle_rates = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(s)
        shuffle_rates[i] = (perm == yr_sign).mean()

    perm_pval = float((shuffle_rates >= observed).mean())
    return {
        "observed_hit_rate": observed,
        "shuffle_mean": float(shuffle_rates.mean()),
        "shuffle_std": float(shuffle_rates.std()),
        "perm_pval": perm_pval,
        "shuffle_rates": shuffle_rates,
    }


# ---------------------------------------------------------------------------
# Summarize — full inference in one call
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline inference on the First-Five-Days Early Warning System.

    Combines hit-rate analysis, mean contrast, and permutation test into a single
    result dict for notebooks and verify.py.
    """
    h = hit_rate_analysis(df)
    c = mean_contrast(df)
    perm = random_label_control(df)
    br = base_rate(df)

    return {
        # Sample
        "n": h["n"],
        "n_up_f5d": h["n_up_f5d"],
        "n_dn_f5d": h["n_dn_f5d"],
        # Hit-rate
        "hit_rate": h["hit_rate"],
        "hit_rate_up_f5d": h["hit_rate_up_f5d"],
        "hit_rate_dn_f5d": h["hit_rate_dn_f5d"],
        "base_rate": br,
        "pval_vs_coin": h["pval_vs_coin"],
        "pval_vs_base_rate": h["pval_vs_base_rate"],
        # Mean contrast
        "mean_yr_up_f5d_pct": c["mean_yr_up_f5d"] * 100,
        "mean_yr_dn_f5d_pct": c["mean_yr_dn_f5d"] * 100,
        "contrast_pct": c["contrast_pct"],
        "tstat_hac": c["tstat_hac"],
        "pval_welsh": c["pval_welsh"],
        # Permutation
        "perm_pval": perm["perm_pval"],
    }


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-stat (inline, no hard quantlib dependency)
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> dict:
    """Newey-West (HAC) t-statistic on the sample mean of ``r``."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return {"tstat": float("nan"), "mean": float("nan"), "se": float("nan")}
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
    }
