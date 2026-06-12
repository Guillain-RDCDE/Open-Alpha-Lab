"""The strategy and its honest controls — Study 80 (Cold-Open).

The January Barometer (Hirsch 1972 / Stock Traders Almanac): *"As goes January, so goes
the year."*  The sign of the S&P 500's January return is said to predict whether the
full year (or at least the February-through-December remainder) will be positive.

We measure this claim two ways:

1. **Hit-rate vs base rate**: does ``sign(jan_ret) == sign(fbd_ret)`` more often than
   would be expected from the unconditional probability that any February-to-December
   period is positive?  The equity market drifts up most of the time — a large base-rate
   for positive rest-of-year — so a raw "75% of up-January years have up rest-of-year"
   headline is almost guaranteed even with no signal.  We compare against a binomial
   test.
2. **Mean rest-of-year return by January sign**: do up-January years have meaningfully
   higher Feb-Dec returns than down-January years, net of that equity drift?  We use a
   HAC t-stat (Newey-West) on the per-year Feb-Dec returns, split by January sign, and
   on the *contrast* (up-Jan mean minus down-Jan mean) as the cleanest test.

The fair null: what would a random-label (shuffled January sign) control produce?  We
also sweep a simple ``sign(jan_ret) * invested_in_rest_of_year`` strategy vs buy-and-
hold, net of an annual re-entry cost.

No look-ahead: January sign is known on the last trading day of January; the position
is entered on 1 February.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def jan_sign(df: pd.DataFrame) -> pd.Series:
    """Return the sign of January (+1 / -1 / 0) for each year.  0 is treated as
    a missing signal and is excluded from hit-rate calculations."""
    return np.sign(df["jan_ret"]).rename("jan_sign")


def base_rate(df: pd.DataFrame) -> float:
    """Unconditional probability that Feb-Dec is positive — the base rate any
    calendar-effect claim must beat to be interesting."""
    return float((df["fbd_ret"] > 0).mean())


# ---------------------------------------------------------------------------
# Hit-rate & binomial test
# ---------------------------------------------------------------------------
def hit_rate_analysis(df: pd.DataFrame) -> dict:
    """Full hit-rate breakdown and a binomial significance test.

    'Hit' = sign(jan_ret) == sign(fbd_ret) (both positive or both negative).
    We report:
    - overall hit-rate and a binomial p-value (H0: hit-rate == 0.5 — a coin).
    - hit-rate conditional on an *up* January (vs the unconditional base rate).
    - hit-rate conditional on a *down* January.
    - effective n (years with a non-zero January return, which is all of them).
    """
    from scipy.stats import binomtest

    s = jan_sign(df)
    valid = df.loc[s != 0]
    sv = s[s != 0]
    hits = (np.sign(valid["fbd_ret"]) == sv).astype(int)

    n_total = int(len(hits))
    n_hits = int(hits.sum())
    hit_rate = float(hits.mean())

    up = sv == 1
    dn = sv == -1
    hr_up = float(hits[up].mean()) if up.any() else float("nan")
    hr_dn = float(hits[dn].mean()) if dn.any() else float("nan")
    n_up = int(up.sum())
    n_dn = int(dn.sum())

    # Binomial test: H0 = p = 0.5 (a coin flip).
    binom_result = binomtest(n_hits, n_total, p=0.5, alternative="greater")
    pval_coin = float(binom_result.pvalue)

    # Binomial test vs base rate (the tougher bar — equity drift makes up-years common).
    br = base_rate(df)
    binom_base = binomtest(n_hits, n_total, p=br, alternative="greater")
    pval_base = float(binom_base.pvalue)

    return {
        "n": n_total,
        "n_up_jan": n_up,
        "n_dn_jan": n_dn,
        "n_hits": n_hits,
        "hit_rate": hit_rate,
        "hit_rate_up_jan": hr_up,
        "hit_rate_dn_jan": hr_dn,
        "base_rate_fbd_pos": br,
        "pval_vs_coin": pval_coin,
        "pval_vs_base_rate": pval_base,
    }


# ---------------------------------------------------------------------------
# Mean contrast & HAC t-stat
# ---------------------------------------------------------------------------
def mean_contrast(df: pd.DataFrame) -> dict:
    """Mean Feb-Dec return for up-Jan vs down-Jan years, and their contrast.

    The contrast = mean(fbd | jan > 0) - mean(fbd | jan < 0) is the primary test
    statistic.  A HAC t-stat (Newey-West) is computed on the per-year contrast
    residuals (demeaned within each group and then combined with signs).  We also
    report the simple per-group means and a Welsh t-test p-value for the contrast.
    """
    from scipy.stats import ttest_ind

    s = jan_sign(df)
    up_mask = s == 1
    dn_mask = s == -1

    fbd = df["fbd_ret"]
    up_rets = fbd[up_mask].to_numpy(dtype=float)
    dn_rets = fbd[dn_mask].to_numpy(dtype=float)

    mean_up = float(up_rets.mean()) if len(up_rets) else float("nan")
    mean_dn = float(dn_rets.mean()) if len(dn_rets) else float("nan")
    contrast = mean_up - mean_dn

    # Welsh t-test on the group difference.
    if len(up_rets) > 1 and len(dn_rets) > 1:
        _, pval_welsh = ttest_ind(up_rets, dn_rets, equal_var=False, alternative="greater")
    else:
        pval_welsh = float("nan")

    # HAC t-stat on the *signed* per-year returns: assign +1 to up-Jan, -1 to dn-Jan,
    # multiply by fbd_ret — this centres the statistic on the contrast.
    s_clean = s[(up_mask | dn_mask)]
    r_clean = fbd[(up_mask | dn_mask)]
    signed = (s_clean * r_clean).to_numpy(dtype=float)
    hac = _hac_tstat(signed)

    return {
        "mean_fbd_up_jan": mean_up,
        "mean_fbd_dn_jan": mean_dn,
        "contrast_pct": contrast * 100.0,
        "tstat_hac": hac["tstat"],
        "pval_welsh": pval_welsh,
        "n_up": int(up_mask.sum()),
        "n_dn": int(dn_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Barometer strategy vs buy-and-hold
# ---------------------------------------------------------------------------
def barometer_strategy(
    df: pd.DataFrame,
    cost_bps: float = 10.0,
) -> dict:
    """Simulate the Jan-Barometer 'tactical call': hold Feb-Dec only after an
    up January; sit in cash (0 % return) after a down January.  Compare to
    buy-and-hold of Feb-Dec every year.

    ``cost_bps`` is the one-way cost (re-entry after a down January, i.e. when
    we come back into the market).  A round-trip is 2 × cost_bps.

    Returns per-year series plus summary statistics.
    """
    s = jan_sign(df)
    fbd = df["fbd_ret"]

    # Strategy: long after up-Jan, flat after down-Jan.
    # Each year the strategy enters on 1 Feb regardless (long or flat), so the
    # one-way cost is paid whenever we change posture (up→flat or flat→up).
    position = (s == 1).astype(float)  # 1 if up-Jan, 0 otherwise
    strat_ret = position * fbd
    # Cost: charged whenever posture flips (once per year where we change).
    prev_pos = position.shift(1)
    changed = (position != prev_pos) & prev_pos.notna()
    cost_series = changed.astype(float) * (cost_bps * 1e-4)
    strat_net = strat_ret - cost_series

    bah = fbd  # buy-and-hold of Feb-Dec every year

    # Summary.
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
# Random-label control (shuffled January signs)
# ---------------------------------------------------------------------------
def random_label_control(
    df: pd.DataFrame,
    n_shuffles: int = 5000,
    seed: int = 80,
) -> dict:
    """Compute the hit-rate distribution under randomly shuffled January signs.

    Shuffling breaks the Jan→Feb-Dec link while preserving the marginal
    distributions.  The fraction of shuffles where the random hit-rate equals or
    exceeds the observed hit-rate is a permutation p-value.
    """
    rng = np.random.default_rng(seed)
    s = jan_sign(df).to_numpy()
    fbd_sign = np.sign(df["fbd_ret"].to_numpy())
    observed = float((s == fbd_sign).mean())

    shuffle_rates = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(s)
        shuffle_rates[i] = (perm == fbd_sign).mean()

    perm_pval = float((shuffle_rates >= observed).mean())
    return {
        "observed_hit_rate": observed,
        "shuffle_mean": float(shuffle_rates.mean()),
        "shuffle_std": float(shuffle_rates.std()),
        "perm_pval": perm_pval,
        "shuffle_rates": shuffle_rates,
    }


# ---------------------------------------------------------------------------
# Summarize — overall inference
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline inference on the January Barometer.

    Combines the hit-rate analysis, mean contrast, and permutation test into
    a single result dict for notebooks and verify.py.
    """
    h = hit_rate_analysis(df)
    c = mean_contrast(df)
    perm = random_label_control(df)
    br = base_rate(df)

    return {
        # Sample size
        "n": h["n"],
        "n_up_jan": h["n_up_jan"],
        "n_dn_jan": h["n_dn_jan"],
        # Hit-rate
        "hit_rate": h["hit_rate"],
        "hit_rate_up_jan": h["hit_rate_up_jan"],
        "hit_rate_dn_jan": h["hit_rate_dn_jan"],
        "base_rate": br,
        "pval_vs_coin": h["pval_vs_coin"],
        "pval_vs_base_rate": h["pval_vs_base_rate"],
        # Mean contrast
        "mean_fbd_up_jan_pct": c["mean_fbd_up_jan"] * 100,
        "mean_fbd_dn_jan_pct": c["mean_fbd_dn_jan"] * 100,
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
