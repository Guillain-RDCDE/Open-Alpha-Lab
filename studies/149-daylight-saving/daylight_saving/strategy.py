"""The Daylight-Saving strategy and its honest controls — Study 149.

The claim (Kamstra, Kramer & Levi 2000): the Monday after a US DST clock change
earns abnormally *negative* returns, attributed to sleep disruption.  Sleep loss
impairs decision-making, and the hour-shorter spring night is more disruptive
than the hour-longer fall night, so the spring-forward Monday is expected to be
worse than the fall-back Monday.

We implement three complementary tests:

1. ``dst_vs_other_mondays`` — headline split: mean return and HAC t-stat on
   post-DST Mondays vs all other Mondays.  The honest baseline is ordinary
   Mondays: the claim is not merely that the market sometimes falls on a Monday,
   but that *this particular Monday* is reliably worse.  A Welch t-stat on the
   mean difference is the decisive test.

2. ``random_monday_split`` — a placebo: randomly designate Mondays as "DST" and
   measure the mean gap across many seeds.  If the true gap barely exceeds the
   shuffle distribution, the effect is indistinguishable from noise.

3. ``split_by_season`` — spring-forward vs fall-back separately.  Kamstra et al.
   predict spring > fall in magnitude; if the fall-back Monday is actually
   *positive*, the mechanism fails.

4. ``split_by_era`` — pre- vs post-2000 (pre- vs post-paper) decay check and
   modern robustness.

No look-ahead: DST labels are assigned by calendar date alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared HAC / Welch helpers
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-statistic for the mean of ``r``."""
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e  = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv  = float(e @ e) / n
    for k in range(1, lags + 1):
        w    = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch (unequal-variance) t-statistic on the mean difference a - b."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def _arm_stats(r: np.ndarray) -> dict:
    """Mean (bps), std, HAC t-stat, and count for a return array."""
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0, "mean_bps": float("nan"), "std_bps": float("nan"),
                "tstat": float("nan")}
    return {
        "n":        int(r.size),
        "mean_bps": float(r.mean() * 1e4),
        "std_bps":  float(r.std(ddof=1) * 1e4),
        "tstat":    _hac_tstat(r),
    }


# ---------------------------------------------------------------------------
# 1. DST Mondays vs other Mondays — the headline test
# ---------------------------------------------------------------------------
def dst_vs_other_mondays(frame: pd.DataFrame) -> dict:
    """Compare post-DST Monday returns against all other Mondays.

    Parameters
    ----------
    frame : DataFrame with columns ``ret``, ``is_dst_monday``, ``is_monday``.

    Returns a dict with stats for each arm and the Welch t on the difference.
    The key signal number: if post-DST Mondays are really worse, the
    ``welch_t`` should be robustly negative and |t| > 2.
    """
    mondays = frame[frame["is_monday"]].copy()
    dst  = mondays[mondays["is_dst_monday"]]["ret"].to_numpy()
    ctrl = mondays[~mondays["is_dst_monday"]]["ret"].to_numpy()

    return {
        "dst":        _arm_stats(dst),
        "other":      _arm_stats(ctrl),
        "diff_bps":   float((dst[np.isfinite(dst)].mean() -
                             ctrl[np.isfinite(ctrl)].mean()) * 1e4),
        "welch_t":    _welch_t(dst, ctrl),
    }


# ---------------------------------------------------------------------------
# 2. Placebo — randomly relabel Mondays as 'DST'
# ---------------------------------------------------------------------------
def random_monday_split(
    frame: pd.DataFrame,
    n_seeds: int = 500,
    seed: int = 149,
) -> dict:
    """Placebo: randomly assign the same number of Mondays as DST and measure the gap.

    If the observed gap is not in the tail of this null distribution, the effect
    is statistically indistinguishable from chance.  Returns the real gap, the
    null distribution mean and std, and a one-sided p-value.
    """
    mondays = frame[frame["is_monday"]].copy()
    n_dst   = int(mondays["is_dst_monday"].sum())
    r       = mondays["ret"].to_numpy(dtype=float)
    r       = r[np.isfinite(r)]
    n       = len(r)

    real_gap = float(
        (r[mondays["is_dst_monday"].values].mean() -
         r[~mondays["is_dst_monday"].values].mean()) * 1e4
    )

    rng  = np.random.default_rng(seed)
    gaps = np.empty(n_seeds)
    for s in range(n_seeds):
        idx       = np.zeros(n, dtype=bool)
        idx[:n_dst] = True
        rng.shuffle(idx)
        gaps[s]   = (r[idx].mean() - r[~idx].mean()) * 1e4

    p_value = float((gaps <= real_gap).mean())   # one-sided: real gap <= null (negative)
    return {
        "real_gap_bps":     real_gap,
        "placebo_mean_bps": float(gaps.mean()),
        "placebo_std_bps":  float(gaps.std(ddof=1)),
        "p_value_one_sided": p_value,
        "n_seeds":          n_seeds,
        "n_dst_mondays":    n_dst,
    }


# ---------------------------------------------------------------------------
# 3. Spring vs fall breakdown
# ---------------------------------------------------------------------------
def split_by_season(frame: pd.DataFrame) -> dict:
    """Compare spring-forward and fall-back Monday returns separately.

    Kamstra et al. predict spring is more negative (shorter sleep) and fall
    is less negative or near-zero.  If both are positive, the mechanism fails.
    """
    dst = frame[frame["is_dst_monday"]].copy()
    spring = dst[dst["season"] == "spring"]["ret"].to_numpy()
    fall   = dst[dst["season"] == "fall"]["ret"].to_numpy()
    other  = frame[frame["is_monday"] & ~frame["is_dst_monday"]]["ret"].to_numpy()
    return {
        "spring":      _arm_stats(spring),
        "fall":        _arm_stats(fall),
        "other":       _arm_stats(other),
        "welch_t_spring_vs_other": _welch_t(spring, other),
        "welch_t_fall_vs_other":   _welch_t(fall, other),
    }


# ---------------------------------------------------------------------------
# 4. Pre- vs post-paper era split
# ---------------------------------------------------------------------------
def split_by_era(
    frame: pd.DataFrame,
    cut: str = "2000-01-01",
) -> dict:
    """Headline gap before and after the paper's publication / data cut (KKL 2000).

    Kamstra et al. used data through ~1997; if the effect was a data artefact it
    should disappear in the post-publication era.  We split at ``cut`` (default
    2000-01-01, the year the paper was published).
    """
    cut_ts = pd.Timestamp(cut)
    pre    = frame[frame.index < cut_ts]
    post   = frame[frame.index >= cut_ts]

    def _gap(df: pd.DataFrame) -> dict:
        mon = df[df["is_monday"]]
        dst  = mon[mon["is_dst_monday"]]["ret"].to_numpy()
        ctrl = mon[~mon["is_dst_monday"]]["ret"].to_numpy()
        if dst.size < 4:
            return {"diff_bps": float("nan"), "welch_t": float("nan"),
                    "n_dst": int(dst.size), "n_other": int(ctrl.size),
                    "dst_mean_bps": float("nan"), "other_mean_bps": float("nan")}
        d, c = dst[np.isfinite(dst)], ctrl[np.isfinite(ctrl)]
        return {
            "diff_bps":       float((d.mean() - c.mean()) * 1e4),
            "welch_t":        _welch_t(d, c),
            "n_dst":          int(d.size),
            "n_other":        int(c.size),
            "dst_mean_bps":   float(d.mean() * 1e4),
            "other_mean_bps": float(c.mean() * 1e4),
        }

    return {
        "pre_publication":  _gap(pre),
        "post_publication": _gap(post),
        "cut": cut,
    }


# ---------------------------------------------------------------------------
# 5. Consolidated summary for the headline results.md table
# ---------------------------------------------------------------------------
def summarize(frame: pd.DataFrame) -> dict:
    """One-stop summary of all headline metrics for ``docs/results.md``.

    Returns a flat dict suitable for both the results file and the notebooks'
    ``R`` dictionary.
    """
    h   = dst_vs_other_mondays(frame)
    pl  = random_monday_split(frame, n_seeds=500, seed=149)
    sea = split_by_season(frame)
    era = split_by_era(frame, cut="2000-01-01")

    return {
        # DST arm
        "n_dst":           h["dst"]["n"],
        "dst_mean_bps":    h["dst"]["mean_bps"],
        "dst_tstat":       h["dst"]["tstat"],
        # Other Monday control
        "n_other":         h["other"]["n"],
        "other_mean_bps":  h["other"]["mean_bps"],
        "other_tstat":     h["other"]["tstat"],
        # Headline gap
        "diff_bps":        h["diff_bps"],
        "welch_t":         h["welch_t"],
        # Placebo
        "placebo_mean_bps": pl["placebo_mean_bps"],
        "placebo_std_bps":  pl["placebo_std_bps"],
        "p_one_sided":      pl["p_value_one_sided"],
        # Season breakdown
        "spring_mean_bps": sea["spring"]["mean_bps"],
        "spring_n":        sea["spring"]["n"],
        "spring_t_vs_other": sea["welch_t_spring_vs_other"],
        "fall_mean_bps":   sea["fall"]["mean_bps"],
        "fall_n":          sea["fall"]["n"],
        "fall_t_vs_other": sea["welch_t_fall_vs_other"],
        # Era split
        "pre_diff_bps":    era["pre_publication"]["diff_bps"],
        "pre_welch_t":     era["pre_publication"]["welch_t"],
        "post_diff_bps":   era["post_publication"]["diff_bps"],
        "post_welch_t":    era["post_publication"]["welch_t"],
    }
