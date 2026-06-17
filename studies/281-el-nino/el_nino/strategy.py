"""Strategy and analysis layer for Study 281 (El-Nino).

The folklore: El Nino (warm Pacific) and La Nina (cold Pacific) move markets --
crop failures and droughts spike soft-commodity and energy prices, and some
commentators extend the claim to equities ("El Nino summers are bullish"). We
implement the ENSO phase as a year-by-year signal and pin it against the honest
benchmarks:

  - The **unconditional mean / base rate**: equities drift up ~7%/yr and rise in
    ~70% of years regardless of the ocean. Any "El-Nino-is-bullish" signal gets
    that drift for free -- it must beat the *unconditional* mean, not zero.
  - A **Welch t-test** (and a HAC/Newey-West t on the long-short return series)
    on El-Nino-year vs La-Nina-year returns: is the gap distinguishable from
    noise with n ~ 25 per phase?
  - A **permutation test**: shuffle the phase labels 10,000 times and see where
    the real El-Nino-minus-La-Nina mean gap lands.
  - A **one-way ANOVA** across the three phases (El Nino / La Nina / Neutral),
    the natural omnibus test for "does phase matter at all".
  - Explicit **multiple-comparisons** accounting: equity vs oil, lag-1 vs lag-0,
    El-Nino-bull vs La-Nina-bull -- a small forest of related hypotheses.

The tiny-n reckoning: 75 ENSO years since 1950, ~25 El Nino / ~25 La Nina /
~25 Neutral. With ~17% equity volatility you need a ~7%/yr structural gap to see
|t| = 2 -- far larger than any plausible weather premium. The same applies to
oil with its ~30%+ vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------
def signal_warm_cold(df: pd.DataFrame) -> pd.Series:
    """Return +1 in El Nino years, -1 in La Nina years, 0 in Neutral years.

    This is the primary ENSO signal: long the asset after a warm (El Nino) winter,
    short after a cold (La Nina) winter, flat in Neutral years.
    """
    m = {"El Nino": 1, "La Nina": -1, "Neutral": 0}
    return df["phase"].map(m).rename("signal_warm_cold").astype(int)


def signal_oni_sign(df: pd.DataFrame) -> pd.Series:
    """Return sign(ONI): +1 when ONI > 0, -1 when ONI < 0, 0 when exactly 0.

    A continuous-threshold variant that ignores the (arbitrary) +/-0.5 cutoff and
    just uses the sign of the anomaly -- another degree of freedom.
    """
    return np.sign(df["oni_ndj"]).astype(int).rename("signal_oni_sign")


# ---------------------------------------------------------------------------
# HAC / Newey-West t-stat on a return series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int = 1) -> float:
    """Newey-West HAC t-stat for the mean of ``x`` against zero.

    With annual data serial correlation is small, so ``lags=1`` is a conservative
    default. Returns the t-stat of the sample mean / HAC standard error.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    s = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = (e[k:] @ e[:-k]) / n
        s += 2.0 * w * cov
    se = np.sqrt(max(s, 1e-18) / n)
    if se == 0:
        return float("nan")
    return float(mu / se)


# ---------------------------------------------------------------------------
# Phase statistics and inference
# ---------------------------------------------------------------------------
def phase_stats(
    df: pd.DataFrame,
    ret_col: str = "asset_return",
    n_permutations: int = 10_000,
    seed: int = 281,
) -> dict:
    """Compute per-phase means, the El-Nino-minus-La-Nina gap, and inference.

    Reports:
      - ``mean_elnino``, ``mean_lanina``, ``mean_neutral``, ``mean_all``
      - ``n_elnino``, ``n_lanina``, ``n_neutral``, ``n_total``
      - ``gap`` : mean(El Nino) - mean(La Nina)
      - ``uncond_up_rate`` : unconditional fraction of up-years
      - ``welch_t``, ``welch_p`` : Welch t-test El Nino vs La Nina returns
      - ``hac_t`` : Newey-West t on the long-short (El Nino - La Nina) signal
        return series (each year contributes +ret in El Nino, -ret in La Nina)
      - ``anova_f``, ``anova_p`` : one-way ANOVA across the three phases
      - ``perm_p`` : permutation p-value for ``|gap|`` (two-sided)
    """
    rets = df[ret_col].to_numpy(dtype=float)
    phase = df["phase"].to_numpy()
    n = len(rets)

    el = rets[phase == "El Nino"]
    la = rets[phase == "La Nina"]
    ne = rets[phase == "Neutral"]

    mean_el = float(el.mean()) if len(el) else float("nan")
    mean_la = float(la.mean()) if len(la) else float("nan")
    mean_ne = float(ne.mean()) if len(ne) else float("nan")
    mean_all = float(rets.mean())
    gap = mean_el - mean_la

    uncond_up_rate = float((rets > 0).mean())

    if len(el) >= 2 and len(la) >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(el, la, equal_var=False)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    # Long-short signal return series: +ret in El Nino, -ret in La Nina, 0 else.
    sig = signal_warm_cold(df).to_numpy()
    ls_ret = sig * rets
    hac_t = hac_tstat(ls_ret[sig != 0], lags=1)

    groups = [g for g in (el, la, ne) if len(g) >= 2]
    if len(groups) >= 2:
        anova_f, anova_p = scipy_stats.f_oneway(*groups)
    else:
        anova_f, anova_p = float("nan"), float("nan")

    # Permutation test on |gap|: shuffle phase labels, recompute gap.
    rng = np.random.default_rng(seed)
    obs = abs(gap)
    perm = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        pp = rng.permutation(phase)
        m_el = rets[pp == "El Nino"]
        m_la = rets[pp == "La Nina"]
        if len(m_el) and len(m_la):
            perm[i] = abs(m_el.mean() - m_la.mean())
        else:
            perm[i] = 0.0
    perm_p = float((perm >= obs).mean())

    return {
        "n_total": int(n),
        "n_elnino": int(len(el)),
        "n_lanina": int(len(la)),
        "n_neutral": int(len(ne)),
        "mean_elnino": round(mean_el, 4),
        "mean_lanina": round(mean_la, 4),
        "mean_neutral": round(mean_ne, 4),
        "mean_all": round(mean_all, 4),
        "gap": round(float(gap), 4),
        "uncond_up_rate": round(uncond_up_rate, 4),
        "welch_t": round(float(welch_t), 3),
        "welch_p": round(float(welch_p), 4),
        "hac_t": round(float(hac_t), 3),
        "anova_f": round(float(anova_f), 3),
        "anova_p": round(float(anova_p), 4),
        "perm_p": round(float(perm_p), 4),
        "perm_gaps": perm,  # full permutation distribution, for plots
    }


# ---------------------------------------------------------------------------
# Backtest the long-short ENSO strategy (with execution lag + costs)
# ---------------------------------------------------------------------------
def backtest(
    df: pd.DataFrame,
    ret_col: str = "asset_return",
    cost_bps_oneway: float = 10.0,
    short_borrow_bps: float = 50.0,
) -> dict:
    """Net long-short ENSO backtest: long after El Nino, short after La Nina.

    Honesty knobs:
      - one-way transaction cost ``cost_bps_oneway`` charged on every position
        change (entering and exiting a year-long position),
      - an annual ``short_borrow_bps`` borrow fee subtracted in short (La Nina) years.

    The phase is already lagged one year in ``fetch_real`` (no look-ahead). Returns
    gross/net annualised figures and the buy-and-hold benchmark for comparison.
    """
    rets = df[ret_col].to_numpy(dtype=float)
    sig = signal_warm_cold(df).to_numpy(dtype=float)
    n = len(rets)

    gross = sig * rets
    # Position turnover: change in |signal| each year, plus the borrow drag on shorts.
    pos_change = np.abs(np.diff(np.concatenate([[0.0], sig])))
    cost = pos_change * cost_bps_oneway * 1e-4
    borrow = (sig < 0).astype(float) * short_borrow_bps * 1e-4
    net = gross - cost - borrow

    def _ann(x):
        return float((1.0 + x).prod() ** (1.0 / n) - 1.0) if n else float("nan")

    bah = _ann(rets)
    return {
        "n_years": int(n),
        "gross_ann": round(_ann(gross), 4),
        "net_ann": round(_ann(net), 4),
        "buy_hold_ann": round(bah, 4),
        "turnover_per_yr": round(float(pos_change.mean()), 3),
        "gross_minus_bah": round(_ann(gross) - bah, 4),
        "net_minus_bah": round(_ann(net) - bah, 4),
    }


# ---------------------------------------------------------------------------
# Multiple-comparisons summary
# ---------------------------------------------------------------------------
def multiple_comparisons_summary(
    frames: dict[str, pd.DataFrame],
    n_permutations: int = 10_000,
    seed: int = 281,
) -> pd.DataFrame:
    """Run phase_stats across several (label -> frame) tapes and Bonferroni-correct.

    ``frames`` maps a descriptive label (e.g. "equity lag1", "oil lag1") to a merged
    DataFrame. Returns one row per tape with the gap, Welch t, HAC t, ANOVA p,
    permutation p, and the Bonferroni-corrected permutation p.
    """
    rows = []
    n_tests = max(1, len(frames))
    for label, frame in frames.items():
        st = phase_stats(frame, n_permutations=n_permutations, seed=seed)
        rows.append({
            "tape": label,
            "n": st["n_total"],
            "n_elnino": st["n_elnino"],
            "n_lanina": st["n_lanina"],
            "gap_pct": round(st["gap"] * 100, 2),
            "welch_t": st["welch_t"],
            "hac_t": st["hac_t"],
            "anova_p": st["anova_p"],
            "perm_p": st["perm_p"],
            "bonferroni_perm_p": min(1.0, st["perm_p"] * n_tests),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarize -- compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 281) -> dict:
    """One-stop summary dict for a merged DataFrame (mirrors R in build_notebooks)."""
    st = phase_stats(df, n_permutations=n_permutations, seed=seed)
    bt = backtest(df)
    return {
        "n": st["n_total"],
        "n_elnino": st["n_elnino"],
        "n_lanina": st["n_lanina"],
        "n_neutral": st["n_neutral"],
        "uncond_up_rate_pct": round(st["uncond_up_rate"] * 100, 1),
        "mean_elnino_pct": round(st["mean_elnino"] * 100, 1),
        "mean_lanina_pct": round(st["mean_lanina"] * 100, 1),
        "mean_neutral_pct": round(st["mean_neutral"] * 100, 1),
        "mean_all_pct": round(st["mean_all"] * 100, 1),
        "gap_pct": round(st["gap"] * 100, 1),
        "welch_t": st["welch_t"],
        "welch_p": st["welch_p"],
        "hac_t": st["hac_t"],
        "anova_f": st["anova_f"],
        "anova_p": st["anova_p"],
        "perm_p": st["perm_p"],
        "gross_ann_pct": round(bt["gross_ann"] * 100, 1),
        "net_ann_pct": round(bt["net_ann"] * 100, 1),
        "buy_hold_ann_pct": round(bt["buy_hold_ann"] * 100, 1),
        "turnover_per_yr": bt["turnover_per_yr"],
    }
