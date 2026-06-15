"""The analysis and its honest controls -- Study 165 (Chinese-Zodiac).

Two concrete sub-claims underlie the Chinese zodiac market folklore:

  (A) PRE-CNY RALLY: the 10 trading days before Chinese New Year show
      significantly higher returns than matched random windows.
  (B) ZODIAC-YEAR EFFECT: certain animals (especially Dragon) produce
      better annual returns on Chinese/Asian equity markets.

This module tests both with clean inference:

  - :func:`pre_cny_windows` -- extract the pre-CNY 10-day windows for each year.
  - :func:`pre_cny_stats` -- mean, vol, and HAC t-stat for the pre-CNY windows
    vs a random-day baseline.
  - :func:`zodiac_year_stats` -- per-animal mean annual return, with a
    Bonferroni-corrected 12-way multiple-comparisons reckoning.
  - :func:`dragon_vs_rest`` -- pairwise Welch t: Dragon years vs the other 11 animals.
  - :func:`permutation_null`` -- how often does a randomly chosen animal look
    as good as Dragon, by chance alone?

No look-ahead: calendar labels are assigned from the hardcoded CNY table in data.py,
which is fully determined before any market data is observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .data import CNY_DF, ANIMALS, PRE_CNY_WINDOW, _dragon_days_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hac_tstat(arr: np.ndarray) -> float:
    """Newey-West HAC t-statistic for the sample mean."""
    n = arr.size
    if n < 6:
        return float("nan")
    mu = arr.mean()
    e = arr - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _get_trading_days_before(date: pd.Timestamp, ret: pd.Series, n: int) -> pd.Series:
    """Return the ``n`` trading days strictly before ``date`` from ``ret``'s index."""
    idx = ret.index[ret.index < date]
    if len(idx) < n:
        return pd.Series(dtype=float)
    return ret.loc[idx[-n:]]


def _annual_return(ret: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Compound annual return for the period [start, end) from daily simple returns."""
    window = ret[(ret.index >= start) & (ret.index < end)]
    if len(window) < 5:
        return float("nan")
    return float((1 + window).prod() - 1)


# ---------------------------------------------------------------------------
# (A) Pre-CNY rally
# ---------------------------------------------------------------------------
def pre_cny_windows(ret: pd.Series, window: int = PRE_CNY_WINDOW) -> pd.DataFrame:
    """Extract the pre-CNY trading-day windows for each CNY year in the data.

    Returns a DataFrame with one row per CNY year: the year, the CNY date,
    the zodiac animal, the window compound return, and the number of trading
    days captured.
    """
    rows = []
    for yr, row in CNY_DF.iterrows():
        cny_date = row["cny_date"]
        window_ret = _get_trading_days_before(cny_date, ret, window)
        if window_ret.empty:
            continue
        compound = float((1 + window_ret).prod() - 1)
        rows.append({
            "year": int(yr),
            "cny_date": cny_date,
            "animal": row["animal"],
            "pre_cny_ret": compound,
            "n_days": len(window_ret),
        })
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def pre_cny_stats(ret: pd.Series, window: int = PRE_CNY_WINDOW, n_random: int = 500,
                  seed: int = 165) -> dict:
    """Mean pre-CNY window return vs a random-window baseline.

    Draws ``n_random`` random windows of the same length from the return series
    (excluding the actual pre-CNY periods) and computes the mean compound return
    for each. Returns observed mean, baseline mean, difference, and HAC t-stat
    on the per-event pre-CNY returns.

    The honest test: if the pre-CNY rally is real, the observed mean should exceed
    the random-window mean AND the HAC t-stat on the event returns should clear 2.
    """
    df = pre_cny_windows(ret, window)
    if df.empty:
        return {"n_events": 0}

    observed = df["pre_cny_ret"].dropna().to_numpy()
    n_obs = observed.size
    if n_obs < 3:
        return {"n_events": int(n_obs), "mean_ret_pct": float("nan"),
                "hac_t": float("nan"), "baseline_mean_pct": float("nan")}

    # Random baseline: draw random windows of the same length
    rng = np.random.default_rng(seed)
    all_dates = ret.index
    n_total = len(all_dates)
    baseline_rets = []
    for _ in range(n_random):
        start_i = rng.integers(0, n_total - window)
        w_ret = ret.iloc[start_i: start_i + window]
        baseline_rets.append(float((1 + w_ret).prod() - 1))

    mean_obs = float(observed.mean())
    mean_base = float(np.mean(baseline_rets))
    hac_t = _hac_tstat(observed)

    return {
        "n_events": int(n_obs),
        "mean_ret_pct": mean_obs * 100,
        "baseline_mean_pct": mean_base * 100,
        "diff_pct": (mean_obs - mean_base) * 100,
        "hac_t": hac_t,
        "pct_above_baseline": float((observed > mean_base).mean() * 100),
    }


# ---------------------------------------------------------------------------
# (B) Zodiac-year effect
# ---------------------------------------------------------------------------
def zodiac_year_stats(ret: pd.Series) -> pd.DataFrame:
    """Annual returns by zodiac animal, with Bonferroni-corrected 12-way inference.

    Each lunar year's return is the compound return from CNY day to the day before
    the next CNY. We compute per-animal mean annual return, sample count (n_years),
    raw Welch t vs the 11-other-animal pooled returns, and Bonferroni-corrected
    p-values (multiplied by 12).

    The effective n per animal over 1994-2026 is ~3. No t-stat on n=3 is meaningful,
    no matter how large it looks -- we say so loudly.
    """
    rows = []
    for i, (yr, row) in enumerate(CNY_DF.iterrows()):
        cny_start = row["cny_date"]
        next_yr = yr + 1
        if next_yr not in CNY_DF.index:
            continue
        cny_end = CNY_DF.loc[next_yr, "cny_date"]
        ann_ret = _annual_return(ret, cny_start, cny_end)
        rows.append({
            "year": int(yr),
            "animal": row["animal"],
            "ann_ret": ann_ret,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    # Per-animal aggregation
    out = []
    for animal in ANIMALS:
        sub = df[df["animal"] == animal]["ann_ret"].dropna()
        rest = df[df["animal"] != animal]["ann_ret"].dropna()
        n = len(sub)
        if n == 0:
            out.append({"animal": animal, "n_years": 0, "mean_ret_pct": float("nan"),
                        "raw_t": float("nan"), "bonferroni_p": float("nan")})
            continue
        mean_ret = float(sub.mean())
        if n >= 2 and len(rest) >= 2:
            tstat, pval = scipy_stats.ttest_ind(sub, rest, equal_var=False)
        else:
            tstat, pval = float("nan"), float("nan")
        bonf_p = min(float(pval) * 12, 1.0) if not np.isnan(pval) else float("nan")
        out.append({
            "animal": animal,
            "n_years": int(n),
            "mean_ret_pct": mean_ret * 100,
            "raw_t": float(tstat),
            "bonferroni_p": bonf_p,
        })
    return pd.DataFrame(out).set_index("animal")


def dragon_vs_rest(ret: pd.Series) -> dict:
    """Welch t-test: Dragon years vs the other 11 animals combined.

    This is the primary falsifiable test for the 'Dragon is lucky' claim.
    n_dragon is only ~3 over the FXI period -- a large t-stat here means nothing.
    """
    df = pd.DataFrame()
    rows = []
    for i, (yr, row) in enumerate(CNY_DF.iterrows()):
        next_yr = yr + 1
        if next_yr not in CNY_DF.index:
            continue
        cny_end = CNY_DF.loc[next_yr, "cny_date"]
        ann_ret = _annual_return(ret, row["cny_date"], cny_end)
        rows.append({"animal": row["animal"], "ann_ret": ann_ret})

    df = pd.DataFrame(rows).dropna(subset=["ann_ret"])
    if df.empty:
        return {}

    dragon_r = df[df["animal"] == "Dragon"]["ann_ret"].to_numpy()
    rest_r = df[df["animal"] != "Dragon"]["ann_ret"].to_numpy()
    n_d, n_r = len(dragon_r), len(rest_r)

    if n_d < 2 or n_r < 2:
        tstat, pval = float("nan"), float("nan")
    else:
        tstat, pval = scipy_stats.ttest_ind(dragon_r, rest_r, equal_var=False)

    return {
        "dragon_mean_pct": float(dragon_r.mean() * 100) if n_d > 0 else float("nan"),
        "rest_mean_pct": float(rest_r.mean() * 100) if n_r > 0 else float("nan"),
        "diff_pct": float((dragon_r.mean() - rest_r.mean()) * 100)
                    if n_d > 0 and n_r > 0 else float("nan"),
        "welch_t": float(tstat),
        "bonferroni_p": min(float(pval) * 12, 1.0) if not np.isnan(pval) else float("nan"),
        "n_dragon": int(n_d),
        "n_rest": int(n_r),
    }


def permutation_null(ret: pd.Series, n_perm: int = 2000, seed: int = 165) -> dict:
    """How often does a randomly chosen animal look as good as Dragon?

    We shuffle the animal labels (while keeping the per-year returns intact) and
    ask what fraction of randomly chosen 'dragon' labels produce a mean as high
    as the observed Dragon mean. A genuinely lucky animal should score a high
    percentile in the real distribution.
    """
    rows = []
    for yr, row in CNY_DF.iterrows():
        next_yr = yr + 1
        if next_yr not in CNY_DF.index:
            continue
        cny_end = CNY_DF.loc[next_yr, "cny_date"]
        ann_ret = _annual_return(ret, row["cny_date"], cny_end)
        rows.append({"animal": row["animal"], "ann_ret": ann_ret})

    df = pd.DataFrame(rows).dropna(subset=["ann_ret"])
    if df.empty or (df["animal"] == "Dragon").sum() == 0:
        return {"dragon_mean_pct": float("nan"), "pct_as_good": float("nan")}

    dragon_rets = df[df["animal"] == "Dragon"]["ann_ret"].to_numpy()
    observed_mean = float(dragon_rets.mean())
    n_dragon = len(dragon_rets)
    all_rets = df["ann_ret"].to_numpy()

    rng = np.random.default_rng(seed)
    count_as_good = 0
    for _ in range(n_perm):
        perm_dragon = rng.choice(all_rets, size=n_dragon, replace=False)
        if perm_dragon.mean() >= observed_mean:
            count_as_good += 1

    return {
        "dragon_mean_pct": observed_mean * 100,
        "pct_as_good": float(count_as_good / n_perm),
        "n_dragon": int(n_dragon),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Annualised summary helper
# ---------------------------------------------------------------------------
def period_summary(ret: pd.Series) -> dict:
    """Headline stats for the full series: CAGR, annualised vol, Sharpe."""
    r = pd.Series(ret).astype(float).dropna()
    mean = r.mean()
    std = r.std(ddof=1)
    cagr = (1.0 + r).prod() ** (252.0 / len(r)) - 1.0
    return {
        "n": int(len(r)),
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(252)),
        "sharpe": float(mean / std * np.sqrt(252)) if std > 0 else float("nan"),
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
    }
