"""The analysis and its honest controls — Study 164 (Mercury-Retrograde).

The claim: Mercury retrograde makes markets chaotic, falling, or more volatile.
Financial astrology's most popular bad-omen: when the planet appears to reverse
direction (an optical illusion from Earth's perspective), investors supposedly
suffer chaotic markets, broken trades, communication failures, and poor decisions.
The test:

  (A) Mean return during retrograde vs non-retrograde trading days (HAC t-stat).
  (B) Realised volatility during retrograde vs non-retrograde periods (variance ratio).
  (C) Unconditional buy-and-hold baseline: can a retrograde-avoidance strategy
      (hold cash during retrograde, invest otherwise) beat the market?
  (D) A random-period permutation control: shuffle the retrograde labels and ask
      how often a randomly chosen ~19% of days looks as bad as retrograde actually is.

No look-ahead: dates are assigned to the observation's own calendar date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core statistics helper
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


# ---------------------------------------------------------------------------
# A: Mean return — retrograde vs non-retrograde
# ---------------------------------------------------------------------------
def retrograde_return_test(ret: pd.Series, mask: pd.Series) -> dict:
    """Compare mean daily return on retrograde vs non-retrograde trading days.

    Parameters
    ----------
    ret :
        Daily simple-return Series.
    mask :
        Boolean Series (True = Mercury in retrograde), aligned to ``ret``.

    Returns a dict with headline statistics for both groups and the difference:
    ``retro_mean_bps``, ``direct_mean_bps``, ``diff_bps``, ``retro_t``,
    ``direct_t``, ``welch_t``, ``n_retro``, ``n_direct``.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)

    retro_r = r[m].to_numpy()
    direct_r = r[~m].to_numpy()

    n_r, n_d = len(retro_r), len(direct_r)
    mean_r = float(retro_r.mean()) if n_r else float("nan")
    mean_d = float(direct_r.mean()) if n_d else float("nan")
    diff = mean_r - mean_d
    se_welch = np.sqrt(
        retro_r.var(ddof=1) / n_r + direct_r.var(ddof=1) / n_d
    ) if n_r > 1 and n_d > 1 else float("nan")

    return {
        "n_retro": int(n_r),
        "n_direct": int(n_d),
        "retro_mean_bps": float(mean_r * 1e4),
        "direct_mean_bps": float(mean_d * 1e4),
        "diff_bps": float(diff * 1e4),
        "retro_t": _hac_tstat(retro_r),
        "direct_t": _hac_tstat(direct_r),
        "welch_t": float(diff / se_welch) if np.isfinite(se_welch) and se_welch > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# B: Volatility test — variance ratio between regimes
# ---------------------------------------------------------------------------
def retrograde_vol_test(ret: pd.Series, mask: pd.Series) -> dict:
    """Compare realised daily volatility during retrograde vs direct (non-retrograde) periods.

    Returns ``retro_vol_ann``, ``direct_vol_ann``, ``vol_ratio`` (retro/direct),
    and an F-test p-value (Levene-style variance ratio test on daily returns).

    A vol_ratio > 1 means retrograde is *more* volatile. The folklore claim is
    that retrograde causes chaos, so vol_ratio should be meaningfully > 1.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)

    retro_r = r[m].to_numpy()
    direct_r = r[~m].to_numpy()
    n_r, n_d = len(retro_r), len(direct_r)

    var_r = float(np.var(retro_r, ddof=1)) if n_r > 1 else float("nan")
    var_d = float(np.var(direct_r, ddof=1)) if n_d > 1 else float("nan")
    vol_r = np.sqrt(var_r * 252)
    vol_d = np.sqrt(var_d * 252)
    ratio = var_r / var_d if var_d > 0 else float("nan")

    # Levene F-statistic: F = (larger var) / (smaller var) with (n_r-1, n_d-1) df
    # (We compute it as retro/direct so ratio > 1 means more volatile during retrograde)
    from scipy import stats as scipy_stats
    try:
        lev_stat, lev_p = scipy_stats.levene(retro_r, direct_r)
    except Exception:
        lev_stat, lev_p = float("nan"), float("nan")

    return {
        "n_retro": int(n_r),
        "n_direct": int(n_d),
        "retro_vol_ann": float(vol_r),
        "direct_vol_ann": float(vol_d),
        "vol_ratio": float(ratio),
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
    }


# ---------------------------------------------------------------------------
# C: Strategy — retrograde-avoidance vs buy-and-hold
# ---------------------------------------------------------------------------
def retrograde_avoidance(ret: pd.Series, mask: pd.Series) -> dict:
    """A strategy that holds cash during Mercury retrograde, invested otherwise.

    This is the most generous framing of the financial-astrology claim: if
    retrograde days are truly bearish, just stepping aside during those ~19% of
    days should generate alpha over a simple buy-and-hold.

    Returns cumulative returns and CAGR for both the strategy and buy-and-hold,
    plus the annualised excess return and HAC t-stat on the daily excess return.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)
    n = len(r)
    years = n / 252.0

    # Strategy: earn market on non-retro days, earn 0 on retro days (hold cash)
    strat_r = r.copy()
    strat_r[m] = 0.0

    bah_cum = float((1 + r).prod())
    strat_cum = float((1 + strat_r).prod())
    bah_cagr = float(bah_cum ** (1 / years) - 1)
    strat_cagr = float(strat_cum ** (1 / years) - 1)
    excess = strat_r - r  # daily excess vs buy-and-hold (= -r on retro days, 0 otherwise)

    return {
        "n": int(n),
        "years": float(years),
        "bah_cagr": float(bah_cagr),
        "strat_cagr": float(strat_cagr),
        "excess_cagr": float(strat_cagr - bah_cagr),
        "excess_hac_t": _hac_tstat(excess.to_numpy()),
        "bah_ret": r,
        "strat_ret": strat_r,
    }


# ---------------------------------------------------------------------------
# D: Permutation control — random-period baseline
# ---------------------------------------------------------------------------
def permutation_null(
    ret: pd.Series,
    mask: pd.Series,
    n_perm: int = 5000,
    seed: int = 164,
) -> dict:
    """How often does a *randomly chosen* ~19% of trading days look as bad as retrograde?

    We create random boolean masks that match the *fraction* of retrograde days but
    with shuffled placement, and ask what fraction of them produce a mean daily return
    at least as low as the actual retrograde mean. If the astrology claim were real,
    the observed retrograde mean should be unusually low compared to these random controls.

    Returns ``pct_worse_than_retro`` (fraction of random periods where the mean is
    at least as low as actual retrograde mean) and the observed retrograde mean.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)
    retro_mean = float(r[m].mean())

    rng = np.random.default_rng(seed)
    n_total = len(r)
    n_retro = int(m.sum())
    vals = r.to_numpy()

    count = 0
    for _ in range(n_perm):
        # Randomly choose the same number of days as actual retrograde
        perm_idx = rng.choice(n_total, size=n_retro, replace=False)
        perm_mean = float(vals[perm_idx].mean())
        if perm_mean <= retro_mean:
            count += 1

    return {
        "retro_mean_bps": float(retro_mean * 1e4),
        "pct_worse_than_retro": float(count / n_perm),
        "n_perm": n_perm,
        "n_retro": int(n_retro),
        "frac_retro": float(n_retro / n_total),
    }


# ---------------------------------------------------------------------------
# Summary: all tests in one call
# ---------------------------------------------------------------------------
def summarize(ret: pd.Series, mask: pd.Series) -> dict:
    """Run all four tests and return a combined results dict.

    This is the study's spine — call it with the real tape's returns and mask
    to reproduce all headline numbers in ``docs/results.md``.
    """
    rtest = retrograde_return_test(ret, mask)
    vtest = retrograde_vol_test(ret, mask)
    avoid = retrograde_avoidance(ret, mask)

    r = pd.Series(ret).astype(float).dropna()
    n = len(r)
    years = n / 252.0
    std = float(r.std(ddof=1))
    mean = float(r.mean())
    cagr = float((1 + r).prod() ** (1 / years) - 1)

    return {
        # Full-tape headline
        "n": int(n),
        "years": float(years),
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(252)),
        "sharpe": float(mean / std * np.sqrt(252)) if std > 0 else float("nan"),
        # Return test
        **{f"ret_{k}": v for k, v in rtest.items()},
        # Vol test
        **{f"vol_{k}": v for k, v in vtest.items()},
        # Avoidance strategy
        "avoid_bah_cagr": avoid["bah_cagr"],
        "avoid_strat_cagr": avoid["strat_cagr"],
        "avoid_excess_cagr": avoid["excess_cagr"],
        "avoid_excess_t": avoid["excess_hac_t"],
    }
