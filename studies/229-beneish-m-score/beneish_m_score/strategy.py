"""Strategy and honest controls — Study 229 (Beneish M-score).

The signal: the Beneish (1999) M-score is an 8-variable composite designed to
detect earnings manipulation.  The equity strategy is:

    Short high-M firms  (likely manipulators, M > −1.78)
    Long  low-M firms   (unlikely manipulators)

Theoretical grounding: if manipulation is exposed and corrected, manipulator
stocks crash; low-M 'clean' firms avoid the crash and earn normal returns.
The literature (Beneish 1999, Beneish et al. 2013) shows in-sample power but
the out-of-sample equity premium is contested — especially on a large-cap
survivorship-biased panel where genuine manipulators rarely survive to appear.

This module computes:

- ``tertile_hedge`` — annual long-low-M / short-high-M return (and breakdowns
  by bucket). Note the sign: low-M is the *long* leg, so hedge = ret_lo − ret_hi
  (reverse of the Altman study, where high-score was the long leg).
- ``summarize`` — HAC (Newey-West) t-stat on the annual hedge series.
- ``mscore_zscore`` — the M-score composite from eight pre-computed component
  DataFrames (for the strategy-unit tests, which use known toy values).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# M-score formula (for unit tests with toy inputs)
# ---------------------------------------------------------------------------

def mscore_from_components(
    DSRI:  pd.DataFrame,
    GMI:   pd.DataFrame,
    AQI:   pd.DataFrame,
    SGI:   pd.DataFrame,
    DEPI:  pd.DataFrame,
    SGAI:  pd.DataFrame,
    TATA:  pd.DataFrame,
    LVGI:  pd.DataFrame,
    intercept: float = -4.84,
    coef: dict | None = None,
) -> pd.DataFrame:
    """Beneish M-score from the eight pre-computed component DataFrames.

    M = intercept + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
                  + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI

    All inputs must be aligned DataFrames (same index × columns).
    NaN propagates: if any component is NaN, M is NaN for that cell.
    """
    if coef is None:
        coef = dict(
            DSRI=0.920, GMI=0.528, AQI=0.404, SGI=0.892,
            DEPI=0.115, SGAI=-0.172, TATA=4.679, LVGI=-0.327,
        )
    return (
        intercept
        + coef["DSRI"]  * DSRI
        + coef["GMI"]   * GMI
        + coef["AQI"]   * AQI
        + coef["SGI"]   * SGI
        + coef["DEPI"]  * DEPI
        + coef["SGAI"]  * SGAI
        + coef["TATA"]  * TATA
        + coef["LVGI"]  * LVGI
    )


# ---------------------------------------------------------------------------
# Annual bucket sort
# ---------------------------------------------------------------------------

def tertile_hedge(
    m_scores: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 1 / 3,
    min_firms: int = 20,
) -> pd.DataFrame:
    """Annual long-low-M / short-high-M hedge vs market.

    For each year in ``m_scores.index``:

    1. Take all firms with a valid M and a valid next-year return.
    2. Split into bottom-q (clean, low-M) and top-q (manipulators, high-M).
    3. Record equal-weighted mean returns for each bucket and the market.

    The hedge = ret_lo − ret_hi (long-clean / short-manipulator).

    Returns a DataFrame indexed by year with columns:
    ``n, n_lo, n_hi, ret_lo, ret_mid, ret_hi, ret_mkt, hedge``.
    """
    rows = []
    for yr in m_scores.index:
        m_yr = m_scores.loc[yr].dropna()
        r_yr = fwd_ret.loc[yr].dropna()
        both = m_yr.index.intersection(r_yr.index)
        m_yr = m_yr.loc[both]
        r_yr = r_yr.loc[both]
        if len(m_yr) < min_firms:
            continue
        cut_lo = m_yr.quantile(q)
        cut_hi = m_yr.quantile(1.0 - q)
        lo  = m_yr <= cut_lo    # clean / low-M → long leg
        hi  = m_yr >= cut_hi    # manipulators / high-M → short leg
        mid = ~lo & ~hi
        rows.append(
            {
                "year":    yr,
                "n":       int(len(m_yr)),
                "n_lo":    int(lo.sum()),
                "n_hi":    int(hi.sum()),
                "ret_lo":  float(r_yr[lo].mean()),
                "ret_mid": float(r_yr[mid].mean()),
                "ret_hi":  float(r_yr[hi].mean()),
                "ret_mkt": float(r_yr.mean()),
                # hedge: long clean (low-M) − short manipulators (high-M)
                "hedge":   float(r_yr[lo].mean() - r_yr[hi].mean()),
                "m_lo":    float(m_yr[lo].mean()),
                "m_hi":    float(m_yr[hi].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# HAC inference
# ---------------------------------------------------------------------------

def summarize(annual_series: pd.Series) -> dict:
    """HAC (Newey-West) t-stat on an annual return series.

    Returns mean, volatility, Sharpe, HAC t-stat, hit-rate, and n.
    ``annual_series`` should be the annual hedge (or leg) return as a
    fraction, one observation per year.
    """
    r = pd.Series(annual_series, dtype=float).dropna().to_numpy()
    n = r.size
    if n < 2:
        return {k: float("nan") for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}
    mean = r.mean()
    vol  = r.std(ddof=1)
    # Newey-West long-run variance (rule-of-thumb lag count)
    e = r - mean
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean":     float(mean),
        "vol":      float(vol),
        "sharpe":   float(mean / vol) if vol > 0 else float("nan"),
        "tstat":    float(mean / se)  if se  > 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "n":        int(n),
    }
