"""Strategy and honest controls -- Study 230 (Ohlson O-score).

The signal: sort stocks into terciles by their Ohlson O-score (computed from
year-*t* EDGAR data). Because higher O = more distressed, the hedge is:
long *low-O* (safe) firms and short *high-O* (distressed) firms. The hedge
return is ``ret_lo - ret_hi`` (safe minus distressed).

Two competing hypotheses exist in the literature -- the same tension as
Altman-Z (Study 123):

- **Distress premium** (risk story): high-O (distressed) firms carry more
  financial risk and should deliver *higher* returns as compensation.
  High-O -> high return => hedge (lo - hi) is *negative*.
- **Distress puzzle** (Dichev 1998, Ohlson himself): high-O firms empirically
  underperform. High-O -> low return => hedge (lo - hi) is *positive*.

This module computes:

- ``ohlson_oscore`` -- the O-score from six EDGAR-derived inputs.
- ``tertile_hedge`` -- annual long-low-O / short-high-O return (safe minus
  distressed) and bucket breakdowns.
- ``summarize`` -- HAC (Newey-West) t-stat on the annual hedge series.

The **control** is buy-and-hold (market portfolio). Because the panel is
survivorship-biased, all results are labelled as upper bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ohlson O-score
# ---------------------------------------------------------------------------

def ohlson_oscore(
    assets: pd.DataFrame,
    assets_cur: pd.DataFrame,
    liab_cur: pd.DataFrame,
    liabilities: pd.DataFrame,
    net_income: pd.DataFrame,
    net_income_prev: pd.DataFrame,
    cfo: pd.DataFrame,
    coef: dict | None = None,
) -> pd.DataFrame:
    """Ohlson (1980) O-score from nine fundamental components.

    O = -1.32 - 0.407*SIZE + 6.03*TLTA - 1.43*WCTA + 0.076*CLCA
           - 1.72*OENEG - 2.37*NITA - 1.83*FUTL + 0.285*INTWO - 0.521*CHIN

    where:
    - SIZE  = log(Total Assets)
    - TLTA  = Total Liabilities / Total Assets
    - WCTA  = (AssetsCurrent - LiabilitiesCurrent) / Total Assets
    - CLCA  = LiabilitiesCurrent / AssetsCurrent
    - OENEG = 1 if Total Liabilities > Total Assets
    - NITA  = Net Income / Total Assets
    - FUTL  = Cash Flow from Operations / Total Liabilities
    - INTWO = 1 if net income negative in both current and prior year
    - CHIN  = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)

    All inputs are year x ticker DataFrames aligned on the same index and
    columns. Entries are NaN wherever Assets <= 0 or Liabilities <= 0.

    Higher O-score = higher predicted distress probability.
    """
    if coef is None:
        coef = dict(
            intercept=-1.32, size=-0.407, tlta=6.03, wcta=-1.43,
            clca=0.076, oeneg=-1.72, nita=-2.37, futl=-1.83,
            intwo=0.285, chin=-0.521,
        )

    valid = (assets > 0) & (liabilities > 0)

    SIZE   = np.log(assets.clip(lower=1.0)).where(valid)
    TLTA   = (liabilities / assets).where(valid)
    WCTA   = ((assets_cur - liab_cur) / assets).where(valid)
    CLCA   = (liab_cur / assets_cur.replace(0, np.nan)).where(valid)
    OENEG  = (liabilities > assets).astype(float).where(valid)
    NITA   = (net_income / assets).where(valid)
    FUTL   = (cfo / liabilities).where(valid)
    INTWO  = ((net_income < 0) & (net_income_prev < 0)).astype(float).where(valid)
    denom  = (net_income.abs() + net_income_prev.abs()).replace(0, np.nan)
    CHIN   = ((net_income - net_income_prev) / denom).where(valid)

    return (
        coef["intercept"]
        + coef["size"]    * SIZE
        + coef["tlta"]    * TLTA
        + coef["wcta"]    * WCTA
        + coef["clca"]    * CLCA
        + coef["oeneg"]   * OENEG
        + coef["nita"]    * NITA
        + coef["futl"]    * FUTL
        + coef["intwo"]   * INTWO
        + coef["chin"]    * CHIN
    )


# ---------------------------------------------------------------------------
# Annual bucket sort
# ---------------------------------------------------------------------------

def tertile_hedge(
    o_scores: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 1 / 3,
    min_firms: int = 20,
) -> pd.DataFrame:
    """Annual long-low-O / short-high-O hedge vs market.

    Because higher O = more distressed, the hedge is long the *safe* firms
    (bottom tercile of O, i.e. low-O) and short the *distressed* firms
    (top tercile of O, i.e. high-O):

        hedge = ret_lo_O - ret_hi_O  (safe minus distressed)

    For each year in ``o_scores.index``:

    1. Take all firms with a valid O and a valid next-year return.
    2. Split into bottom-``q`` (low-O = safe) and top-``q`` (high-O = distressed) by O.
    3. Record equal-weighted mean returns for each bucket and the market.

    Returns a DataFrame indexed by year with columns:
    ``n, ret_lo, ret_mid, ret_hi, ret_mkt, hedge`` (= ret_lo - ret_hi,
    i.e. safe minus distressed).
    """
    rows = []
    for yr in o_scores.index:
        o_yr = o_scores.loc[yr].dropna()
        r_yr = fwd_ret.loc[yr].dropna()
        both = o_yr.index.intersection(r_yr.index)
        o_yr = o_yr.loc[both]
        r_yr = r_yr.loc[both]
        if len(o_yr) < min_firms:
            continue
        cut_lo = o_yr.quantile(q)       # low-O boundary (safe)
        cut_hi = o_yr.quantile(1.0 - q)  # high-O boundary (distressed)
        lo = o_yr <= cut_lo   # safe: low O-score
        hi = o_yr >= cut_hi   # distressed: high O-score
        mid = ~lo & ~hi
        rows.append(
            {
                "year":    yr,
                "n":       int(len(o_yr)),
                "n_lo":    int(lo.sum()),
                "n_hi":    int(hi.sum()),
                "ret_lo":  float(r_yr[lo].mean()),   # safe (low-O) return
                "ret_mid": float(r_yr[mid].mean()),
                "ret_hi":  float(r_yr[hi].mean()),   # distressed (high-O) return
                "ret_mkt": float(r_yr.mean()),
                "hedge":   float(r_yr[lo].mean() - r_yr[hi].mean()),  # safe - distressed
                "o_lo":    float(o_yr[lo].mean()),
                "o_hi":    float(o_yr[hi].mean()),
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
    # Newey-West long-run variance: standard rule-of-thumb lag count
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
