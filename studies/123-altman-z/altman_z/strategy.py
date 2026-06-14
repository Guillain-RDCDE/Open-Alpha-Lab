"""Strategy and honest controls — Study 123 (Altman-Z).

The signal: sort stocks into terciles by their Altman Z-score (computed from
year-*t* EDGAR data). Hold the bottom tercile (distressed, low-Z) and the top
tercile (safe, high-Z) through the following calendar year. The question: does Z
separate winners from losers?

Two competing hypotheses exist in the literature:

- **Distress premium** (Campbell et al. 2008, Dichev 1998): low-Z firms carry
  more financial risk, so they should deliver *higher* returns — risk
  compensation. Low-Z → high return.
- **Distress puzzle** (Altman & Hotchkiss 2006): contrary to the above, low-Z
  firms empirically underperform — their stocks are overpriced relative to
  distress risk. Low-Z → low return.

This module computes:

- ``altman_zscore`` — the Z-score from the five EDGAR-derived ratios.
- ``tertile_hedge`` — annual long-top-Z / short-bottom-Z return (and breakdowns
  by bucket).
- ``summarize`` — HAC (Newey-West) t-stat on the annual hedge series.

The **control** is buy-and-hold (market portfolio): if high-Z consistently
beats market and low-Z consistently trails it, the signal adds something.
Because the panel is survivorship-biased, all results are labelled as upper
bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Altman Z-score
# ---------------------------------------------------------------------------

def altman_zscore(
    assets: pd.DataFrame,
    assets_cur: pd.DataFrame,
    liab_cur: pd.DataFrame,
    retained_earnings: pd.DataFrame,
    ebit: pd.DataFrame,
    liabilities: pd.DataFrame,
    revenues: pd.DataFrame,
    mkt_equity: pd.DataFrame,
    coef: dict | None = None,
) -> pd.DataFrame:
    """Altman (1968) Z-score from five normalised fundamental ratios.

    Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5, where

    - X1 = (AssetsCurrent − LiabilitiesCurrent) / Assets   (working-capital ratio)
    - X2 = RetainedEarnings / Assets                        (accumulated profitability)
    - X3 = OperatingIncome / Assets                         (EBIT / TA, return on assets)
    - X4 = MktEquity / TotalLiabilities                     (leverage / solvency)
    - X5 = Revenues / Assets                                (asset turnover)

    All inputs are year × ticker DataFrames aligned on the same index and
    columns. Entries are NaN wherever Assets ≤ 0 or Liabilities ≤ 0.

    Original thresholds (public-company model): Z < 1.81 distress zone,
    1.81 ≤ Z < 2.99 grey zone, Z ≥ 2.99 safe zone. This module does not
    hard-code a classification; the signal is the continuous Z value.
    """
    if coef is None:
        coef = dict(X1=1.2, X2=1.4, X3=3.3, X4=0.6, X5=1.0)

    valid = (assets > 0) & (liabilities > 0)
    X1 = ((assets_cur - liab_cur) / assets).where(valid)
    X2 = (retained_earnings / assets).where(valid)
    X3 = (ebit / assets).where(valid)
    X4 = (mkt_equity / liabilities).where(valid)
    X5 = (revenues / assets).where(valid)

    return (
        coef["X1"] * X1
        + coef["X2"] * X2
        + coef["X3"] * X3
        + coef["X4"] * X4
        + coef["X5"] * X5
    )


# ---------------------------------------------------------------------------
# Annual bucket sort
# ---------------------------------------------------------------------------

def tertile_hedge(
    z_scores: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 1 / 3,
    min_firms: int = 20,
) -> pd.DataFrame:
    """Annual long-high-Z / short-low-Z hedge vs market.

    For each year in ``z_scores.index``:

    1. Take all firms with a valid Z and a valid next-year return.
    2. Split into bottom-``q`` (distressed) and top-``q`` (safe) by Z.
    3. Record equal-weighted mean returns for each bucket and the market.

    Firms in the middle tercile are not traded (wide bucket avoids the
    coarseness problem, and the mid-zone is the Altman 'grey zone').

    Returns a DataFrame indexed by year with columns:
    ``n, ret_lo, ret_mid, ret_hi, ret_mkt, hedge`` (= ret_hi − ret_lo).
    """
    rows = []
    for yr in z_scores.index:
        z_yr = z_scores.loc[yr].dropna()
        r_yr = fwd_ret.loc[yr].dropna()
        both = z_yr.index.intersection(r_yr.index)
        z_yr = z_yr.loc[both]
        r_yr = r_yr.loc[both]
        if len(z_yr) < min_firms:
            continue
        cut_lo = z_yr.quantile(q)
        cut_hi = z_yr.quantile(1.0 - q)
        lo = z_yr <= cut_lo
        hi = z_yr >= cut_hi
        mid = ~lo & ~hi
        rows.append(
            {
                "year":    yr,
                "n":       int(len(z_yr)),
                "n_lo":    int(lo.sum()),
                "n_hi":    int(hi.sum()),
                "ret_lo":  float(r_yr[lo].mean()),
                "ret_mid": float(r_yr[mid].mean()),
                "ret_hi":  float(r_yr[hi].mean()),
                "ret_mkt": float(r_yr.mean()),
                "hedge":   float(r_yr[hi].mean() - r_yr[lo].mean()),
                "z_lo":    float(z_yr[lo].mean()),
                "z_hi":    float(z_yr[hi].mean()),
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
