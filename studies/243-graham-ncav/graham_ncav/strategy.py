"""Strategy and honest controls — Study 243 (Graham NCAV).

The signal: each year, identify firms where NCAV/MktCap > 1 (price < NCAV/share,
the Graham net-net criterion). Equal-weight those firms and measure the next-year
return vs the broad market.

Graham's insight (Security Analysis, 1934): firms trading below their net current
asset value offer a margin of safety because even in liquidation, a creditor
receiving 100 cents on the dollar for current assets while retiring all liabilities
leaves equity-holders with more than the market price. Graham found many such firms
in the 1930s depression era among small, obscure stocks.

The steelman thesis today: NCAV screens identify deeply undervalued, overlooked
small-caps; the "January effect" and value premium should still reward them.

Reality in our universe: the S&P 500 large-cap panel. Net-nets are:
- Extremely rare (0-7 per year, zero in 2020-2023).
- Not the cheap cigar-butts Graham intended — they are asset-light tech firms
  (NVDA, AAPL, LRCX, MA) where high intangible value explains the gap.
- Dominated by survivorship bias: we know ex-post these are winners.

This module computes:
- ``ncav_screen`` — binary screen: which firms pass the NCAV test each year.
- ``net_net_portfolio`` — annual EW returns of net-net stocks vs market.
- ``summarize`` — HAC (Newey-West) t-stat on the annual hedge/portfolio series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# NCAV screen
# ---------------------------------------------------------------------------

def ncav_screen(
    ncav_ratio: pd.DataFrame,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Boolean DataFrame: True where NCAV/MktCap >= threshold.

    threshold = 1.0 (default): price <= NCAV/share (strict Graham net-net).
    threshold = 1.5 (= 1/(2/3)): price <= 2/3 NCAV/share (Graham's rule).

    Parameters
    ----------
    ncav_ratio : DataFrame, year x ticker
        NCAV-to-market-cap ratio (from data.fetch_panel).
    threshold : float
        Minimum NCAV/MktCap ratio to pass the screen.

    Returns
    -------
    DataFrame of booleans, same shape as ncav_ratio.
    """
    return ncav_ratio >= threshold


# ---------------------------------------------------------------------------
# Annual portfolio return
# ---------------------------------------------------------------------------

def net_net_portfolio(
    ncav_ratio: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    threshold: float = 1.0,
    min_firms: int = 1,
) -> pd.DataFrame:
    """Annual equal-weighted net-net portfolio vs the broad market.

    For each year in ``ncav_ratio.index``:

    1. Identify firms passing the NCAV screen (ncav_ratio >= threshold).
    2. Compute the equal-weighted next-year return of those firms.
    3. Compute the equal-weighted next-year return of all firms with valid data
       (broad market proxy).

    Returns a DataFrame indexed by year with columns:
    ``n_netnets, n_market, ret_netnets, ret_market, hedge``
    (hedge = ret_netnets - ret_market).

    Years with fewer than ``min_firms`` net-nets are still included with
    NaN for ret_netnets and hedge (to transparently show the scarcity
    problem — a key finding of this study).
    """
    rows = []
    for yr in ncav_ratio.index:
        nr = ncav_ratio.loc[yr]
        r = fwd_ret.loc[yr]

        valid = nr.notna() & r.notna()
        nn_mask = (nr >= threshold) & r.notna()

        n_nn = int((nr >= threshold).sum())
        n_mkt = int(valid.sum())

        ret_nn = float(r[nn_mask].mean()) if nn_mask.sum() >= min_firms else float("nan")
        ret_mkt = float(r[valid].mean()) if valid.sum() > 0 else float("nan")

        rows.append({
            "year":        yr,
            "n_netnets":   n_nn,
            "n_market":    n_mkt,
            "ret_netnets": ret_nn,
            "ret_market":  ret_mkt,
            "hedge":       float(ret_nn - ret_mkt) if not (np.isnan(ret_nn) or np.isnan(ret_mkt)) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# HAC inference
# ---------------------------------------------------------------------------

def summarize(annual_series: pd.Series) -> dict:
    """HAC (Newey-West) t-stat on an annual return series.

    Returns mean, volatility, Sharpe, HAC t-stat, hit-rate, and n.
    Operates on non-NaN values only.
    """
    r = pd.Series(annual_series, dtype=float).dropna().to_numpy()
    n = r.size
    if n < 2:
        return {k: float("nan") for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}
    mean = r.mean()
    vol = r.std(ddof=1)
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
