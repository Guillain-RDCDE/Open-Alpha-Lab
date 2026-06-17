"""Strategy and honest controls — Study 240 (Dividend-Initiation).

The claim: firms that pay their FIRST dividend (initiators) earn a durable re-rating
premium vs matched non-initiators in the same universe over the following year.  The
signalling hypothesis (Bhattacharya 1979, Miller & Rock 1985) predicts that dividend
initiation reveals management's private information about future earnings.

Two arms tested each year:

- **Initiators** — names for whom the current year is the first calendar year with
  positive dividends.  Equal-weight, one-year holding period (calendar year *y+1*).
- **Control (non-initiators)** — names in the same universe that have NEVER paid a
  dividend through year *y*.  This is the closest feasible matched control without a
  bespoke matching algorithm on market-cap/industry: both arms share the same "growth
  stock" universe profile.

Inference uses a HAC t-stat (Newey-West) on the year-by-year spread series.

Design notes
------------
- Execution lag: signal detected at fiscal year-end (December 31 of year *y*); position
  entered January 1 of year *y+1*.  No look-ahead.
- Small-sample caveat: initiation events are rare (~3–8 per year in a 50-stock universe),
  so the annual mean return of the initiator arm is noisy.  Years with fewer than
  ``min_events`` initiators are excluded from the spread test.
- Gross returns only.  Borrow costs and transaction costs are NOT deducted; a real
  implementation would face higher costs on the growth-stock names that initiates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Arm return construction
# ---------------------------------------------------------------------------
def arm_returns(
    fwd_ret: pd.DataFrame,
    mask: pd.DataFrame,
    min_names: int = 1,
) -> pd.Series:
    """Equal-weight average next-year return of the names selected by ``mask`` each year.

    Years with fewer than ``min_names`` in the mask are excluded.
    """
    out: dict = {}
    for y in mask.index:
        if y not in fwd_ret.index:
            continue
        sel = mask.columns[mask.loc[y]].tolist()
        if len(sel) < min_names:
            continue
        r = fwd_ret.loc[y, sel].dropna()
        if len(r) < min_names:
            continue
        out[y] = float(r.mean())
    return pd.Series(out, name="ret").sort_index()


# ---------------------------------------------------------------------------
# Spread / hedge
# ---------------------------------------------------------------------------
def hedge_returns(
    initiator_ret: pd.Series,
    control_ret: pd.Series,
) -> pd.Series:
    """Year-by-year initiator return minus control return (the re-rating premium spread).

    Aligns on the common years.  A positive mean with a significant HAC t-stat would be
    the evidence needed to claim "initiators outperform non-initiators".
    """
    a, b = initiator_ret.align(control_ret, join="inner")
    spread = a - b
    spread.name = "initiator_minus_control"
    return spread


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary(returns: pd.Series) -> dict:
    """Headline annual-return statistics with a HAC t-stat (Newey-West).

    Reports mean, vol, Sharpe, HAC t-stat, hit rate, max drawdown, and n.
    |t| >= 2 is the minimum inference bar for a real signal.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate",
                                    "max_dd", "n")}
    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    sharpe = float(mean / vol) if vol > 0 else np.nan

    # Newey-West HAC t-stat
    e = r.to_numpy() - mean
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = float(np.sqrt(max(lrv, 0.0) / n))
    tstat = float(mean / se) if se > 0 else np.nan

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean,
        "vol": vol,
        "sharpe": sharpe,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_dd": max_dd,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Event count summary
# ---------------------------------------------------------------------------
def event_counts(initiation_mask: pd.DataFrame) -> pd.Series:
    """Number of initiation events per year."""
    return initiation_mask.sum(axis=1).rename("n_initiations")
