"""The gross-profitability long-short and its honest equal-weight market baseline.

Novy-Marx (2013): sort firms on GP/A each year; go long the top quintile, short
the bottom quintile, annual rebalance. We benchmark against the equal-weight
universe return — not an absolute backtest — so the verdict is "vs the market",
never a raw return claim.

Engine is deliberately kept close to the desk's EDGAR-factor studies (52-smoke-screen,
65-scorecard): ``quintile_hedge`` + ``summary`` are drop-in compatible.

No look-ahead: the signal at year *y* is the fiscal year-*y* gross profitability ratio
(as in :func:`data.fetch_panel`); the strategy acts on *y+1* calendar returns. The
EDGAR data lag is baked into the data layer.

The panel is **survivorship-biased** (current S&P 500 projected backwards): we name
this on the Signal axis and treat every positive result as an upper bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core signal
# ---------------------------------------------------------------------------
def gross_profitability(
    gross_profit: pd.DataFrame, assets: pd.DataFrame
) -> pd.DataFrame:
    """GP/A = GrossProfit / Assets, (year × ticker) frame.

    Handles missing data gracefully: a ticker must have both GrossProfit and Assets
    to get a signal value; otherwise NaN. Financial firms with negative or near-zero
    Assets are not explicitly filtered — survivorship bias means most pathological
    cases have already been selected out, but the user should be aware.
    """
    common = gross_profit.columns.intersection(assets.columns)
    return gross_profit[common] / assets[common]


# ---------------------------------------------------------------------------
# Long-short sorter
# ---------------------------------------------------------------------------
def quintile_hedge(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
) -> pd.DataFrame:
    """Annual top-q / bottom-q long-short on a (year × ticker) GP/A signal.

    Long the top-``q`` gross-profitability quintile, short the bottom-``q`` quintile.
    The hedge return for year *y* is ``mean(return_y+1 | top-q) − mean(return_y+1 | bot-q)``
    with equal weighting within each leg.

    Returns a DataFrame indexed by year with columns:
        ``high`` — mean next-year return of the high-GP/A leg
        ``low`` — mean next-year return of the low-GP/A leg
        ``hedge`` — long high minus short low (the portfolio of interest)
        ``market`` — equal-weight universe return (the baseline)
        ``n`` — number of stocks in the universe for that year
    """
    rows: dict[int, dict] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        common = s.index.intersection(nxt.index)
        if len(common) < 20:
            continue
        sc = s.reindex(common)
        nc = nxt.reindex(common)
        hi_idx = sc[sc >= sc.quantile(1.0 - q)].index
        lo_idx = sc[sc <= sc.quantile(q)].index
        r_hi = nc.reindex(hi_idx).dropna().mean()
        r_lo = nc.reindex(lo_idx).dropna().mean()
        if np.isnan(r_hi) or np.isnan(r_lo):
            continue
        rows[int(y)] = {
            "high": float(r_hi),
            "low": float(r_lo),
            "hedge": float(r_hi - r_lo),
            "market": float(nc.mean()),
            "n": int(len(common)),
        }
    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------
# Equal-weight market baseline
# ---------------------------------------------------------------------------
def market_annual(fwd_ret: pd.DataFrame) -> pd.Series:
    """Equal-weight universe mean return for each year — the honest baseline."""
    return fwd_ret.mean(axis=1).rename("market").dropna()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary(annual_returns: pd.Series) -> dict:
    """Annualised statistics for a yearly return series.

    Returns mean, vol, Sharpe, a Newey-West HAC t-stat (robust to the modest
    autocorrelation of annual quality-factor returns), hit-rate (years > 0),
    max drawdown, and *n*. The HAC t-stat is the inference-bar number.
    """
    r = pd.Series(annual_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    sr = mean / vol if vol > 0 else np.nan
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())

    # Newey-West HAC t-stat on annual returns (lags rule-of-thumb).
    arr = r.to_numpy()
    mu = arr.mean()
    e = arr - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else np.nan

    return {
        "mean": mean,
        "vol": vol,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": n,
    }
