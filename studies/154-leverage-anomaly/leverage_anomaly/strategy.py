"""Strategy for Study 154 (Leverage-Anomaly) — low-minus-high leverage, annual rebalance.

The leverage-anomaly recipe (Penman, Richardson & Tuna 2007):

  1. At end of fiscal year y, rank S&P 500 firms by financial-leverage proxy
     (LTD/Assets or Liabilities/Equity).
  2. Form a *low-leverage* quintile portfolio (bottom 20% of leverage) and a
     *high-leverage* quintile portfolio (top 20%).
  3. Measure calendar-year y+1 equal-weight returns.
  4. The anomaly: low-leverage outperforms high-leverage — the *opposite* of
     Modigliani-Miller risk compensation.

**Honest controls**:

1. Equal-weight market: all firms with valid leverage data that year.
2. Random portfolio of the same quintile size (seeded) — the null that any
   decile-size concentrated subset might outperform randomly, without any
   information in the leverage rank.

**No look-ahead**: signal from fiscal year y → forward return in calendar year y+1.

**Survivorship bias**: the EDGAR panel is current S&P 500 projected backwards.
All results are *upper bounds*; the true live effect is weaker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core sorting and portfolio construction
# ---------------------------------------------------------------------------
def quintile_spread(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    low_is_good: bool = True,
) -> pd.DataFrame:
    """Annual low-minus-high quintile spread on a (year × ticker) leverage signal.

    ``q`` is the tail fraction (default 0.20 = quintiles).  When ``low_is_good``
    is True (the anomaly direction), the *bottom* quintile of leverage is the
    "good" (long) portfolio; the *top* quintile is the "bad" (short) portfolio.

    Returns a DataFrame with columns ``low``, ``high``, ``market``, and ``spread``
    (low − high when ``low_is_good``, else high − low), indexed by signal year.
    """
    rows: dict[int, dict] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < 20:
            continue

        hi_thresh = s.quantile(1.0 - q)
        lo_thresh = s.quantile(q)
        low_ret = nxt[s <= lo_thresh].mean()   # low-leverage portfolio
        high_ret = nxt[s >= hi_thresh].mean()  # high-leverage portfolio
        mkt_ret = nxt.mean()

        if np.isnan(low_ret) or np.isnan(high_ret) or np.isnan(mkt_ret):
            continue

        spread = float(low_ret - high_ret) if low_is_good else float(high_ret - low_ret)
        rows[int(y)] = {
            "low": float(low_ret),
            "high": float(high_ret),
            "market": float(mkt_ret),
            "spread": spread,
            "n": int(len(s)),
            "n_low": int((s <= lo_thresh).sum()),
            "n_high": int((s >= hi_thresh).sum()),
        }
    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_excess(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    seed: int = 154,
) -> pd.Series:
    """Null distribution of random-portfolio excess returns (quintile size).

    For each year, draw ``n_draws`` random subsets of the same count as the
    top/bottom quintile and record their excess return vs the equal-weight market.
    Returns a Series of excess-return-vs-market across all (year × draw) pairs —
    the null distribution that any concentrated-subset effect must exceed.
    """
    rng = np.random.default_rng(seed)
    excesses: list[float] = []
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < 20:
            continue
        n_q = max(1, int(round(len(s) * q)))
        mkt = nxt.to_numpy()
        mkt_mean = mkt.mean()
        all_idx = np.arange(len(mkt))
        for _ in range(n_draws):
            pick = rng.choice(all_idx, size=n_q, replace=False)
            excesses.append(float(mkt[pick].mean() - mkt_mean))
    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(annual_series: pd.Series) -> dict:
    """Headline statistics for an annual return or spread series.

    Returns mean, vol, Sharpe, HAC t-stat (Newey-West, appropriate for short
    annual series), hit rate, max drawdown, and count.
    """
    r = pd.Series(annual_series).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate",
                                    "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    # Newey-West HAC t-stat (suitable for short series with potential autocorrelation)
    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r.clip(lower=-1.0)).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": mu,
        "vol": std,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }


def market_annual(fwd_ret: pd.DataFrame, years: list[int] | None = None) -> pd.Series:
    """Equal-weight annual market return across all tickers in the panel."""
    if years is not None:
        fwd_ret = fwd_ret.reindex(years)
    return fwd_ret.mean(axis=1).rename("market").dropna()
