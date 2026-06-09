"""Cross-sectional sorting machinery — and the load-bearing question: **is the SML really too flat?**

The whole anomaly rests on one cross-sectional fact: rank stocks by past volatility and the *calm*
ones go on to earn a higher risk-adjusted return than the *wild* ones — the opposite of the naive
"more risk, more reward" (Ang–Hodrick–Xing–Zhang 2006; Baker–Bradley–Wurgler 2011; Frazzini–Pedersen
2014). If that gradient weren't there, sorting on vol would just be sorting on noise. So before any
backtest we measure the thing directly: across the universe, does *lower* past vol line up with a
*higher* Sharpe?

Everything here is a pure function of a ``dates × stock`` daily-returns panel. The one rule the
portfolio builder never breaks: the weight a stock carries on day ``t`` is fixed from returns up to
day ``t−1`` (a past-only vol rank), so there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Per-stock realized vol and the security-market-line diagnostic
# --------------------------------------------------------------------------- #

def realized_vol(panel: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Trailing per-stock realized volatility — the rolling std of each column over ``window`` days.

    ``window=126`` ≈ six trading months, the shorter end of the canonical low-vol lookback (Ang et al.
    use 1–12 months). Per-bar (not annualised); leading rows are NaN until the window fills. This is
    the single signal the sort ranks on.
    """
    return panel.rolling(window).std(ddof=1)


def _ann(mean_daily, std_daily):
    return mean_daily * TRADING_DAYS_PER_YEAR, std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)


def security_market_line(panel: pd.DataFrame, n_buckets: int = 10) -> dict:
    """Sort the universe by full-sample vol into ``n_buckets`` and ask if Sharpe *falls* with vol.

    For each stock we take its full-sample annualised return, annualised vol and Sharpe, then sort the
    names into equal-count vol buckets (bucket 0 = calmest, ``n_buckets−1`` = wildest) and average
    within each. The anomaly shows as a **decreasing** Sharpe across buckets — the calm decile earning
    *more* per unit of risk than the wild one. We also fit the cross-sectional OLS of per-stock Sharpe
    on per-stock vol; a **negative ``sharpe_vol_slope``** is the flat/inverted SML in one number, the
    engine the long-short harvests.

    Returns the per-bucket table, the low-minus-high Sharpe gap, the OLS slope and its sign, plus the
    raw per-stock frame. A textbook-CAPM null panel gives a slope ≈ 0 and a flat bucket profile.
    """
    mu = panel.mean()
    sd = panel.std(ddof=1)
    ann_ret, ann_vol = _ann(mu, sd)
    sharpe = (mu / sd * np.sqrt(TRADING_DAYS_PER_YEAR)).replace([np.inf, -np.inf], np.nan)
    per = pd.DataFrame({"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}).dropna()

    order = per["ann_vol"].rank(method="first")
    per = per.assign(bucket=pd.qcut(order, n_buckets, labels=False))
    table = per.groupby("bucket").agg(
        ann_ret=("ann_ret", "mean"),
        ann_vol=("ann_vol", "mean"),
        sharpe=("sharpe", "mean"),
        n=("sharpe", "size"),
    )

    # cross-sectional OLS of Sharpe on vol: negative slope == low-vol earns more per unit risk
    x = per["ann_vol"].to_numpy()
    y = per["sharpe"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    lo_sharpe = float(table["sharpe"].iloc[0])
    hi_sharpe = float(table["sharpe"].iloc[-1])
    return {
        "table": table,
        "low_minus_high_sharpe": lo_sharpe - hi_sharpe,
        "low_bucket_sharpe": lo_sharpe,
        "high_bucket_sharpe": hi_sharpe,
        "sharpe_vol_slope": float(slope),
        "anomaly_present": bool(slope < 0),
        "per_stock": per,
        "n_stocks": int(len(per)),
    }


# --------------------------------------------------------------------------- #
# Look-ahead-safe quantile portfolios
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SortedBook:
    """The daily return streams of a vol-sorted book, plus the weights behind them (for turnover)."""
    low: pd.Series             # equal-weight bottom-decile (low-vol) long-only stream
    high: pd.Series            # equal-weight top-decile (high-vol) stream
    long_short: pd.Series      # dollar-neutral: +1 low − 1 high
    market: pd.Series          # equal-weight whole universe (the cross-section's "market")
    w_low: pd.DataFrame
    w_high: pd.DataFrame
    rebalances: int


def quantile_portfolios(
    panel: pd.DataFrame,
    lookback: int = 126,
    rebal: int = 21,
    decile_frac: float = 0.1,
) -> SortedBook:
    """Form the low-vol, high-vol, dollar-neutral and equal-weight-market daily streams, no look-ahead.

    At each rebalance (every ``rebal`` days, after a ``lookback`` warm-up) we rank the names that have
    a *complete* trailing window by realized vol, equal-weight the bottom ``decile_frac`` (low-vol) and
    the top ``decile_frac`` (high-vol), and hold them until the next rebalance. The vol that sets a
    weight uses returns strictly *before* the day the weight earns a return, so the book is
    investable-as-described. ``long_short = low − high`` is the dollar-neutral anomaly portfolio;
    ``market`` is the equal-weight universe.
    """
    cols = panel.columns
    vals = panel.to_numpy()
    n, m = vals.shape
    w_low = np.zeros((n, m))
    w_high = np.zeros((n, m))

    rebal_pts = range(lookback, n, rebal)
    count = 0
    for p in rebal_pts:
        win = vals[p - lookback:p, :]                     # rows p-lookback .. p-1 (past-only)
        full = ~np.isnan(win).any(axis=0)                 # names with a complete window
        if full.sum() < max(10, int(2 / decile_frac)):
            continue
        rv = np.full(m, np.nan)
        rv[full] = win[:, full].std(axis=0, ddof=1)
        valid = np.where(~np.isnan(rv))[0]
        k = max(1, int(len(valid) * decile_frac))
        order = valid[np.argsort(rv[valid])]
        low_idx, high_idx = order[:k], order[-k:]
        end = min(p + rebal, n)
        w_low[p:end, low_idx] = 1.0 / k
        w_high[p:end, high_idx] = 1.0 / k
        count += 1

    w_low = pd.DataFrame(w_low, index=panel.index, columns=cols)
    w_high = pd.DataFrame(w_high, index=panel.index, columns=cols)

    filled = panel.fillna(0.0)
    low = (w_low * filled).sum(axis=1).rename("low")
    high = (w_high * filled).sum(axis=1).rename("high")
    market = panel.mean(axis=1).rename("market")

    # trim the warm-up (all-zero weights) so summaries see only invested days
    active = (w_low.abs().sum(axis=1) > 0)
    low, high = low[active], high[active]
    long_short = (low - high).rename("long_short")
    market = market.reindex(low.index)
    return SortedBook(
        low=low, high=high, long_short=long_short, market=market,
        w_low=w_low.loc[low.index], w_high=w_high.loc[low.index], rebalances=count,
    )


def turnover_ann(weights: pd.DataFrame, rebal: int = 21) -> float:
    """Annualised one-way turnover of a weight book — what the cost model charges against.

    Sums ``½·Σ|Δw|`` across rebalances (a full swap of the decile = 1.0) and scales to a year. Low-vol
    deciles are sticky, so this is small — part of why the *long-only* leg is cheap to run (the
    expensive part is the short leg's borrow, priced separately in :mod:`strategy`).
    """
    dw = weights.diff().abs().sum(axis=1)
    per_rebal = dw[dw > 0]
    if per_rebal.empty:
        return 0.0
    rebals_per_year = TRADING_DAYS_PER_YEAR / rebal
    return float(0.5 * per_rebal.mean() * rebals_per_year)
