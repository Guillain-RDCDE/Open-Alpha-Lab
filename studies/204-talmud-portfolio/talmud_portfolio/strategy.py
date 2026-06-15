"""The Talmud Portfolio engine and its honest controls — Study 204.

The ancient recipe attributed to Rabbi Isaac Bar Aha (Babylonian Talmud, Bava Metzia
42a): *"A man should always divide his money into three parts — a third in land, a
third in business, and a third in reserve."* Rendered in ETFs: 1/3 VNQ (real estate),
1/3 SPY (stocks), 1/3 BND (bonds/cash), rebalanced annually.

The three benchmarks are chosen to bracket the Talmud blending claim fairly:

- **100 % SPY** (buy-and-hold stocks) — the return ceiling; the Talmud should trail
  in bull markets but cushion in crashes.
- **60/40 (SPY/BND)** — the canonical "diversified" portfolio; the Talmud differs
  only in splitting 60 equity into 1/3 REIT + 1/3 equity. Does real estate add value
  to a classic 60/40?
- **Equal-weight SPY/VNQ** (50/50 stocks + REITs, no bonds) — tests whether the bond
  leg or the REIT leg is doing the diversification work.

No look-ahead: annual rebalance on the first trading day of each calendar year.
Costs: one-way rebalance turnover charged at ``cost_bps`` on each rebalance.

All Sharpe ratios are computed on **excess-of-BND** returns so that the bond leg's
total return is used consistently as the cash proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a total-return price frame (first row dropped)."""
    return prices.pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _worst_year(daily_ret: pd.Series) -> float:
    """Worst calendar-year total return from a daily return series."""
    annual = (1.0 + daily_ret).groupby(daily_ret.index.year).prod() - 1.0
    return float(annual.min())


# ---------------------------------------------------------------------------
# Fixed-weight blended portfolio engine — annual rebalance
# ---------------------------------------------------------------------------
def blended_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "annual",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily net return of a fixed-weight blend, rebalanced on a calendar schedule.

    Weights drift between rebalances; on each rebalance date the book is reset to
    ``weights`` and the one-way turnover (sum of absolute weight changes) is charged at
    ``cost_bps``. ``rebalance='annual'`` resets on the first trading day of each
    calendar year; ``'none'`` lets the mix drift forever (buy-and-hold style). Returns
    a daily net-return Series aligned to ``returns.index``.

    No look-ahead: weights are computed from ``weights`` (fixed targets), not from
    any future price information. The signal is formed before the period; positions
    are held during the period.
    """
    cols = list(weights.keys())
    R = returns[cols].to_numpy()
    w_target = np.array([weights[c] for c in cols], dtype=float)
    w_target /= w_target.sum()  # normalise to 1 in case of rounding
    n = R.shape[0]
    idx = returns.index

    if rebalance == "annual":
        marks = idx.to_series().groupby(idx.year).head(1).index
    elif rebalance == "none":
        marks = idx[:1]
    else:
        raise ValueError(f"unknown rebalance schedule: {rebalance!r}")
    rebal_set = set(marks)

    w = w_target.copy()
    out = np.empty(n)
    cost = cost_bps * 1e-4
    for t in range(n):
        turn = 0.0
        if idx[t] in rebal_set:
            turn = float(np.abs(w_target - w).sum())
            w = w_target.copy()
        port_ret = float(w @ R[t]) - turn * cost
        out[t] = port_ret
        # Update weights for drift
        w = w * (1.0 + R[t])
        s = w.sum()
        if s > 0:
            w = w / s
    return pd.Series(out, index=idx, name="blend")


def talmud_portfolio(
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    reit: str = "VNQ",
    stk: str = "SPY",
    bond: str = "BND",
) -> pd.Series:
    """The Talmud 1/3–1/3–1/3 portfolio (VNQ / SPY / BND), annually rebalanced."""
    weights = {reit: 1 / 3, stk: 1 / 3, bond: 1 / 3}
    return blended_portfolio(returns, weights, rebalance="annual", cost_bps=cost_bps)


def sixty_forty(
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    stk: str = "SPY",
    bond: str = "BND",
) -> pd.Series:
    """Classic 60/40 (SPY/BND), annually rebalanced."""
    weights = {stk: 0.60, bond: 0.40}
    return blended_portfolio(returns, weights, rebalance="annual", cost_bps=cost_bps)


def spy_only(returns: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    """100% SPY — no rebalancing, no costs — the return ceiling."""
    return returns[ticker].rename("SPY")


def spy_vnq_50_50(
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    stk: str = "SPY",
    reit: str = "VNQ",
) -> pd.Series:
    """50/50 SPY/VNQ (no bond leg), annually rebalanced — isolates the REIT contribution."""
    weights = {stk: 0.50, reit: 0.50}
    return blended_portfolio(returns, weights, rebalance="annual", cost_bps=cost_bps)


# ---------------------------------------------------------------------------
# Synthetic-tape helpers for the positive control
# ---------------------------------------------------------------------------
def synthetic_talmud(
    frame: pd.DataFrame,
    cost_bps: float = 0.0,
    reit: str = "REIT",
    stk: str = "STK",
    bond: str = "BOND",
) -> pd.Series:
    """The Talmud blend applied to a synthetic three-asset frame (REIT/STK/BOND)."""
    ret = to_returns(frame)
    weights = {reit: 1 / 3, stk: 1 / 3, bond: 1 / 3}
    return blended_portfolio(ret, weights, rebalance="annual", cost_bps=cost_bps)


# ---------------------------------------------------------------------------
# Stats & inference
# ---------------------------------------------------------------------------
def portfolio_stats(
    net: pd.Series,
    rf: pd.Series | None = None,
) -> dict:
    """Headline annual stats for a daily-return series.

    ``rf`` is the daily return of the bond/cash proxy (BND); if given, Sharpe is
    computed on excess-of-rf so all arms are on a like-for-like risk-adjusted basis.
    Returns CAGR, annualised vol, Sharpe, max drawdown, worst calendar-year return,
    and the length of the series in years.
    """
    net = net.dropna().astype(float)
    if len(net) < 2:
        nan = float("nan")
        return dict(cagr=nan, vol=nan, sharpe=nan, max_dd=nan, worst_year=nan, years=0.0)

    equity = (1.0 + net).cumprod()
    years = len(net) / TRADING_DAYS
    total = float(equity.iloc[-1])
    cagr = (total ** (1.0 / years) - 1.0) if years > 0 and total > 0 else float("nan")

    ann_vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float)
    else:
        ex = net
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
              if ex.std(ddof=1) > 0 else float("nan"))

    return {
        "cagr": float(cagr),
        "vol": ann_vol,
        "sharpe": float(sharpe),
        "max_dd": _max_drawdown(equity.to_numpy()),
        "worst_year": _worst_year(net),
        "years": years,
    }


def annual_returns(daily: pd.Series) -> pd.Series:
    """Calendar-year total returns from a daily-return series."""
    return (1.0 + daily.dropna()).groupby(daily.dropna().index.year).prod() - 1.0


# ---------------------------------------------------------------------------
# Inference: Newey-West HAC t-stat on annual return differences
# ---------------------------------------------------------------------------
def hac_tstat_annual(r1: pd.Series, r2: pd.Series) -> float:
    """HAC (Newey-West) t-stat on mean annual return difference (r1 − r2).

    Annual granularity is appropriate because: (a) the rebalance cadence is
    annual, (b) daily returns of a fixed-weight blend are highly autocorrelated
    within each year, and (c) the power of interest is in regime-level differences.
    """
    ar1 = annual_returns(r1)
    ar2 = annual_returns(r2)
    idx = ar1.index.intersection(ar2.index)
    diff = (ar1.loc[idx] - ar2.loc[idx]).dropna().to_numpy(dtype=float)
    n = len(diff)
    if n < 4:
        return float("nan")
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mu = diff.mean()
    e = diff - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Drawdown analysis — the core defensive claim
# ---------------------------------------------------------------------------
def drawdown_episodes(
    returns: pd.DataFrame,
    stock: str,
    thresh: float = -0.15,
) -> list[dict]:
    """Equity-crash episodes (SPY peak-to-trough >= thresh) with contemporaneous leg returns.

    For each episode (peak→trough in the stock leg), returns the stock loss and the
    total return of every other column over the same window — the direct test of
    whether the Talmud's REIT and bond legs cushioned or amplified the crash.
    """
    px = (1.0 + returns).cumprod()
    s = px[stock]
    peak = s.cummax()
    dd = s / peak - 1.0

    episodes: list[dict] = []
    in_dd = False
    peak_date = s.index[0]
    for i in range(len(s)):
        if not in_dd and dd.iloc[i] < 0:
            in_dd = True
            peak_date = s.index[max(0, i - 1)]
        elif in_dd and dd.iloc[i] >= 0:
            in_dd = False
            seg = px.loc[peak_date : s.index[i - 1]]
            trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
            loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
            if loss <= thresh:
                window_df = px.loc[peak_date:trough_date]
                others = {
                    c: float(window_df[c].iloc[-1] / window_df[c].iloc[0] - 1.0)
                    for c in returns.columns
                    if c != stock
                }
                episodes.append(
                    {"peak": peak_date, "trough": trough_date,
                     "stock_loss": loss, "others": others}
                )
    # Open drawdown at tape-end
    if in_dd:
        seg = px.loc[peak_date:]
        trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
        loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
        if loss <= thresh:
            window_df = px.loc[peak_date:trough_date]
            others = {
                c: float(window_df[c].iloc[-1] / window_df[c].iloc[0] - 1.0)
                for c in returns.columns
                if c != stock
            }
            episodes.append(
                {"peak": peak_date, "trough": trough_date,
                 "stock_loss": loss, "others": others}
            )
    return episodes
