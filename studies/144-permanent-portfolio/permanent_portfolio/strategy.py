"""The Permanent Portfolio engine and its honest controls — Study 144.

Harry Browne's recipe: 25% stocks / 25% long Treasuries / 25% gold / 25% cash,
rebalanced annually. The design intent is **regime-proof diversification**: stocks
thrive in prosperity, bonds in deflation, gold in inflation / crisis, and cash
preserves optionality in recession. The bet is not about return maximisation but
about *surviving any regime* with minimum drawdown.

We pin the PP against the one comparison that honestly grades this claim:

- **100% SPY** — the return yardstick; the PP should trail here.
- **60/40 (SPY/TLT)** — the mainstream alternative with similar diversification
  appeal; the PP should improve on drawdown and Sharpe vs 60/40.

No look-ahead: annual rebalance on the first trading day of each calendar year.
Costs: one-way rebalance turnover charged at ``cost_bps`` on each rebalance leg.
Cash (SHY) receives a realistic total return because we use the SHY ETF total-
return price (which includes coupon income), so no synthetic cash yield is needed.

Conventions
-----------
- Inputs are **price frames** (total-return, e.g. from data.load_real).
- Returns are simple daily: ``pct_change().dropna()``.
- The Sharpe is annualised and computed on **excess-of-SHY** returns so all arms
  are compared on a like-for-like risk-adjusted basis.
- The HAC t-stat is Newey-West on the annual return differences (PP − benchmark);
  annual granularity avoids the daily-autocorrelation issue and matches the
  rebalance cadence.
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
# The blended portfolio engine — fixed weights, annual rebalance
# ---------------------------------------------------------------------------
def blended_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "annual",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily net return of a fixed-weight blend, rebalanced on a calendar schedule.

    Weights drift between rebalances; on each rebalance date the book is reset
    to ``weights`` and the one-way turnover (sum of absolute weight changes) is
    charged at ``cost_bps``. ``rebalance='annual'`` resets on the first trading
    day of each calendar year; ``'none'`` lets the mix drift forever (buy-and-
    hold). Returns a daily net-return Series aligned to ``returns.index``.
    """
    cols = list(weights.keys())
    R = returns[cols].to_numpy()
    w_target = np.array([weights[c] for c in cols], dtype=float)
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
        w = w * (1.0 + R[t])
        s = w.sum()
        if s > 0:
            w = w / s
    return pd.Series(out, index=idx, name="blend")


def permanent_portfolio(
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    stk: str = "SPY",
    bond: str = "TLT",
    gold: str = "GLD",
    cash: str = "SHY",
) -> pd.Series:
    """Harry Browne's 25/25/25/25 Permanent Portfolio, annually rebalanced."""
    weights = {stk: 0.25, bond: 0.25, gold: 0.25, cash: 0.25}
    return blended_portfolio(returns, weights, rebalance="annual", cost_bps=cost_bps)


def sixty_forty(
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    stk: str = "SPY",
    bond: str = "TLT",
) -> pd.Series:
    """Classic 60/40 (SPY/TLT), annually rebalanced, for comparison."""
    weights = {stk: 0.60, bond: 0.40}
    return blended_portfolio(returns, weights, rebalance="annual", cost_bps=cost_bps)


def spy_only(returns: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    """100% SPY — no rebalancing, no costs — the return ceiling."""
    return returns[ticker].rename("SPY")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def portfolio_stats(
    net: pd.Series,
    rf: pd.Series | None = None,
) -> dict:
    """Headline annual stats for a daily-return series.

    ``rf`` is the daily return of the cash proxy (SHY); if given, Sharpe is
    computed on excess-of-cash so all arms are compared like-for-like.
    """
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float)
    else:
        ex = net
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
              if ex.std(ddof=1) > 0 else float("nan"))
    worst_yr = _worst_year(net)
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "worst_year": worst_yr,
        "years": years,
    }


def annual_returns(daily: pd.Series) -> pd.Series:
    """Calendar-year total returns from a daily-return series."""
    return (1.0 + daily).groupby(daily.index.year).prod() - 1.0


# ---------------------------------------------------------------------------
# Inference: Newey-West HAC t-stat on annual return differences
# ---------------------------------------------------------------------------
def hac_tstat_annual(pp_daily: pd.Series, bench_daily: pd.Series) -> float:
    """Newey-West HAC t-stat for mean(PP annual return − benchmark annual return).

    Annual granularity is appropriate here because: (a) the rebalance cadence is
    annual, (b) daily returns of a fixed-weight blend are highly autocorrelated
    within each year, and (c) the power of interest is in regime-level
    differences, not day-to-day noise.  The HAC correction still applies to the
    annual series (serial correlation from overlapping macro cycles).
    """
    ar_pp = annual_returns(pp_daily)
    ar_bk = annual_returns(bench_daily)
    common = ar_pp.index.intersection(ar_bk.index)
    diff = ar_pp.loc[common] - ar_bk.loc[common]
    x = diff.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return float("nan")
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Bootstrap CI for Sharpe difference
# ---------------------------------------------------------------------------
def bootstrap_sharpe_diff(
    a: pd.Series,
    b: pd.Series,
    rf: pd.Series | None = None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 144,
) -> dict:
    """Circular block bootstrap CI for the Sharpe *difference* (arm ``a`` − arm ``b``).

    Resamples the two aligned excess-return series jointly in circular blocks so
    that the cross-correlation and volatility clustering survive the resampling,
    recomputes each arm's annualised Sharpe per resample, and returns the point
    difference with a 95% CI and the bootstrap fraction in which ``a`` wins.
    """
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f

    def _sharpe(r: np.ndarray) -> float:
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    n = len(ra)
    point = _sharpe(ra) - _sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = []
    wins = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _sharpe(ra[sel]), _sharpe(rb[sel])
        d = sa - sb
        diffs.append(d)
        if sa > sb:
            wins += 1
    diffs_arr = np.array(diffs)
    lo, hi = float(np.nanpercentile(diffs_arr, 2.5)), float(np.nanpercentile(diffs_arr, 97.5))
    return {
        "point": float(point),
        "ci95": (lo, hi),
        "frac_a_wins": wins / n_boot,
        "n": n,
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Regime-drawdown analysis — the core defensive claim
# ---------------------------------------------------------------------------
def equity_drawdowns(
    returns: pd.DataFrame,
    stock: str,
    thresh: float = -0.10,
) -> list[dict]:
    """SPY drawdown episodes deeper than ``thresh``, with contemporaneous returns.

    For each episode (peak→trough in the stock leg) returns the SPY loss and the
    total return of every other column over the same window — the direct test of
    whether the PP's non-stock legs cushioned or piled on.
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
            peak_date = s.index[i - 1] if i > 0 else s.index[0]
        elif in_dd and dd.iloc[i] >= 0:
            in_dd = False
            seg = px.loc[peak_date:s.index[i - 1]]
            trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
            loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
            if loss <= thresh:
                window_df = px.loc[peak_date:trough_date]
                others = {c: float(window_df[c].iloc[-1] / window_df[c].iloc[0] - 1.0)
                          for c in returns.columns if c != stock}
                episodes.append({
                    "peak": peak_date,
                    "trough": trough_date,
                    "stock_loss": loss,
                    "others": others,
                })
    # open drawdown at tape end
    if in_dd:
        seg = px.loc[peak_date:]
        trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
        loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
        if loss <= thresh:
            window_df = px.loc[peak_date:trough_date]
            others = {c: float(window_df[c].iloc[-1] / window_df[c].iloc[0] - 1.0)
                      for c in returns.columns if c != stock}
            episodes.append({
                "peak": peak_date,
                "trough": trough_date,
                "stock_loss": loss,
                "others": others,
            })
    return episodes
