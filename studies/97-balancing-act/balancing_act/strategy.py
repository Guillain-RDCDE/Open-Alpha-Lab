"""The 60/40 engine and its honest controls — Study 97 (Balancing-Act).

The folk claim: 'the classic 60/40 stock/bond portfolio is the sensible default — most of
the stock return with much less risk, a better risk-adjusted outcome than 100% stocks, and
bonds cushion every equity crash.' We build the fixed-weight blend and pin it against the
thing it claims to improve on:

- **100% stocks (SPY, total return)** — the return yardstick.
- The **60/40 blend, rebalanced annually** — 60% stocks, 40% Treasuries, reset to weights
  on the first trading day of each calendar year (drift in between). This is the *fixed*
  60/40 — NOT volatility-weighted risk parity (that is Study 68, All-Weather).

Beyond the headline race we test the central selling point — *"bonds cushion every equity
crash"* — by tracking the **rolling stock/bond correlation** across regimes and zooming on
**2022**, when stocks and bonds fell together. We compare **excess-of-cash to
excess-of-cash** (SHY as the cash proxy) so the Sharpe race is fair, and we put the Sharpe
*difference* through a Newey-West (HAC) *t* and a **circular block bootstrap** CI.

Conventions: returns are simple daily total returns from the price frame; the blend is a
calendar rebalance (no signal, no execution lag needed); rebalance turnover is charged
one-way at ``cost_bps``.
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


# ---------------------------------------------------------------------------
# The blend engine — fixed weights, annual rebalance
# ---------------------------------------------------------------------------
def rebalanced_blend(
    returns: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "annual",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily net return of a fixed-weight blend rebalanced on a calendar schedule.

    Weights drift with the assets between rebalances; on each rebalance date the book is
    reset to ``weights`` and the one-way turnover (sum of absolute weight changes) is
    charged at ``cost_bps``. ``rebalance='annual'`` resets on the first trading day of each
    calendar year; ``'monthly'`` on the first of each month; ``'none'`` lets it drift
    forever (buy-and-hold the initial mix).

    Returns a daily net-return Series aligned to ``returns.index``.
    """
    cols = list(weights.keys())
    R = returns[cols].to_numpy()
    w_target = np.array([weights[c] for c in cols], dtype=float)
    n = R.shape[0]
    idx = returns.index

    if rebalance == "annual":
        marks = idx.to_series().groupby(idx.year).head(1).index
    elif rebalance == "monthly":
        marks = idx.to_series().groupby([idx.year, idx.month]).head(1).index
    elif rebalance == "none":
        marks = idx[:1]
    else:
        raise ValueError(f"unknown rebalance schedule: {rebalance!r}")
    rebal = pd.Index(marks)

    w = w_target.copy()
    out = np.empty(n)
    cost = cost_bps * 1e-4
    for t in range(n):
        if idx[t] in rebal:
            turn = np.abs(w_target - w).sum()
            w = w_target.copy()
        else:
            turn = 0.0
        port_ret = float(w @ R[t]) - turn * cost
        out[t] = port_ret
        # drift the weights with realised returns, renormalise
        w = w * (1.0 + R[t])
        s = w.sum()
        if s != 0:
            w = w / s
    return pd.Series(out, index=idx, name="blend")


def single_asset(returns: pd.DataFrame, ticker: str) -> pd.Series:
    """The daily return series of one asset (e.g. 100% stocks)."""
    return returns[ticker].rename(ticker)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def stats(net: pd.Series, rf: pd.Series | None = None) -> dict:
    """Headline stats for a daily return series.

    If ``rf`` (a daily risk-free / cash return series) is given, the Sharpe is computed on
    the **excess-of-cash** return so two arms with different cash exposure are compared
    like-for-like; CAGR/vol/drawdown stay on the raw total-return series.
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
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "final": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# The crash-cushion test — "bonds cushion every equity crash"?
# ---------------------------------------------------------------------------
def rolling_correlation(
    returns: pd.DataFrame, a: str, b: str, window: int = 63
) -> pd.Series:
    """Rolling correlation between two daily return series (default ~quarter window)."""
    return returns[a].rolling(window, min_periods=window).corr(returns[b])


def equity_drawdowns(
    returns: pd.DataFrame, stock: str, thresh: float = -0.10
) -> list[dict]:
    """Identify peak-to-trough stock drawdowns deeper than ``thresh`` (e.g. -10%).

    For each such episode (peak date → trough date) returns the stock loss and the
    *contemporaneous* total return of every other column over the same window — the direct
    test of whether bonds cushioned (positive) or piled on (negative) during the crash.
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
                window = px.loc[peak_date:trough_date]
                others = {c: float(window[c].iloc[-1] / window[c].iloc[0] - 1.0)
                          for c in returns.columns if c != stock}
                episodes.append({"peak": peak_date, "trough": trough_date,
                                 "stock_loss": loss, "others": others})
    if in_dd:  # an open drawdown at the end of the tape
        seg = px.loc[peak_date:]
        trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
        loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
        if loss <= thresh:
            window = px.loc[peak_date:trough_date]
            others = {c: float(window[c].iloc[-1] / window[c].iloc[0] - 1.0)
                      for c in returns.columns if c != stock}
            episodes.append({"peak": peak_date, "trough": trough_date,
                             "stock_loss": loss, "others": others})
    return episodes


def window_return(returns: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Total return of every column over a calendar window [start, end] (inclusive)."""
    seg = returns.loc[start:end]
    px = (1.0 + seg).cumprod()
    return {c: float(px[c].iloc[-1] - 1.0) for c in seg.columns}


# ---------------------------------------------------------------------------
# Inference: HAC t-stat and circular block bootstrap on a difference series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _annualised_sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_sharpe_diff(
    a: pd.Series,
    b: pd.Series,
    rf: pd.Series | None = None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 97,
) -> dict:
    """Circular block bootstrap CI for the Sharpe *difference* (arm ``a`` − arm ``b``).

    Resamples the two aligned excess-return series **jointly** in circular blocks (so the
    cross-correlation and the volatility clustering both survive), recomputes each arm's
    annualised Sharpe per resample, and returns the point difference with a 95% CI and the
    bootstrap fraction of resamples in which ``a``'s Sharpe exceeds ``b``'s.
    """
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f
    n = len(idx)
    point = _annualised_sharpe(ra) - _annualised_sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    a_wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _annualised_sharpe(ra[sel]), _annualised_sharpe(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            a_wins += 1
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": float(point),
        "ci95": (float(lo), float(hi)),
        "frac_a_wins": a_wins / n_boot,
        "n": n, "block": block, "n_boot": n_boot,
    }
