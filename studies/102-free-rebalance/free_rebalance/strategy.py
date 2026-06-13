"""The rebalanced-vs-drift engine and the diversification-return decomposition — Study 102.

The folk claim, steelmanned: *'rebalancing is a free lunch — the rebalancing bonus
(a.k.a. volatility harvesting / diversification return) mechanically ADDS return on top
of the risk control.'* (Fernholz & Shay 1982; Booth & Fama 1992; the 'Shannon's Demon'
framing.) We implement two portfolios from the *same* fixed initial weights and pin one
against the other:

- **Rebalanced** — held back to the fixed target weights on a schedule (annual or
  quarterly), paying a one-way cost on the traded turnover at each rebalance.
- **Drift (buy-and-hold)** — the same initial weights left to drift; the winners grow
  their share, the losers shrink. No trading after inception, so no rebalancing cost.

The **bonus** is (rebalanced CAGR) − (drift CAGR). We also report the classic
*diversification-return identity* (Booth-Fama / Willenbrock), which states that a
**continuously** rebalanced portfolio's geometric return exceeds the weighted average of
the assets' geometric returns by

    DR = 0.5 * ( sum_i w_i sigma_i^2  -  sigma_p^2 )

i.e. half the *reduction in variance* rebalancing buys. This is an exact identity for the
*continuously rebalanced* portfolio vs the *weighted-average-of-assets* baseline — it is
**always >= 0** by construction. The honest point of this study is that the identity's
baseline is NOT the drift portfolio: a drift portfolio also earns a diversification
benefit (and lets winners run), so the *realised* bonus over drift can be small, regime
-dependent, and **negative** when one asset persistently trends. We show both numbers.

Costs: turnover is one-way x NAV (the desk convention), charged once per rebalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------
def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """The set of dates on which we rebalance back to target.

    ``freq`` in {'A','annual','Q','quarterly','none'/'drift'}: the *last trading day*
    of each calendar year (annual) or quarter (quarterly). 'none' / 'drift' returns an
    empty set (pure buy-and-hold). The schedule is calendar-known — no look-ahead.
    """
    f = freq.lower()
    if f in ("none", "drift", "buy_and_hold", "bh"):
        return set()
    if f in ("a", "annual", "y", "yearly"):
        key = index.year.to_numpy()
    elif f in ("q", "quarterly"):
        key = index.year.to_numpy() * 4 + (index.quarter.to_numpy() - 1)
    else:
        raise ValueError(f"unknown freq {freq!r}")
    s = pd.Series(index=np.arange(len(index)), data=index)
    return set(s.groupby(key).last().tolist())


# ---------------------------------------------------------------------------
# Portfolio engines — rebalanced and drift, from the same initial weights
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def run_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray | None = None,
    freq: str = "annual",
    cost_bps: float = 10.0,
) -> dict:
    """Simulate a fixed-weight portfolio rebalanced on ``freq`` (or left to drift).

    Starting from target ``weights`` (default equal), the per-asset holdings drift with
    daily total returns. On each rebalance date the book is traded back to target,
    charging ``cost_bps`` (one-way) on the traded fraction of NAV. ``freq='drift'``
    rebalances never — buy-and-hold of the initial weights.

    Returns a dict with the net daily portfolio return series, the equity curve, and
    headline stats (CAGR, vol, Sharpe, max drawdown, number of rebalances, total cost
    drag in pts of final NAV).
    """
    px = prices.dropna(how="any")
    rets = px.pct_change().fillna(0.0)
    n_assets = px.shape[1]
    if weights is None:
        weights = np.full(n_assets, 1.0 / n_assets)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    rb = rebalance_dates(px.index, freq)
    # Holdings expressed as a value vector summing to NAV; start at NAV = 1.
    nav = 1.0
    hold = weights * nav
    daily_ret = np.empty(len(px))
    equity = np.empty(len(px))
    n_rebal = 0
    total_cost = 0.0
    r = rets.to_numpy()
    idx = px.index
    for t in range(len(px)):
        if t > 0:
            hold = hold * (1.0 + r[t])  # grow each leg by its return
        new_nav = hold.sum()
        daily_ret[t] = new_nav / nav - 1.0 if nav > 0 else 0.0
        nav = new_nav
        if idx[t] in rb:
            target = weights * nav
            turnover = np.abs(target - hold).sum() / nav  # one-way x NAV
            cost = turnover * cost_bps * 1e-4
            nav *= 1.0 - cost
            total_cost += cost
            # net the cost back into the day's return, then reset to target
            daily_ret[t] = (1.0 + daily_ret[t]) * (1.0 - cost) - 1.0
            hold = weights * nav
            n_rebal += 1
        equity[t] = nav

    net = pd.Series(daily_ret, index=px.index)
    return _stats(net, n_rebal, total_cost, equity)


def _stats(net: pd.Series, n_rebal: int, total_cost: float, equity: np.ndarray) -> dict:
    net = net.astype(float)
    eq = pd.Series(equity, index=net.index)
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if net.std() > 0
        else float("nan")
    )
    return {
        "net": net,
        "equity": eq,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity),
        "n_rebal": n_rebal,
        "cost_drag": float(total_cost),
        "final": float(eq.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# The bonus and its decomposition
# ---------------------------------------------------------------------------
def rebalancing_bonus(
    prices: pd.DataFrame,
    weights: np.ndarray | None = None,
    freq: str = "annual",
    cost_bps: float = 10.0,
) -> dict:
    """The realised rebalancing bonus (rebalanced CAGR - drift CAGR) and the arms.

    Returns ``{'rebalanced':..., 'drift':..., 'bonus_cagr':..., 'bonus_sharpe':...}``
    where ``bonus_cagr`` is the headline 'free lunch' number — positive only if
    rebalancing actually *added* return net of cost over simply holding the basket.
    """
    reb = run_portfolio(prices, weights, freq=freq, cost_bps=cost_bps)
    dri = run_portfolio(prices, weights, freq="drift", cost_bps=cost_bps)
    return {
        "rebalanced": reb,
        "drift": dri,
        "bonus_cagr": reb["cagr"] - dri["cagr"],
        "bonus_sharpe": reb["sharpe"] - dri["sharpe"],
        "dd_diff": reb["max_dd"] - dri["max_dd"],
        "vol_diff": reb["vol"] - dri["vol"],
    }


def diversification_return(
    prices: pd.DataFrame, weights: np.ndarray | None = None
) -> dict:
    """The Booth-Fama / Willenbrock diversification-return identity (exact, parameter-free).

    For a **continuously** rebalanced fixed-weight portfolio, the geometric return exceeds
    the weighted average of the assets' own geometric returns by

        DR = 0.5 * ( sum_i w_i * var_i  -  var_p )

    where ``var_i`` is asset i's daily-return variance, ``var_p`` the variance of the
    fixed-weight portfolio's daily return, both annualised. DR is **>= 0 by
    construction** (rebalancing reduces variance). This is the textbook 'bonus' — but its
    baseline is the *weighted average of constituents*, NOT the drift portfolio, which is
    why the *realised* bonus over drift (see :func:`rebalancing_bonus`) can be smaller or
    negative. Returns the annualised DR and its pieces.
    """
    px = prices.dropna(how="any")
    rets = px.pct_change().dropna()
    n_assets = px.shape[1]
    if weights is None:
        weights = np.full(n_assets, 1.0 / n_assets)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    cov = rets.cov().to_numpy() * TRADING_DAYS  # annualised covariance
    var_i = np.diag(cov)
    var_p = float(weights @ cov @ weights)
    wavg_var = float(weights @ var_i)
    dr = 0.5 * (wavg_var - var_p)
    return {
        "dr_annual": float(dr),
        "wavg_var": wavg_var,
        "var_p": var_p,
        "var_reduction": wavg_var - var_p,
    }


# ---------------------------------------------------------------------------
# Inference: HAC t-stat and a circular block bootstrap CI for a mean
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


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 63,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 102,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of a daily series ``x``.

    Resamples overlapping blocks of length ``block`` (i.i.d. resampling would destroy the
    autocorrelation the inference must respect) and returns the ``(1-alpha)`` percentile
    interval of the bootstrap mean. Default ``block=63`` (~one quarter of trading days),
    matched to the rebalancing horizon. Used to put a CI on the *daily bonus* series so
    the sign claim carries uncertainty.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        sample = x[idx.ravel()][:n]
        means[b] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi
