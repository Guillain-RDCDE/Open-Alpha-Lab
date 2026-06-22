"""Strategy + inference for Study 357 — the Coffee-Can portfolio.

The parable: equal-weight a basket of quality names, then **never trade** for a decade. We
compare three things on the same monthly total-return tape:

* **Coffee-can (zero turnover).** Buy equal-weight once, then let the weights drift forever.
  No rebalancing, no costs after the single entry. This is Kirby's "can".
* **Rebalanced.** The same basket, reset to equal weight every ``rebal`` months — and charged a
  realistic one-way cost on the turnover that reset implies (the part the can avoids).
* **Index.** SPY total return — the active manager's benchmark.

Two honesty layers, both first-class on the verdict:

1. **Cost / turnover (real, defensible).** The cost model charges ``cost_bps`` one-way on the
   traded fraction of NAV. The can pays it once; the rebalanced book pays it every reset. That
   gap is the *legitimate* coffee-can tailwind.
2. **Survivorship (look-ahead, the catch).** The basket's names are chosen with hindsight. We
   quantify it on the synthetic panel via :func:`survivorship_gap`: the SAME selection rule run
   look-ahead (famous survivors) vs ex-ante (honest) on a universe that still contains the dead
   names. The gap is the fake edge.

Inference: a **paired t-test** of the can's monthly excess return over the index (Newey-West /
Lo-style SE optional via ``hac_lag``) — the Signal-axis statistic. Pure numpy / pandas; an
optional scipy import is used only for the t-distribution p-value and degrades gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = 12


# --------------------------------------------------------------------------- #
# Portfolio engines
# --------------------------------------------------------------------------- #
def buy_and_hold(returns: pd.DataFrame, weights=None) -> pd.Series:
    """Zero-turnover equal-weight buy-and-hold: weights drift with the holdings forever.

    ``returns`` is an ``(T, k)`` monthly simple-return frame for the basket. We seed equal
    weights, then evolve each name's weight by its own gross return (a coffee can never
    rebalances). Returns the monthly portfolio simple-return series."""
    R = returns.to_numpy()
    T, k = R.shape
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    out = np.empty(T)
    for t in range(T):
        gross = 1.0 + R[t]
        port = float(w @ gross)         # portfolio gross return this month
        out[t] = port - 1.0
        w = w * gross / port            # drift the weights (no trade)
    return pd.Series(out, index=returns.index, name="coffee_can")


def rebalanced(returns: pd.DataFrame, rebal: int = 12, cost_bps: float = 10.0,
               weights=None) -> pd.Series:
    """Equal-weight basket reset to target every ``rebal`` months, charged a one-way cost.

    At each reset we pay ``cost_bps`` (one-way) on the *traded fraction of NAV* — the L1 weight
    change needed to return to equal weight. Between resets the weights drift like the can.
    Returns the monthly NET simple-return series."""
    R = returns.to_numpy()
    T, k = R.shape
    tgt = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, float) / np.sum(weights)
    w = tgt.copy()
    c = cost_bps / 1e4
    out = np.empty(T)
    for t in range(T):
        gross = 1.0 + R[t]
        port = float(w @ gross)
        w = w * gross / port            # drift
        ret = port - 1.0
        if (t + 1) % rebal == 0:        # reset at the end of this month
            turn = float(np.abs(w - tgt).sum())   # one-way traded fraction of NAV
            ret -= c * turn
            w = tgt.copy()
        out[t] = ret
    return pd.Series(out, index=returns.index, name=f"rebal_{rebal}m")


def entry_cost(k: int, cost_bps: float = 10.0) -> float:
    """One-off cost of building an equal-weight basket of ``k`` names (one-way, on full NAV)."""
    return cost_bps / 1e4 * 1.0       # buy 100% of NAV once


# --------------------------------------------------------------------------- #
# Summary stats
# --------------------------------------------------------------------------- #
def cagr(rets: pd.Series) -> float:
    g = float(np.prod(1.0 + rets.to_numpy()))
    yrs = len(rets) / TRADING_MONTHS
    return g ** (1.0 / yrs) - 1.0 if yrs > 0 else float("nan")


def ann_vol(rets: pd.Series) -> float:
    return float(rets.std(ddof=1) * np.sqrt(TRADING_MONTHS))


def sharpe(rets: pd.Series, rf_ann: float = 0.0) -> float:
    ex = rets - rf_ann / TRADING_MONTHS
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_MONTHS)) if sd > 0 else float("nan")


def max_drawdown(rets: pd.Series) -> float:
    eq = (1.0 + rets).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def summary(rets: pd.Series) -> dict:
    return {"cagr": cagr(rets), "vol": ann_vol(rets),
            "sharpe": sharpe(rets), "mdd": max_drawdown(rets), "n": int(len(rets))}


# --------------------------------------------------------------------------- #
# Inference — paired t-test of the can's excess over the index
# --------------------------------------------------------------------------- #
def _nw_se(x: np.ndarray, lag: int) -> float:
    """Newey-West (HAC) standard error of the mean of ``x`` with ``lag`` lags."""
    x = np.asarray(x, float)
    n = len(x)
    xc = x - x.mean()
    g0 = float(xc @ xc) / n
    s = g0
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        cov = float(xc[L:] @ xc[:-L]) / n
        s += 2.0 * w * cov
    return float(np.sqrt(s / n))


def paired_t(strat: pd.Series, bench: pd.Series, hac_lag: int = 0) -> dict:
    """Paired t-test that the strategy's mean monthly EXCESS return over ``bench`` is > 0.

    ``hac_lag = 0`` is the plain paired-t (i.i.d. SE); ``hac_lag > 0`` uses a Newey-West SE
    (the autocorrelation-robust statistic the Signal axis requires). Returns the mean monthly
    excess (annualised too), the t-stat, n, and a two-sided p-value (scipy if present)."""
    d = (strat - bench).dropna().to_numpy()
    n = len(d)
    mean = float(d.mean())
    se = _nw_se(d, hac_lag) if hac_lag > 0 else float(d.std(ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else float("nan")
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), df=n - 1))
    except Exception:                              # pragma: no cover - scipy optional
        p = float("nan")
    return {"mean_m": mean, "mean_ann": mean * TRADING_MONTHS, "t": float(t),
            "se": se, "n": n, "p": p, "hac_lag": hac_lag}


# --------------------------------------------------------------------------- #
# Survivorship decomposition (on the synthetic panel)
# --------------------------------------------------------------------------- #
def basket_buy_and_hold(prices: pd.DataFrame, names: list[str]) -> pd.Series:
    """Buy-and-hold a named sub-basket on a price panel that may contain delistings.

    A dead name's price becomes NaN; once dead it contributes its last (crashed) value and then
    drops out — exactly what a real coffee-can holder would have eaten. We forward-fill the
    crashed terminal level so the wipe is realised, not silently erased."""
    sub = prices[names].copy()
    sub = sub.ffill()                              # hold the crashed value after delist
    rets = sub.pct_change().iloc[1:].fillna(0.0)
    return buy_and_hold(rets)


def survivorship_gap(prices: pd.DataFrame, k: int = 10, lookback: int = 24) -> dict:
    """The fake edge: look-ahead 'famous' basket minus honest ex-ante basket, same universe.

    On a panel where every firm shares the *same* true drift, any out-performance of the
    famous-survivor basket over the honest basket (and over the equal-weight whole universe) is
    pure survivorship. Returns CAGRs and the gap."""
    from . import data
    famous = data.famous_names(prices, k=k)
    honest = data.honest_names(prices, k=k, lookback=lookback)
    all_names = list(prices.columns)

    can_famous = basket_buy_and_hold(prices, famous)
    can_honest = basket_buy_and_hold(prices, honest)
    can_all = basket_buy_and_hold(prices, all_names)

    cf, ch, ca = cagr(can_famous), cagr(can_honest), cagr(can_all)
    return {"cagr_famous": cf, "cagr_honest": ch, "cagr_all": ca,
            "gap_vs_honest": cf - ch, "gap_vs_all": cf - ca,
            "famous": famous, "honest": honest}
