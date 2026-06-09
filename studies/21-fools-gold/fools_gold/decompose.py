"""The teardown that earns the stamps — is the crossover *informative*, or just less exposure?

Three legs:

  1. :func:`spread_tstat` — the Newey-West t-stat that golden-state days really out-return death-state
     days. This is the direct test of the folk claim; a small/insignificant t says the crossover event
     carries no information beyond the drift the asset already had.
  2. :func:`vs_buy_hold` — regress the timing stream on buy-and-hold. A timing book that's out of the
     market half the time has a **beta below 1**, so its lower vol and drawdown are mostly *less
     exposure*, not skill; the **alpha** (and its HAC t) is the part that isn't just a smaller bet.
  3. :func:`risk_matched` — give buy-and-hold the *same average exposure* as the timing book (scale it
     by the timing book's average weight) and compare. If the crossover adds nothing, a constant lower
     exposure matches or beats it — the honest "it's just a cash blend" control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import timing_returns, buy_hold, timing_weights, summary
from .cross import cross_state

TRADING_DAYS_PER_YEAR = 252


def _ols_nw(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    y = np.asarray(y, float); x = np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x]); n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ (X.T @ y); resid = y - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * resid[:, None]; M = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0); G = u[k:].T @ u[:-k]; M += w * (G + G.T)
    cov = XtX_inv @ M @ XtX_inv; se = np.sqrt(np.diag(cov))
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "beta_t": float(beta[1] / se[1]) if se[1] > 0 else np.nan, "lags": int(lags), "n": int(n)}


def spread_tstat(close: pd.Series, fast: int = 50, slow: int = 200,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Newey-West t that golden-state next-day returns exceed death-state ones (the folk claim itself).

    Builds the per-day series ``state * future_return`` over the days that have a defined state, and
    HAC-tests its mean. A positive, significant value means being golden really does predict a higher
    return; a flat result means the crossover is uninformative.
    """
    state = cross_state(close, fast, slow).shift(1)
    fwd = close.pct_change()
    df = pd.DataFrame({"state": state, "fwd": fwd}).dropna()
    s = (df["state"] * df["fwd"]).to_numpy()              # +fwd in golden, -fwd in death
    n = s.size; mu = s.mean(); e = s - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e / n)
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0); lrv += 2.0 * wk * float(e[k:] @ e[:-k] / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"spread_ann_pct": float(mu * periods_per_year * 100.0),
            "t_stat": float(mu / se) if se > 0 else np.nan, "n": int(n)}


def vs_buy_hold(close: pd.Series, cost_bps: float = 2.0,
                periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Regress the timing stream on buy-and-hold: alpha (annualised, HAC t) and beta.

    A beta well below 1 is the tell that the book's calmer ride is *less exposure*; alpha is what's left
    once that's accounted for. The folk claim survives only if alpha > 0 with |t| > 2.
    """
    t = timing_returns(close, cost_bps=cost_bps, **kw)
    bh = buy_hold(close).reindex(t.index)
    reg = _ols_nw(t.to_numpy(), bh.to_numpy())
    reg["alpha_ann_pct"] = float(reg["alpha"] * periods_per_year * 100.0)
    return reg


def risk_matched(close: pd.Series, cost_bps: float = 2.0,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Compare the timing book to buy-and-hold scaled to the *same average exposure* (a cash blend).

    If the crossover adds no information, a static position at the timing book's average weight earns
    the same risk-adjusted return without any trading. Reports both Sharpes and the gap — the honest
    'is it more than a smaller constant bet?' control.
    """
    t = timing_returns(close, cost_bps=cost_bps, **kw)
    bh = buy_hold(close).reindex(t.index)
    avg_w = timing_weights(close, **kw).reindex(t.index).mean()
    blend = (avg_w * bh).rename("cash_blend")
    s_t, s_blend = summary(t, periods_per_year), summary(blend, periods_per_year)
    return {
        "avg_exposure": float(avg_w),
        "timing_sharpe": s_t["sharpe"],
        "cash_blend_sharpe": s_blend["sharpe"],
        "sharpe_edge_over_blend": float(s_t["sharpe"] - s_blend["sharpe"]),
        "timing_maxdd": s_t["max_drawdown"],
        "cash_blend_maxdd": s_blend["max_drawdown"],
    }
