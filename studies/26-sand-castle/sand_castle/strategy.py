"""The investable books — the 'optimal' mean-variance stat-arb portfolio, and the naive version.

We trade the reversion signal two ways, both **causal** (estimated from trailing data):

    * **optimized** — ``w ∝ C^{-1} E`` with the sample covariance ``C`` (the §3.18 recipe).
    * **naive** — ``w ∝ E`` (dollar-neutral, no covariance inversion).

The optimizer is supposed to do *better* — it accounts for correlations. The study's question is whether,
once ``C`` is *estimated* (noisily) and the book is *rebalanced daily* (huge turnover), the inversion
helps or hurts. The P&L is the dollar-neutral residual return ``w · ε_{t+1}``, net of cost on each day's
weight change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .statarb import residual_returns, reversion_signal, optimal_weights, naive_weights

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "ann_return": float(mean * periods_per_year), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "cagr": float(cagr), "max_drawdown": float(dd.min()), "n_days": int(len(r))}


def _run(panel, market, optimized, shrink, window, lookback, cost_bps):
    """Daily causal stat-arb loop. Returns (net daily return series, annualised turnover)."""
    res = residual_returns(panel, market)
    E = reversion_signal(res, lookback)
    R = res.to_numpy()
    Em = E.to_numpy()
    n, m = R.shape
    rets = np.full(n, np.nan)
    turn = []
    prev_full = np.zeros(m)
    start = max(window, lookback + 1)
    for t in range(start, n - 1):
        win = R[t - window:t, :]
        valid = ~np.isnan(win).any(axis=0) & ~np.isnan(Em[t]) & ~np.isnan(R[t + 1])
        if valid.sum() < 5:
            rets[t + 1] = 0.0; turn.append(0.0); continue
        Wv = win[:, valid]
        Ev = Em[t, valid]
        if optimized:
            C = np.cov(Wv, rowvar=False)
            wv = optimal_weights(Ev, C, shrink=shrink)
        else:
            wv = naive_weights(Ev)
        w_full = np.zeros(m); w_full[valid] = wv
        rets[t + 1] = float(np.nansum(w_full * R[t + 1]))
        turn.append(float(np.abs(w_full - prev_full).sum()))
        prev_full = w_full
    s = pd.Series(rets, index=panel.index).dropna()
    turnover_ann = float(np.mean(turn) * TRADING_DAYS_PER_YEAR) if turn else 0.0
    return s.rename("statarb"), turnover_ann


def statarb_returns(panel: pd.DataFrame, market: pd.Series | None = None, optimized: bool = True,
                    shrink: float = 0.0, window: int = 252, lookback: int = 5, cost_bps: float = 5.0) -> pd.Series:
    """Net daily return of the (optimized or naive) stat-arb book, charged ``cost_bps`` per unit traded.

    Cost is the average per-day turnover charge (``cost_bps`` × annual turnover / 252) — a flat haircut
    consistent with the measured (very high) daily turnover; the point is that any realistic cost erases
    the gross edge, not the second decimal of the haircut.
    """
    gross, turn_ann = _run(panel, market, optimized, shrink, window, lookback, cost_bps)
    daily_cost = (cost_bps * 1e-4) * (turn_ann / TRADING_DAYS_PER_YEAR)
    return (gross - daily_cost).rename("statarb")


def turnover_ann(panel: pd.DataFrame, market: pd.Series | None = None, optimized: bool = True,
                 shrink: float = 0.0, window: int = 252, lookback: int = 5) -> float:
    return _run(panel, market, optimized, shrink, window, lookback, 0.0)[1]


def compare(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
            shrink: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Optimized vs naive stat-arb — Sharpe and turnover, gross and net."""
    opt_g, opt_turn = _run(panel, market, True, shrink, kw.get("window", 252), kw.get("lookback", 5), 0.0)
    nai_g, nai_turn = _run(panel, market, False, shrink, kw.get("window", 252), kw.get("lookback", 5), 0.0)
    c = cost_bps * 1e-4
    opt_net = opt_g - c * (opt_turn / periods_per_year)
    nai_net = nai_g - c * (nai_turn / periods_per_year)
    return {
        "optimized_gross": summary(opt_g, periods_per_year), "optimized_net": summary(opt_net, periods_per_year),
        "naive_gross": summary(nai_g, periods_per_year), "naive_net": summary(nai_net, periods_per_year),
        "optimized_turnover": float(opt_turn), "naive_turnover": float(nai_turn),
        "opt_minus_naive_net_sharpe": float(summary(opt_net, periods_per_year)["sharpe"] - summary(nai_net, periods_per_year)["sharpe"]),
    }
