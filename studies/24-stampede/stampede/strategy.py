"""The investable book — long the winners, short the losers (WML), rebalanced monthly.

The classic momentum factor: each month, rank the cross-section by 12-1 return, go **long the top
decile** (winners) and **short the bottom decile** (losers), equal-weight, dollar-neutral. Two reads
matter: the dollar-neutral **WML** spread (the academic factor) and the **long-only winners** book a
real-money investor who can't short would actually run. The benchmark is the equal-weight universe.

Momentum turns over fast (the winners list churns), so costs bite — but the headline catch is not cost,
it is the **crash**: WML is short the losers, so when beaten-down names violently rebound (a bear-market
bottom), the short leg explodes against you. :mod:`decompose` and :mod:`extension` price that tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import momentum_score

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "ann_return": float(mean * periods_per_year),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "skew": float(r.skew()),
        "n_days": int(len(r)),
    }


def _decile_weights(panel, lookback, skip, rebal, decile_frac):
    """Equal-weight winner (+) and loser (−) decile weights, refreshed every ``rebal`` days, lagged."""
    score = momentum_score(panel, lookback, skip)
    cols = panel.columns
    n, m = panel.shape
    w_win = np.zeros((n, m)); w_los = np.zeros((n, m))
    for p in range(lookback, n, rebal):
        s = score.iloc[p].dropna()
        if len(s) < max(10, int(2 / decile_frac)):
            continue
        k = max(1, int(len(s) * decile_frac))
        order = s.sort_values()
        los, win = order.index[:k], order.index[-k:]
        end = min(p + rebal, n)
        wi = cols.get_indexer(win); li = cols.get_indexer(los)
        w_win[p:end, wi] = 1.0 / k
        w_los[p:end, li] = 1.0 / k
    return (pd.DataFrame(w_win, index=panel.index, columns=cols),
            pd.DataFrame(w_los, index=panel.index, columns=cols))


def books(panel: pd.DataFrame, lookback: int = 252, skip: int = 21, rebal: int = 21,
          decile_frac: float = 0.1, cost_bps: float = 5.0) -> dict:
    """Return the daily net streams: winners (long-only), losers, WML (dollar-neutral), and market."""
    w_win, w_los = _decile_weights(panel, lookback, skip, rebal, decile_frac)
    filled = panel.fillna(0.0)
    win = (w_win.shift(1) * filled).sum(axis=1)
    los = (w_los.shift(1) * filled).sum(axis=1)
    c = cost_bps * 1e-4
    cost_win = c * w_win.diff().abs().sum(axis=1)
    cost_los = c * w_los.diff().abs().sum(axis=1)
    wml = (win - los - cost_win - cost_los)
    active = w_win.abs().sum(axis=1) > 0
    return {
        "winners": (win - cost_win)[active].rename("winners"),
        "losers": los[active].rename("losers"),
        "wml": wml[active].rename("wml"),
        "market": panel.mean(axis=1)[active].rename("market"),
        "w_win": w_win, "w_los": w_los,
    }


def turnover_ann(panel: pd.DataFrame, **kw) -> float:
    w_win, _ = _decile_weights(panel, kw.get("lookback", 252), kw.get("skip", 21),
                               kw.get("rebal", 21), kw.get("decile_frac", 0.1))
    return float(w_win.diff().abs().sum(axis=1).mean() * TRADING_DAYS_PER_YEAR)


def compare(panel: pd.DataFrame, cost_bps: float = 5.0, periods_per_year: int = TRADING_DAYS_PER_YEAR,
            **kw) -> dict:
    """WML and long-only winners vs the equal-weight market — the scoreboard."""
    b = books(panel, cost_bps=cost_bps, **kw)
    return {
        "wml": summary(b["wml"], periods_per_year),
        "winners": summary(b["winners"], periods_per_year),
        "market": summary(b["market"], periods_per_year),
        "turnover_ann": float(turnover_ann(panel, **kw)),
        "n_days": int(summary(b["wml"], periods_per_year)["n_days"]),
    }


def cost_sweep(panel: pd.DataFrame, roundtrip_bps=(0, 5, 10, 20, 40), periods_per_year=TRADING_DAYS_PER_YEAR, **kw):
    rows = {}
    for c in roundtrip_bps:
        s = summary(books(panel, cost_bps=c, **kw)["wml"], periods_per_year)
        rows[c] = {"wml_sharpe": s["sharpe"], "wml_ann_return": s["ann_return"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out
