"""The investable book — time-series momentum, sized by inverse vol and rebalanced slowly.

The §10.4 rule, one line per asset::

    w_i = gamma * sign(R_i^T) / sigma_i,    sum_i |w_i| = 1

``R_i^T`` is the trailing-T return (the trend), ``sign`` makes the bet directional (long an uptrend,
short a downtrend), ``1/sigma_i`` equalises risk across assets, and ``gamma`` normalises gross exposure
to 1. Crucially the signal is *slow* — a months-long return sign — so the book turns over a handful of
times a year, the opposite of Study 19's daily churn. That low turnover is why, if the trend is real,
costs *don't* kill this one.

The benchmark is the **equal-weight long-only basket**: trend-following has to beat simply *holding*
the diversified menu to be worth the shorting and the timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import trailing_return, realized_vol

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline stats for a return stream: Sharpe is the number that matters; maxDD the one you feel."""
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
        "n_days": int(len(r)),
    }


def tsmom_weights(
    panel: pd.DataFrame,
    lookback: int = 252,
    vol_window: int = 63,
    rebal: int = 21,
) -> pd.DataFrame:
    """Past-only TSMOM weights ``w_i = sign(R_i^T)/sigma_i`` (gross-normalised), held between rebalances.

    Signal and vol are both computed from data up to the rebalance and **lagged one day**, so the
    weight set at a rebalance earns the *following* days' returns with no look-ahead. Weights are
    refreshed every ``rebal`` days and held flat in between (the slow turnover that makes the book
    cheap to run).
    """
    sig = np.sign(trailing_return(panel, lookback))
    inv_vol = 1.0 / realized_vol(panel, vol_window).replace(0.0, np.nan)
    raw = (sig * inv_vol)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    w = raw.div(gross, axis=0)                                   # sum |w| = 1 each day

    # refresh only on rebalance days, hold flat in between, then lag one day
    idx = np.arange(len(w))
    keep = (idx % rebal == 0)
    w_reb = w.where(pd.Series(keep, index=w.index), other=np.nan).ffill()
    return w_reb.shift(1)


def tsmom_returns(
    panel: pd.DataFrame,
    cost_bps: float = 2.0,
    lookback: int = 252,
    vol_window: int = 63,
    rebal: int = 21,
) -> pd.Series:
    """The TSMOM book's daily return stream, net of ``cost_bps`` charged on rebalancing turnover."""
    w = tsmom_weights(panel, lookback, vol_window, rebal)
    gross = (w * panel).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).dropna().rename("tsmom")


def long_only_basket(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight long-only basket return — the benchmark trend-following must beat."""
    return panel.mean(axis=1).rename("basket")


def turnover_ann(panel: pd.DataFrame, **kw) -> float:
    """Annualised turnover of the TSMOM book — low, because the trend signal moves slowly."""
    w = tsmom_weights(panel, **kw)
    return float(w.diff().abs().sum(axis=1).mean() * TRADING_DAYS_PER_YEAR)


def compare(panel: pd.DataFrame, cost_bps: float = 2.0, periods_per_year: int = TRADING_DAYS_PER_YEAR,
            **kw) -> dict:
    """TSMOM (net) vs the equal-weight long-only basket — the scoreboard."""
    tsm = tsmom_returns(panel, cost_bps=cost_bps, **kw)
    bench = long_only_basket(panel).reindex(tsm.index)
    s_t, s_b = summary(tsm, periods_per_year), summary(bench, periods_per_year)
    return {
        "tsmom": s_t,
        "basket": s_b,
        "sharpe_gain": float(s_t["sharpe"] - s_b["sharpe"]),
        "turnover_ann": float(turnover_ann(panel, **kw)),
        "n_days": int(s_t["n_days"]),
    }


def cost_sweep(panel: pd.DataFrame, roundtrip_bps=(0, 2, 5, 10, 20, 40),
               periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> pd.DataFrame:
    """Net TSMOM Sharpe as cost per unit traded rises — flat-ish, because turnover is low.

    The mirror image of Study 19's cost cliff: here the slow signal means the break-even cost is *far*
    above any realistic level, so costs are not what threatens this strategy (decay is).
    """
    rows = {}
    for c in roundtrip_bps:
        s = summary(tsmom_returns(panel, cost_bps=c, **kw), periods_per_year)
        rows[c] = {"sharpe": s["sharpe"], "ann_return": s["ann_return"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out
