"""Worked extension (beat 7) — the crisis-convexity test: does trend pay when the basket crashes?

A trend book's headline Sharpe is only half its story. The reason real allocators hold managed futures
despite a modest standalone Sharpe is **convexity**: trend tends to be *short* a falling market by the
time the fall is under way, so it makes money in exactly the months a long-only basket bleeds — a
"long volatility" / crisis-alpha profile (Fung & Hsieh; Hutchinson & O'Brien). That diversification can
be worth more than the standalone return.

So we make the caveat a measurement. :func:`crisis_convexity` splits months into the basket's *worst*
decile and the rest, and compares the TSMOM return in each — a real diversifier earns its keep (or at
least doesn't bleed) when the basket is down hardest. :func:`down_market_capture` reports the average
TSMOM return in down-basket months. Two baked checks pin it: the null tape shows no convexity, the
trend tape shows TSMOM holding up (or gaining) when the basket falls.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import tsmom_returns, long_only_basket, summary

TRADING_DAYS_PER_YEAR = 252


def _monthly(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).resample("ME").prod() - 1.0


def crisis_convexity(panel: pd.DataFrame, cost_bps: float = 2.0, worst_decile: float = 0.1, **kw) -> dict:
    """Average TSMOM monthly return in the basket's worst months vs the rest — the convexity read.

    Resamples both books to months, flags the basket's worst ``worst_decile`` of months, and compares
    the TSMOM average return inside vs outside that set. A positive (or merely non-negative) crisis-
    month return against a deeply negative basket is the diversification trend is bought for.
    """
    tsm = tsmom_returns(panel, cost_bps=cost_bps, **kw)
    bench = long_only_basket(panel).reindex(tsm.index)
    mt, mb = _monthly(tsm).dropna(), _monthly(bench).dropna()
    common = mt.index.intersection(mb.index)
    mt, mb = mt.loc[common], mb.loc[common]
    thresh = mb.quantile(worst_decile)
    crisis = mb <= thresh
    return {
        "tsmom_crisis_mean_pct": float(mt[crisis].mean() * 100.0),
        "basket_crisis_mean_pct": float(mb[crisis].mean() * 100.0),
        "tsmom_calm_mean_pct": float(mt[~crisis].mean() * 100.0),
        "basket_calm_mean_pct": float(mb[~crisis].mean() * 100.0),
        "n_crisis_months": int(crisis.sum()),
        "convexity_pp": float((mt[crisis].mean() - mt[~crisis].mean()) * 100.0),
    }


def down_market_capture(panel: pd.DataFrame, cost_bps: float = 2.0, **kw) -> dict:
    """Average TSMOM monthly return in *all* down-basket months — a simpler diversification read."""
    tsm = tsmom_returns(panel, cost_bps=cost_bps, **kw)
    bench = long_only_basket(panel).reindex(tsm.index)
    mt, mb = _monthly(tsm).dropna(), _monthly(bench).dropna()
    common = mt.index.intersection(mb.index)
    mt, mb = mt.loc[common], mb.loc[common]
    down = mb < 0
    return {
        "tsmom_in_down_months_pct": float(mt[down].mean() * 100.0),
        "basket_in_down_months_pct": float(mb[down].mean() * 100.0),
        "n_down_months": int(down.sum()),
    }
