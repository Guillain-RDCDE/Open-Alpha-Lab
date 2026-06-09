"""The teardown that earns the stamps — does the pair *stay* cointegrated, and is it real or selection?

Two legs, two ways a pairs trade dies:

  1. :func:`in_sample_vs_oos` — split the history in half. A genuinely cointegrated pair works in *both*
     halves; a pair whose relationship was a lucky in-sample fit breaks out of sample. The gap between
     the first-half and second-half Sharpe is how hard the tether snaps.
  2. :func:`spurious_pairs` — the selection trap. Among many *independent* random walks (no real
     relationship at all), a fraction will look "cointegrated" — a short spread half-life — purely by
     chance. If you scan a universe and trade the best-looking pairs, you are mostly harvesting that
     luck, and it evaporates live. This quantifies the false-positive rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spread import half_life, spread
from .strategy import pairs_returns, summary

TRADING_DAYS_PER_YEAR = 252


def in_sample_vs_oos(a: pd.Series, b: pd.Series, cost_bps: float = 2.0,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """First-half vs second-half Sharpe of the (causal) pairs book — does the relationship persist?

    A real, stable pair shows a similar Sharpe in both halves; a fragile or spurious one shows a strong
    first half and a collapsing second. Returns both Sharpes and the drop.
    """
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    mid = len(a) // 2
    a1, b1, a2, b2 = a.iloc[:mid], b.iloc[:mid], a.iloc[mid:], b.iloc[mid:]
    s1 = summary(pairs_returns(a1, b1, cost_bps=cost_bps, **kw), periods_per_year)["sharpe"]
    s2 = summary(pairs_returns(a2, b2, cost_bps=cost_bps, **kw), periods_per_year)["sharpe"]
    return {
        "first_half_sharpe": float(s1),
        "second_half_sharpe": float(s2),
        "sharpe_drop": float(s1 - s2),
        "survives_oos": bool(s2 > 0.3),
    }


def spurious_pairs(n_series: int = 20, n_bars: int = 2000, hl_threshold: float = 60.0,
                   seed: int = 0) -> dict:
    """False-positive rate: among ``n_series`` *independent* random walks, how many pairs look cointegrated?

    Builds independent random-walk log-prices (no real relationship), forms every pair, and counts the
    fraction whose spread half-life falls below ``hl_threshold`` days — i.e. looks tradably mean-reverting
    *by chance*. A high fraction means a universe scan will surface mostly spurious "pairs".
    """
    rng = np.random.default_rng(seed)
    series = []
    idx = pd.bdate_range(start="2010-01-04", periods=n_bars, name="date")
    for i in range(n_series):
        p = np.exp(np.cumsum(0.01 * rng.standard_normal(n_bars)) + 4.0)
        series.append(pd.Series(p, index=idx, name=f"S{i}"))
    n_pass, n_total = 0, 0
    hls = []
    for i in range(n_series):
        for j in range(i + 1, n_series):
            hl = half_life(spread(series[i], series[j]))
            hls.append(hl if np.isfinite(hl) else np.nan)
            n_total += 1
            if np.isfinite(hl) and hl < hl_threshold:
                n_pass += 1
    return {
        "n_pairs": int(n_total),
        "n_look_cointegrated": int(n_pass),
        "false_positive_rate": float(n_pass / n_total) if n_total else np.nan,
        "median_half_life": float(np.nanmedian(hls)) if hls else np.nan,
    }
