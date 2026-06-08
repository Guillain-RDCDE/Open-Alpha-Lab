"""Pair formation — the GGR (1999) rule, exactly.

Gatev, Goetzmann & Rouwenhorst form pairs with a single, parameter-free yardstick:
over a 12-month **formation** window, build each name's *normalized price* — a total
return index that starts at 1 — and measure the **sum of squared deviations** (SSD)
between every eligible pair of normalized series. The smallest SSD = the two paths that
hugged each other tightest. Take the top-N by SSD; those are the twins you'll trade in
the following window. No regression, no cointegration test, no fitted parameter — that
parameter-freeness is precisely why the result is hard to dismiss as data-mining.

The one number carried out of formation into trading is each pair's **spread standard
deviation** ``sigma`` (the std of ``norm_a − norm_b`` over the formation window). The
trading rule (see :mod:`pairs_trading.backtest`) opens when the live spread diverges by
more than ``k·sigma`` and closes when it crosses back through zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pair:
    """A formed pair, with everything trading needs carried from the formation window."""
    a: str
    b: str
    ssd: float          # sum of squared deviations of the normalized prices (the rank key)
    sigma: float        # std of the (norm_a - norm_b) spread over formation -> the trigger unit


def normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Total-return index per column, re-based to 1.0 at the first row of ``prices``.

    ``norm = cumprod(1 + r)`` from the window's first session, so two names that earned
    the *same* cumulative return trace the *same* path regardless of price level — the
    GGR normalization that makes a \\$20 stock and a \\$200 stock directly comparable.
    """
    rets = prices.pct_change().fillna(0.0)
    return (1.0 + rets).cumprod()


def _eligible(prices: pd.DataFrame, min_obs: int) -> list[str]:
    """Names fully populated across the formation window (no pre-IPO / halted blanks)."""
    good = prices.notna().sum(axis=0)
    return [c for c in prices.columns if good[c] >= min_obs and prices[c].notna().all()]


def select_pairs(
    formation_prices: pd.DataFrame,
    top_n: int = 20,
    min_obs: int | None = None,
) -> list[Pair]:
    """Rank every eligible pair by formation SSD and return the ``top_n`` tightest.

    ``formation_prices`` is the close panel sliced to the formation window. Only names
    populated across the whole window are eligible (a partial series would forge a
    spurious match). Returns pairs sorted by ascending SSD — the GGR "minimum-distance"
    ordering — each carrying the spread ``sigma`` the trader will trigger on.
    """
    n_rows = len(formation_prices)
    if min_obs is None:
        min_obs = n_rows
    cols = _eligible(formation_prices, min_obs)
    if len(cols) < 2:
        return []

    norm = normalized_prices(formation_prices[cols])
    mat = norm.to_numpy()                      # (T, M) normalized paths
    m = mat.shape[1]

    pairs: list[Pair] = []
    for i in range(m):
        # Vectorize the inner loop: diffs of column i against all j > i at once.
        if i + 1 >= m:
            break
        diff = mat[:, i + 1:] - mat[:, i:i + 1]          # (T, m-i-1)
        ssd = np.einsum("tj,tj->j", diff, diff)          # sum of squared deviations
        sigma = diff.std(axis=0, ddof=1)
        for off, j in enumerate(range(i + 1, m)):
            pairs.append(Pair(cols[i], cols[j], float(ssd[off]), float(sigma[off])))

    pairs.sort(key=lambda p: p.ssd)
    return pairs[:top_n]
