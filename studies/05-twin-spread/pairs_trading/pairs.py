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


def df_tstat(spread: np.ndarray) -> float:
    """Dickey–Fuller t-statistic for a unit root in ``spread`` (constant, no lags).

    Regress Δsₜ = α + ρ·sₜ₋₁ + εₜ and return the t-stat of ρ̂. A strongly *negative*
    value rejects the random-walk null in favour of **mean reversion** — the economic
    anchor the raw minimum-SSD rule never checks. Dependency-free (no statsmodels): the
    desk keeps its own stats, like the Acklam normal in `robustness.py`. This is a plain
    DF, not an augmented ADF — adequate as a coarse cointegration gate, and named as such.
    """
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        return float("nan")
    s_lag = s[:-1]
    ds = np.diff(s)
    dof = len(ds) - 2
    if dof <= 0 or np.var(s_lag) <= 0:        # constant spread -> not a mean-reverting pair
        return float("nan")
    X = np.column_stack([np.ones_like(s_lag), s_lag])      # [const, level]
    beta, *_ = np.linalg.lstsq(X, ds, rcond=None)
    resid = ds - X @ beta
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float("nan")
    se_rho = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    return float(beta[1] / se_rho) if se_rho > 0 else float("nan")


def select_pairs(
    formation_prices: pd.DataFrame,
    top_n: int = 20,
    min_obs: int | None = None,
    cointegration: bool = False,
    df_crit: float = -2.86,
) -> list[Pair]:
    """Rank every eligible pair by formation SSD and return the ``top_n`` tightest.

    ``formation_prices`` is the close panel sliced to the formation window. Only names
    populated across the whole window are eligible (a partial series would forge a
    spurious match). Returns pairs sorted by ascending SSD — the GGR "minimum-distance"
    ordering — each carrying the spread ``sigma`` the trader will trigger on.

    With ``cointegration=True`` a pair is kept only if its formation spread passes a
    Dickey–Fuller mean-reversion gate (DF t-stat < ``df_crit``; −2.86 ≈ 5% with a
    constant) — the beat-7 fix that demands an economic reason to revert, not just a
    lucky formation year. The gate is applied *before* the top-N truncation, so you get
    the tightest ``top_n`` *survivors*.
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
            if cointegration and not (df_tstat(diff[:, off]) < df_crit):
                continue
            pairs.append(Pair(cols[i], cols[j], float(ssd[off]), float(sigma[off])))

    pairs.sort(key=lambda p: p.ssd)
    return pairs[:top_n]
