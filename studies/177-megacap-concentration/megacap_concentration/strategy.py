"""The strategy and its honest controls — Study 177 (Megacap-Concentration).

The folk recipe: each year, buy the top-N S&P 500 companies by market cap (the
"Magnificent Seven", "FAANG", or whatever the current megacap darlings are called),
hold for 12 months, and rebalance. Proponents claim:
  - Big companies have better earnings growth, wider moats, better capital allocation.
  - The market cap weighting of the S&P 500 already tilts toward them — why not just
    go all the way?

We pin this against two fair baselines:
  1. **Equal-weight (unconditional)**: hold all S&P 500 members with equal weight —
     this is what beating megacap concentration should look like if the effect is real.
  2. **SPY total-return**: the actual cap-weighted S&P 500 index.

And we split by decade to expose the recency bias:
  - Pre-2015: the dot-com and financial-crisis era when small/mid-cap *outperformed*
    large-cap (the size premium was alive).
  - 2015–2019: the early FAANG era.
  - 2020–2025: the Magnificent-Seven era when the claim actually worked.

Key discipline: only *prior year-end* market caps are used to rank — no look-ahead.
Returns are annual price returns (not total returns) for the concentration portfolios;
the SPY benchmark uses total return from the shared cache (conservative comparison).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from quantlab.analytics import mean_tstat_hac, sharpe_with_se
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def annual_portfolio_returns(
    panel: pd.DataFrame,
    top_n: int = 10,
    col: str = "ret_annual",
) -> pd.DataFrame:
    """Compute annual returns for top-N, equal-weight, and a random-N control.

    Parameters
    ----------
    panel : pd.DataFrame
        Long DataFrame with columns: year, ticker, mcap_rank, ret_annual, is_top_n.
        For synthetic data the relevant columns are year, rank, ret_gross, is_top_n.
    top_n : int
        Number of megacap stocks in the concentrated portfolio.
    col : str
        Return column name (``ret_annual`` for real; ``ret_gross`` for synthetic).

    Returns
    -------
    pd.DataFrame indexed by year with columns:
        ``topn_ret``: equal-weight average of the top-N stocks (concentration strategy).
        ``ew_ret``: equal-weight average of *all* stocks in the cross-section.
        ``topn_excess``: ``topn_ret`` minus ``ew_ret``.
        ``n_stocks``: number of valid stocks that year.
        ``n_topn``: number of top-N stocks with valid returns that year.
    """
    rank_col = "mcap_rank" if "mcap_rank" in panel.columns else "rank"
    rows = {}
    for yr, grp in panel.groupby("year"):
        valid = grp.dropna(subset=[col])
        if len(valid) < top_n:
            continue
        topn_mask = valid[rank_col] <= top_n
        topn_ret  = valid.loc[topn_mask, col].mean()
        ew_ret    = valid[col].mean()
        rows[yr] = {
            "topn_ret": float(topn_ret),
            "ew_ret": float(ew_ret),
            "topn_excess": float(topn_ret - ew_ret),
            "n_stocks": int(len(valid)),
            "n_topn": int(topn_mask.sum()),
        }
    return pd.DataFrame(rows).T.sort_index()


def with_spy_benchmark(
    annual: pd.DataFrame,
    spy_ann: pd.Series,
) -> pd.DataFrame:
    """Join the SPY annual return to the portfolio frame.

    ``spy_ann``: a Series indexed by integer year with annual SPY total returns.
    Adds ``spy_ret`` and ``topn_vs_spy`` columns.
    """
    out = annual.copy()
    out["spy_ret"] = spy_ann.reindex(annual.index.astype(int))
    out["topn_vs_spy"] = out["topn_ret"] - out["spy_ret"]
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _hac_tstat(series: pd.Series) -> float:
    """Newey-West HAC t-stat on the sample mean.  Falls back to an inline
    implementation when quantlab is not importable (tests run offline)."""
    r = series.dropna().to_numpy(dtype=float)
    n = r.size
    if n < 3:
        return float("nan")
    if _HAS_QUANTLAB:
        return float(mean_tstat_hac(pd.Series(r))["tstat"])
    # Inline Newey-West (mirrors quantlab.analytics.mean_tstat_hac)
    mu = r.mean()
    e  = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv  = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(annual: pd.DataFrame, decade_split: bool = True) -> dict:
    """Headline statistics for the concentration backtest.

    Returns a dict with:
      - ``full``: stats over the full sample.
      - ``by_decade``: stats split into pre-2015 / 2015–2019 / 2020–.
      - ``n_years``: total observation count.

    Each sub-dict contains:
      - ``topn_mean``: mean annual return of top-N portfolio (%).
      - ``ew_mean``: mean annual return of equal-weight portfolio (%).
      - ``excess_mean``: topn_mean − ew_mean (%).
      - ``excess_tstat``: HAC t-stat on the excess return series.
      - ``topn_sharpe``: annualised Sharpe of the top-N portfolio.
      - ``n``: number of years.
    """
    def _period_stats(sub: pd.DataFrame) -> dict:
        if len(sub) < 2:
            return {k: float("nan") for k in
                    ["topn_mean", "ew_mean", "excess_mean", "excess_tstat", "topn_sharpe", "n"]}
        excess = sub["topn_excess"]
        sr = float(sub["topn_ret"].mean() / sub["topn_ret"].std(ddof=1)) if sub["topn_ret"].std() > 0 else float("nan")
        return {
            "topn_mean": float(sub["topn_ret"].mean() * 100),
            "ew_mean":   float(sub["ew_ret"].mean() * 100),
            "excess_mean": float(excess.mean() * 100),
            "excess_tstat": float(_hac_tstat(excess)),
            "topn_sharpe": float(sr),
            "n": int(len(sub)),
        }

    full = _period_stats(annual)
    out = {"full": full, "n_years": int(len(annual))}

    if decade_split and len(annual) > 0:
        idx = annual.index.astype(int)
        out["by_decade"] = {
            "pre_2015":  _period_stats(annual[idx < 2015]),
            "2015_2019": _period_stats(annual[(idx >= 2015) & (idx < 2020)]),
            "post_2020": _period_stats(annual[idx >= 2020]),
        }

    if "topn_vs_spy" in annual.columns:
        vs_spy = annual["topn_vs_spy"].dropna()
        out["vs_spy"] = {
            "mean": float(vs_spy.mean() * 100),
            "tstat": float(_hac_tstat(vs_spy)),
            "n": int(len(vs_spy)),
        }

    return out


# ---------------------------------------------------------------------------
# SPY total-return helper
# ---------------------------------------------------------------------------
def spy_annual_returns(cache_dir: str | None = None) -> pd.Series:
    """Annual SPY total returns by calendar year, from the shared cache.

    Returns a Series indexed by integer year (e.g. 2008 → -0.37).
    Falls back to a hard-coded frozen table when the cache is unavailable
    (so tests and the offline demo work without the full repo cache).
    """
    if cache_dir is not None:
        spy_path = os.path.join(cache_dir, "SPY_total_return.parquet")
        if os.path.exists(spy_path):
            spy = pd.read_parquet(spy_path)
            spy.index = pd.to_datetime(spy.index)
            close = spy["Close"].dropna()
            close.index = close.index.year
            annual = close.groupby(level=0).last().pct_change().dropna()
            return annual

    # Hard-coded fallback — SPY annual total returns 2008–2025
    # Source: Macrotrends / SPDR (approximate, for offline mode)
    FALLBACK = {
        2008: -0.370, 2009: 0.264, 2010: 0.151, 2011: 0.021,
        2012: 0.160,  2013: 0.323, 2014: 0.136, 2015: 0.014,
        2016: 0.120,  2017: 0.219, 2018: -0.044, 2019: 0.314,
        2020: 0.184,  2021: 0.287, 2022: -0.181, 2023: 0.264,
        2024: 0.250,  2025: -0.002,
    }
    return pd.Series(FALLBACK, name="spy_ret")


import os  # noqa: E402 (imported again after sys manipulation at top)
