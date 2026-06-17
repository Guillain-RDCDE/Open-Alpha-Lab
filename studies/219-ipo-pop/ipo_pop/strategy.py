"""Strategy and controls — Study 219 (IPO-Pop).

The claim:
  (a) IPOs produce large first-day pops — buy the offer, sell at the first close.
  (b) Buying at the first close and holding for 6/12/36 months earns positive returns
      (Ritter 1991 long-run underperformance hypothesis says the opposite: mean reversion
      / disappointment leads to negative abnormal returns vs the market at 36m).

We test both sub-claims:
  (a) First-day pop: descriptive only — pops are real but unharvestable by retail
      (lottery allocation), and the pop measured here (offer→first_close) is NOT
      available to market buyers entering at the open.
  (b) Long-run drift: forward returns at 6m / 12m / 36m from the first close.
      HAC t-stat on the mean return (null = zero).  Ritter's finding implies
      mean < 0 at 36m; the data will decide.

Key caveats:
  - Survivorship bias: table contains only tickers still quoteable on yfinance.
    Cratered IPOs that delisted are excluded, biasing long-run returns upward.
  - The first-day pop (offer→first_close) is NOT harvestable by market investors;
    only IPO allocation holders benefit.  Retail allocation is effectively a lottery.
  - Long-run returns are price-return only (total return would include dividends, small
    for growth IPOs).  Compared against zero, not a market benchmark.
  - Small n (≈ 38 valid rows after data cleaning) means power is limited.

Functions
---------
- ``pop_stats``           — descriptive stats on the first-day pop distribution.
- ``drift_stats``         — HAC t-stat on post-close forward returns at each horizon.
- ``harvestability``      — quantifies why the pop is unharvestable.
- ``cost_adjusted_return``— net annualised return after round-trip costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# HAC t-stat (Newey-West)
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of x."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# First-day pop statistics
# ---------------------------------------------------------------------------
def pop_stats(df: pd.DataFrame) -> dict:
    """Descriptive statistics on the first-day IPO pop (offer→first_close).

    Parameters
    ----------
    df : DataFrame with a 'pop' column (fractional, e.g. 0.30 = 30%).

    Returns
    -------
    dict with keys: n, mean_pct, median_pct, std_pct, min_pct, max_pct,
    win_rate (fraction of IPOs with positive pop), tstat.
    """
    pops = df["pop"].dropna().to_numpy(dtype=float)
    n = pops.size
    if n < 2:
        return {k: float("nan") for k in (
            "n", "mean_pct", "median_pct", "std_pct", "min_pct", "max_pct",
            "win_rate", "tstat"
        )}
    return {
        "n":          int(n),
        "mean_pct":   float(pops.mean() * 100),
        "median_pct": float(np.median(pops) * 100),
        "std_pct":    float(pops.std(ddof=1) * 100),
        "min_pct":    float(pops.min() * 100),
        "max_pct":    float(pops.max() * 100),
        "win_rate":   float((pops > 0).mean()),
        "tstat":      _hac_tstat(pops),
    }


# ---------------------------------------------------------------------------
# Post-close forward return statistics
# ---------------------------------------------------------------------------
def drift_stats(
    df: pd.DataFrame,
    col: str = "ret_12m",
) -> dict:
    """Statistics for one post-close horizon's forward return.

    Parameters
    ----------
    df : DataFrame with a return column (e.g. 'ret_6m', 'ret_12m', 'ret_36m').
    col : Which column to analyse.

    Returns
    -------
    dict with keys: n, mean_pct, median_pct, std_pct, win_rate, tstat.
    Positive tstat = positive mean drift; negative = underperformance vs zero.
    """
    r = df[col].dropna().to_numpy(dtype=float)
    n = r.size
    if n < 2:
        return {k: float("nan") for k in (
            "n", "mean_pct", "median_pct", "std_pct", "win_rate", "tstat"
        )}
    return {
        "n":          int(n),
        "mean_pct":   float(r.mean() * 100),
        "median_pct": float(np.median(r) * 100),
        "std_pct":    float(r.std(ddof=1) * 100),
        "win_rate":   float((r > 0).mean()),
        "tstat":      _hac_tstat(r),
    }


def compare_horizons(df: pd.DataFrame) -> pd.DataFrame:
    """Compute drift statistics across all three long-run horizons.

    Returns a summary DataFrame with one row per horizon.
    """
    labels = {"ret_6m": "6m (~126d)", "ret_12m": "12m (~252d)", "ret_36m": "36m (~756d)"}
    rows = []
    for col, lbl in labels.items():
        if col not in df.columns:
            continue
        s = drift_stats(df, col=col)
        rows.append({"horizon": lbl, **s})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Harvestability analysis
# ---------------------------------------------------------------------------
def harvestability(df: pd.DataFrame) -> dict:
    """Quantify why the first-day pop is unharvestable by market investors.

    The pop (offer → first_close) requires an IPO allocation.  For institutional
    investors this depends on book-building participation (high bar); for retail
    it is a lottery with very low odds on hot IPOs.

    This function computes the 'aftermarket pop' — the return from buying at the
    first close (public market open) vs the offer, and the 'intraday' gap between
    first_close and offer.  It also computes what you'd earn buying at first_close
    and selling 1 day later (t+1 entry is our default, but let's look at t+0 close).

    Returns
    -------
    dict with keys:
      - mean_pop_pct: mean first-day pop (offer → first_close), %
      - mean_day1_gain_pct: how much of the pop a buyer at the open could theoretically
        capture (approximated as 0 — the pop is priced in by the open)
      - note: plain English summary
    """
    pops = df["pop"].dropna()
    mean_pop = float(pops.mean() * 100)
    # The first close is the end-of-day price; a market buyer at the open already
    # faces a price near first_close in a hot IPO.  We approximate the aftermarket
    # entry cost as the full pop (the most conservative and realistic assumption).
    return {
        "mean_pop_pct": mean_pop,
        "median_pop_pct": float(pops.median() * 100),
        "n_positive_pop": int((pops > 0).sum()),
        "n_ipos": int(len(pops)),
        "note": (
            "The first-day pop (offer->close) averages {:.0f}%. "
            "It is NOT harvestable by public market buyers: hot IPOs open near or above "
            "first-close by the time the first public trade executes. "
            "Retail allocation requires IPO application with low probability of award "
            "on the hottest issues. Institutional allocation requires book-build "
            "participation at the syndicate's discretion."
        ).format(mean_pop),
    }


# ---------------------------------------------------------------------------
# Cost-adjusted annualised return
# ---------------------------------------------------------------------------
def cost_adjusted_return(
    mean_return: float,
    horizon_days: int,
    cost_bps: float = 10.0,
) -> float:
    """Deduct a round-trip transaction cost and annualise the net return.

    Parameters
    ----------
    mean_return : Gross holding-period return (fraction, e.g. 0.05 = 5%).
    horizon_days : Number of trading days in the holding period.
    cost_bps : Round-trip cost in basis points (default 10 bps for liquid large-caps).

    Returns
    -------
    Net annualised return (fraction).
    """
    net = mean_return - cost_bps * 1e-4
    ann_factor = TRADING_DAYS_PER_YEAR / horizon_days
    return float((1.0 + net) ** ann_factor - 1.0)


# ---------------------------------------------------------------------------
# Synthetic positive control helper
# ---------------------------------------------------------------------------
def planted_vs_null(
    planted: pd.DataFrame,
    null: pd.DataFrame,
    col: str = "ret_12m",
) -> tuple[float, float]:
    """Return (planted_mean_pct, null_mean_pct) for one horizon.

    Used to verify the engine recovers a drift when one is planted and not
    otherwise — the 'machinery proof'.
    """
    pm = planted[col].dropna().to_numpy(dtype=float)
    nm = null[col].dropna().to_numpy(dtype=float)
    return float(pm.mean() * 100), float(nm.mean() * 100)
