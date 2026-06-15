"""The three-fund strategy and its honest baselines — Study 205 (Three-Fund).

The Bogleheads "three-fund portfolio" recipe:
  - ~60% VTI  (US total market equity)
  - ~30% VXUS (total international ex-US equity)
  - ~10% BND  (US total bond market)
  (weights approximate a global market-cap allocation with a bond sleeve)

We compare it honestly against two baselines on a *risk-adjusted* basis:

  1. **US-only (SPY)**      — 100% US equity; the simplest alternative.
  2. **60/40 (SPY + BND)** — a classic balanced portfolio; the "standard lazy".

All three are *passive*, long-only, rebalanced annually. Costs are charged on
the one-way turnover at each rebalance. No look-ahead: weights are pre-specified
and constant (the study design, not a signal), so rebalancing is purely
mechanical.

The interesting sub-questions:

  H1 (vs US-only):   Does adding international + bonds improve the risk-adjusted
                     return (Sharpe)? Or did the US-only strategy dominate over the
                     sample?
  H2 (vs 60/40):    Does the VXUS sleeve add value beyond a simple 60/40?
  H3 (international contribution): Did the international sleeve help or hurt — was
                     VXUS positive-alpha relative to holding more SPY instead?

A multiple-comparisons correction (Bonferroni across 3 hypotheses) is applied.
Inference: Newey-West HAC t-stat on annual return differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Canonical portfolio weight sets
# ---------------------------------------------------------------------------
# Three-fund Bogleheads default: ~60% US equity, ~30% int'l, ~10% bond.
# Mimics global market-cap weighting with a bond sleeve.
THREE_FUND_WEIGHTS = {"VTI": 0.60, "VXUS": 0.30, "BND": 0.10}

# 60/40: classic balanced portfolio
SIXTY_FORTY_WEIGHTS = {"SPY": 0.60, "BND": 0.40}

# 100% US equity
US_ONLY_WEIGHTS = {"SPY": 1.00}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def _rebalance_cost(prev_w: dict, new_w: dict, cost_bps: float) -> float:
    """One-way turnover cost for a rebalance from ``prev_w`` to ``new_w``.

    Turnover = 0.5 * sum(|w_new - w_old|) — half the gross one-way trade.
    Charged at ``cost_bps`` per 1 unit of turnover (one-way).
    """
    all_tickers = set(prev_w) | set(new_w)
    turnover = sum(
        abs(new_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_tickers
    )
    return turnover * cost_bps * 1e-4


def run_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    rebalance_freq: str = "YE",
    cost_bps: float = 5.0,
) -> pd.Series:
    """Daily total-return index for a fixed-weight portfolio with periodic rebalancing.

    On each rebalance date (year-end by default) the portfolio is reset to ``weights``.
    The one-way turnover cost is charged as a fraction of NAV on the rebalance day.
    Between rebalances the portfolio drifts with the underlying prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted-close price matrix. Must contain all columns named in ``weights``.
    weights : dict[str, float]
        Target allocation {ticker: weight}. Weights need not sum to 1 — they are
        normalised automatically.
    rebalance_freq : str
        Pandas resample frequency for rebalance dates (default 'YE' = calendar year-end).
    cost_bps : float
        One-way rebalance cost in basis points applied to the gross turnover.

    Returns
    -------
    pd.Series
        Daily NAV starting at 1.0 on the first date of prices.
    """
    tickers = list(weights.keys())
    # Normalise weights
    total = sum(weights.values())
    w = {t: weights[t] / total for t in tickers}

    # Subset and align prices
    px = prices[tickers].dropna()
    if px.empty:
        return pd.Series(dtype=float, name="nav")

    # Rebalance calendar: last trading day of each calendar year
    annual_dates = set(
        px.resample(rebalance_freq).last().index
    )

    # Forward simulation: track dollar holdings per ticker
    shares = {t: w[t] / px[t].iloc[0] for t in tickers}  # notional $1 start
    nav = np.empty(len(px))
    nav[0] = 1.0

    prev_w = w.copy()

    for i in range(1, len(px)):
        date = px.index[i]
        row = px.iloc[i]

        # Current NAV before any rebalance
        current_nav = sum(shares[t] * row[t] for t in tickers)

        # Rebalance?
        if date in annual_dates:
            # Compute actual weights at current prices
            actual_w = {t: shares[t] * row[t] / current_nav for t in tickers}
            cost = _rebalance_cost(actual_w, w, cost_bps)
            current_nav *= (1.0 - cost)
            # Reset shares to target weights
            shares = {t: w[t] * current_nav / row[t] for t in tickers}
            prev_w = actual_w

        nav[i] = sum(shares[t] * row[t] for t in tickers)

    return pd.Series(nav, index=px.index, name="nav")


def nav_to_returns(nav: pd.Series) -> pd.Series:
    """Convert a NAV series to daily simple returns."""
    return nav.pct_change().dropna()


# ---------------------------------------------------------------------------
# Metrics and inference
# ---------------------------------------------------------------------------
def _max_drawdown(nav: pd.Series) -> float:
    arr = nav.to_numpy(dtype=float)
    peak = np.maximum.accumulate(arr)
    return float((arr / peak - 1.0).min())


def annual_returns(daily_ret: pd.Series) -> pd.Series:
    """Compound annual returns from a daily series."""
    r = daily_ret.dropna()
    return (1.0 + r).resample("YE").prod() - 1.0


def summarize(nav: pd.Series) -> dict:
    """Headline risk/return metrics for a NAV series.

    Returns CAGR, annualised Sharpe, max drawdown, Calmar ratio, and a
    Newey-West HAC t-stat on the mean annual return (the inference bar).
    """
    r = nav_to_returns(nav)
    r = r.dropna()
    if len(r) < 2:
        return {k: float("nan") for k in (
            "n_years", "cagr", "ann_vol", "sharpe", "max_dd", "calmar", "tstat_annual"
        )}

    total = float((1.0 + r).prod())
    n_years = len(r) / TRADING_DAYS
    cagr = total ** (1.0 / n_years) - 1.0 if (n_years > 0 and total > 0) else float("nan")
    ann_mean = r.mean() * TRADING_DAYS
    ann_vol  = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe   = ann_mean / ann_vol if ann_vol > 0 else float("nan")
    mdd      = _max_drawdown(nav)
    calmar   = cagr / abs(mdd) if mdd < 0 else float("nan")

    # HAC t-stat on annual return observations
    ann_r = annual_returns(r).to_numpy(dtype=float)
    ann_r = ann_r[np.isfinite(ann_r)]
    n_ann = ann_r.size
    tstat = float("nan")
    if n_ann > 2:
        mu = ann_r.mean()
        e  = ann_r - mu
        lags = max(1, int(np.floor(4.0 * (n_ann / 100.0) ** (2.0 / 9.0))))
        lrv  = float(e @ e) / n_ann
        for k in range(1, lags + 1):
            wk = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n_ann
        se = np.sqrt(max(lrv, 0.0) / n_ann)
        tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "n_years": float(n_years),
        "cagr":    float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe":  float(sharpe),
        "max_dd":  float(mdd),
        "calmar":  float(calmar),
        "tstat_annual": float(tstat),
    }


def hac_tstat_diff(nav_a: pd.Series, nav_b: pd.Series) -> dict:
    """HAC t-stat on the per-year return difference (A minus B).

    Tests whether strategy A significantly outperforms strategy B using annual
    return differences; Newey-West long-run variance. This is the honest test
    for 'does the three-fund beat 60/40?'.
    """
    r_a = nav_to_returns(nav_a)
    r_b = nav_to_returns(nav_b)
    ann_a = annual_returns(r_a.dropna())
    ann_b = annual_returns(r_b.dropna())
    aligned = ann_a.align(ann_b, join="inner")
    diff = (aligned[0] - aligned[1]).dropna().to_numpy(dtype=float)
    n = diff.size
    if n < 3:
        return {"mean_diff": float("nan"), "tstat": float("nan"), "n": n}
    mu = diff.mean()
    e  = diff - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv  = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean_diff": float(mu),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "n": n,
    }


def run_all_portfolios(
    prices: pd.DataFrame,
    rebalance_freq: str = "YE",
    cost_bps: float = 5.0,
) -> dict[str, pd.Series]:
    """Run all three portfolio strategies on the same price matrix.

    Returns a dict of NAV series: 'three_fund', 'sixty_forty', 'us_only'.
    """
    navs = {}
    navs["three_fund"] = run_portfolio(prices, THREE_FUND_WEIGHTS, rebalance_freq, cost_bps)
    navs["sixty_forty"] = run_portfolio(prices, SIXTY_FORTY_WEIGHTS, rebalance_freq, cost_bps)
    navs["us_only"]     = run_portfolio(prices, US_ONLY_WEIGHTS,     rebalance_freq, cost_bps)
    return navs


def bonferroni_threshold(n_tests: int = 3, alpha: float = 0.05) -> float:
    """Bonferroni-corrected |t|-threshold for n_tests comparisons at significance alpha.

    With annual return observations (n_ann ~ 10-15 years) the t-distribution's
    critical value at the Bonferroni level is well approximated by the normal.
    """
    from scipy.stats import norm
    return float(norm.ppf(1.0 - alpha / (2.0 * n_tests)))


def international_attribution(prices: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """Did the VXUS sleeve add value?

    Compares:
      A) Three-fund (60 VTI / 30 VXUS / 10 BND)
      B) No-VXUS alternative (90 VTI / 10 BND — same equity/bond split, no intl)

    Returns the per-year return difference (A - B) and its HAC t-stat.
    """
    nav_3f  = run_portfolio(prices, THREE_FUND_WEIGHTS, cost_bps=cost_bps)
    nav_nointl = run_portfolio(prices, {"VTI": 0.90, "BND": 0.10}, cost_bps=cost_bps)
    diff = hac_tstat_diff(nav_3f, nav_nointl)
    return diff
