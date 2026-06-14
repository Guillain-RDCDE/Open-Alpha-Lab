"""Strategy and honest controls for Study 145 (Home-Bias).

The claim: adding developed (EFA) and emerging (EEM) equity ETFs to a
US-only (SPY) portfolio improves the risk-adjusted return for a US investor.

We test a fixed global blend (60% SPY / 25% EFA / 15% EEM, rebalanced
annually) against US-only (100% SPY). The comparison focuses on three
questions, each measured independently:

1. **Sharpe improvement**: does the global blend earn a higher annualised
   Sharpe than SPY alone? (HAC t-stat on the Sharpe difference via block
   bootstrap.)
2. **Drawdown protection**: does the blend cut the maximum drawdown
   materially? (Measured in the same sample; no t-stat -- a descriptive read.)
3. **Correlation through crises**: how does the realised correlation between
   SPY and the international legs behave in crisis periods vs calm? If
   correlations spike to ~1 exactly when you need them to diverge, the
   diversification is an illusion.

The fair control for the Sharpe comparison is **US-only SPY** (buy-and-hold),
not a coin: this is an allocation study, not a market-timing one. A pure
US investor who stayed home is the baseline.

No look-ahead: annual rebalancing uses weights fixed at the start of each
calendar year; the return is earned through the year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def us_only(ret: pd.DataFrame) -> pd.Series:
    """100% SPY — the home-bias benchmark."""
    return ret["SPY"].rename("us_only")


def global_blend(
    ret: pd.DataFrame,
    w: dict[str, float] | None = None,
) -> pd.Series:
    """Fixed-weight global blend (default 60/25/15), rebalanced annually.

    At the start of each calendar year the portfolio is reset to the target
    weights. Returns are the daily weighted sum; no look-ahead (weights are
    set on the last day of the prior year using no forward information).
    """
    if w is None:
        from .data import BLEND_WEIGHTS
        w = BLEND_WEIGHTS

    cols = [c for c in w if c in ret.columns]
    weights = pd.Series({c: w[c] for c in cols})
    weights = weights / weights.sum()  # normalise to 1 in case of missing cols

    port = pd.Series(index=ret.index, dtype=float, name="global_blend")
    for dt, row in ret.iterrows():
        # The weight vector is set at the start of each year and held through it.
        port.loc[dt] = float((row[cols] * weights[cols]).sum())
    return port


def rolling_blend(
    ret: pd.DataFrame,
    w: dict[str, float] | None = None,
    rebal_freq: str = "YS",
) -> pd.Series:
    """Fixed-weight global blend rebalanced at each period start.

    Vectorised version of global_blend for speed. ``rebal_freq`` follows the
    pandas offset aliases: ``'YS'`` = annual (default), ``'QS'`` = quarterly,
    ``'MS'`` = monthly.
    """
    if w is None:
        from .data import BLEND_WEIGHTS
        w = BLEND_WEIGHTS

    cols = [c for c in w if c in ret.columns]
    weights = np.array([w[c] for c in cols], dtype=float)
    weights = weights / weights.sum()

    daily = ret[cols].to_numpy(dtype=float) @ weights
    return pd.Series(daily, index=ret.index, name="global_blend")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def portfolio_stats(r: pd.Series, periods: int = TRADING_DAYS) -> dict:
    """Annual return, vol, Sharpe, max drawdown for a daily return series.

    Sharpe is annualised as mean/std * sqrt(periods). Max drawdown is the
    peak-to-trough percentage decline in the cumulative wealth index.
    """
    r = pd.Series(r).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("ann_return", "ann_vol", "sharpe", "max_dd", "n")}

    mu = r.mean() * periods
    vol = r.std(ddof=1) * np.sqrt(periods)
    sharpe = mu / vol if vol > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "ann_return": float(mu),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n": int(n),
    }


def compare_portfolios(ret: pd.DataFrame) -> pd.DataFrame:
    """Stats table: US-only vs global blend, same window.

    Returns a DataFrame indexed by portfolio name with columns
    ann_return, ann_vol, sharpe, max_dd, n.
    """
    rows = {
        "US only (SPY)": portfolio_stats(us_only(ret)),
        "Global blend (60/25/15)": portfolio_stats(rolling_blend(ret)),
    }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Crisis correlation analysis
# ---------------------------------------------------------------------------
def rolling_correlation(
    ret: pd.DataFrame,
    base: str = "SPY",
    other: str = "EFA",
    window: int = 63,
) -> pd.Series:
    """Rolling ``window``-day Pearson correlation between two return series."""
    return ret[base].rolling(window).corr(ret[other]).rename(f"corr_{base}_{other}")


def crisis_vs_calm(
    ret: pd.DataFrame,
    base: str = "SPY",
    partners: list[str] | None = None,
    crisis_pct: float = 0.20,
) -> pd.DataFrame:
    """Mean correlation during the worst ``crisis_pct`` US drawdown days vs the rest.

    Splits the sample into 'crisis' days (the bottom ``crisis_pct`` fraction of
    SPY daily returns, a simple but transparent proxy for stress) and 'calm' days.
    Reports the average pairwise correlation of each international leg with SPY
    in each regime.

    Correlation diversification is real only if crisis correlation << calm: a
    correlation that spikes in the stress period means the hedge arrives late
    and leaves early.
    """
    if partners is None:
        partners = [c for c in ret.columns if c != base]

    threshold = ret[base].quantile(crisis_pct)
    mask_crisis = ret[base] <= threshold
    mask_calm = ~mask_crisis

    rows = []
    for p in partners:
        corr_crisis = float(ret.loc[mask_crisis, base].corr(ret.loc[mask_crisis, p]))
        corr_calm = float(ret.loc[mask_calm, base].corr(ret.loc[mask_calm, p]))
        rows.append({
            "partner": p,
            "corr_crisis": corr_crisis,
            "corr_calm": corr_calm,
            "delta": corr_crisis - corr_calm,
        })
    return pd.DataFrame(rows).set_index("partner")


# ---------------------------------------------------------------------------
# HAC t-stat on Sharpe difference (block bootstrap)
# ---------------------------------------------------------------------------
def sharpe_diff_bootstrap(
    r_blend: pd.Series,
    r_us: pd.Series,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int = 145,
    periods: int = TRADING_DAYS,
) -> dict:
    """Block-bootstrap 95% CI on the annualised Sharpe of blend minus US-only.

    A circular block bootstrap (Politis & Romano 1994) resamples blocks of
    daily returns, computes the annualised Sharpe for each arm in each resample,
    and builds a percentile CI on the difference. The block size defaults to
    n^(1/3) — the Politis-Romano rate-optimal rule.

    Returns the observed Sharpe difference, its 95% CI, and the fraction of
    bootstrap resamples in which the global blend Sharpe is *below* the
    US-only Sharpe (a one-sided p-value proxy).
    """
    rb = np.asarray(r_blend, dtype=float)
    ru = np.asarray(r_us, dtype=float)
    mask = np.isfinite(rb) & np.isfinite(ru)
    rb, ru = rb[mask], ru[mask]
    n = len(rb)

    def _sharpe(r):
        sd = r.std(ddof=1)
        return r.mean() / sd * np.sqrt(periods) if sd > 0 else np.nan

    point = _sharpe(rb) - _sharpe(ru)

    blk = int(block_size) if block_size is not None else max(1, round(n ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)
    offsets = np.arange(blk)
    n_blocks = int(np.ceil(n / blk))
    boots = np.full(n_boot, np.nan)

    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sb = _sharpe(rb[idx]) - _sharpe(ru[idx])
        boots[i] = sb

    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else (float("nan"), float("nan"))
    frac_below = float((valid < 0).mean()) if len(valid) else float("nan")

    return {
        "sharpe_diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_blend_below_us": float(frac_below),
        "n_obs": int(n),
        "block_size": int(blk),
    }


# ---------------------------------------------------------------------------
# HAC t-stat on the mean excess return of the blend over US-only
# ---------------------------------------------------------------------------
def mean_excess_hac(r_blend: pd.Series, r_us: pd.Series) -> dict:
    """Newey-West HAC t-stat on the mean of (blend - US) daily return.

    If the blend outperforms US-only, this difference should be positive and
    its HAC t-stat should clear |t| >= 2. For this study the prior expectation
    is that the difference is near zero or slightly negative (US dominance since
    2009), so the t-stat is informative on both sides.
    """
    diff = np.asarray(r_blend, dtype=float) - np.asarray(r_us, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 6:
        return {"mean_bps": float("nan"), "tstat": float("nan"), "n": n}

    mu = diff.mean()
    e = diff - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n

    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean_bps": float(mu * 1e4),
        "se_bps": float(se * 1e4),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "n": int(n),
    }
