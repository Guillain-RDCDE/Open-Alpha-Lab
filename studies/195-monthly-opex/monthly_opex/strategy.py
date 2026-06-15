"""The strategy and honest controls — Study 195 (Monthly-OpEx).

The folk hypothesis: the monthly options-expiration week (the 3rd Friday of every month
and the surrounding Mon–Fri window) causes mechanical hedging, delta-hedging pinning, and
roll activity that elevates **volume** and **intraday range** on the expiry week.  A
tradable *directional* return edge ("pin risk", "expiry drift") is a more speculative but
common retail claim.

We test three legs:

1. **The vol/volume leg** — do OpEx weeks have materially elevated daily range and year-
   demeaned volume vs baseline weeks?  Tested via Newey-West HAC t-stats.
2. **The return leg** — do OpEx days / weeks carry a systematic directional return?
   Same HAC machinery on daily close-to-close returns.
3. **Monthly-only incremental effect** — separate non-quarterly OpEx weeks from quarterly
   triple-witching weeks (already studied in Study 82).  If the quarterly effect drives
   everything, the incremental monthly effect should be null.

Honest baseline: all *non-OpEx-week* trading days in the same year (so a secular trend
or year-specific event does not contaminate the comparison).

Pre/post-2010 split: the rapid growth of 0DTE (zero days to expiry) and weekly options
after 2010 may have changed the dynamics.  We flag this structural break explicitly.

No look-ahead: all OpEx masks are computed from the Gregorian calendar — known before
the month begins — so there is no data look-ahead in the signal construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    opex_day_mask,
    opex_week_mask,
    monthly_only_opex_week_mask,
    quarterly_opex_week_mask,
)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def daily_range_pct(bars: pd.DataFrame) -> pd.Series:
    """Intraday range as a fraction of the close: (high − low) / close.

    A direction-agnostic volatility measure.  Elevated OpEx-week range supports the
    mechanical-activity hypothesis regardless of direction.
    """
    return ((bars["high"] - bars["low"]) / bars["close"]).rename("range_pct")


def log_volume_demeaned(bars: pd.DataFrame) -> pd.Series:
    """Year-demeaned log volume: log(volume) minus that year's annual mean log(volume).

    Year-demeaning removes the strong secular trend in ETF/index volume (which has
    declined as SPY matured), delivering a clean within-year comparison of OpEx vs
    non-OpEx weeks.  Without this, a naïve log-volume test is contaminated by the trend.
    """
    lv = np.log(bars["volume"].clip(lower=1.0))
    year = pd.DatetimeIndex(bars.index).year
    annual_mean = lv.groupby(pd.Series(year, index=bars.index)).transform("mean")
    return (lv - annual_mean).rename("log_volume_dm")


def daily_return(bars: pd.DataFrame) -> pd.Series:
    """Overnight-inclusive daily close-to-close simple return."""
    return bars["close"].pct_change().rename("ret")


# ---------------------------------------------------------------------------
# Core comparison engine
# ---------------------------------------------------------------------------

def compare_opex_vs_baseline(
    series: pd.Series,
    mask: pd.Series,
    label: str = "opex",
) -> dict:
    """Compare ``series`` on OpEx days/weeks vs non-OpEx days.

    Returns means, standard deviations, counts, the mean difference, and a
    Newey-West HAC t-stat on the difference.  The HAC approach is conservative and
    respects serial correlation in daily financial data.

    The HAC t-stat is the decisive number for the inference bar: |t| >= 2 on the
    real tape is the minimum for ``REAL``; below that, ``WEAK`` or ``NONE``.
    """
    s = series.astype(float).dropna()
    m = mask.reindex(s.index).fillna(False).astype(bool)
    opex_vals = s[m]
    base_vals = s[~m]

    mean_o = float(opex_vals.mean()) if len(opex_vals) else float("nan")
    mean_b = float(base_vals.mean()) if len(base_vals) else float("nan")
    diff = mean_o - mean_b

    t = float("nan")
    if len(opex_vals) > 5 and len(base_vals) > 5:
        n = len(s)
        x = m.values.astype(float)
        e = s.values - (mean_b + diff * x)  # OLS residuals (binary regressor)
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        xx = float(x @ x)
        xe = x * e
        lrv = float(xe @ xe)
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(xe[k:] @ xe[:-k])
        se = np.sqrt(max(lrv, 0.0) / (xx ** 2))
        t = float(diff / se) if se > 0 else float("nan")

    return {
        "label": label,
        "n_opex": int(len(opex_vals)),
        "n_baseline": int(len(base_vals)),
        "mean_opex": mean_o,
        "mean_baseline": mean_b,
        "std_opex": float(opex_vals.std(ddof=1)) if len(opex_vals) > 1 else float("nan"),
        "std_baseline": float(base_vals.std(ddof=1)) if len(base_vals) > 1 else float("nan"),
        "diff": float(diff),
        "tstat": float(t),
    }


# ---------------------------------------------------------------------------
# Three main test legs
# ---------------------------------------------------------------------------

def opex_vol_test(bars: pd.DataFrame) -> dict:
    """HAC t-tests: are range and year-demeaned volume elevated in OpEx weeks?

    These are the *mechanical* effects — from options delta-hedging, gamma pinning,
    and institutional roll activity around expiry.  A real range effect without a
    return effect is the expected finding for a pure mechanical/liquidity story.
    """
    idx = pd.DatetimeIndex(bars.index)
    rng = daily_range_pct(bars)
    lvol_dm = log_volume_demeaned(bars)

    day_mask = opex_day_mask(idx)
    week_mask = opex_week_mask(idx)

    return {
        "range_day": compare_opex_vs_baseline(rng, day_mask, "range_opex_day"),
        "range_week": compare_opex_vs_baseline(rng, week_mask, "range_opex_week"),
        "vol_day": compare_opex_vs_baseline(lvol_dm, day_mask, "vol_dm_opex_day"),
        "vol_week": compare_opex_vs_baseline(lvol_dm, week_mask, "vol_dm_opex_week"),
    }


def opex_return_test(bars: pd.DataFrame) -> dict:
    """HAC t-tests: do OpEx days / weeks carry a systematic directional return?

    Tests the day itself and the full expiry week, against a non-OpEx baseline.
    """
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)

    day_mask = opex_day_mask(idx)
    week_mask = opex_week_mask(idx)

    return {
        "day": compare_opex_vs_baseline(ret, day_mask, "opex_day_vs_rest"),
        "week": compare_opex_vs_baseline(ret, week_mask, "opex_week_vs_rest"),
    }


def opex_incremental_monthly_test(bars: pd.DataFrame) -> dict:
    """Incremental monthly vs quarterly: does the non-quarterly OpEx week add signal?

    Compares non-quarterly OpEx weeks (Jan/Feb/Apr/May/Jul/Aug/Oct/Nov) against
    baseline AND against quarterly OpEx weeks — isolating the monthly cycle from
    the Study-82 triple-witching effect.
    """
    idx = pd.DatetimeIndex(bars.index)
    rng = daily_range_pct(bars)
    lvol_dm = log_volume_demeaned(bars)
    ret = daily_return(bars).dropna()

    monthly_mask = monthly_only_opex_week_mask(idx)
    quarterly_mask = quarterly_opex_week_mask(idx)

    return {
        "range_monthly_only": compare_opex_vs_baseline(
            rng, monthly_mask, "range_monthly_only_opex_week"
        ),
        "vol_monthly_only": compare_opex_vs_baseline(
            lvol_dm, monthly_mask, "vol_monthly_only_opex_week"
        ),
        "ret_monthly_only": compare_opex_vs_baseline(
            ret.reindex(idx).dropna(), monthly_mask.reindex(ret.index).fillna(False),
            "ret_monthly_only_opex_week"
        ),
        "range_quarterly": compare_opex_vs_baseline(
            rng, quarterly_mask, "range_quarterly_opex_week"
        ),
        "vol_quarterly": compare_opex_vs_baseline(
            lvol_dm, quarterly_mask, "vol_quarterly_opex_week"
        ),
    }


def opex_pre_post_2010_split(bars: pd.DataFrame) -> dict:
    """Split the data at 2010 to capture the 0DTE / weekly-options era.

    0DTE options grew explosively from ~2010 onward; weekly options were introduced
    by CBOE in 2005 (broad adoption post-2010).  This structural break may change the
    gamma pinning / hedging dynamics around monthly OpEx.  We report vol-week and
    return-week t-stats separately for each sub-period.
    """
    idx = pd.DatetimeIndex(bars.index)
    week_mask = opex_week_mask(idx)
    rng = daily_range_pct(bars)
    lvol_dm = log_volume_demeaned(bars)
    ret = daily_return(bars).dropna()

    cut = pd.Timestamp("2010-01-01")
    pre = bars.index < cut
    post = bars.index >= cut

    def _test_sub(b: pd.DataFrame, lbl: str) -> dict:
        sidx = pd.DatetimeIndex(b.index)
        wm = opex_week_mask(sidx)
        return {
            "range": compare_opex_vs_baseline(daily_range_pct(b), wm, f"range_{lbl}"),
            "vol": compare_opex_vs_baseline(log_volume_demeaned(b), wm, f"vol_{lbl}"),
            "ret": compare_opex_vs_baseline(
                daily_return(b).dropna(), opex_week_mask(pd.DatetimeIndex(daily_return(b).dropna().index)),
                f"ret_{lbl}"
            ),
        }

    return {
        "pre_2010": _test_sub(bars[pre], "pre2010"),
        "post_2010": _test_sub(bars[post], "post2010"),
    }


# ---------------------------------------------------------------------------
# Tradability check: a simple OpEx-week long rule
# ---------------------------------------------------------------------------

def opex_week_trade_returns(bars: pd.DataFrame, long_week: bool = True) -> pd.Series:
    """Daily returns from being long (or short) SPY only in OpEx weeks.

    Position is fully determined by the exchange calendar — no signal from price data.
    We enter at the *prior close* (no look-ahead): the full Mon–Fri OpEx-week return.
    ``long_week=True`` goes long; ``long_week=False`` goes short.
    """
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)
    week_mask = opex_week_mask(idx).reindex(ret.index).fillna(False)
    direction = 1.0 if long_week else -1.0
    return (ret * week_mask.values * direction).rename("strat")


def summarize(returns: pd.Series) -> dict:
    """Headline statistics for a daily return series: Sharpe, CAGR, and HAC t-stat.

    The HAC t-stat uses Newey-West lags calibrated to sample size, matching the
    inference bar in ``compare_opex_vs_baseline``.
    """
    r = returns.astype(float).dropna().to_numpy()
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return {k: float("nan") for k in ("n", "mean_bps", "sharpe_ann", "cagr", "tstat")}

    mu = r.mean()
    sig = r.std(ddof=1)
    cagr = float(np.prod(1.0 + r) ** (TRADING_DAYS_PER_YEAR / n) - 1.0)

    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "n": int(n),
        "mean_bps": float(mu * 1e4),
        "sharpe_ann": float(mu / sig * np.sqrt(TRADING_DAYS_PER_YEAR)) if sig > 0 else float("nan"),
        "cagr": float(cagr),
        "tstat": float(tstat),
    }
