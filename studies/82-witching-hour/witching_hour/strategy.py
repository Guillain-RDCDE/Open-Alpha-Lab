"""The strategy and honest controls — Study 82 (Witching-Hour).

The folk hypothesis: the quarterly expiry (triple/quadruple witching — the 3rd
Friday of Mar/Jun/Sep/Dec, when index options, stock options, and futures all
expire together) causes *mechanical* hedging and roll activity that elevates both
**volume** and **intraday range** on the expiry day and in the expiry week.  A
tradable *directional* return edge is claimed by some (usually "sell the witch
and buy back Monday") but is mechanically less justified.

We test two legs separately:

1. **The vol/volume leg** — do witching days / weeks have materially elevated daily
   range (high-low / close) and volume vs matched baseline weeks?  This is tested
   via a simple mean-difference t-stat with Newey-West HAC correction.
2. **The return leg** — do witching days / weeks carry a systematic directional
   return (vs the rest of the year)?  Same HAC t-stat machinery on daily returns.

Fair baseline: "baseline weeks" are all non-witching weeks in the same year (so
seasonality doesn't contaminate).  We also compute a matched-quarter control
(non-witching Fridays in the same calendar quarter, to rule out simple
Friday-is-special effects).

No look-ahead: witching dates are purely calendar-derived — known before the
month begins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import witching_day_mask, witching_week_mask

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def daily_range_pct(bars: pd.DataFrame) -> pd.Series:
    """Intraday range as a fraction of the close: (high − low) / close.

    A direction-agnostic measure of intraday volatility.  Elevated witching-day
    range supports the mechanical-activity hypothesis regardless of direction.
    """
    return ((bars["high"] - bars["low"]) / bars["close"]).rename("range_pct")


def log_volume(bars: pd.DataFrame) -> pd.Series:
    """Log of daily share volume (natural log, units: log-shares).

    Log-volume is approximately normally distributed and compares multiplicatively:
    a +0.30 difference implies a ~35% volume increase.
    """
    return np.log(bars["volume"].clip(lower=1.0)).rename("log_volume")


def log_volume_demeaned(bars: pd.DataFrame) -> pd.Series:
    """Year-demeaned log volume: log(volume) minus that year's annual mean log(volume).

    Year-demeaning removes the strong secular trend in ETF/index volume (which declines
    over time as the ETF matures and assets grow relative to daily activity), giving a
    clean within-year comparison of witching vs non-witching days.  Without this
    correction, a naïve log-volume t-test is contaminated by the trend.
    """
    lv = np.log(bars["volume"].clip(lower=1.0))
    year = pd.DatetimeIndex(bars.index).year
    lv_dm = lv - pd.Series(year, index=bars.index).map(
        lv.groupby(pd.Series(year, index=bars.index)).mean()
    )
    return lv_dm.rename("log_volume_dm")


def daily_return(bars: pd.DataFrame) -> pd.Series:
    """Overnight-inclusive daily simple return: close-to-close.

    This is the return a buy-at-prior-close / sell-at-close strategy earns on a
    witching day (the classic ''buy-and-hold over the event'' directional bet).
    """
    return bars["close"].pct_change().rename("ret")


# ---------------------------------------------------------------------------
# Core comparison engine
# ---------------------------------------------------------------------------

def compare_witching_vs_baseline(
    series: pd.Series,
    mask: pd.Series,
    label: str = "witching",
) -> dict:
    """Compare ``series`` on witching days vs non-witching days.

    Returns means, standard deviations, counts, the mean difference, and a
    Newey-West HAC t-stat on the difference (computed by regressing the series
    on the mask indicator — the HAC approach is conservative and respects the
    serial correlation inherent in daily financial data).

    The HAC t-stat is the decisive number for the inference bar: |t| >= 2 on the
    real tape is the minimum bar for ``REAL``; below that, ``WEAK`` or ``NONE``.
    """
    s = series.astype(float).dropna()
    m = mask.reindex(s.index).fillna(False).astype(bool)
    witch = s[m]
    base = s[~m]

    mean_w = float(witch.mean()) if len(witch) else float("nan")
    mean_b = float(base.mean()) if len(base) else float("nan")
    diff = mean_w - mean_b

    # Newey-West HAC t-stat on the difference (regression on indicator):
    # r_t = alpha + beta * I_t + eps_t, test H0: beta = 0.
    # With a binary indicator, beta = diff; we only need its HAC SE.
    t = float("nan")
    if len(witch) > 5 and len(base) > 5:
        n = len(s)
        x = m.values.astype(float)
        e = s.values - (mean_b + diff * x)  # OLS residuals (regressor = indicator)
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        # Heteroskedasticity-robust (HC) sandwich with NW autocorrelation correction
        xx = float(x @ x)
        xe = x * e
        lrv = float(xe @ xe)  # lag 0 term (already scaled by 1/n below)
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(xe[k:] @ xe[:-k])
        se = np.sqrt(max(lrv, 0.0) / (xx ** 2))
        t = float(diff / se) if se > 0 else float("nan")

    return {
        "label": label,
        "n_witching": int(len(witch)),
        "n_baseline": int(len(base)),
        "mean_witching": mean_w,
        "mean_baseline": mean_b,
        "std_witching": float(witch.std(ddof=1)) if len(witch) > 1 else float("nan"),
        "std_baseline": float(base.std(ddof=1)) if len(base) > 1 else float("nan"),
        "diff": float(diff),
        "tstat": float(t),
    }


def compare_friday_effect(
    series: pd.Series,
    index: pd.DatetimeIndex,
) -> dict:
    """Compare witching Fridays vs all other Fridays.

    Rules out the hypothesis that witching-day effects are just a Friday
    seasonality — if the effect persists vs other Fridays, the witching calendar
    itself carries explanatory power.
    """
    all_fridays = index.dayofweek == 4
    w_day = witching_day_mask(index)
    friday_not_witch = all_fridays & ~w_day.values
    return compare_witching_vs_baseline(
        series,
        w_day,
        label="witching_vs_other_fridays",
    )


# ---------------------------------------------------------------------------
# Return-leg: directional bias test
# ---------------------------------------------------------------------------

def witching_return_test(bars: pd.DataFrame) -> dict:
    """HAC t-test: do witching *days* have a statistically significant return?

    Tests both the day itself and the expiry week, and flags whether the effect
    clears the inference bar (|t| >= 2).
    """
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)

    day_mask = witching_day_mask(idx)
    week_mask = witching_week_mask(idx)

    day_res = compare_witching_vs_baseline(ret, day_mask, "witching_day_vs_rest")
    week_res = compare_witching_vs_baseline(ret, week_mask, "witching_week_vs_rest")
    return {"day": day_res, "week": week_res}


# ---------------------------------------------------------------------------
# Volume / volatility leg
# ---------------------------------------------------------------------------

def witching_vol_test(bars: pd.DataFrame) -> dict:
    """HAC t-tests: are range and volume elevated on witching days / weeks?

    These are the *mechanical* effects — the ones that follow directly from
    options expiry, delta-hedging, and institutional roll activity.

    Volume is measured as **year-demeaned log volume** — this removes the strong
    secular decline in SPY volume (as the ETF matures) and delivers a clean
    within-year comparison.  The raw log-volume test is also included for
    completeness.  Range is measured as (high − low) / close.
    """
    idx = pd.DatetimeIndex(bars.index)
    rng = daily_range_pct(bars)
    lvol_dm = log_volume_demeaned(bars)
    lvol_raw = log_volume(bars)

    day_mask = witching_day_mask(idx)
    week_mask = witching_week_mask(idx)

    return {
        "range_day": compare_witching_vs_baseline(rng, day_mask, "range_witching_day"),
        "range_week": compare_witching_vs_baseline(rng, week_mask, "range_witching_week"),
        "vol_day": compare_witching_vs_baseline(lvol_dm, day_mask, "volume_dm_witching_day"),
        "vol_week": compare_witching_vs_baseline(lvol_dm, week_mask, "volume_dm_witching_week"),
        "vol_raw_day": compare_witching_vs_baseline(lvol_raw, day_mask, "volume_raw_witching_day"),
    }


# ---------------------------------------------------------------------------
# Tradability: a simple witching-week long/short rule
# ---------------------------------------------------------------------------

def witching_week_returns(bars: pd.DataFrame, long_week: bool = True) -> pd.Series:
    """Daily returns from being long (or short) the SPY only in witching weeks.

    The position is fully determined by the exchange calendar — no signal from
    price data.  Because the calendar is known before the week begins, we enter
    at the *prior close* (no look-ahead).  Returns a daily return Series.

    ``long_week=True`` goes long; ``long_week=False`` goes short.
    Used to ask: "even if the vol effect is real, can you earn alpha trading around it?"
    """
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)
    week_mask = witching_week_mask(idx).reindex(ret.index).fillna(False)
    direction = 1.0 if long_week else -1.0
    return (ret * week_mask.values * direction).rename("strat")


def summarize(returns: pd.Series) -> dict:
    """Headline statistics for a daily return series: Sharpe, CAGR, and HAC t-stat on the mean.

    The HAC t-stat uses Newey-West lags calibrated to sample size, matching the
    inference bar in ``compare_witching_vs_baseline``.
    """
    r = returns.astype(float).dropna().to_numpy()
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return {k: float("nan") for k in ("n", "mean_bps", "sharpe_ann", "cagr", "tstat")}

    mu = r.mean()
    sig = r.std(ddof=1)
    cagr = float(np.prod(1.0 + r) ** (TRADING_DAYS_PER_YEAR / n) - 1.0)

    # Newey-West HAC t-stat on the mean
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
