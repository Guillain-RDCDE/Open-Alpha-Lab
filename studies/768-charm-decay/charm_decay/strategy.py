"""The charm-decay strategy and its honest controls — Study 768 (Charm-Decay).

The folk hypothesis: in the last week before monthly options expiration, the delta of the
dealer book decays with time (the Greek *charm*), forcing systematic re-hedging that pushes
SPY **up into OpEx** and lets it **give back after**.  We turn that into four falsifiable
measurements on the daily SPY close-to-close tape:

1. **Pre-OpEx drift.**  Mean return on the charm window (the 5 sessions ending on OpEx)
   vs all other days, Newey-West HAC t-stat.
2. **Post-OpEx give-back.**  Same, for the 5 sessions after OpEx.
3. **Pre-minus-post asymmetry.**  The difference of the two window means — the single number
   the "rally then fade" story predicts to be large and positive.
4. **Placebo / calendar randomisation.**  Slide the OpEx anchor by a fake number of trading
   days and re-measure the pre-window drift, building a null distribution of t-stats.  If the
   true OpEx anchor is not special, its t sits inside the placebo cloud — the honest
   falsification for any calendar-anchored claim.

Everything is direction-only close-to-close return; the position is set by the exchange
calendar, so no price look-ahead exists.  One execution lag is never needed for a
calendar-known window (the dates are fixed before the month begins), but we still enter at
the *prior* close so the first window day's full return is earned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    charm_window_mask,
    post_opex_mask,
    pre_opex_mask,
    quarterly_pre_opex_mask,
)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def daily_return(bars: pd.DataFrame) -> pd.Series:
    """Overnight-inclusive daily close-to-close simple return (total-return proxy)."""
    return bars["close"].pct_change().rename("ret")


def _hac_lags(n: int) -> int:
    """Newey-West rule-of-thumb lag length: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


# ---------------------------------------------------------------------------
# Core comparison engine — window vs baseline with a HAC t-stat
# ---------------------------------------------------------------------------

def compare_window_vs_baseline(
    series: pd.Series,
    mask: pd.Series,
    label: str = "window",
) -> dict:
    """Compare ``series`` on the window (``mask`` True) vs all other days.

    Returns means, counts, the mean difference, and a Newey-West HAC t-stat on the
    difference (regression of the outcome on the window indicator).  The HAC t-stat is the
    decisive number for the inference bar: |t| >= 2 on the real tape is the minimum for
    ``REAL``; below that, ``WEAK`` or ``NONE``.
    """
    s = series.astype(float).dropna()
    m = mask.reindex(s.index).fillna(False).astype(bool)
    win, base = s[m], s[~m]

    mean_w = float(win.mean()) if len(win) else float("nan")
    mean_b = float(base.mean()) if len(base) else float("nan")
    diff = mean_w - mean_b

    t = float("nan")
    if len(win) > 5 and len(base) > 5:
        n = len(s)
        x = m.values.astype(float)
        e = s.values - (mean_b + diff * x)      # OLS residuals, binary regressor
        lags = _hac_lags(n)
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
        "n_window": int(len(win)),
        "n_baseline": int(len(base)),
        "mean_window": mean_w,
        "mean_baseline": mean_b,
        "diff": float(diff),
        "tstat": float(t),
    }


# ---------------------------------------------------------------------------
# The four charm tests
# ---------------------------------------------------------------------------

def charm_drift_test(bars: pd.DataFrame, ndays: int = 5) -> dict:
    """Pre-OpEx and post-OpEx directional-drift HAC t-tests vs baseline."""
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)
    pre = pre_opex_mask(idx, ndays=ndays)
    post = post_opex_mask(idx, ndays=ndays)
    return {
        "pre": compare_window_vs_baseline(ret, pre, "pre_opex"),
        "post": compare_window_vs_baseline(ret, post, "post_opex"),
    }


def pre_post_asymmetry(bars: pd.DataFrame, ndays: int = 5) -> dict:
    """The 'rally then fade' number: mean pre-OpEx minus mean post-OpEx return.

    Tested with a HAC t-stat on a signed contrast series (+1 on pre days, -1 on post days,
    0 elsewhere) regressed against its own mean — i.e. the difference of the two window
    means, with autocorrelation-robust inference on the difference itself.
    """
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)
    pre = pre_opex_mask(idx, ndays=ndays).reindex(ret.index).fillna(False).values
    post = post_opex_mask(idx, ndays=ndays).reindex(ret.index).fillna(False).values

    both = pre | post
    r = ret.values[both]
    sign = np.where(pre[both], 1.0, -1.0)
    # difference of means == mean of (sign * demeaned) up to weighting; use two-sample HAC
    pre_vals = ret.values[pre]
    post_vals = ret.values[post]
    diff = float(pre_vals.mean() - post_vals.mean())

    # HAC t on the difference via the pooled window regression on a +1/-1 regressor
    x = sign
    xc = x - x.mean()
    y = r - r.mean()
    n = len(r)
    beta = float((xc @ y) / (xc @ xc)) if (xc @ xc) > 0 else float("nan")
    resid = y - beta * xc
    lags = _hac_lags(n)
    xe = xc * resid
    lrv = float(xe @ xe)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(xe[k:] @ xe[:-k])
    se = np.sqrt(max(lrv, 0.0) / ((xc @ xc) ** 2))
    t = float(beta / se) if se > 0 else float("nan")
    return {
        "diff_bps": diff * 1e4,
        "mean_pre_bps": float(pre_vals.mean() * 1e4),
        "mean_post_bps": float(post_vals.mean() * 1e4),
        "tstat": t,
        "n_pre": int(pre.sum()),
        "n_post": int(post.sum()),
    }


def placebo_randomization(
    bars: pd.DataFrame,
    ndays: int = 5,
    shifts: tuple[int, ...] | None = None,
) -> dict:
    """Slide the OpEx anchor by fake trading-day offsets and re-measure pre-window drift.

    Builds a null distribution of the pre-window HAC t-stat over anchors that are NOT the
    real OpEx (``shift`` in ``shifts``, excluding a neighbourhood of 0 that would overlap
    the true window).  The empirical two-sided p-value asks: is the real-anchor t-stat
    unusual against the cloud of arbitrary calendar anchors?  A true charm effect should
    put the real anchor in the tail; noise puts it in the body.
    """
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)
    if shifts is None:
        shifts = tuple(s for s in range(-30, 31) if abs(s) >= ndays)

    true_t = compare_window_vs_baseline(
        ret, pre_opex_mask(idx, ndays=ndays), "pre_true"
    )["tstat"]

    placebo_t = []
    for s in shifts:
        m = charm_window_mask(idx, lo=-(ndays - 1), hi=0, shift=s, name=f"pl{s}")
        placebo_t.append(compare_window_vs_baseline(ret, m)["tstat"])
    placebo_t = np.array([t for t in placebo_t if np.isfinite(t)])

    p_two = float((np.abs(placebo_t) >= abs(true_t)).mean()) if len(placebo_t) else float("nan")
    return {
        "true_t": float(true_t),
        "placebo_mean_abs_t": float(np.abs(placebo_t).mean()),
        "placebo_max_abs_t": float(np.abs(placebo_t).max()),
        "n_placebo": int(len(placebo_t)),
        "p_value": p_two,
        "placebo_t": placebo_t,
    }


def quarterly_split(bars: pd.DataFrame, ndays: int = 5) -> dict:
    """Pre-OpEx drift restricted to quarterly triple-witching months vs all-months."""
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)
    return {
        "all": compare_window_vs_baseline(ret, pre_opex_mask(idx, ndays=ndays), "pre_all"),
        "quarterly": compare_window_vs_baseline(
            ret, quarterly_pre_opex_mask(idx, ndays=ndays), "pre_quarterly"
        ),
    }


def pre_post_2012_split(bars: pd.DataFrame, ndays: int = 5) -> dict:
    """Split at 2012 — the era in which charm/vanna flow narratives became mainstream.

    Weekly options broadened after 2010 and 0DTE volume exploded post-2019; if the charm
    drift is a modern microstructure artefact it should be stronger in the later sub-period.
    """
    cut = pd.Timestamp("2012-01-01")

    def _sub(b: pd.DataFrame) -> dict:
        r = daily_return(b).dropna()
        i = pd.DatetimeIndex(r.index)
        return compare_window_vs_baseline(r, pre_opex_mask(i, ndays=ndays), "pre")

    return {
        "pre_2012": _sub(bars[bars.index < cut]),
        "post_2012": _sub(bars[bars.index >= cut]),
    }


# ---------------------------------------------------------------------------
# Tradability: the charm overlay
# ---------------------------------------------------------------------------

def charm_overlay_returns(
    bars: pd.DataFrame,
    ndays: int = 5,
    short_post: bool = False,
) -> pd.Series:
    """Daily returns from being long SPY in the pre-OpEx window (optionally short post).

    Position is fully calendar-determined — no price signal, no look-ahead.  With
    ``short_post=True`` the overlay also shorts the post-OpEx window (the full 'rally then
    fade' trade); shorts pay borrow, charged in :func:`summarize` via the cost sweep, not here.
    """
    ret = daily_return(bars).dropna()
    idx = pd.DatetimeIndex(ret.index)
    pos = pre_opex_mask(idx, ndays=ndays).reindex(ret.index).fillna(False).astype(float)
    if short_post:
        pos = pos - post_opex_mask(idx, ndays=ndays).reindex(ret.index).fillna(False).astype(float)
    return (ret * pos.values).rename("charm_overlay")


def summarize(returns: pd.Series) -> dict:
    """Headline statistics for a daily return series: Sharpe, CAGR, and HAC t-stat."""
    r = returns.astype(float).dropna().to_numpy()
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return {k: float("nan") for k in ("n", "mean_bps", "sharpe_ann", "cagr", "tstat")}

    mu = r.mean()
    sig = r.std(ddof=1)
    cagr = float(np.prod(1.0 + r) ** (TRADING_DAYS_PER_YEAR / n) - 1.0)

    e = r - mu
    lags = _hac_lags(n)
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


def cost_sweep(
    gross_mean_bps: float,
    trades_per_year: float = 24.0,
    active_days_per_year: float = 60.0,
    costs_bps=(0.0, 0.5, 1.0, 2.0),
) -> list[tuple]:
    """Net mean bps per active day after a one-way cost.

    ``trades_per_year`` counts one-way legs (the long-only charm overlay enters and exits
    once a month ~ 24 legs/yr) spread over ``active_days_per_year`` invested sessions, so the
    per-active-day drag of a one-way cost ``c`` is ``c * trades_per_year / active_days_per_year``.
    Returns ``[(cost_bps, net_mean_bps), ...]``.
    """
    return [
        (c, gross_mean_bps - c * (trades_per_year / active_days_per_year))
        for c in costs_bps
    ]
