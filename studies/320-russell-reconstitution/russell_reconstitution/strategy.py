"""Strategy and analysis layer for Study 320 (Russell-Reconstitution).

The folklore: the late-June Russell reconstitution forces a giant, one-directional,
*pre-announced* rebalancing flow into the small-cap index, so you can front-run it — buy
the index in the run-up window, sell on (or just after) the reconstitution Friday close.
We implement the event study on the index ETF itself (IWM / IWO) and pin each event window
against the only honest baselines:

  - The **unconditional daily mean** — the small-cap ETF drifts; an "is it positive?" claim
    that ignores the drift gets paid for the drift, not for any reconstitution magic.
  - The **same-month (June, non-event) mean** — a control that shares any June seasonality
    so we are not crediting "June" to "reconstitution". (Existence-only: a same-month
    control can support that an effect *exists* but never certifies its magnitude.)
  - A **one-sample t-test vs the unconditional mean** and its **Newey-West (HAC) t-stat** —
    the desk's REAL-signal hurdle is |t| >= 2 on the *excess* return, not the raw return.
  - A **permutation test**: draw n random run-up windows from the same tape and see where
    the event mean lands in the null.
  - An explicit **trading column**: buy at the close ``runup_days`` before reconstitution
    Friday, exit on the event close — gross and net of one-way costs on NAV.

The tiny-n reckoning: there are only ~26 reconstitutions (2000-2025). At ~1.3% daily vol a
single-session window's mean has SE ≈ 1.3%/√26 ≈ 0.26%, and a 5-session cumulative window's
SE is larger still — so a front-run edge must be sizeable to clear |t| = 2. That small n is
itself the headline risk for the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import data as _data


# ---------------------------------------------------------------------------
# Newey-West (HAC) t-stat
# ---------------------------------------------------------------------------
def hac_tstat(x, lag: int = 2) -> float:
    """Newey-West HAC t-stat for H0: mean(x) = 0 (Bartlett kernel, ``lag`` lags).

    For annual event windows the autocorrelation is mild; pass ``x - mu`` to test the
    *excess* return.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    u = x - m
    s = (u @ u) / n
    for l in range(1, lag + 1):
        if l >= n:
            break
        w = 1.0 - l / (lag + 1)
        cov = (u[l:] @ u[:-l]) / n
        s += 2.0 * w * cov
    se = np.sqrt(s / n)
    if se == 0:
        return float("nan")
    return float(m / se)


# ---------------------------------------------------------------------------
# Event-window construction
# ---------------------------------------------------------------------------
def _event_locs(index: pd.DatetimeIndex) -> list[int]:
    """Integer positions of the reconstitution session (Friday at/just before each event).

    For each calendar event date we snap to the session at or before it (handles holidays
    and the fact the cached ETF tape starts in 2000)."""
    locs = []
    for ev in _data.RECON_DATES:
        loc = index.searchsorted(ev, side="right") - 1
        if 0 <= loc < len(index) and index[loc] >= ev - pd.Timedelta(days=6):
            locs.append(int(loc))
    return locs


def build_windows(
    bars: pd.DataFrame,
    runup_days: int = 5,
    post_days: int = 5,
) -> pd.DataFrame:
    """One row per reconstitution event with the run-up, event-day, and post returns.

    Columns (all simple cumulative returns over the named window):
      - ``year``               : the reconstitution year
      - ``event_date``         : the session used as reconstitution Friday
      - ``runup_ret``          : close ``runup_days`` sessions before → event close
                                 (the front-run trade; the run-up EXCLUDES the next-day pop)
      - ``event_day_ret``      : the single event-Friday close-to-close return
      - ``post_ret``           : event close → ``post_days`` sessions after (the give-back)
    No execution lag is applied: the date is calendar-known years ahead.
    """
    close = bars["close"]
    idx = bars.index
    locs = _event_locs(idx)
    rows = []
    for loc in locs:
        lo = loc - runup_days
        if lo < 0:
            continue
        runup = float(close.iat[loc] / close.iat[lo] - 1.0)
        event_day = float(close.iat[loc] / close.iat[loc - 1] - 1.0)
        hi = loc + post_days
        post = float(close.iat[hi] / close.iat[loc] - 1.0) if hi < len(close) else np.nan
        rows.append(
            {
                "year": int(idx[loc].year),
                "event_date": idx[loc],
                "runup_ret": runup,
                "event_day_ret": event_day,
                "post_ret": post,
            }
        )
    return pd.DataFrame(rows)


def runup_windows_baseline(
    bars: pd.DataFrame,
    runup_days: int = 5,
    month: int | None = None,
) -> pd.Series:
    """A baseline distribution of *all* rolling ``runup_days``-session cumulative returns.

    This is the unconditional comparison for a multi-session run-up window: every
    ``runup_days``-bar block return on the tape. If ``month`` is given (e.g. 6 for June),
    restrict to blocks ENDING in that month — the same-month control that shares any June
    seasonality.
    """
    close = bars["close"]
    block = close / close.shift(runup_days) - 1.0
    block = block.dropna()
    if month is not None:
        block = block[block.index.month == month]
    return block


# ---------------------------------------------------------------------------
# Event-study statistics
# ---------------------------------------------------------------------------
def event_stats(
    event_ret: pd.Series,
    baseline_ret: pd.Series,
    n_permutations: int = 10_000,
    seed: int = 320,
    hac_lag: int = 2,
) -> dict:
    """Compare an event-window return series against a baseline distribution.

    ``event_ret``    : the per-year event-window returns (run-up, event-day, or post).
    ``baseline_ret`` : the matched unconditional distribution of same-length windows.

    Returns the event mean/std, baseline mean/std, excess, up-rates, one-sample t vs 0 and
    vs the baseline mean, the Newey-West HAC t of the excess (the headline hurdle), and a
    two-sided permutation p-value with the full permutation distribution.
    """
    ev = np.asarray(event_ret.dropna().to_numpy(dtype=float))
    base = np.asarray(baseline_ret.dropna().to_numpy(dtype=float))
    n = len(ev)

    ev_mean = float(ev.mean())
    ev_std = float(ev.std(ddof=1)) if n > 1 else float("nan")
    base_mean = float(base.mean())
    base_std = float(base.std(ddof=1))
    excess_mean = ev_mean - base_mean

    up_rate = float((ev > 0).mean())
    base_up_rate = float((base > 0).mean())

    t0 = scipy_stats.ttest_1samp(ev, 0.0)
    tb = scipy_stats.ttest_1samp(ev, base_mean)
    hac_t = hac_tstat(ev - base_mean, lag=hac_lag)

    rng = np.random.default_rng(seed)
    perm_means = np.empty(n_permutations, dtype=float)
    replace = len(base) < n
    for i in range(n_permutations):
        perm_means[i] = rng.choice(base, size=n, replace=replace).mean()
    obs_dev = abs(ev_mean - base_mean)
    perm_p = float((np.abs(perm_means - base_mean) >= obs_dev).mean())

    return {
        "n": int(n),
        "ev_mean": round(ev_mean, 8),
        "ev_std": round(ev_std, 8),
        "base_mean": round(base_mean, 8),
        "base_std": round(base_std, 8),
        "excess_mean": round(excess_mean, 8),
        "up_rate": round(up_rate, 4),
        "base_up_rate": round(base_up_rate, 4),
        "t_vs_zero": round(float(t0.statistic), 3),
        "p_vs_zero": round(float(t0.pvalue), 4),
        "t_vs_base": round(float(tb.statistic), 3),
        "p_vs_base": round(float(tb.pvalue), 4),
        "hac_t_excess": round(float(hac_t), 3),
        "perm_p": round(float(perm_p), 4),
        "perm_means": perm_means,
    }


# ---------------------------------------------------------------------------
# Sub-period breakdown (decay check)
# ---------------------------------------------------------------------------
def subperiod_stats(
    windows: pd.DataFrame,
    baseline_ret: pd.Series,
    col: str = "runup_ret",
    breaks: tuple[tuple[int, int], ...] = ((2000, 2008), (2009, 2016), (2017, 2025)),
    hac_lag: int = 2,
) -> pd.DataFrame:
    """HAC t-stat of the excess return for each sub-period (decay check).

    ``windows`` is the frame from :func:`build_windows`; the mean of ``baseline_ret`` is
    subtracted to form the excess. Columns: period, n, mean_bps, excess_bps, hac_t.
    """
    mu = float(np.asarray(baseline_ret.dropna()).mean())
    rows = []
    for lo, hi in breaks:
        sub = windows[(windows["year"] >= lo) & (windows["year"] <= hi)][col].dropna()
        ev = np.asarray(sub.to_numpy(dtype=float))
        if len(ev) == 0:
            continue
        ex = ev - mu
        rows.append(
            {
                "period": f"{lo}-{hi}",
                "n": int(len(ev)),
                "mean_bps": round(float(ev.mean()) * 1e4, 2),
                "excess_bps": round(float(ex.mean()) * 1e4, 2),
                "hac_t": round(hac_tstat(ex, lag=hac_lag), 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tradability: front-run rule, gross and net
# ---------------------------------------------------------------------------
def tradability(
    event_ret: pd.Series,
    cost_bps_oneway: float = 1.0,
) -> dict:
    """A 'buy the run-up window once a year, sell on the event close' rule, gross and net.

    One round-trip per year; round-trip cost = 2 * ``cost_bps_oneway`` per year on NAV.
    Costs are counted one-way × NAV (long-only, no borrow). Returns per-event mean
    gross/net, compounded gross/net growth, per-event cost drag, and per-event CAGR.
    """
    ev = np.asarray(event_ret.dropna().to_numpy(dtype=float))
    n = len(ev)
    rt_cost = 2.0 * cost_bps_oneway * 1e-4

    gross_mean = float(ev.mean())
    net_ev = ev - rt_cost
    net_mean = float(net_ev.mean())
    gross_cum = float(np.prod(1.0 + ev))
    net_cum = float(np.prod(1.0 + net_ev))

    return {
        "n_events": int(n),
        "gross_mean_per_event": round(gross_mean, 8),
        "net_mean_per_event": round(net_mean, 8),
        "cost_drag_per_event": round(rt_cost, 8),
        "gross_cum_growth": round(gross_cum, 6),
        "net_cum_growth": round(net_cum, 6),
        "gross_cagr": round(gross_cum ** (1.0 / n) - 1.0, 6) if n else float("nan"),
        "net_cagr": round(net_cum ** (1.0 / n) - 1.0, 6) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Summarize — compact dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    bars: pd.DataFrame,
    runup_days: int = 5,
    n_permutations: int = 10_000,
    seed: int = 320,
) -> dict:
    """One-stop summary dict mirroring the R = dict(...) in build_notebooks.py.

    Builds the event windows from ``bars``, compares the run-up against the unconditional
    rolling-window baseline, and packages the headline numbers (returns in %, excess in bps).
    """
    win = build_windows(bars, runup_days=runup_days)
    base = runup_windows_baseline(bars, runup_days=runup_days)
    es = event_stats(win["runup_ret"], base, n_permutations=n_permutations, seed=seed)

    # event-day and post (give-back) windows
    base1 = runup_windows_baseline(bars, runup_days=1)
    es_ev = event_stats(win["event_day_ret"], base1, n_permutations=n_permutations, seed=seed)
    base_post = runup_windows_baseline(bars, runup_days=runup_days)
    es_post = event_stats(win["post_ret"], base_post, n_permutations=n_permutations, seed=seed)

    tr = tradability(win["runup_ret"])
    return {
        "n": es["n"],
        "runup_days": runup_days,
        "runup_mean_pct": round(es["ev_mean"] * 100, 4),
        "runup_std_pct": round(es["ev_std"] * 100, 4),
        "base_mean_pct": round(es["base_mean"] * 100, 4),
        "excess_bps": round(es["excess_mean"] * 1e4, 2),
        "up_rate_pct": round(es["up_rate"] * 100, 2),
        "base_up_rate_pct": round(es["base_up_rate"] * 100, 2),
        "t_vs_zero": es["t_vs_zero"],
        "t_vs_base": es["t_vs_base"],
        "p_vs_base": es["p_vs_base"],
        "hac_t_excess": es["hac_t_excess"],
        "perm_p": es["perm_p"],
        "event_day_mean_pct": round(es_ev["ev_mean"] * 100, 4),
        "event_day_hac_t": es_ev["hac_t_excess"],
        "post_mean_pct": round(es_post["ev_mean"] * 100, 4),
        "post_hac_t": es_post["hac_t_excess"],
        "net_mean_per_event_bps": round(tr["net_mean_per_event"] * 1e4, 2),
        "gross_mean_per_event_bps": round(tr["gross_mean_per_event"] * 1e4, 2),
    }
