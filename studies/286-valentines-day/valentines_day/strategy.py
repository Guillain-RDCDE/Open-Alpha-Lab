"""Strategy and analysis layer for Study 286 (Valentines-Day).

The Valentine's Day folklore: the market is "in love" on Feb 14, so the S&P 500
prints an unusually positive return that session. We implement the event study and
pin the Valentine-session return against the only honest baseline:

  - The **unconditional daily mean** -- how much does the S&P move on an *average*
    day? An "is the day positive?" claim that ignores the market's tiny positive
    daily drift gets credit for the drift, not for any Valentine's magic.
  - A **one-sample t-test vs the unconditional mean** (the excess return), and its
    **Newey-West (HAC) t-stat** -- the desk's REAL-signal hurdle is |t| >= 2 on the
    excess return, not on the raw return.
  - A **permutation test**: draw n random sessions from the same window 10,000
    times and see where the Valentine mean lands in the null distribution.
  - An explicit **trading column**: a "buy the Valentine's session" rule, gross and
    net of one-way costs, against buy-and-hold-this-one-day.

The tiny-n / tiny-effect reckoning: there are 76 Valentine sessions (1950-2025).
With ~1% daily volatility, the mean of 76 days has a standard error of ~0.11%/day,
so a "Valentine premium" would need to be ~0.23%/day to clear |t| = 2 -- about six
times the market's whole-day average drift. The folklore is nowhere near that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Newey-West (HAC) t-stat for the mean of a series
# ---------------------------------------------------------------------------
def hac_tstat(x, lag: int = 4) -> float:
    """Newey-West HAC t-stat for H0: mean(x) = 0.

    Bartlett-kernel long-run variance with ``lag`` lags. Robust to the mild serial
    correlation in daily returns. To test the *excess* return, pass ``x - mu``.
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
# Event-study statistics
# ---------------------------------------------------------------------------
def event_stats(
    event_ret: pd.Series,
    uncond_ret: pd.Series,
    n_permutations: int = 10_000,
    seed: int = 286,
    hac_lag: int = 4,
) -> dict:
    """Compare the Valentine-session returns against the unconditional daily baseline.

    Parameters
    ----------
    event_ret : Series of Valentine-session returns (one per year)
    uncond_ret : Series of *all* daily returns over the window (the baseline)
    n_permutations : permutation draws
    seed : RNG seed for permutations
    hac_lag : Newey-West lag

    Returns a dict with:
      - ``n``: number of Valentine sessions
      - ``val_mean``, ``val_std``: Valentine mean/std (simple return)
      - ``uncond_mean``, ``uncond_std``: unconditional daily mean/std
      - ``excess_mean``: val_mean - uncond_mean
      - ``up_rate``, ``uncond_up_rate``: fraction of positive days
      - ``t_vs_zero``, ``p_vs_zero``: one-sample t of Valentine returns vs 0
      - ``t_vs_uncond``, ``p_vs_uncond``: one-sample t of Valentine vs unconditional mean
      - ``hac_t_excess``: Newey-West HAC t of the excess return (the headline hurdle)
      - ``perm_p``: two-sided permutation p-value vs the unconditional mean
      - ``perm_means``: the full permutation distribution of n-session means
    """
    ev = np.asarray(event_ret.dropna().to_numpy(dtype=float))
    allr = np.asarray(uncond_ret.dropna().to_numpy(dtype=float))
    n = len(ev)

    val_mean = float(ev.mean())
    val_std = float(ev.std(ddof=1))
    uncond_mean = float(allr.mean())
    uncond_std = float(allr.std(ddof=1))
    excess_mean = val_mean - uncond_mean

    up_rate = float((ev > 0).mean())
    uncond_up_rate = float((allr > 0).mean())

    t0 = scipy_stats.ttest_1samp(ev, 0.0)
    tu = scipy_stats.ttest_1samp(ev, uncond_mean)

    hac_t = hac_tstat(ev - uncond_mean, lag=hac_lag)

    rng = np.random.default_rng(seed)
    perm_means = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm_means[i] = rng.choice(allr, size=n, replace=False).mean()
    obs_dev = abs(val_mean - uncond_mean)
    perm_p = float((np.abs(perm_means - uncond_mean) >= obs_dev).mean())

    return {
        "n": int(n),
        "val_mean": round(val_mean, 8),
        "val_std": round(val_std, 8),
        "uncond_mean": round(uncond_mean, 8),
        "uncond_std": round(uncond_std, 8),
        "excess_mean": round(excess_mean, 8),
        "up_rate": round(up_rate, 4),
        "uncond_up_rate": round(uncond_up_rate, 4),
        "t_vs_zero": round(float(t0.statistic), 3),
        "p_vs_zero": round(float(t0.pvalue), 4),
        "t_vs_uncond": round(float(tu.statistic), 3),
        "p_vs_uncond": round(float(tu.pvalue), 4),
        "hac_t_excess": round(float(hac_t), 3),
        "perm_p": round(float(perm_p), 4),
        "perm_means": perm_means,
    }


# ---------------------------------------------------------------------------
# Tradability: buy-the-Valentine-session rule, gross and net
# ---------------------------------------------------------------------------
def tradability(
    event_ret: pd.Series,
    cost_bps_oneway: float = 1.0,
) -> dict:
    """A 'buy the Valentine's session once a year' rule, gross and net of costs.

    The rule: enter at the prior close, exit at the Valentine close (a one-session
    hold). Round-trip cost = 2 * ``cost_bps_oneway`` per year applied to NAV. We
    compare to the *same one day* of buy-and-hold (which is just the gross event
    return -- the point being there is no edge to harvest).

    Returns mean gross/net per-event return, annualised compounded gross/net, and
    the per-event cost drag.
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
        # "Annualised" here means: average compounded growth per event-year.
        "gross_cagr": round(gross_cum ** (1.0 / n) - 1.0, 6) if n else float("nan"),
        "net_cagr": round(net_cum ** (1.0 / n) - 1.0, 6) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Summarize -- compact dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    event_ret: pd.Series,
    uncond_ret: pd.Series,
    n_permutations: int = 10_000,
    seed: int = 286,
) -> dict:
    """One-stop summary dict mirroring the R = dict(...) in build_notebooks.py.

    All return numbers are reported in percent (×100) for readability.
    """
    es = event_stats(event_ret, uncond_ret, n_permutations=n_permutations, seed=seed)
    tr = tradability(event_ret)
    return {
        "n": es["n"],
        "val_mean_pct": round(es["val_mean"] * 100, 4),
        "val_std_pct": round(es["val_std"] * 100, 4),
        "uncond_mean_pct": round(es["uncond_mean"] * 100, 4),
        "uncond_std_pct": round(es["uncond_std"] * 100, 4),
        "excess_mean_bps": round(es["excess_mean"] * 1e4, 2),
        "up_rate_pct": round(es["up_rate"] * 100, 2),
        "uncond_up_rate_pct": round(es["uncond_up_rate"] * 100, 2),
        "t_vs_zero": es["t_vs_zero"],
        "t_vs_uncond": es["t_vs_uncond"],
        "p_vs_uncond": es["p_vs_uncond"],
        "hac_t_excess": es["hac_t_excess"],
        "perm_p": es["perm_p"],
        "net_mean_per_event_bps": round(tr["net_mean_per_event"] * 1e4, 2),
        "gross_mean_per_event_bps": round(tr["gross_mean_per_event"] * 1e4, 2),
    }
