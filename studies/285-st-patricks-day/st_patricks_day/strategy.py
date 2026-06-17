"""The strategy and its honest controls — Study 285 (St-Patricks-Day).

The folk claim: the S&P 500 enjoys a "St. Patrick's Day bump" — March 17 (or the
nearest trading session) is supposed to deliver an unusually green day. We run an
event study and four honest checks:

1. **St. Patrick's session vs all other trading days** (``stpat_comparison``).
   HAC Newey-West t-stat on the event-day returns, plus a Welch t-test on the
   contrast against every non-event day.

2. **St. Patrick's session vs all other MARCH days** — the tighter control that
   removes any generic March/spring seasonality and isolates the day itself.

3. **Placebo: March 24 (one week later)** (``placebo_comparison``). Same month,
   same late-March proximity, no folklore. If the placebo looks just as green,
   the St. Patrick result is noise.

4. **Multiple-comparisons sweep** (``day_of_march_sweep``). The St. Patrick's Day
   isn't pre-registered before anyone looked — it was picked *because* of the
   holiday. A snooper testing many March day-of-month slots would find the most
   extreme one by chance. We Bonferroni-correct across a grid of March anchor days.

5. **Sub-period split** (``subperiod_table``) — the kill shot for this particular
   tape: any "effect" that lives in one regime and reverses in the others is a
   regime artifact, not a signal.

No look-ahead: the event label (today is the St. Patrick session) is known before
the open; the return is that day's close-to-close log-return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import STPAT_DAY, PLACEBO_DAY, first_trading_on_or_after


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-statistic on a sample mean
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> dict:
    """Newey-West (HAC) t-statistic on the sample mean of ``r``."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return {"tstat": float("nan"), "mean": float(r.mean()) if n else float("nan"),
                "se": float("nan"), "n": n}
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "mean": float(mu),
        "se": float(se),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Primary test: St. Patrick's session vs all other days
# ---------------------------------------------------------------------------
def stpat_comparison(df: pd.DataFrame) -> dict:
    """St. Patrick's session returns vs all other trading days, and vs other March days.

    Parameters
    ----------
    df : DataFrame with columns ``ret, is_stpat, is_placebo, is_march``.

    Returns
    -------
    dict with the event mean, HAC t-stat, the two control means, and Welch
    contrasts (vs all other days and vs other March days).
    """
    from scipy.stats import ttest_ind

    stpat = df.loc[df["is_stpat"], "ret"].to_numpy(dtype=float)
    stpat = stpat[np.isfinite(stpat)]
    all_other = df.loc[~df["is_stpat"], "ret"].to_numpy(dtype=float)
    all_other = all_other[np.isfinite(all_other)]
    march_other = df.loc[df["is_march"] & ~df["is_stpat"], "ret"].to_numpy(dtype=float)
    march_other = march_other[np.isfinite(march_other)]

    h = _hac_tstat(stpat)

    if len(stpat) > 1 and len(all_other) > 1:
        t_all, p_all = ttest_ind(stpat, all_other, equal_var=False)
    else:
        t_all, p_all = float("nan"), float("nan")
    if len(stpat) > 1 and len(march_other) > 1:
        t_mar, p_mar = ttest_ind(stpat, march_other, equal_var=False)
    else:
        t_mar, p_mar = float("nan"), float("nan")

    return {
        "n_stpat": int(len(stpat)),
        "mean_stpat_bps": float(stpat.mean() * 1e4) if len(stpat) else float("nan"),
        "tstat_stpat_hac": h["tstat"],
        "n_all_other": int(len(all_other)),
        "mean_all_bps": float(all_other.mean() * 1e4) if len(all_other) else float("nan"),
        "contrast_vs_all_bps": float((stpat.mean() - all_other.mean()) * 1e4)
        if len(stpat) and len(all_other) else float("nan"),
        "tstat_welch_all": float(t_all),
        "pval_welch_all": float(p_all),
        "n_march_other": int(len(march_other)),
        "mean_march_other_bps": float(march_other.mean() * 1e4) if len(march_other) else float("nan"),
        "contrast_vs_march_bps": float((stpat.mean() - march_other.mean()) * 1e4)
        if len(stpat) and len(march_other) else float("nan"),
        "tstat_welch_march": float(t_mar),
        "pval_welch_march": float(p_mar),
    }


# ---------------------------------------------------------------------------
# Placebo: March 24 (one week later) vs other March days
# ---------------------------------------------------------------------------
def placebo_comparison(df: pd.DataFrame) -> dict:
    """Same test as ``stpat_comparison`` but for the March-24 placebo session."""
    from scipy.stats import ttest_ind

    plac = df.loc[df["is_placebo"], "ret"].to_numpy(dtype=float)
    plac = plac[np.isfinite(plac)]
    march_other = df.loc[df["is_march"] & ~df["is_placebo"], "ret"].to_numpy(dtype=float)
    march_other = march_other[np.isfinite(march_other)]

    h = _hac_tstat(plac)
    if len(plac) > 1 and len(march_other) > 1:
        t_mar, p_mar = ttest_ind(plac, march_other, equal_var=False)
    else:
        t_mar, p_mar = float("nan"), float("nan")

    return {
        "n_placebo": int(len(plac)),
        "mean_placebo_bps": float(plac.mean() * 1e4) if len(plac) else float("nan"),
        "tstat_placebo_hac": h["tstat"],
        "contrast_vs_march_bps": float((plac.mean() - march_other.mean()) * 1e4)
        if len(plac) and len(march_other) else float("nan"),
        "tstat_welch_march": float(t_mar),
        "pval_welch_march": float(p_mar),
    }


# ---------------------------------------------------------------------------
# Multiple-comparisons sweep over March anchor days
# ---------------------------------------------------------------------------
def day_of_march_sweep(df: pd.DataFrame, days: list[int] | None = None) -> pd.DataFrame:
    """Test the 'first trading day on/after day-N of March' for several N.

    The St. Patrick's Day (N=17) is one of many candidate March anchor days a
    snooper could have tested. We Bonferroni-correct the Welch p-values across the
    grid. Returns a DataFrame sorted by raw p-value so the most extreme slot is on
    top.

    This is the multiple-comparisons kill shot: a raw p of 0.30 on the 17th is not
    impressive against a Bonferroni bar of 0.05/len(days).
    """
    from scipy.stats import ttest_ind

    if days is None:
        days = [3, 10, 17, 24, 31]
    idx = pd.DatetimeIndex(df.index)
    n_tests = len(days)
    rows = []
    for d in days:
        ev = first_trading_on_or_after(idx, d)
        mask = idx.isin(ev)
        g = df.loc[mask, "ret"].to_numpy(dtype=float)
        g = g[np.isfinite(g)]
        rest = df.loc[df["is_march"] & ~mask, "ret"].to_numpy(dtype=float)
        rest = rest[np.isfinite(rest)]
        if len(g) > 1 and len(rest) > 1:
            _, pval = ttest_ind(g, rest, equal_var=False)
        else:
            pval = float("nan")
        rows.append({
            "anchor_day": int(d),
            "n": int(len(g)),
            "mean_bps": float(g.mean() * 1e4) if len(g) else float("nan"),
            "contrast_bps": float((g.mean() - rest.mean()) * 1e4)
            if len(g) and len(rest) else float("nan"),
            "pval_raw": float(pval),
            "pval_bonferroni": float(min(pval * n_tests, 1.0)) if np.isfinite(pval) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("pval_raw").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sub-period split — the regime-stability kill shot
# ---------------------------------------------------------------------------
def subperiod_table(df: pd.DataFrame, edges: list[int] | None = None) -> pd.DataFrame:
    """St. Patrick's session mean + HAC t-stat within each sub-period.

    A genuine signal should be reasonably stable across regimes. If the whole
    'effect' lives in one slice and reverses in the others, it is a regime
    artifact. ``edges`` are the (lo, hi) calendar-year boundaries.
    """
    if edges is None:
        edges = [(1928, 1960), (1961, 1990), (1991, 2026)]
    years = pd.DatetimeIndex(df.index).year
    rows = []
    for lo, hi in edges:
        mask = df["is_stpat"].to_numpy() & (years >= lo) & (years <= hi)
        g = df.loc[mask, "ret"].to_numpy(dtype=float)
        g = g[np.isfinite(g)]
        h = _hac_tstat(g)
        rows.append({
            "period": f"{lo}-{hi}",
            "n": int(len(g)),
            "mean_bps": float(g.mean() * 1e4) if len(g) else float("nan"),
            "tstat_hac": h["tstat"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost-aware "trade the bump" P&L
# ---------------------------------------------------------------------------
def strategy_pnl(df: pd.DataFrame, one_way_bps: float = 1.0) -> dict:
    """Long the index only on the St. Patrick's session each year; gross & net.

    The 'strategy' goes flat (cash) every day except the St. Patrick's session,
    when it is fully long the index close-to-close. That is one round-trip per
    year, so the cost is ``2 * one_way_bps`` per event. We report the gross and
    net mean event return and HAC t-stats, plus the buy-and-hold benchmark.
    """
    stpat = df.loc[df["is_stpat"], "ret"].to_numpy(dtype=float)
    stpat = stpat[np.isfinite(stpat)]
    roundtrip = 2.0 * one_way_bps * 1e-4
    net = stpat - roundtrip

    gross_h = _hac_tstat(stpat)
    net_h = _hac_tstat(net)

    all_ret = df["ret"].to_numpy(dtype=float)
    all_ret = all_ret[np.isfinite(all_ret)]
    n_days = len(all_ret)
    bah_ann = float(np.expm1(all_ret.sum() / n_days * 252)) if n_days else float("nan")
    # Strategy is long ~1 day/yr; annualised contribution if compounded once/yr.
    n_events = len(stpat)
    strat_ann = float(np.expm1(net.sum() / max(n_events, 1))) if n_events else float("nan")

    return {
        "n_events": int(n_events),
        "gross_mean_bps": float(stpat.mean() * 1e4) if n_events else float("nan"),
        "gross_tstat_hac": gross_h["tstat"],
        "net_mean_bps": float(net.mean() * 1e4) if n_events else float("nan"),
        "net_tstat_hac": net_h["tstat"],
        "roundtrip_bps": float(roundtrip * 1e4),
        "bah_daily_ann_pct": round(bah_ann * 100, 2),
        "strat_per_event_pct": round(strat_ann * 100, 4),
    }


# ---------------------------------------------------------------------------
# Summarize — all checks in one call (mirrors docs/results.md)
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers for the St. Patrick's Day bump — all checks combined."""
    pri = stpat_comparison(df)
    plac = placebo_comparison(df)
    mc = day_of_march_sweep(df)
    sub = subperiod_table(df)
    pnl = strategy_pnl(df, one_way_bps=1.0)

    row17 = mc[mc["anchor_day"] == STPAT_DAY]
    bonf17 = float(row17["pval_bonferroni"].iloc[0]) if len(row17) else float("nan")

    return {
        # Primary
        "n_stpat": pri["n_stpat"],
        "mean_stpat_bps": pri["mean_stpat_bps"],
        "tstat_stpat_hac": pri["tstat_stpat_hac"],
        "mean_all_bps": pri["mean_all_bps"],
        "contrast_vs_all_bps": pri["contrast_vs_all_bps"],
        "tstat_welch_all": pri["tstat_welch_all"],
        "pval_welch_all": pri["pval_welch_all"],
        "mean_march_other_bps": pri["mean_march_other_bps"],
        "contrast_vs_march_bps": pri["contrast_vs_march_bps"],
        "tstat_welch_march": pri["tstat_welch_march"],
        "pval_welch_march": pri["pval_welch_march"],
        # Placebo
        "n_placebo": plac["n_placebo"],
        "mean_placebo_bps": plac["mean_placebo_bps"],
        "pval_placebo_march": plac["pval_welch_march"],
        # Multiple comparisons
        "pval_bonferroni_17": bonf17,
        "mc_table": mc,
        # Sub-periods
        "subperiod_table": sub,
        # Strategy
        "net_mean_bps": pnl["net_mean_bps"],
        "net_tstat_hac": pnl["net_tstat_hac"],
        "strat_per_event_pct": pnl["strat_per_event_pct"],
    }
