"""The FOMC-blackout engine and its honest controls — Study 322 (FOMC-Blackout).

The claim under test: the Fed's pre-meeting communications **blackout** — the ~10-day
quiet period before each FOMC decision when officials may not speak publicly — is a
"calm before the storm" window that carries an *excess* drift (and lower volatility)
over the rest of the year.

The decisive number is the *difference*: blackout mean minus non-blackout mean, with a
Welch t. A blackout window that merely earns the same equity drift as every other day is
not an edge — it is just being long the market. We therefore report each arm's HAC t
*and* the diff's Welch t side by side, and we never sell a non-blackout-baseline gain as
a blackout edge.

Tools:
    blackout_table  — blackout vs non-blackout vs all-days means, HAC t, Welch diff
    placebo_shift   — slide the window forward/back and re-measure the diff (a placebo)
    split_by_era    — pre/post-2011 contrast (post-publication decay of any drift)
    vol_table       — the "calm?" check: blackout vs non-blackout daily volatility
    cost_sweep      — a long-only-in-blackout timing book, net of one-way costs x NAV
    excess_sharpe   — excess-of-cash Sharpe of the timing book vs buy-and-hold

One execution lag is unnecessary here: the blackout flag is *calendar-known* months in
advance (the schedule is public), so a day's label uses no information from that day's
return. Costs are counted one-way x NAV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HAC (Newey-West) t-stat — no hard dependency on quantlab being importable
# ---------------------------------------------------------------------------
def hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of ``r`` (Bartlett kernel, rule-of-thumb lag)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch (unequal-variance) two-sample t-stat on ``mean(a) - mean(b)``."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def _arm_stats(r: np.ndarray, ann: int = 252) -> dict:
    """Mean (bps), annualised vol/Sharpe, and HAC t for a daily-return array."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return {"n": int(r.size), "mean_bps": float("nan"), "vol_ann_bps": float("nan"),
                "sharpe_ann": float("nan"), "tstat": float("nan")}
    mu = r.mean()
    sd = r.std(ddof=1)
    return {
        "n": int(r.size),
        "mean_bps": float(mu * 1e4),
        "vol_ann_bps": float(sd * np.sqrt(ann) * 1e4),
        "sharpe_ann": float(mu / sd * np.sqrt(ann)) if sd > 0 else float("nan"),
        "tstat": hac_tstat(r),
    }


# ---------------------------------------------------------------------------
# 1. Headline blackout vs non-blackout split
# ---------------------------------------------------------------------------
def blackout_table(frame: pd.DataFrame, ann: int = 252) -> dict:
    """Headline blackout vs non-blackout vs all-days comparison.

    ``frame`` has columns ``ret`` and ``blackout`` (bool). Returns a dict with arm
    stats for ``blackout``, ``other`` and ``all``, the diff in bps, and the Welch t on
    the *difference* — the decisive test. The all-days baseline makes explicit that the
    blackout arm only counts as an *edge* if it beats the rest of the year, not merely if
    it is positive (being long the market is positive).
    """
    blk = frame.loc[frame["blackout"], "ret"].to_numpy()
    oth = frame.loc[~frame["blackout"], "ret"].to_numpy()
    allr = frame["ret"].to_numpy()
    return {
        "blackout": _arm_stats(blk, ann),
        "other": _arm_stats(oth, ann),
        "all": _arm_stats(allr, ann),
        "diff_bps": float(np.nanmean(blk) * 1e4 - np.nanmean(oth) * 1e4),
        "welch_t": _welch_t(blk, oth),
    }


# ---------------------------------------------------------------------------
# 2. Placebo — slide the window by a calendar offset and re-measure
# ---------------------------------------------------------------------------
def placebo_shift(
    frame: pd.DataFrame,
    fomc_dates: pd.DatetimeIndex,
    offsets: list[int] | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Calendar-shift placebo: slide the blackout window and re-measure the diff.

    For each ``offset`` (calendar days, 0 = the true rule), rebuild the blackout flag
    shifted by that many days and recompute the blackout-minus-other diff and its Welch
    t. A real blackout effect should peak at offset 0 and fade as the window slides off
    the meeting; a flat profile across offsets says the "blackout" label carries no
    special information.

    ``frame`` must be indexed by trading date. ``window`` defaults to the package's
    ``BLACKOUT_DAYS``.
    """
    from .data import BLACKOUT_DAYS, in_blackout

    if offsets is None:
        offsets = [-20, -10, -5, 0, 5, 10, 20]
    if window is None:
        window = BLACKOUT_DAYS

    ret = frame["ret"]
    rows = []
    for off in offsets:
        flag = in_blackout(ret.index, fomc_dates, window=window, offset=off)
        blk = ret[flag.values].to_numpy()
        oth = ret[~flag.values].to_numpy()
        rows.append({
            "offset_days": off,
            "n_blackout": int(np.isfinite(blk).sum()),
            "diff_bps": float(np.nanmean(blk) * 1e4 - np.nanmean(oth) * 1e4),
            "welch_t": _welch_t(blk, oth),
        })
    return pd.DataFrame(rows).set_index("offset_days")


# ---------------------------------------------------------------------------
# 3. Pre- vs post-2011 split (decay / robustness)
# ---------------------------------------------------------------------------
def split_by_era(frame: pd.DataFrame, cut: str = "2011-01-01", ann: int = 252) -> dict:
    """Blackout-minus-other diff before and after ``cut``.

    The pre-FOMC drift literature (Lucca-Moench 2015) was published mid-2010s; if any
    blackout drift were a tradable anomaly we would expect post-publication decay
    (McLean-Pontiff 2016). We split at ``cut`` and report the diff and Welch t in each
    era.
    """
    cut_ts = pd.Timestamp(cut)
    pre = frame[frame.index < cut_ts]
    post = frame[frame.index >= cut_ts]

    def _gap(df: pd.DataFrame) -> dict:
        blk = df.loc[df["blackout"], "ret"].to_numpy()
        oth = df.loc[~df["blackout"], "ret"].to_numpy()
        return {
            "diff_bps": float(np.nanmean(blk) * 1e4 - np.nanmean(oth) * 1e4)
            if blk.size and oth.size else float("nan"),
            "welch_t": _welch_t(blk, oth),
            "blackout_mean_bps": float(np.nanmean(blk) * 1e4) if blk.size else float("nan"),
            "other_mean_bps": float(np.nanmean(oth) * 1e4) if oth.size else float("nan"),
            "n_blackout": int(np.isfinite(blk).sum()),
        }

    return {"pre": _gap(pre), "post": _gap(post), "cut": cut}


# ---------------------------------------------------------------------------
# 4. The "calm?" check — volatility inside vs outside the window
# ---------------------------------------------------------------------------
def vol_table(frame: pd.DataFrame, ann: int = 252) -> dict:
    """Daily volatility, blackout vs non-blackout (the 'calm before the storm' check).

    Returns each arm's annualised vol (bps) and the ratio. ``ratio < 1`` means the
    blackout window is genuinely calmer; ``>= 1`` busts the calm framing.
    """
    blk = frame.loc[frame["blackout"], "ret"].to_numpy()
    oth = frame.loc[~frame["blackout"], "ret"].to_numpy()
    blk = blk[np.isfinite(blk)]; oth = oth[np.isfinite(oth)]
    v_blk = float(blk.std(ddof=1) * np.sqrt(ann) * 1e4) if blk.size > 1 else float("nan")
    v_oth = float(oth.std(ddof=1) * np.sqrt(ann) * 1e4) if oth.size > 1 else float("nan")
    return {
        "blackout_vol_bps": v_blk,
        "other_vol_bps": v_oth,
        "ratio": v_blk / v_oth if v_oth > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# 5. Tradability — long-only-in-blackout timing book vs buy-and-hold
# ---------------------------------------------------------------------------
def cost_sweep(
    frame: pd.DataFrame,
    costs_bps: list[float] | None = None,
    rf_annual: float = 0.0,
    ann: int = 252,
) -> pd.DataFrame:
    """Net annualised return of a 'long the market only during blackout' timing book.

    The book holds the index on blackout days and sits in cash (earning ``rf_annual``)
    otherwise. It rebalances at each window edge (enter at the start, exit at the end),
    so turnover is two one-way trips per meeting; we charge ``cost_bps`` one-way x NAV at
    each transition. Compared against buy-and-hold of the same tape on an
    **excess-of-cash** basis (both legs measured net of the same cash rate) so the race
    is apples-to-apples.

    Returns a DataFrame indexed by cost, with the timing book's gross/net annualised
    return and the buy-and-hold benchmark for reference.
    """
    if costs_bps is None:
        costs_bps = [0.0, 1.0, 2.0, 5.0]

    r = frame["ret"].to_numpy(dtype=float)
    blk = frame["blackout"].to_numpy()
    rf_daily = rf_annual / ann

    # Position: 1 on blackout days, 0 otherwise. One execution lag is unnecessary —
    # the schedule is known in advance — so the position applies to the same day's return.
    pos = blk.astype(float)
    # Transitions = number of position changes (one-way trips).
    switches = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], pos])))))
    n = len(r)
    yrs = n / ann
    trips_per_year = switches / yrs if yrs > 0 else 0.0

    # Gross daily book return: market on blackout days, cash otherwise.
    book_gross = np.where(blk, r, rf_daily)
    gross_ann = float(np.nanmean(book_gross) * ann)

    bh_ann = float(np.nanmean(r) * ann)

    rows = []
    for c in costs_bps:
        annual_cost = trips_per_year * c * 1e-4
        net_ann = gross_ann - annual_cost
        rows.append({
            "cost_bps": c,
            "book_gross_ann_pct": gross_ann * 100,
            "book_net_ann_pct": net_ann * 100,
            "buyhold_ann_pct": bh_ann * 100,
            "trips_per_year": trips_per_year,
        })
    return pd.DataFrame(rows).set_index("cost_bps")


def excess_sharpe(frame: pd.DataFrame, rf_annual: float = 0.0, ann: int = 252) -> dict:
    """Excess-of-cash annualised Sharpe of the blackout timing book vs buy-and-hold.

    Both the timing book (long on blackout days, cash otherwise) and buy-and-hold are
    measured net of the same daily cash rate, so the comparison is excess-vs-excess.
    """
    r = frame["ret"].to_numpy(dtype=float)
    blk = frame["blackout"].to_numpy()
    rf_daily = rf_annual / ann

    book = np.where(blk, r, rf_daily) - rf_daily
    bh = r - rf_daily

    def _sh(x):
        x = x[np.isfinite(x)]
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")

    return {"book_excess_sharpe": _sh(book), "buyhold_excess_sharpe": _sh(bh)}
