"""Strategy + inference for Study 733 — Kentucky-Derby-Effect.

The claim: the first-Saturday-in-May Kentucky Derby is a market event — either a broad
**market seasonal** (Derby weekend = the "Sell in May" boundary) or, more sharply, a
**pop in the one stock that runs the race**, Churchill Downs Inc. (``CHDN``).

The Derby is run on a **Saturday night** (post) — a non-trading day — so the result
(and the weekend of wall-to-wall coverage, betting handle, attendance records) cannot
be acted on until markets reopen. That gives the study its single, unavoidable execution
lag for free:

* **day(-1)** = the last trading close *before* the race (ordinarily the Friday) — the
  last price that does NOT yet know how the weekend went.
* **day(0)**  = the first trading close *after* the race (ordinarily the Monday) — the
  first price that DOES. Entering at day(0)'s close is zero-look-ahead by construction.

But the Derby's *date* is fixed years in advance (always the first Saturday in May), so
a **run-up** window — day(-6)->day(-1), the trading week *into* the race — is *also*
fully tradable with no look-ahead at all (you know the date, you just don't yet know the
result). We test both.

Two legs, each measured as a windowed abnormal return over horizon k:

* **CHDN (the gambling name).** Abnormal = CHDN's window return minus SPY's (a beta=1
  market model): does the company that runs the Derby beat the market around its marquee
  event? Signal windows run from day(-1); the tradable capture enters at day(0) (post-
  result) or over the calendar-known run-up, net of one round trip of costs.
* **Market (the seasonal).** SPY's own window return with its full-sample daily drift
  removed (a constant-mean model): does the broad market do anything unusual in Derby
  week? The random-window placebo is the primary test here — a random same-length window
  carries the same drift, so it is the honest null.

Each Derby year is a single, independent, non-overlapping event, so the primary
statistic is a **one-sample t** across events (not a daily panel). A random-window
placebo (same-length windows drawn at random points in the same ticker's own history)
checks whether the observed mean sits outside the ordinary luck cloud; a Wilson interval
bounds the hit rate; and a costed capture test asks whether any of it is bankable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

RUNUP_K = 5    # trading sessions into the race: day(-6) -> day(-1) (calendar-known)
WEEK_K = 5     # trading sessions ~ 1 week after
MONTH_K = 21   # trading sessions ~ 1 month after
COST_BPS = 10.0  # one-way, per leg -- a mid-cap single stock (CHDN), wider than an ETF


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event abnormal returns, both legs
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per (year, leg) Derby event: windowed abnormal returns + inclusion flag.

    ``leg`` is "chdn" (the gambling name, abnormal vs SPY) or "market" (the SPY seasonal,
    drift-removed). A row is INCLUDED only if the tape covers
    [day(-6) .. day(-1)+MONTH_K] around that event. The market leg additionally requires
    ``ran_in_may`` (the 2020 September running is not a first-Saturday-in-May seasonal
    observation and is dropped from that leg, kept for CHDN).
    """
    spy = prices[dt.MARKET]
    chdn = prices[dt.GAMBLING]
    common = spy.index.intersection(chdn.index).sort_values()
    spy = spy.reindex(common)
    chdn = chdn.reindex(common)

    # full-sample SPY daily drift (for the constant-mean market-seasonal model)
    spy_daily_mean = float(spy.pct_change().mean())

    rows = []
    for year, date_str, ran_in_may in dt.EVENTS:
        d = pd.Timestamp(date_str)
        before = common[common < d]
        for leg in ("chdn", "market"):
            row = dict(year=year, leg=leg, date=date_str, ran_in_may=ran_in_may)
            if leg == "market" and not ran_in_may:
                row.update(included=False, reason="2020 ran in September, not May")
                rows.append(row)
                continue
            if len(before) < RUNUP_K + 1:
                row.update(included=False, reason="insufficient leading history")
                rows.append(row)
                continue
            p = common.get_loc(before[-1])                 # day(-1)
            if p - RUNUP_K < 0 or p + MONTH_K >= len(common):
                row.update(included=False, reason="window runs off the tape")
                rows.append(row)
                continue

            d_pre = common[p - RUNUP_K]     # day(-6): run-up anchor
            d_m1 = common[p]                # day(-1): pre-race close
            d_0 = common[p + 1]             # day(0): first post-race close
            d_wk = common[p + WEEK_K]
            d_mo = common[p + MONTH_K]

            def ab(anchor, end, leg=leg):
                """Windowed abnormal return anchor->end. CHDN: relative to SPY (beta=1).
                Market: SPY window return minus expected drift (constant-mean model)."""
                n = common.get_loc(end) - common.get_loc(anchor)
                r_spy = spy.loc[end] / spy.loc[anchor] - 1.0
                if leg == "chdn":
                    r_chdn = chdn.loc[end] / chdn.loc[anchor] - 1.0
                    return float(r_chdn - r_spy)
                return float(r_spy - n * spy_daily_mean)

            # signal windows (from day(-1); include the un-tradable weekend jump)
            ar_runup = ab(d_pre, d_m1)      # day(-6) -> day(-1): the week INTO the race
            ar_week = ab(d_m1, d_wk)
            ar_month = ab(d_m1, d_mo)
            # tradable capture (enter day(0), after the result is public), net of costs
            rt = 2.0 * cost_bps / 1e4
            cap_week_g = ab(d_0, d_wk)
            cap_month_g = ab(d_0, d_mo)
            # run-up is calendar-known: enter day(-6), exit day(-1) -> tradable as-is
            row.update(
                included=True, reason="",
                anchor_pre=str(d_pre.date()), anchor_m1=str(d_m1.date()),
                day0_date=str(d_0.date()),
                ar_runup=ar_runup, ar_week=ar_week, ar_month=ar_month,
                cap_runup_g=ar_runup, cap_runup_n=ar_runup - rt,
                cap_week_g=cap_week_g, cap_week_n=cap_week_g - rt,
                cap_month_g=cap_month_g, cap_month_n=cap_month_g - rt,
            )
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    yearly events (not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances) -- the "CHDN beats the market
    seasonal?" third-axis comparison. NaN if either group has < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hit_rate(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Random-window placebo: is the observed mean abnormal return inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], leg: str,
                   col: str, k: int, entry_offset: int = 0, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 733,
                   tail: str = "right") -> dict:
    """For each INCLUDED event of ``leg``, redraw a random same-length window on the tape
    and recompute the abnormal return with the SAME construction as ``col``; average
    across the same n events; repeat n_seeds x n_draws_per_seed times.

    ``entry_offset`` matches the column: 0 for the "ar_*"/run-up signal windows (window
    anchored the session BEFORE a random point, mirroring day(-1)->day(-1)+k), 1 for the
    "cap_*" capture windows (anchored ON a random point, mirroring the day(0) entry).
    ``tail``: "right" if the claim predicts a positive effect, else "left".
    """
    spy = prices[dt.MARKET]
    chdn = prices[dt.GAMBLING]
    common = spy.index.intersection(chdn.index).sort_values()
    spy = spy.reindex(common)
    chdn = chdn.reindex(common)
    spy_daily_mean = float(spy.pct_change().mean())

    inc = events[(events["included"]) & (events["leg"] == leg)]
    n_ev = len(inc)
    obs = float(inc[col].mean())
    rt = 2.0 * cost_bps / 1e4

    def ab_pos(a_pos, e_pos):
        n = e_pos - a_pos
        a, e = common[a_pos], common[e_pos]
        r_spy = spy.loc[e] / spy.loc[a] - 1.0
        if leg == "chdn":
            r_chdn = chdn.loc[e] / chdn.loc[a] - 1.0
            return float(r_chdn - r_spy)
        return float(r_spy - n * spy_daily_mean)

    lo, hi = RUNUP_K, len(common) - k - entry_offset - 1
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw = []
            for _e in range(n_ev):
                p = int(rng.integers(lo, hi)) + entry_offset
                draw.append(ab_pos(p, p + k) - rt)
            if draw:
                means.append(np.mean(draw))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative abnormal return by trading-day offset
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], leg: str,
            pre: int = RUNUP_K, post: int = MONTH_K) -> pd.Series:
    """Mean cumulative abnormal return at each offset -pre..+post, averaged over all
    INCLUDED events of ``leg`` and re-anchored so the run-up start (offset -pre) == 0 —
    the standard event-study convention (nothing has happened before the window opens),
    giving an intuitive left-to-right path through the race (offset 0 = day(-1))."""
    spy = prices[dt.MARKET]
    chdn = prices[dt.GAMBLING]
    common = spy.index.intersection(chdn.index).sort_values()
    spy = spy.reindex(common)
    chdn = chdn.reindex(common)
    spy_daily_mean = float(spy.pct_change().mean())

    inc = events[(events["included"]) & (events["leg"] == leg)]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, r in inc.iterrows():
        p = common.get_loc(pd.Timestamp(r["anchor_m1"]))
        anchor = common[p]
        vals = []
        for off in offsets:
            end = common[p + off]
            n = off
            r_spy = spy.loc[end] / spy.loc[anchor] - 1.0
            if leg == "chdn":
                r_chdn = chdn.loc[end] / chdn.loc[anchor] - 1.0
                vals.append(float(r_chdn - r_spy))
            else:
                vals.append(float(r_spy - n * spy_daily_mean))
        paths.append(vals)
    arr = np.asarray(paths)
    mean_path = arr.mean(axis=0)
    mean_path = mean_path - mean_path[0]   # re-anchor: offset -pre == 0
    return pd.Series(mean_path, index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = WEEK_K) -> dict:
    """Run the one-sample-t capture detector on a synthetic paired world with a planted
    Derby bump on the first session after each synthetic race (day(0)->day(0)+k)."""
    a, b, events = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in events:
        if p + 1 + k >= len(a):
            continue
        ra = a.iloc[p + 1:p + 1 + k].sum()   # cumulative CHDN-like log-return, day(0)->+k
        rb = b.iloc[p + 1:p + 1 + k].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
