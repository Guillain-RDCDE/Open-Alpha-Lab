"""Strategy + inference for Study 735 — Ryder-Cup-Effect.

The claim: **the losing continent's stock market underperforms the winner's the Monday
after the Ryder Cup** -- a two-sided, cross-Atlantic cousin of the Edmans-García-Norli
sports-sentiment effect (real, one-directional, for football World Cup elimination
losses). Team USA is proxied by ``SPY``, Team Europe by ``VGK``.

The result is decided on a **Sunday** (a non-trading day; the one 2010 exception
concluded on a Monday, handled below), so nobody can act on it until markets reopen.
That gives the study its single, unavoidable execution lag for free:

* **day(-1)** = the last common (SPY & VGK) trading close *before* the result date --
  ordinarily the Friday. The last price that does NOT yet know who won.
* **day(0)**  = the first common close *after* day(-1) -- ordinarily the Monday. The
  first price that DOES know the result. (For 2010, whose match ended on Monday morning
  US time, day(0) is that Monday's close, which already knew the outcome -- still zero
  look-ahead.)

Two measurements per event, both **paired loser-minus-winner** so the common global
weekend move cancels out:

* **Signal (the announcement effect).** ``spread(k) = r_loser - r_winner`` cumulated
  from day(-1) to day(-1)+k, for k = 1 (the Monday itself) and k = 5 (~1 week). The
  folklore predicts ``spread < 0`` (the loser lags). This is the honest size of the
  cross-Atlantic sentiment gap, including the un-tradable Friday->Monday weekend jump.
* **Tradability (can you bank it?).** Long the winner, short the loser, entered at
  day(0)'s close (AFTER the result is public), held k-1 further sessions, net of a
  round trip on BOTH legs plus borrow on the short. This is what a trader reacting to
  Sunday-night headlines could actually capture -- pure continuation, no weekend gift.

Because each Ryder Cup is a single independent, non-overlapping event (not a daily
series), the primary statistic is a **one-sample t** of the paired spread across events.
A random-window placebo (redrawing the same signed loser-minus-winner spread on random,
non-Ryder-Cup dates) checks whether the observed mean sits outside the pair's ordinary
week-to-week noise. A third axis decomposes the spread into each side's own
constant-mean **absolute** abnormal return -- is the loser actually *slumping* (the
Edmans mechanism), or is any gap just the winner drifting up / noise?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

MON_K = 1        # trading sessions -- "the Monday after"
WEEK_K = 5       # trading sessions ~ 1 week
MONTH_K = 21     # trading sessions ~ 1 month (for the anatomy / coverage floor)
COST_BPS = 2.0        # one-way, per leg (SPY & VGK are among the most liquid ETFs)
BORROW_ANNUAL = 0.005  # 50 bps/yr borrow on the short leg


# --------------------------------------------------------------------------- #
# Per-ticker mean daily return (constant-mean market model, for the abs. legs)
# --------------------------------------------------------------------------- #
def mean_daily_return(prices: dict[str, pd.Series]) -> dict[str, float]:
    """Full-sample mean simple daily return per ticker -- the constant-mean 'normal'."""
    return {t: float(s.pct_change().mean()) for t, s in prices.items()}


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar + side map -> per-event paired spreads
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per Ryder Cup: paired loser-minus-winner spreads + inclusion reason.

    A row is INCLUDED only if it has a losing side (not the 1989 tie) AND both SPY and
    VGK cover [day(-1) .. day(-1)+MONTH_K] around the result. Excluded rows are kept
    (with a reason) so the funnel is auditable -- almost every exclusion is "VGK
    coverage postdates the Ryder Cup" (VGK inception 2005-03-10).
    """
    mu = mean_daily_return(prices)
    usa = prices[dt.USA_TICKER]
    eur = prices[dt.EUROPE_TICKER]
    common = usa.index.intersection(eur.index).sort_values()
    rows = []
    for year, result_date, winner, loser in dt.EVENTS:
        row = dict(year=year, result_date=result_date, winner=winner, loser=loser)
        if loser is None:
            row.update(included=False, reason="14-14 tie -- no losing side")
            rows.append(row)
            continue
        loser_tkr, winner_tkr = dt.SIDE_TICKER[loser], dt.SIDE_TICKER[winner]
        before = common[common < pd.Timestamp(result_date)]
        if len(before) == 0:
            row.update(included=False, reason="SPY/VGK coverage postdates the Ryder Cup")
            rows.append(row)
            continue
        p = common.get_loc(before[-1])
        if p + MONTH_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue
        # signal windows are anchored at day(-1); capture windows at day(0)
        d_m1, d_0 = common[p], common[p + 1]
        d_mon, d_wk = common[p + MON_K], common[p + WEEK_K]        # day(-1)+1, day(-1)+5
        d_cap1, d_cap5 = common[p + 1 + 1], common[p + 1 + WEEK_K]  # day(0)+1, day(0)+5
        L, W = prices[loser_tkr], prices[winner_tkr]

        def spread(t_end):
            r_l = L.loc[t_end] / L.loc[d_m1] - 1.0
            r_w = W.loc[t_end] / W.loc[d_m1] - 1.0
            return float(r_l - r_w)             # folklore: < 0 (loser lags)

        def cap(t_end, k):
            # long winner / short loser, entered at day(0) close (result already public),
            # exited k sessions later -- pure post-Monday continuation, no weekend gift
            r_l = L.loc[t_end] / L.loc[d_0] - 1.0
            r_w = W.loc[t_end] / W.loc[d_0] - 1.0
            gross = float(r_w - r_l)            # profit if the loser keeps lagging
            # two legs, one round trip each (4x one-way) + borrow on the short leg
            costs = 4.0 * cost_bps / 1e4 + BORROW_ANNUAL * (k / 252.0)
            return gross, gross - costs

        # absolute constant-mean abnormal return of each leg over the Monday window
        def abn(tkr, t_end, k):
            r = prices[tkr].loc[t_end] / prices[tkr].loc[d_m1] - 1.0
            return float(r - mu[tkr] * k)

        cap_day_g, cap_day_n = cap(d_cap1, 1)
        cap_wk_g, cap_wk_n = cap(d_cap5, WEEK_K)
        row.update(
            included=True, reason="",
            anchor_date=str(d_m1.date()), day0_date=str(d_0.date()),
            loser_ticker=loser_tkr, winner_ticker=winner_tkr,
            spread_mon=spread(d_mon), spread_wk=spread(d_wk),
            loser_ab_mon=abn(loser_tkr, d_mon, MON_K),
            winner_ab_mon=abn(winner_tkr, d_mon, MON_K),
            cap_day_gross=cap_day_g, cap_day_net=cap_day_n,
            cap_week_gross=cap_wk_g, cap_week_net=cap_wk_n,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit of inference for independent,
    non-overlapping biennial events (not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances) -- used for the USA-loss vs
    Europe-loss asymmetry check. NaN if either group has < 2 obs."""
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


def hit_rate(x: np.ndarray, direction: str = "neg") -> dict:
    """Fraction of events consistent with the folklore. ``direction='neg'`` counts
    spreads < 0 (loser underperformed); ``'pos'`` counts > 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    k = int((x < 0).sum()) if direction == "neg" else int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Random-window placebo: is the observed mean spread inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], col: str,
                   k: int, entry_offset: int = 0, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 735,
                   tail: str = "left") -> dict:
    """For each INCLUDED event, redraw a random (non-Ryder-Cup) k-session window on the
    SAME pair, keeping that event's loser/winner assignment, and recompute the
    loser-minus-winner spread; average across the same events; repeat
    n_seeds x n_draws_per_seed times.

    ``entry_offset``: 0 for the ``spread_*`` signal columns (window anchored the session
    BEFORE a random point, mirroring day(-1)->day(-1)+k), 1 for the ``cap_*`` capture
    columns (window anchored ON a random point, mirroring the zero-look-ahead
    day(0)->day(0)+k entry). ``tail``: "left" (the folklore predicts a NEGATIVE spread
    -> p = share of null means <= observed) or "right".
    """
    usa = prices[dt.USA_TICKER]
    eur = prices[dt.EUROPE_TICKER]
    common = usa.index.intersection(eur.index).sort_values()
    inc = events[events["included"]]
    obs = float(inc[col].mean())
    is_cap = col.startswith("cap_")

    pairs = list(zip(inc["loser_ticker"], inc["winner_ticker"]))
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for loser_tkr, winner_tkr in pairs:
                lo = 0
                hi = len(common) - k - entry_offset - 1
                if hi <= lo:
                    continue
                pos = int(rng.integers(lo, hi)) + entry_offset
                d_start, d_end = common[pos], common[pos + k]
                L, W = prices[loser_tkr], prices[winner_tkr]
                r_l = L.loc[d_end] / L.loc[d_start] - 1.0
                r_w = W.loc[d_end] / W.loc[d_start] - 1.0
                if is_cap:
                    val = float(r_w - r_l) - (4.0 * cost_bps / 1e4 + BORROW_ANNUAL * (k / 252.0))
                else:
                    val = float(r_l - r_w)
                draw_vals.append(val)
            if draw_vals:
                means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means <= obs).mean()) if tail == "left" else float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative spread by trading-day offset (for the notebook chart)
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], max_k: int = WEEK_K
             ) -> pd.Series:
    """Mean cumulative loser-minus-winner spread at each offset 0..max_k from day(-1),
    averaged across all INCLUDED events."""
    usa = prices[dt.USA_TICKER]
    eur = prices[dt.EUROPE_TICKER]
    common = usa.index.intersection(eur.index).sort_values()
    inc = events[events["included"]]
    paths = []
    for _, row in inc.iterrows():
        L, W = prices[row["loser_ticker"]], prices[row["winner_ticker"]]
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        d_m1 = common[p]
        vals = []
        for k in range(0, max_k + 1):
            d_k = common[p + k]
            r_l = L.loc[d_k] / L.loc[d_m1] - 1.0
            r_w = W.loc[d_k] / W.loc[d_m1] - 1.0
            vals.append(float(r_l - r_w))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=range(0, max_k + 1))


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(slump: float, seed: int, k: int = 1) -> dict:
    """Run the paired one-sample-t detector on a synthetic (USA, Europe) world with a
    planted loser slump on its synthetic Ryder-Cup calendar."""
    u, e, events, losers = dt.synthetic_world(slump=slump, seed=seed)
    series = {"USA": u, "Europe": e}
    sp = []
    for pos, loser in zip(events, losers):
        if pos + k >= len(u):
            continue
        winner = "Europe" if loser == "USA" else "USA"
        r_l = series[loser].iloc[pos:pos + k + 1].sum()   # cumulative log-return day(-1)->+k
        r_w = series[winner].iloc[pos:pos + k + 1].sum()
        sp.append(float(r_l - r_w))
    return one_sample_t(np.asarray(sp))
