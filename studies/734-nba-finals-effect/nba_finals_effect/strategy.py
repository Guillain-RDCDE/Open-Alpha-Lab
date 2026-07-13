"""Strategy + inference for Study 734 — NBA-Finals-Effect.

The claim: **the losing team's home-city market dips (and the champion's pops) around the
NBA Finals** — the folklore cousin of the Edmans-Garcia-Norli (2007) elimination-shock
effect, applied to basketball instead of World Cup soccer.

The series-clinching game tips ~9pm local and ends ~11:30pm — AFTER the market close on
the game date. That gives the study its single, unavoidable execution lag for free:

* **day(-1)** = the last trading close on or before the clinching-game date. The game is a
  night event, so this close does NOT yet know the result. (For a weekday game this is the
  game day's own 4pm close, five hours before tip-off; for a weekend/holiday game it is the
  prior Friday.)
* **day(0)**  = the first trading close AFTER the game — the first price that DOES know the
  result. By the time markets reopen the outcome is hours old and wall-to-wall news, so
  entering at day(0)'s close (not before) is zero-look-ahead by construction.

Two roles per Finals (mirroring EGN's asymmetry): ``role = "loser"`` (the runner-up's
metro proxy — EGN predicts a NEGATIVE abnormal return, the deflated-fanbase dip) and
``role = "champion"`` (the winner's metro proxy — the folklore feel-good pop, predicted
POSITIVE). Both are abnormal (metro proxy minus the SPY US benchmark, both total-return).

* **Signal (the announcement effect).** Cumulative abnormal return from day(-1) to
  day(-1)+k, for k = 1 (~next day, the EGN horizon) and k = 5 (~1 week). Includes the
  un-tradable overnight/weekend jump — the honest size of the mood shock, not a strategy.
* **Tradability (can retail bank it?).** The same abnormal return but entered at day(0)'s
  close (AFTER the result is public and priced), held k-1 further sessions, net of one
  round-trip's costs — what a fan reacting to the morning-after headlines could capture.

Because each Finals is a single independent event (not a daily series), the primary
statistic is a **one-sample t** of the abnormal return across events (n = number of Finals
with proxy + benchmark coverage). A random-window placebo (drawing the same number of
random, non-Finals anchor dates from each ticker's own history) checks whether the observed
mean sits inside or outside the tickers' ordinary week-to-week tracking noise against SPY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

DAY_K = 1      # trading sessions ~ next day (the EGN elimination-shock horizon)
WEEK_K = 5     # trading sessions ~ 1 week
COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar + team->proxy map -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per (year, role) NBA Finals event: abnormal returns + inclusion reason.

    ``role`` is "loser" (runner-up) or "champion". A row is INCLUDED only if the team's
    mapped proxy (``data.TEAM_PROXY``) AND the SPY benchmark both cover
    [day(-1) .. day(-1)+WEEK_K] for that event. Excluded rows are kept (with a reason) so
    the funnel is auditable.
    """
    bench = prices[dt.BENCHMARK]
    rows = []
    for year, clinch_date, champion, runner_up in dt.EVENTS:
        for role, team in (("loser", runner_up), ("champion", champion)):
            ticker = dt.TEAM_PROXY.get(team)
            row = dict(year=year, role=role, team=team, ticker=ticker,
                       clinch_date=clinch_date)
            if ticker is None:
                row.update(included=False, reason=f"no proxy mapped for {team}")
                rows.append(row)
                continue
            proxy = prices[ticker]
            common = proxy.index.intersection(bench.index).sort_values()
            # day(-1) = last trading close ON OR BEFORE the (night) game date
            on_or_before = common[common <= pd.Timestamp(clinch_date)]
            if len(on_or_before) == 0:
                row.update(included=False, reason="proxy/benchmark predate the Finals")
                rows.append(row)
                continue
            p = common.get_loc(on_or_before[-1])
            if p + 1 + WEEK_K >= len(common):
                row.update(included=False, reason="insufficient trailing history")
                rows.append(row)
                continue
            d_m1, d_0 = common[p], common[p + 1]
            # SIGNAL exits are day(-1)+k; CAPTURE exits are day(0)+k (a real k-session hold
            # entered AFTER the result is already public -- the announcement's overnight
            # jump, day(-1)->day(0), is un-tradable by construction and is excluded here).
            d_day, d_wk = common[p + DAY_K], common[p + WEEK_K]
            c_day, c_wk = common[p + 1 + DAY_K], common[p + 1 + WEEK_K]

            def ar(t_end):
                r_proxy = proxy.loc[t_end] / proxy.loc[d_m1] - 1.0
                r_bench = bench.loc[t_end] / bench.loc[d_m1] - 1.0
                return float(r_proxy - r_bench)

            def cap(t_end):
                # entered at day(0)'s close -- AFTER the result is public: zero look-ahead
                r_proxy = proxy.loc[t_end] / proxy.loc[d_0] - 1.0
                r_bench = bench.loc[t_end] / bench.loc[d_0] - 1.0
                gross = float(r_proxy - r_bench)
                return gross, gross - 2.0 * cost_bps / 1e4

            ar_day, ar_week = ar(d_day), ar(d_wk)
            cap_day_gross, cap_day_net = cap(c_day)
            cap_week_gross, cap_week_net = cap(c_wk)
            row.update(
                included=True, reason="",
                anchor_date=str(d_m1.date()), day0_date=str(d_0.date()),
                ar_day=ar_day, ar_week=ar_week,
                cap_day_gross=cap_day_gross, cap_day_net=cap_day_net,
                cap_week_gross=cap_week_gross, cap_week_net=cap_week_net,
            )
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Broad-market test: does the whole US tape (SPY) move the day after the Finals?
# EGN predicts ~zero for a within-country event (one US city elated, one deflated).
# --------------------------------------------------------------------------- #
def broad_market_events(prices: dict[str, pd.Series], trail: int = 60) -> pd.DataFrame:
    """Per-event SPY market-model abnormal return on day(0): SPY's day(-1)->day(0) return
    minus its trailing-``trail``-session mean daily return. One row per Finals."""
    bench = prices[dt.BENCHMARK].sort_index()
    ret = bench.pct_change()
    rows = []
    for year, clinch_date, champion, runner_up in dt.EVENTS:
        idx = bench.index
        on_or_before = idx[idx <= pd.Timestamp(clinch_date)]
        if len(on_or_before) == 0:
            continue
        p = idx.get_loc(on_or_before[-1])
        if p + 1 >= len(idx) or p - trail < 0:
            continue
        d0 = idx[p + 1]
        r0 = float(bench.iloc[p + 1] / bench.iloc[p] - 1.0)
        mu = float(ret.iloc[p - trail:p].mean())
        rows.append(dict(year=year, day0=str(d0.date()), spy_ret=r0,
                         expected=mu, abn=r0 - mu))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit of inference for independent,
    non-overlapping yearly events (not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances) -- used for the "champion vs
    loser" third-axis comparison. NaN if either group has < 2 obs."""
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


def hit_rate(x: np.ndarray, positive: bool = True) -> dict:
    """Fraction of events with the predicted sign. ``positive=True`` counts x>0 (the
    champion/feel-good prediction); ``positive=False`` counts x<0 (the EGN loser dip)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    k = int((x > 0).sum()) if positive else int((x < 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Random-window placebo: is the observed mean AR inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], col: str,
                   k: int, entry_offset: int = 0, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 734,
                   tail: str = "left") -> dict:
    """For each INCLUDED event, redraw a random (non-Finals) k-session window on the SAME
    ticker's own history and recompute the abnormal return; average across the same n
    events; repeat n_seeds x n_draws_per_seed times.

    ``entry_offset`` matches the column tested: 0 for the "ar_*" signal columns (window
    anchored the session BEFORE a random point, mirroring day(-1)->day(-1)+k), 1 for the
    "cap_*" tradable-capture columns (window anchored ON a random point, mirroring the
    zero-look-ahead day(0)->day(0)+k entry). ``tail``: "left" (the EGN loser prediction is
    NEGATIVE -> p = share of null means <= observed) or "right" (the champion prediction is
    POSITIVE -> p = share of null means >= observed).
    """
    bench = prices[dt.BENCHMARK]
    inc = events[events["included"]]
    obs = float(inc[col].mean())

    tickers = inc["ticker"].tolist()
    common_by_ticker = {t: prices[t].index.intersection(bench.index).sort_values()
                        for t in set(tickers)}
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for t in tickers:
                common = common_by_ticker[t]
                lo = 0
                hi = len(common) - k - entry_offset - 1
                if hi <= lo:
                    continue
                p = int(rng.integers(lo, hi)) + entry_offset
                d_start = common[p]
                d_end = common[p + k]
                proxy = prices[t]
                r_proxy = proxy.loc[d_end] / proxy.loc[d_start] - 1.0
                r_bench = bench.loc[d_end] / bench.loc[d_start] - 1.0
                draw_vals.append(float(r_proxy - r_bench) - 2.0 * cost_bps / 1e4)
            if draw_vals:
                means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means <= obs).mean()) if tail == "left" else float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative AR by trading-day offset (for the notebook chart)
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], role: str,
             max_k: int = WEEK_K) -> pd.Series:
    """Mean cumulative abnormal return (proxy - SPY) at each offset 0..max_k from day(-1)
    (the pre-result anchor), averaged across all INCLUDED events of ``role``."""
    bench = prices[dt.BENCHMARK]
    inc = events[(events["included"]) & (events["role"] == role)]
    paths = []
    for _, row in inc.iterrows():
        proxy = prices[row["ticker"]]
        common = proxy.index.intersection(bench.index).sort_values()
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        d_m1 = common[p]
        vals = []
        for k in range(0, max_k + 1):
            d_k = common[p + k]
            r_proxy = proxy.loc[d_k] / proxy.loc[d_m1] - 1.0
            r_bench = bench.loc[d_k] / bench.loc[d_m1] - 1.0
            vals.append(float(r_proxy - r_bench))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=range(0, max_k + 1))


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = 1) -> dict:
    """Run the one-sample-t detector on a synthetic paired (proxy, benchmark) world with a
    planted bump on the day AFTER each synthetic (night-game) event."""
    a, b, events = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in events:
        if p + k >= len(a):
            continue
        # day(-1) = p, shock lands at p+1; cumulative log-AR from p to p+k
        ra = a.iloc[p + 1:p + 1 + k].sum()
        rb = b.iloc[p + 1:p + 1 + k].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
