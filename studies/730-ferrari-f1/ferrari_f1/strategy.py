"""Strategy + inference for Study 730 — Ferrari-F1.

The claim: **RACE (Ferrari N.V.) gets a fan-sentiment / brand-halo pop the Monday after
Scuderia Ferrari wins a Formula 1 Grand Prix** -- the Ferrari-specific cousin of the
Edmans-Garcia-Norli sports-sentiment effect, on the one listed company whose brand *is*
a race team.

Every Grand Prix runs on a **Sunday**, a non-trading day, so nobody can act on the
result until markets reopen on Monday. That gives the study its single, unavoidable
execution lag for free:

* **day(-1)** = the last trading close *before* the race (ordinarily the Friday) --
  the last price that does NOT yet know the result.
* **day(0)**  = the first trading close *after* the race (ordinarily the Monday) --
  the first price that DOES know it. By Monday's open the result is ~12-40 hours old and
  globally reported, so entering at day(0)'s close (not before) is zero-look-ahead by
  construction.

Two measurements per win, both abnormal (RACE minus the SPY market benchmark, both
total-return):

* **Signal (the reaction).** Cumulative abnormal return from day(-1) to day(-1)+k, for
  k = 1 trading session (the immediate Friday-close -> Monday-close "win pop", the headline
  and the only fully non-overlapping horizon) and k = 5 (~1 week). This is the full price
  reaction to the win becoming public, including the un-tradable weekend jump -- the
  honest size of "the pop", not a strategy.
* **Tradability (can a fan bank it?).** The same abnormal return but entered at day(0)'s
  close instead of day(-1)'s -- i.e. AFTER the win is already public and priced -- held k
  further sessions, net of one round-trip's costs. This is what a Ferrari fan reacting to
  Sunday-evening headlines could actually capture.

Because each Ferrari win is a single independent event, the primary statistic is a
**one-sample t** of the abnormal return across wins (n = 24). At the DAY(0) horizon every
event is a distinct Monday, so independence holds exactly; at the 1-week horizon three
back-to-back win pairs overlap (Belgium/Monza 2019, Britain/Austria 2022, USA/Mexico
2024) -- named, and re-run dropping the second of each pair. A random-calendar placebo
(drawing the same number of random, non-race anchor dates from RACE's own history) checks
whether the observed mean is inside or outside the luck cloud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

DAY_K = 1      # the immediate Monday reaction (day(-1) -> day(0))
WEEK_K = 5     # trading sessions ~ 1 week
COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded win calendar -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per Ferrari win: abnormal returns at day(0) and 1 week + inclusion flag.

    A row is INCLUDED only if RACE and SPY both cover [day(-1) .. day(-1)+WEEK_K] for
    that race. Every win in the frozen calendar postdates the 2015-10-21 RACE listing,
    so all 24 resolve; the funnel is kept anyway so the structure matches the desk's
    event-study template (and would exclude a pre-listing race if one were ever added).
    """
    race = prices[dt.TICKER]
    bench = prices[dt.BENCHMARK]
    common = race.index.intersection(bench.index).sort_values()
    rows = []
    for season, race_date, gp, driver, era in dt.EVENTS:
        row = dict(season=season, race_date=race_date, gp=gp, driver=driver, era=era)
        before = common[common < pd.Timestamp(race_date)]
        if len(before) == 0:
            row.update(included=False, reason="RACE/SPY predate the race")
            rows.append(row)
            continue
        p = common.get_loc(before[-1])
        if p + WEEK_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue
        d_m1 = common[p]                    # last close before the Sunday race (Friday)
        d_0 = common[p + 1]                 # first close after (Monday)
        d_day = common[p + DAY_K]           # == d_0
        d_wk = common[p + WEEK_K]

        def ar(t_end):
            r_race = race.loc[t_end] / race.loc[d_m1] - 1.0
            r_bench = bench.loc[t_end] / bench.loc[d_m1] - 1.0
            return float(r_race - r_bench)

        def cap(t_end):
            # entered at day(0)'s close -- AFTER the result is public: zero look-ahead
            r_race = race.loc[t_end] / race.loc[d_0] - 1.0
            r_bench = bench.loc[t_end] / bench.loc[d_0] - 1.0
            gross = float(r_race - r_bench)
            return gross, gross - 2.0 * cost_bps / 1e4

        cap_wk_gross, cap_wk_net = cap(d_wk)
        row.update(
            included=True, reason="",
            anchor_date=str(d_m1.date()), day0_date=str(d_0.date()),
            ar_day=ar(d_day), ar_week=ar(d_wk),
            weekly_overlap=(race_date in dt.WEEKLY_OVERLAP_DROP),
            cap_week_gross=cap_wk_gross, cap_week_net=cap_wk_net,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit of inference for independent,
    non-overlapping single-race events (not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances) -- used for the
    contender-vs-sporadic era contrast. NaN if either group has < 2 obs."""
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
# Random-calendar placebo: is the observed mean AR inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], col: str,
                   k: int, entry_offset: int = 0, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 500, base_seed: int = 730,
                   tail: str = "right") -> dict:
    """For each INCLUDED win, redraw a random (non-race) k-session window on RACE's own
    history and recompute the abnormal return vs SPY; average across the same n events;
    repeat n_seeds x n_draws_per_seed times.

    ``entry_offset`` matches the column: 0 for the "ar_*" signal columns (window anchored
    the session BEFORE a random point, mirroring day(-1)->day(-1)+k), 1 for the "cap_*"
    tradable-capture columns (window anchored ON a random point, mirroring the zero-look-
    ahead day(0)->day(0)+k entry). ``tail``: "right" (the claim predicts a POSITIVE pop)
    or "left".
    """
    race = prices[dt.TICKER]
    bench = prices[dt.BENCHMARK]
    common = race.index.intersection(bench.index).sort_values()
    inc = events[events["included"]]
    obs = float(inc[col].mean())
    n_ev = len(inc)

    lo = 0
    hi = len(common) - k - entry_offset - 1
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            ps = rng.integers(lo, hi, n_ev) + entry_offset
            draw_vals = []
            for p in ps:
                d_start = common[p]
                d_end = common[p + k]
                r_race = race.loc[d_end] / race.loc[d_start] - 1.0
                r_bench = bench.loc[d_end] / bench.loc[d_start] - 1.0
                draw_vals.append(float(r_race - r_bench) - 2.0 * cost_bps / 1e4)
            means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative AR by trading-day offset (for the notebook chart)
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], max_k: int = WEEK_K,
             era: str | None = None) -> pd.Series:
    """Mean cumulative abnormal return (RACE - SPY) at each offset 0..max_k from
    day(-1) (the pre-race anchor), averaged across all INCLUDED wins (optionally
    filtered to one ``era``)."""
    race = prices[dt.TICKER]
    bench = prices[dt.BENCHMARK]
    common = race.index.intersection(bench.index).sort_values()
    inc = events[events["included"]]
    if era is not None:
        inc = inc[inc["era"] == era]
    paths = []
    for _, row in inc.iterrows():
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        d_m1 = common[p]
        vals = []
        for k in range(0, max_k + 1):
            d_k = common[p + k]
            r_race = race.loc[d_k] / race.loc[d_m1] - 1.0
            r_bench = bench.loc[d_k] / bench.loc[d_m1] - 1.0
            vals.append(float(r_race - r_bench))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=range(0, max_k + 1))


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = 1) -> dict:
    """Run the one-sample-t detector on a synthetic paired (asset, benchmark) world
    with a planted bump on its synthetic win calendar."""
    a, b, events = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in events:
        if p + k >= len(a):
            continue
        # cumulative log-return day(-1)->day(-1)+k; the bump lands on day(-1)+1 == day(0)
        ra = a.iloc[p:p + k + 1].sum()
        rb = b.iloc[p:p + k + 1].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
