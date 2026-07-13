"""Strategy + inference for Study 770 — Concert-Economy.

The claim: **Live Nation (LYV) rallies INTO festival season** — the stock front-runs
the summer touring quarter, popping in the weeks before Coachella opens the US festival
calendar. Two things to separate:

* **The run-up (the "rally into").** LYV's abnormal return (LYV total-return minus the
  SPY total-return benchmark) over the K trading sessions ENDING on the last close
  on/before Coachella's weekend-1 Friday. Tested at K = 21 (~1 month) and K = 42
  (~2 months). Because Coachella's opening date is announced each January, this window
  is **calendar-known** — a "buy K sessions before Coachella, sell the day it opens"
  rule is zero-look-ahead by construction, so the signal and the tradable capture are
  the SAME window (gross vs net of costs), with no surprise-day lag to argue about.

* **Sell-the-news? (the myth-check third axis).** The folklore's natural corollary is
  "buy the rumour, sell the news" — the run-up is supposed to fade once the season is
  actually underway. So we also measure the abnormal return from Coachella's Friday
  through the end of the summer touring season (~SEASON_K sessions, roughly to Labor
  Day). If the run-up is real *and* it reverses in-season, that is a coherent
  sell-the-news story; if the in-season return is just noise, the corollary is BUSTED.

Because each festival year is a single independent event (not a daily series), the
primary statistic is a **one-sample t** of the abnormal return across events (n = the
number of years with LYV coverage and a held Coachella — small by construction, and
said out loud). A random-window placebo (drawing many random, non-Coachella K-session
windows from LYV's own history vs SPY) checks whether the observed mean sits inside or
outside the stock's ordinary tracking noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

MONTH_K = 21     # trading sessions ~ 1 month  (the 1-month run-up)
TWO_MONTH_K = 42  # trading sessions ~ 2 months (the 2-month run-up)
SEASON_K = 95    # trading sessions ~ Coachella -> Labor Day (the in-season window)
COST_BPS = 5.0   # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per Coachella year: run-up + in-season abnormal returns + inclusion.

    A row is INCLUDED only if LYV and SPY both have cached history covering
    [anchor - TWO_MONTH_K .. anchor + SEASON_K] for that year. Cancelled years (2020,
    2021) and years whose 2-month run-up predates LYV's IPO are excluded with a reason,
    so the funnel is auditable.
    """
    lyv = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = lyv.index.intersection(spy.index).sort_values()
    rt = 2.0 * cost_bps / 1e4    # round-trip cost (one-way x 2), applied to the net leg
    rows = []
    for year, friday, cancelled in dt.EVENTS:
        row = dict(year=year, friday=friday)
        if cancelled is not None:
            row.update(included=False, reason=cancelled)
            rows.append(row)
            continue
        anchor_ts = pd.Timestamp(friday)
        on_or_before = common[common <= anchor_ts]
        if len(on_or_before) == 0:
            row.update(included=False, reason="no LYV/SPY coverage at the anchor")
            rows.append(row)
            continue
        p = common.get_loc(on_or_before[-1])
        if p - TWO_MONTH_K < 0:
            row.update(included=False, reason="2-month run-up predates LYV IPO")
            rows.append(row)
            continue
        if p + SEASON_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue

        def ar(i_start, i_end):
            r_lyv = lyv.loc[common[i_end]] / lyv.loc[common[i_start]] - 1.0
            r_spy = spy.loc[common[i_end]] / spy.loc[common[i_start]] - 1.0
            return float(r_lyv - r_spy)

        ru_1mo = ar(p - MONTH_K, p)
        ru_2mo = ar(p - TWO_MONTH_K, p)
        during = ar(p, p + SEASON_K)      # Coachella Friday -> ~Labor Day (in-season)
        row.update(
            included=True, reason="",
            anchor_date=str(common[p].date()),
            entry_1mo=str(common[p - MONTH_K].date()),
            entry_2mo=str(common[p - TWO_MONTH_K].date()),
            ru_1mo=ru_1mo, ru_1mo_net=ru_1mo - rt,
            ru_2mo=ru_2mo, ru_2mo_net=ru_2mo - rt,
            during=during, during_net=during - rt,
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
# Random-window placebo: is the observed mean run-up inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], col: str,
                   k: int, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 770,
                   tail: str = "right") -> dict:
    """For each INCLUDED event, redraw a random (non-Coachella) k-session window on
    LYV vs SPY and recompute the abnormal return; average across the same n events;
    repeat n_seeds x n_draws_per_seed times.

    ``tail``: "right" (the claim predicts a POSITIVE run-up -> p = share of null means
    >= observed) or "left" (observed is negative -> p = share of null means <= observed).
    """
    lyv = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = lyv.index.intersection(spy.index).sort_values()
    n_events = int(events["included"].sum()) if "included" in events else len(events)
    inc = events[events["included"]] if "included" in events else events
    obs = float(inc[col].mean())

    lo, hi = 0, len(common) - k - 1
    rt = 2.0 * cost_bps / 1e4
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for _e in range(n_events):
                ppos = int(rng.integers(lo, hi))
                d_start, d_end = common[ppos], common[ppos + k]
                r_lyv = lyv.loc[d_end] / lyv.loc[d_start] - 1.0
                r_spy = spy.loc[d_end] / spy.loc[d_start] - 1.0
                draw_vals.append(float(r_lyv - r_spy) - rt)
            means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative AR by trading-day offset around the anchor
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series],
             pre: int = TWO_MONTH_K, post: int = SEASON_K) -> pd.Series:
    """Mean cumulative abnormal return (LYV - SPY) at each offset from -pre..+post
    relative to the anchor (Coachella Friday), normalised so offset 0 = 0%, averaged
    across all INCLUDED events. Negative offsets are the run-up; positive is in-season.
    """
    lyv = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = lyv.index.intersection(spy.index).sort_values()
    inc = events[events["included"]]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, row in inc.iterrows():
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        base_lyv, base_spy = lyv.loc[common[p]], spy.loc[common[p]]
        vals = []
        for o in offsets:
            d = common[p + o]
            r_lyv = lyv.loc[d] / base_lyv - 1.0
            r_spy = spy.loc[d] / base_spy - 1.0
            vals.append(float(r_lyv - r_spy))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = MONTH_K) -> dict:
    """Run the one-sample-t run-up detector on a synthetic paired (asset, benchmark)
    world with a planted pre-festival bump on its synthetic calendar."""
    a, b, fests = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in fests:
        if p - k < 0:
            continue
        ra = a.iloc[p - k:p].sum()   # cumulative run-up log-return, [p-k .. p)
        rb = b.iloc[p - k:p].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
