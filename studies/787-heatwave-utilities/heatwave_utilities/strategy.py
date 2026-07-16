"""Strategy + inference for Study 787 — Heatwave-Utilities.

The claim: **the utilities sector (XLU) rallies through peak-summer heat.** Two windows,
both anchored on the fixed climatological peak-heat date (July 22, calendar-known):

* **Into the heat (the run-up).** XLU's abnormal return (XLU total-return minus the SPY
  total-return benchmark) over the K trading sessions ENDING on the last close on/before
  the peak-heat anchor — the ramp *into* peak cooling demand. Tested at K = 10 (~2 weeks)
  and K = 21 (~1 month).

* **Through / past the heat.** XLU's abnormal return over the K sessions STARTING from the
  peak-heat close — late July into August, the core cooling-demand weeks. Tested at
  K = 10 and K = 21. If the folklore is right, at least one of these windows is reliably
  positive; if both are noise, the "rally through the heat" story is busted.

Because each summer is a single independent event (not a daily series), the primary
statistic is a **one-sample t** of the abnormal return across events (n = the number of
years with XLU+SPY coverage). A random-window placebo (drawing many random, non-peak
K-session windows from XLU's own history vs SPY) checks whether the observed mean sits
inside or outside the sector's ordinary tracking noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

PRE_SHORT_K = 10    # ~2 weeks into the heat
PRE_LONG_K = 21     # ~1 month into the heat
POST_SHORT_K = 10   # ~2 weeks through/past the peak
POST_LONG_K = 21    # ~1 month through/past the peak
COST_BPS = 5.0      # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per summer: into-the-heat + past-peak abnormal returns + inclusion.

    A row is INCLUDED only if XLU and SPY both have cached history covering
    [anchor - PRE_LONG_K .. anchor + POST_LONG_K] for that year. Years whose window
    falls outside coverage are excluded with a reason, so the funnel is auditable.
    """
    xlu = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = xlu.index.intersection(spy.index).sort_values()
    rt = 2.0 * cost_bps / 1e4    # round-trip cost (one-way x 2), applied to the net leg
    rows = []
    for year, keydate in dt.EVENTS:
        row = dict(year=year, peak=keydate)
        anchor_ts = pd.Timestamp(keydate)
        on_or_before = common[common <= anchor_ts]
        if len(on_or_before) == 0:
            row.update(included=False, reason="no XLU/SPY coverage at the anchor")
            rows.append(row)
            continue
        p = common.get_loc(on_or_before[-1])
        if p - PRE_LONG_K < 0:
            row.update(included=False, reason="into-the-heat window predates coverage")
            rows.append(row)
            continue
        if p + POST_LONG_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue

        def ar(i_start, i_end):
            r_a = xlu.loc[common[i_end]] / xlu.loc[common[i_start]] - 1.0
            r_s = spy.loc[common[i_end]] / spy.loc[common[i_start]] - 1.0
            return float(r_a - r_s)

        pre_s = ar(p - PRE_SHORT_K, p)
        pre_l = ar(p - PRE_LONG_K, p)
        post_s = ar(p, p + POST_SHORT_K)
        post_l = ar(p, p + POST_LONG_K)
        row.update(
            included=True, reason="",
            anchor_date=str(common[p].date()),
            pre_s=pre_s, pre_s_net=pre_s - rt,
            pre_l=pre_l, pre_l_net=pre_l - rt,
            post_s=post_s, post_s_net=post_s - rt,
            post_l=post_l, post_l_net=post_l - rt,
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
# Random-window placebo: is the observed mean inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], col: str,
                   k: int, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 814,
                   tail: str = "right") -> dict:
    """For each INCLUDED event, redraw a random (non-peak) k-session window on XLU vs SPY
    and recompute the abnormal return; average across the same n events; repeat
    n_seeds x n_draws_per_seed times.

    ``tail``: "right" (a claim of POSITIVE mean -> p = share of null means >= observed) or
    "left" (a claim of NEGATIVE mean -> p = share of null means <= observed).
    """
    xlu = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = xlu.index.intersection(spy.index).sort_values()
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
                r_a = xlu.loc[d_end] / xlu.loc[d_start] - 1.0
                r_s = spy.loc[d_end] / spy.loc[d_start] - 1.0
                draw_vals.append(float(r_a - r_s) - rt)
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
             pre: int = PRE_LONG_K, post: int = POST_LONG_K) -> pd.Series:
    """Mean cumulative abnormal return (XLU - SPY) at each offset from -pre..+post
    relative to the peak-heat anchor, normalised so offset 0 = 0%, averaged across all
    INCLUDED events. Negative offsets are the into-the-heat ramp; positive is the
    past-peak window.
    """
    xlu = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = xlu.index.intersection(spy.index).sort_values()
    inc = events[events["included"]]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, row in inc.iterrows():
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        base_a, base_s = xlu.loc[common[p]], spy.loc[common[p]]
        vals = []
        for o in offsets:
            d = common[p + o]
            r_a = xlu.loc[d] / base_a - 1.0
            r_s = spy.loc[d] / base_s - 1.0
            vals.append(float(r_a - r_s))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = PRE_SHORT_K,
                     side: str = "pre") -> dict:
    """Run the one-sample-t detector on a synthetic paired world with a planted
    into-the-heat bump. ``side='pre'`` measures the run-up window [p-k..p); ``side='post'``
    measures the past-peak window [p..p+k)."""
    a, b, keys = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in keys:
        if side == "pre":
            if p - k < 0:
                continue
            ra = a.iloc[p - k:p].sum()
            rb = b.iloc[p - k:p].sum()
        else:
            if p + k >= len(a):
                continue
            ra = a.iloc[p:p + k].sum()
            rb = b.iloc[p:p + k].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
