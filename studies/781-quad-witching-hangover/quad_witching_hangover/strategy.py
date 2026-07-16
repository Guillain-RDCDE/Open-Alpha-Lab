"""Strategy + inference for Study 781 — Quad-Witching-Hangover.

The claim: **the week after quad-witching underperforms.** Quad-witching is the third
Friday of Mar/Jun/Sep/Dec; the folklore says the post-expiration "hangover" week drifts
lower. Because SPY is the index itself, this is a **self-benchmarked single-tape** study —
we test SPY's own forward return, and the "is it abnormal?" question is answered by a
random-window placebo against SPY's own history (SPY drifts up, so "underperform" means
*below that drift*, which the placebo cloud measures directly).

Three windows, all anchored on the last close on/before the quad-witching Friday:

* **The hangover week (the claim).** SPY's raw return over the K = 5 sessions STARTING
  from the quad-witching close. If the folklore is right this is reliably negative — or at
  least well below SPY's ordinary +drift.
* **The two-week give-back.** K = 10 sessions after — a longer read on any post-expiration
  sag.
* **The run-in (contrast).** SPY's return over the K = 5 sessions ENDING on the quad-
  witching close, as a placebo-of-convenience: if *every* window around the date is noisy,
  the "hangover" is just calendar pareidolia.

Because each quad-witching Friday is an independent, non-overlapping quarterly event, the
primary statistic is a **one-sample t** of the forward return across events (n = quarters
with SPY coverage). A random-window placebo (drawing many random, non-quad-witching
K-session windows from SPY's own history) checks whether the observed mean sits inside or
outside the index's ordinary weekly noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

PRE_K = 5        # ~1 week run-in (contrast)
POST_SHORT_K = 5   # ~1 week hangover (the claim)
POST_LONG_K = 10   # ~2 week give-back
COST_BPS = 5.0     # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event forward returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per quad-witching quarter: run-in + post-event returns + inclusion.

    A row is INCLUDED only if SPY has cached history covering
    [anchor - PRE_K .. anchor + POST_LONG_K] for that quarter. Quarters whose window falls
    outside coverage are excluded with a reason, so the funnel is auditable.
    """
    spy = prices[dt.INSTRUMENT]
    common = spy.index.sort_values()
    rt = 2.0 * cost_bps / 1e4    # round-trip cost (one-way x 2), applied to the net leg
    rows = []
    for keydate, year, quarter in dt.EVENTS:
        row = dict(date=keydate, year=year, quarter=quarter)
        anchor_ts = pd.Timestamp(keydate)
        on_or_before = common[common <= anchor_ts]
        if len(on_or_before) == 0:
            row.update(included=False, reason="no SPY coverage at the anchor")
            rows.append(row)
            continue
        p = common.get_loc(on_or_before[-1])
        if p - PRE_K < 0:
            row.update(included=False, reason="run-in window predates coverage")
            rows.append(row)
            continue
        if p + POST_LONG_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue

        def ret(i_start, i_end):
            return float(spy.loc[common[i_end]] / spy.loc[common[i_start]] - 1.0)

        pre = ret(p - PRE_K, p)
        post_s = ret(p, p + POST_SHORT_K)
        post_l = ret(p, p + POST_LONG_K)
        row.update(
            included=True, reason="",
            anchor_date=str(common[p].date()),
            pre=pre, pre_net=pre - rt,
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
    quarterly events (not a daily panel)."""
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
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 793,
                   tail: str = "left") -> dict:
    """For each INCLUDED event, redraw a random (non-quad-witching) k-session window on SPY
    and recompute the raw return; average across the same n events; repeat
    n_seeds x n_draws_per_seed times.

    ``tail``: "left" (a claim of NEGATIVE / below-drift mean, e.g. the hangover -> p = share
    of null means <= observed) or "right" (a claim of POSITIVE mean -> p = share of null
    means >= observed).
    """
    spy = prices[dt.INSTRUMENT]
    common = spy.index.sort_values()
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
                r = spy.loc[d_end] / spy.loc[d_start] - 1.0
                draw_vals.append(float(r) - rt)
            means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means <= obs).mean()) if tail == "left" else float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative return by trading-day offset around the anchor
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series],
             pre: int = PRE_K, post: int = POST_LONG_K) -> pd.Series:
    """Mean cumulative SPY return at each offset from -pre..+post relative to the quad-
    witching close, normalised so offset 0 = 0%, averaged across all INCLUDED events.
    Negative offsets are the run-in; positive is the post-expiration "hangover" window.
    """
    spy = prices[dt.INSTRUMENT]
    common = spy.index.sort_values()
    inc = events[events["included"]]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, row in inc.iterrows():
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        base = spy.loc[common[p]]
        paths.append([float(spy.loc[common[p + o]] / base - 1.0) for o in offsets])
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(dip: float, seed: int, k: int = POST_SHORT_K,
                     side: str = "post") -> dict:
    """Run the one-sample-t detector on a synthetic single-name world with a planted
    post-event hangover. ``side='post'`` measures the hangover window [p..p+k);
    ``side='pre'`` measures the run-in [p-k..p)."""
    a, keys = dt.synthetic_world(dip=dip, seed=seed)
    r = []
    for p in keys:
        if side == "post":
            if p + k >= len(a):
                continue
            r.append(float(a.iloc[p:p + k].sum()))
        else:
            if p - k < 0:
                continue
            r.append(float(a.iloc[p - k:p].sum()))
    return one_sample_t(np.asarray(r))
