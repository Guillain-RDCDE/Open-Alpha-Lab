"""Strategy + inference for Study 771 — Box-Office-Bomb.

The claim: **sell Disney (DIS) after one of its movies flops.** The event is a wide
theatrical release's opening weekend; the flop is common knowledge by the following
Monday (weekend box-office estimates print Sunday, actuals Monday). So the trade anchor is
the **first trading session on/after the Monday after opening** — calendar-known,
zero-look-ahead. Two windows, both anchored there:

* **The post-flop (the "sell the news").** DIS's abnormal return (DIS total-return minus
  the SPY total-return benchmark) over the K trading sessions STARTING from the anchor
  close. Tested at K = 10 (~2 weeks) and K = 21 (~1 month). If the folklore is right, this
  is negative.

* **The pre-release drift (context).** DIS's abnormal return over the K sessions ENDING at
  the anchor. Tested at K = 10 and K = 21. Did the stock already sag into the release, or
  is any move purely post-flop? Reported for completeness.

Because each flop is a single independent event (not a daily series), the primary
statistic is a **one-sample t** of the abnormal return across events (n = the number of
flops with DIS+SPY coverage). A random-window placebo (drawing many random, non-flop
K-session windows from DIS's own history vs SPY) checks whether the observed mean sits
inside or outside the stock's ordinary tracking noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

PRE_SHORT_K = 10    # ~2 weeks pre-release drift
PRE_LONG_K = 21     # ~1 month pre-release drift
POST_SHORT_K = 10   # ~2 weeks post-flop
POST_LONG_K = 21    # ~1 month post-flop
COST_BPS = 5.0      # one-way, per leg


def _anchor_ts(open_date: str) -> pd.Timestamp:
    """The first *calendar* day at which the flop is public: the Monday after the opening
    weekend. Opening Friday + 3 days = Monday; a Wednesday/Thanksgiving open + 3 days lands
    on the weekend, and the trading-calendar snap below moves it to the following Monday
    all the same. The trading-session snap happens in build_event_table."""
    return pd.Timestamp(open_date) + pd.Timedelta(days=3)


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per flop: pre-release + post-flop abnormal returns + inclusion.

    The anchor is the first trading session on/after the Monday after the opening weekend
    (the first session at which the flop is common knowledge). A row is INCLUDED only if DIS
    and SPY both have cached history covering [anchor - PRE_LONG_K .. anchor + POST_LONG_K]
    for that event. Events whose window falls outside coverage are excluded with a reason,
    so the funnel is auditable.
    """
    dis = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = dis.index.intersection(spy.index).sort_values()
    rt = 2.0 * cost_bps / 1e4    # round-trip cost (one-way x 2), applied to the net leg
    rows = []
    for title, open_date in dt.EVENTS:
        row = dict(title=title, opening=open_date)
        anchor_target = _anchor_ts(open_date)
        on_or_after = common[common >= anchor_target]
        if len(on_or_after) == 0:
            row.update(included=False, reason="no DIS/SPY coverage at the anchor")
            rows.append(row)
            continue
        anchor = on_or_after[0]
        p = common.get_loc(anchor)
        if p - PRE_LONG_K < 0:
            row.update(included=False, reason="pre-release window predates coverage")
            rows.append(row)
            continue
        if p + POST_LONG_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue

        def ar(i_start, i_end):
            r_a = dis.loc[common[i_end]] / dis.loc[common[i_start]] - 1.0
            r_s = spy.loc[common[i_end]] / spy.loc[common[i_start]] - 1.0
            return float(r_a - r_s)

        pre_s = ar(p - PRE_SHORT_K, p)
        pre_l = ar(p - PRE_LONG_K, p)
        post_s = ar(p, p + POST_SHORT_K)
        post_l = ar(p, p + POST_LONG_K)
        # The trade is a SHORT (sell the studio), so the net-of-cost tradable return is
        # -post + cost drag: we report post_*_net as the SHORT's net return (-post - rt).
        row.update(
            included=True, reason="",
            anchor_date=str(anchor.date()),
            pre_s=pre_s, pre_l=pre_l,
            post_s=post_s, post_l=post_l,
            short_s_net=-post_s - rt, short_l_net=-post_l - rt,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    flop events (not a daily panel)."""
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
    """Share of events with a POSITIVE abnormal return. For the 'sell after the flop'
    claim the interesting quantity is 1 - rate (share negative), but we report the positive
    rate for consistency with the rest of the lab."""
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
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 773,
                   tail: str = "left") -> dict:
    """For each INCLUDED event, redraw a random (non-flop) k-session window on DIS vs SPY
    and recompute the abnormal return; average across the same n events; repeat
    n_seeds x n_draws_per_seed times.

    ``tail``: "left" (a claim of NEGATIVE mean, e.g. the post-flop sell-off -> p = share of
    null means <= observed) or "right" (a claim of POSITIVE mean -> p = share of null means
    >= observed).
    """
    dis = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = dis.index.intersection(spy.index).sort_values()
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
                r_a = dis.loc[d_end] / dis.loc[d_start] - 1.0
                r_s = spy.loc[d_end] / spy.loc[d_start] - 1.0
                draw_vals.append(float(r_a - r_s) - rt)
            means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means <= obs).mean()) if tail == "left" else float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative AR by trading-day offset around the anchor
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series],
             pre: int = PRE_LONG_K, post: int = POST_LONG_K) -> pd.Series:
    """Mean cumulative abnormal return (DIS - SPY) at each offset from -pre..+post relative
    to the anchor (the Monday-after), normalised so offset 0 = 0%, averaged across all
    INCLUDED events. Negative offsets are the pre-release drift; positive is the post-flop
    window.
    """
    dis = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = dis.index.intersection(spy.index).sort_values()
    inc = events[events["included"]]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, row in inc.iterrows():
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        base_a, base_s = dis.loc[common[p]], spy.loc[common[p]]
        vals = []
        for o in offsets:
            d = common[p + o]
            r_a = dis.loc[d] / base_a - 1.0
            r_s = spy.loc[d] / base_s - 1.0
            vals.append(float(r_a - r_s))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(drop: float, seed: int, k: int = POST_SHORT_K,
                     side: str = "post") -> dict:
    """Run the one-sample-t detector on a synthetic paired world with a planted post-flop
    sell-off. ``side='post'`` measures the post-flop window [p..p+k); ``side='pre'``
    measures [p-k..p). A larger planted ``drop`` should drive the post-window mean (and t)
    more NEGATIVE, monotonically."""
    a, b, keys = dt.synthetic_world(drop=drop, seed=seed)
    ar = []
    for p in keys:
        if side == "post":
            if p + k >= len(a):
                continue
            ra = a.iloc[p:p + k].sum()
            rb = b.iloc[p:p + k].sum()
        else:
            if p - k < 0:
                continue
            ra = a.iloc[p - k:p].sum()
            rb = b.iloc[p - k:p].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
