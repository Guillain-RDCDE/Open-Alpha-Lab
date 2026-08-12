"""Strategy + inference for Study 844 — Madden-Cover-Curse.

The claim: **"Buy the hype into the new Madden / NBA 2K."** The gaming "curse" is about
the *cover athlete*; the finance transplant asks whether the *publisher* (EA for Madden,
TTWO for NBA 2K) earns an abnormal return around the annual game launch. Two windows,
both anchored on the real launch date and measured against SPY:

* **The run-up (the "buy the hype").** The publisher's abnormal return (publisher
  total-return minus the SPY total-return benchmark) over the K trading sessions ENDING
  on the last close on/before the launch. Tested at K = 5 (~1 week) and K = 10 (~2 weeks).

* **The launch drift (the "sell / ride the news").** The publisher's abnormal return over
  the K sessions STARTING from the launch-day close. Tested at K = 5 and K = 10. If the
  folklore is right the run-up is positive; the drift says whether a good ship keeps
  ripping or the news is sold.

Each launch is a single, non-overlapping event pooled across the two publishers (a
Madden launch → EA, a 2K launch → TTWO), so the primary statistic is a **one-sample t**
of the abnormal return across events (n = the number of launches with publisher+SPY
coverage). A **Newey-West (HAC) t** on the date-ordered per-event series is reported as a
robustness cross-check — with independent, well-spaced events it should land close to the
one-sample t. A **random-window placebo** (drawing many random, non-launch K-session
windows on the *same* publisher vs SPY) checks whether the observed mean sits inside or
outside the stock's ordinary tracking noise. With only 20 events, power is low — a mild
lean will not clear |t| ≥ 2, and we say so.

Survivorship note (Signal axis): EA and TTWO are both **survivors** — large, still-listed
publishers — so there is no delisting bias in the panel, but by the same token the sample
cannot speak to a franchise/publisher that failed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

PRE_SHORT_K = 5     # ~1 week run-up
PRE_LONG_K = 10     # ~2 weeks run-up
POST_SHORT_K = 5    # ~1 week launch drift
POST_LONG_K = 10    # ~2 weeks launch drift
COST_BPS = 5.0      # one-way, per leg
EXEC_LAG = 0        # launch date is calendar-known years ahead; no announcement lag


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per game launch: run-up + post-launch abnormal returns + inclusion.

    Each event is anchored to *its own* publisher (Madden → EA, 2K → TTWO) versus SPY. A
    row is INCLUDED only if the publisher and SPY both have cached history covering
    [anchor − PRE_LONG_K .. anchor + POST_LONG_K] for that event, so the funnel is
    auditable.
    """
    spy = prices[dt.BENCHMARK]
    rt = 2.0 * cost_bps / 1e4    # round-trip cost (one-way x 2), applied to the net leg
    rows = []
    for launch, publisher, title, athlete in dt.EVENTS:
        row = dict(launch=launch, publisher=publisher, title=title, athlete=athlete)
        pub = prices[publisher]
        common = pub.index.intersection(spy.index).sort_values()
        anchor_ts = pd.Timestamp(launch)
        on_or_before = common[common <= anchor_ts]
        if len(on_or_before) == 0:
            row.update(included=False, reason=f"no {publisher}/SPY coverage at the anchor")
            rows.append(row)
            continue
        p = common.get_loc(on_or_before[-1])
        if p - PRE_LONG_K < 0:
            row.update(included=False, reason="run-up window predates coverage")
            rows.append(row)
            continue
        if p + POST_LONG_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue

        def ar(i_start, i_end, _pub=pub, _spy=spy, _common=common):
            r_a = _pub.loc[_common[i_end]] / _pub.loc[_common[i_start]] - 1.0
            r_s = _spy.loc[_common[i_end]] / _spy.loc[_common[i_start]] - 1.0
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
# Inference primitives (canonical desk set — mirror of 803's)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    events (not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch (unequal-variance) two-sample t of mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 3) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 on the date-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


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
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 844,
                   tail: str = "right") -> dict:
    """For each INCLUDED event, redraw a random (non-launch) k-session window on the
    event's OWN publisher vs SPY and recompute the abnormal return; average across the
    same n events; repeat n_seeds x n_draws_per_seed times.

    ``tail``: "right" (a claim of POSITIVE mean -> p = share of null means >= observed) or
    "left" (a claim of NEGATIVE mean -> p = share of null means <= observed).
    """
    spy = prices[dt.BENCHMARK]
    inc = events[events["included"]] if "included" in events else events
    obs = float(inc[col].mean())

    # Precompute, per publisher, the aligned (pub, spy) arrays over their common index.
    pub_arrays = {}
    for publisher in dt.PUBLISHERS:
        pub = prices[publisher]
        common = pub.index.intersection(spy.index).sort_values()
        pub_arrays[publisher] = (pub.reindex(common).to_numpy(dtype=float),
                                 spy.reindex(common).to_numpy(dtype=float),
                                 len(common))

    rt = 2.0 * cost_bps / 1e4
    event_pubs = inc["publisher"].to_numpy()
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for publisher in event_pubs:
                pa, sa, m = pub_arrays[publisher]
                ppos = int(rng.integers(0, m - k - 1))
                r_a = pa[ppos + k] / pa[ppos] - 1.0
                r_s = sa[ppos + k] / sa[ppos] - 1.0
                draw_vals.append(float(r_a - r_s) - rt)
            means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative AR by trading-day offset around the launch
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series],
             pre: int = PRE_LONG_K, post: int = POST_LONG_K) -> pd.Series:
    """Mean cumulative abnormal return (publisher − SPY) at each offset from -pre..+post
    relative to the launch, normalised so offset 0 = 0%, averaged across all INCLUDED
    events. Negative offsets are the run-up; positive is the post-launch drift window.
    """
    spy = prices[dt.BENCHMARK]
    inc = events[events["included"]]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, row in inc.iterrows():
        pub = prices[row["publisher"]]
        common = pub.index.intersection(spy.index).sort_values()
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        base_a, base_s = pub.loc[common[p]], spy.loc[common[p]]
        vals = []
        for o in offsets:
            d = common[p + o]
            r_a = pub.loc[d] / base_a - 1.0
            r_s = spy.loc[d] / base_s - 1.0
            vals.append(float(r_a - r_s))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(drift: float, seed: int, k: int = POST_SHORT_K,
                     side: str = "post") -> dict:
    """Run the one-sample-t detector on a synthetic paired world with a planted
    launch-week drift. ``side='post'`` measures the launch window [p..p+k); ``side='pre'``
    measures the run-up [p-k..p)."""
    a, b, keys = dt.synthetic_world(drift=drift, seed=seed)
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
