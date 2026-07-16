"""Strategy + inference for Study 780 — Long-Weekend-Drift.

The claim: **SPY drifts up on the pre-holiday session.** Because SPY is tested on its own
(a single-series, self-benchmarked calendar effect), "abnormal" means *excess over the
sample's own mean daily return* — does the holiday-eve session beat an ordinary SPY day?

Three windows, all anchored on the pre-holiday session (the last SPY close strictly before
each NYSE full-day closure, calendar-known years ahead):

* **pre1 (the classic).** SPY's 1-session return on the holiday-eve close, minus the sample
  mean daily return. This is the textbook pre-holiday effect (Ariel 1990).
* **pre3 (broader run-up).** SPY's cumulative return over the 3 sessions ending on the
  holiday-eve close, minus 3x the mean daily return.
* **post1 (reversal check).** SPY's 1-session return on the FIRST session after the holiday,
  minus the mean daily return. If the eve pops and then gives it back, that is a round-trip,
  not an edge.

Because each holiday is a single, essentially independent event, the primary statistic is a
**one-sample t** of the excess return across events (n = the number of holidays with SPY
coverage). A random-window placebo (drawing many random, non-holiday sessions from SPY's own
history) checks whether the observed mean sits inside or outside the market's ordinary
day-to-day noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

PRE1_K = 1      # the classic single holiday-eve session
PRE3_K = 3      # a 3-session run-up into the holiday
POST1_K = 1     # first session after the holiday (reversal check)
COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Baseline: the sample's own mean daily return (the "ordinary day")
# --------------------------------------------------------------------------- #
def baseline_daily(prices: dict[str, pd.Series]) -> float:
    """Mean daily simple return of SPY over the full cached sample — the drift an ordinary
    session earns, which the pre-holiday excess is measured against."""
    spy = prices[dt.INSTRUMENT]
    return float(spy.pct_change().dropna().mean())


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded holiday calendar -> per-event excess returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per holiday: holiday-eve excess returns (pre1, pre3) + a post-holiday reversal
    check (post1) + inclusion.

    A row is INCLUDED only if SPY has history covering [eve - PRE3_K .. eve + POST1_K] for
    that holiday. Holidays whose window falls outside coverage are excluded with a reason,
    so the funnel is auditable. "eve" is the last SPY close strictly before the holiday.
    """
    spy = prices[dt.INSTRUMENT]
    idx = spy.index.sort_values()
    mu = baseline_daily(prices)
    rt = 2.0 * cost_bps / 1e4     # round-trip cost (one-way x 2), applied to the net leg
    rows = []
    for holiday, name in dt.EVENTS:
        row = dict(holiday=holiday, name=name)
        anchor_ts = pd.Timestamp(holiday)
        before = idx[idx < anchor_ts]
        if len(before) == 0:
            row.update(included=False, reason="no SPY coverage before the holiday")
            rows.append(row)
            continue
        p = idx.get_loc(before[-1])           # holiday-eve session position
        if p - PRE3_K < 0:
            row.update(included=False, reason="run-up window predates coverage")
            rows.append(row)
            continue
        if p + POST1_K >= len(idx):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue

        def win(i_start, i_end):
            return float(spy.iloc[i_end] / spy.iloc[i_start] - 1.0)

        pre1 = win(p - PRE1_K, p) - PRE1_K * mu
        pre3 = win(p - PRE3_K, p) - PRE3_K * mu
        post1 = win(p, p + POST1_K) - POST1_K * mu
        row.update(
            included=True, reason="",
            eve_date=str(idx[p].date()),
            pre1=pre1, pre1_net=pre1 - rt,
            pre3=pre3, pre3_net=pre3 - rt,
            post1=post1,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    holiday events (not a daily panel)."""
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
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 792,
                   tail: str = "right") -> dict:
    """For each INCLUDED event, redraw a random (non-holiday) k-session SPY window and
    recompute the excess return (over k x the mean daily return); average across the same n
    events; repeat n_seeds x n_draws_per_seed times.

    ``tail``: "right" (a claim of POSITIVE mean -> p = share of null means >= observed) or
    "left" (a claim of NEGATIVE mean -> p = share of null means <= observed).
    """
    spy = prices[dt.INSTRUMENT]
    idx = spy.index.sort_values()
    mu = baseline_daily(prices)
    n_events = int(events["included"].sum()) if "included" in events else len(events)
    inc = events[events["included"]] if "included" in events else events
    obs = float(inc[col].mean())

    lo, hi = k, len(idx) - k - 1
    rt = 2.0 * cost_bps / 1e4
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for _e in range(n_events):
                ppos = int(rng.integers(lo, hi))
                r = float(spy.iloc[ppos] / spy.iloc[ppos - k] - 1.0) - k * mu
                draw_vals.append(r - rt)
            means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative excess return by trading-day offset around the eve
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series],
             pre: int = 5, post: int = 5) -> pd.Series:
    """Mean cumulative excess return (SPY minus its own drift) at each offset from -pre..+post
    relative to the holiday-eve session, normalised so offset 0 = 0%, averaged across all
    INCLUDED events. Negative offsets lead into the holiday; positive offsets are post-holiday.
    """
    spy = prices[dt.INSTRUMENT]
    idx = spy.index.sort_values()
    mu = baseline_daily(prices)
    inc = events[events["included"]]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, row in inc.iterrows():
        p = idx.get_loc(pd.Timestamp(row["eve_date"]))
        if p - pre < 0 or p + post >= len(idx):
            continue
        base = spy.iloc[p]
        vals = [float(spy.iloc[p + o] / base - 1.0) - o * mu for o in offsets]
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = PRE1_K, side: str = "pre") -> dict:
    """Run the one-sample-t detector on a synthetic single-series world with a planted
    pre-holiday bump. ``side='pre'`` measures the run-up window ending on the eve [p-k+1..p];
    ``side='post'`` measures the session after the holiday [p+1]."""
    r, evs = dt.synthetic_world(bump=bump, seed=seed)
    mu = float(r.mean())
    ar = []
    for p in evs:
        if side == "pre":
            if p - (k - 1) < 0:
                continue
            val = float(r.iloc[p - k + 1:p + 1].sum()) - k * mu
        else:
            if p + 1 >= len(r):
                continue
            val = float(r.iloc[p + 1]) - mu
        ar.append(val)
    return one_sample_t(np.asarray(ar))
