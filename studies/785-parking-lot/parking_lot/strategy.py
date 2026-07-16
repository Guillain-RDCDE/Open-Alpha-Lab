"""Strategy + inference for Study 785 — Parking-Lot (shape C: labelled-proxy -> forward return).

The claim: **satellite parking-lot counts beat the earnings print for Walmart (WMT).** The
signal is the LABELLED-PROXY quarterly foot-traffic index (see ``data.py``), used ordinally: a
quarter is 'busy' if the year-over-year parking change is positive, 'slow' if negative. The
tradable idea is a "long busy-quarter, short slow-quarter" timing rule entered at the WMT
earnings-print close (the alt-data verdict is 'known' before the print) and held forward:

* **Short forward window, K = 5 (~one week post-print drift).**
* **Long forward window, K = 21 (~one month post-print drift).**

We measure WMT's forward abnormal return (WMT total-return minus SPY total-return) over the K
sessions AFTER the print anchor. If the folklore is right, BUSY quarters drift up, SLOW quarters
drift down, and the long/short spread (busy minus slow) is positive. The primary statistic is a
**one-sample t** of the direction-signed forward return ``sign(yoy) * fwd`` across events (the
long/short return series), plus a Welch two-sample t of the busy-vs-slow spread. A sign-shuffle
placebo (randomly relabelling which quarters were 'busy') checks whether the observed spread is
inside or outside ordinary luck given these same forward returns.

CAVEAT, named up front: the parking signal is a LABELLED PROXY, not real satellite data, and the
print anchors are stylised reporting windows (not exact release timestamps). So this study tests
the *method and the proxy*, not a live orbital feed — a positive result here would still need
the real, paywalled panel to bank. The synthetic control shows the detector recovers a real
planted link and stays quiet on the null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

FWD_SHORT_K = 5      # ~one week post-print drift
FWD_LONG_K = 21      # ~one month post-print drift
COST_BPS = 5.0       # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: LABELLED-PROXY parking events -> forward abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS) -> pd.DataFrame:
    """One row per parking quarter: direction + forward abnormal returns + inclusion.

    The anchor is the last session on/before the stylised WMT reporting date for that quarter
    (zero look-ahead — the reporting cadence and the pre-print parking verdict are both known by
    then). A row is INCLUDED only if WMT and SPY both cover [anchor .. anchor+FWD_LONG_K].
    'flat' events are kept in the table but excluded from the busy/slow tests.
    """
    wmt = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = wmt.index.intersection(spy.index).sort_values()
    rt = 2.0 * cost_bps / 1e4    # round-trip cost (one-way x 2), applied to the net leg
    ev = dt.parking_events()
    rows = []
    for _, e in ev.iterrows():
        row = dict(year=int(e["year"]), tag=e["tag"], anchor=e["anchor"],
                   yoy=float(e["yoy"]), direction=e["direction"])
        anchor_ts = pd.Timestamp(e["anchor"])
        on_or_before = common[common <= anchor_ts]
        if len(on_or_before) == 0:
            row.update(included=False, reason="no WMT/SPY coverage at the anchor")
            rows.append(row)
            continue
        p = common.get_loc(on_or_before[-1])
        if p + FWD_LONG_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue

        def fwd_ar(i_start, i_end):
            r_a = wmt.loc[common[i_end]] / wmt.loc[common[i_start]] - 1.0
            r_s = spy.loc[common[i_end]] / spy.loc[common[i_start]] - 1.0
            return float(r_a - r_s)

        fwd_s = fwd_ar(p, p + FWD_SHORT_K)
        fwd_l = fwd_ar(p, p + FWD_LONG_K)
        row.update(
            included=True, reason="",
            anchor_date=str(common[p].date()),
            fwd_s=fwd_s, fwd_s_net=fwd_s - rt,
            fwd_l=fwd_l, fwd_l_net=fwd_l - rt,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def busy(events: pd.DataFrame) -> pd.DataFrame:
    return events[(events["included"]) & (events["direction"] == "busy")]


def slow(events: pd.DataFrame) -> pd.DataFrame:
    return events[(events["included"]) & (events["direction"] == "slow")]


def longshort_returns(events: pd.DataFrame, col: str) -> np.ndarray:
    """The per-event long/short return series: +fwd on busy quarters, -fwd on slow quarters
    (flat excluded). Its mean is the busy-minus-slow timing P&L; feed it to one_sample_t."""
    inc = events[events["included"]]
    inc = inc[inc["direction"] != "flat"]
    w = np.where(inc["direction"].values == "busy", 1.0, -1.0)
    return w * inc[col].values.astype(float)


# --------------------------------------------------------------------------- #
# Inference primitives (shared with the repo's other event studies)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent signal events (not a
    daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def two_sample_t(x: np.ndarray, y: np.ndarray) -> dict:
    """Welch two-sample t of mean(x) - mean(y) -- the 'busy minus slow' spread."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return {"nx": nx, "ny": ny, "diff": float("nan"), "t": float("nan")}
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    diff = float(x.mean() - y.mean())
    return {"nx": nx, "ny": ny, "diff": diff, "t": float(diff / se) if se > 0 else float("nan")}


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


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between the parking YoY and the forward return (numpy-only)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Sign-shuffle placebo: is the observed long/short spread inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, col: str,
                   n_seeds: int = 40, n_draws_per_seed: int = 250, base_seed: int = 810,
                   tail: str = "right") -> dict:
    """Hold the forward returns of the non-flat events fixed; randomly RELABEL which quarters
    were 'busy' vs 'slow' (independent +/-1 sign flips) and recompute the long/short mean. Repeat
    n_seeds x n_draws_per_seed times. If the observed busy-minus-slow timing P&L sits in the tail
    of that null, the parking labels carry information beyond luck.

    ``tail``: "right" (claim of POSITIVE long/short mean -> p = share of null means >= observed).
    """
    inc = events[events["included"]]
    inc = inc[inc["direction"] != "flat"]
    fwd = inc[col].values.astype(float)
    w = np.where(inc["direction"].values == "busy", 1.0, -1.0)
    obs = float(np.mean(w * fwd))
    n = len(fwd)

    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            wp = rng.choice((-1.0, 1.0), size=n)
            means.append(float(np.mean(wp * fwd)))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative forward AR by trading-day offset from the print anchor
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], subset: str = "busy",
             post: int = FWD_LONG_K) -> pd.Series:
    """Mean cumulative abnormal return (WMT - SPY) at each offset from 0..+post relative to the
    print anchor, normalised so offset 0 = 0%, averaged across all INCLUDED events in the subset
    ('busy' or 'slow')."""
    wmt = prices[dt.INSTRUMENT]
    spy = prices[dt.BENCHMARK]
    common = wmt.index.intersection(spy.index).sort_values()
    sub = busy(events) if subset == "busy" else slow(events)
    offsets = list(range(0, post + 1))
    paths = []
    for _, row in sub.iterrows():
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        base_a, base_s = wmt.loc[common[p]], spy.loc[common[p]]
        vals = []
        for o in offsets:
            d = common[p + o]
            r_a = wmt.loc[d] / base_a - 1.0
            r_s = spy.loc[d] / base_s - 1.0
            vals.append(float(r_a - r_s))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = FWD_LONG_K) -> dict:
    """Run the one-sample-t long/short detector on a synthetic paired world whose print
    positions carry a planted two-sided forward link. For each print, the long/short return is
    ``dir * forward_AR`` over [p .. p+k); ``bump = 0`` is the null."""
    a, b, sig, prints = dt.synthetic_world(bump=bump, seed=seed)
    ls = []
    for p in prints:
        if p + k >= len(a):
            continue
        d = float(sig.iloc[p])
        ra = a.iloc[p:p + k].sum()
        rs = b.iloc[p:p + k].sum()
        ls.append(d * float(ra - rs))
    return one_sample_t(np.asarray(ls))
