"""Strategy + inference for Study 783 — IPO-Deal-Of-Year.

The claim: **the banks' celebrated 'IPO of the year' then underperforms.** For each marquee
debut we anchor on its **first trading close** and measure the **forward abnormal return**
(name total-return minus SPY total-return) over three horizons:

* **3 months (K = 63 sessions)** — the immediate post-pop / lockup-approach window.
* **6 months (K = 126 sessions)** — post-lockup drift.
* **12 months (K = 252 sessions)** — the classic first-year window (Ritter's underperformance
  horizon, compressed from his 3-year cut so recent 2023/24 debuts still qualify).

Because each debut is a single independent event (not a daily series), the primary statistic
is a **one-sample t** of the forward abnormal return across names (n = the number of marquee
IPOs with full forward coverage). A random-window placebo (drawing many random, non-IPO
K-session windows from *each name's own* post-listing history vs SPY) checks whether the
observed mean sits inside or outside these names' ordinary abnormal-return noise.

NOTE ON LOOK-AHEAD. Anchoring on the first close is *descriptive*, not tradable: the "IPO of
the year" crown is awarded months later, so you could not have bought the basket at its opens.
The forward return is netted of a round-trip cost so the tradability axis is at least costed,
but the honest reading is a Mirage regardless of the gross sign — see docs/results.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

H_3M = 63      # ~3 months
H_6M = 126     # ~6 months
H_12M = 252    # ~12 months
COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event forward abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per marquee IPO: forward abnormal returns at 3/6/12 months + inclusion.

    A row is INCLUDED only if the name and SPY both have cached history covering
    [first_close .. first_close + H_12M] for that debut. Names whose 12-month window falls
    outside coverage are excluded with a reason, so the funnel is auditable.
    """
    spy = prices[dt.BENCHMARK]
    rt = 2.0 * cost_bps / 1e4    # round-trip cost (one-way x 2), applied to the net leg
    rows = []
    for tkr, ipo_date, label in dt.EVENTS:
        row = dict(ticker=tkr, ipo=ipo_date, label=label)
        if tkr not in prices:
            row.update(included=False, reason="no cached tape for name")
            rows.append(row)
            continue
        name = prices[tkr]
        common = name.index.intersection(spy.index).sort_values()
        anchor_ts = pd.Timestamp(ipo_date)
        on_or_after = common[common >= anchor_ts]
        if len(on_or_after) == 0:
            row.update(included=False, reason="no name/SPY coverage at the debut")
            rows.append(row)
            continue
        p = common.get_loc(on_or_after[0])
        if p + H_12M >= len(common):
            row.update(included=False, reason="insufficient forward history (<12m)")
            rows.append(row)
            continue

        def fr(k):
            r_a = name.loc[common[p + k]] / name.loc[common[p]] - 1.0
            r_s = spy.loc[common[p + k]] / spy.loc[common[p]] - 1.0
            return float(r_a - r_s)

        f3, f6, f12 = fr(H_3M), fr(H_6M), fr(H_12M)
        row.update(
            included=True, reason="",
            anchor_date=str(common[p].date()),
            fwd_3m=f3, fwd_3m_net=f3 - rt,
            fwd_6m=f6, fwd_6m_net=f6 - rt,
            fwd_12m=f12, fwd_12m_net=f12 - rt,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    IPO events (not a daily panel)."""
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
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 802,
                   tail: str = "left") -> dict:
    """For each INCLUDED event, redraw a random (non-IPO-anchored) k-session window on THAT
    NAME'S own post-listing history vs SPY and recompute the abnormal return; average across
    the same n events; repeat n_seeds x n_draws_per_seed times.

    Drawing from each name's own tape controls for the fact that new listings are simply
    more volatile — the null is "a random slice of this stock's life," not "a random slice of
    the market." ``tail``: "left" (a claim of NEGATIVE mean, i.e. underperformance -> p =
    share of null means <= observed) or "right".
    """
    spy = prices[dt.BENCHMARK]
    inc = events[events["included"]] if "included" in events else events
    obs = float(inc[col].mean())
    rt = 2.0 * cost_bps / 1e4

    # pre-index each included name's common calendar with SPY
    per_name = []
    for _, r in inc.iterrows():
        name = prices[r["ticker"]]
        common = name.index.intersection(spy.index).sort_values()
        if len(common) > k + 1:
            per_name.append((name, common))

    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for name, common in per_name:
                ppos = int(rng.integers(0, len(common) - k - 1))
                d_start, d_end = common[ppos], common[ppos + k]
                r_a = name.loc[d_end] / name.loc[d_start] - 1.0
                r_s = spy.loc[d_end] / spy.loc[d_start] - 1.0
                draw_vals.append(float(r_a - r_s) - rt)
            means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means <= obs).mean()) if tail == "left" else float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative AR by trading-day offset AFTER the debut
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series],
             post: int = H_12M) -> pd.Series:
    """Mean cumulative abnormal return (name - SPY) at each offset from 0..+post relative to
    the first trading close, normalised so offset 0 = 0%, averaged across all INCLUDED
    events. Positive offsets trace the post-IPO forward path.
    """
    spy = prices[dt.BENCHMARK]
    inc = events[events["included"]]
    offsets = list(range(0, post + 1))
    paths = []
    for _, row in inc.iterrows():
        name = prices[row["ticker"]]
        common = name.index.intersection(spy.index).sort_values()
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        base_a, base_s = name.loc[common[p]], spy.loc[common[p]]
        vals = []
        for o in offsets:
            d = common[p + o]
            r_a = name.loc[d] / base_a - 1.0
            r_s = spy.loc[d] / base_s - 1.0
            vals.append(float(r_a - r_s))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = H_12M) -> dict:
    """Run the one-sample-t detector on a synthetic paired world with a planted post-IPO
    forward drift. Measures the forward abnormal return over [p .. p+k) across the synthetic
    IPO anchors."""
    a, b, ipos = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in ipos:
        if p + k >= len(a):
            continue
        ra = a.iloc[p:p + k].sum()
        rb = b.iloc[p:p + k].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
