"""Strategy + inference for Study 719 — Met-Gala-Luxury.

The claim: **the big European luxury houses get a spotlight bump around the Met Gala**
— fashion's biggest night, held the first Monday in May, where LVMH/Kering/Hermes/
Richemont brands dominate the red carpet. The folklore says the free global attention
lifts the luxury complex in the days around the event.

The Met Gala runs on a **Monday evening in New York** (~7pm ET red carpet). The European
exchanges that list all four names (Euronext Paris, SIX Swiss) close ~17:30 CET — i.e.
~7-8 hours BEFORE the red carpet even begins. That hands the study its single,
unavoidable execution lag for free:

* **day(-1)** = the European close on the gala Monday itself — the last price that does
  NOT yet know anything about the gala (the market shut hours before the event).
* **day(0)**  = the next European close (ordinarily Tuesday) — the first price that DOES
  reflect the gala. Entering at day(0)'s close (not before) is zero-look-ahead by
  construction.

Two measurements per event, both abnormal (equal-weighted luxury basket minus the VGK
Europe benchmark, both total-return):

* **Signal (the spotlight effect).** Cumulative abnormal return from day(-1) to
  day(-1)+k, for k = 5 trading sessions (~1 week) and k = 21 (~1 month). The honest size
  of "the bump", including the un-tradable Monday-close -> Tuesday-close overnight jump.
* **Tradability (can you bank it?).** The same abnormal return but entered at day(0)'s
  close instead of day(-1)'s -- i.e. AFTER the gala is already public and priced -- held
  k-1 further sessions, net of one round-trip's costs.

Because each Met Gala year is a single independent event (not a daily series), the
primary statistic is a **one-sample t** of the abnormal return across events (n is the
number of years with both the basket and VGK coverage -- small by construction, and said
out loud; VGK's 2005 inception is the binding floor). A random-window placebo (drawing
the same number of random, non-gala anchor dates from the basket's own history) checks
whether the observed mean is inside or outside the luck cloud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

WEEK_K = 5      # trading sessions ~ 1 week
MONTH_K = 21    # trading sessions ~ 1 month
COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per Met Gala year: basket + per-name abnormal returns + inclusion reason.

    A row is INCLUDED only if the gala was held AND both the luxury basket and the VGK
    benchmark cover [day(-1) .. day(-1)+MONTH_K] around the gala date. Excluded rows are
    kept (with a reason) so the funnel is auditable.
    """
    bench = prices[dt.EUROPE_BENCHMARK]
    basket = dt.basket_close(prices)
    common = basket.index.intersection(bench.index).sort_values()
    rows = []
    for year, gala_date, note in dt.EVENTS:
        if gala_date is None:
            rows.append(dict(year=year, gala_date=None, included=False,
                              reason=f"no gala held ({note})"))
            continue
        before = common[common <= pd.Timestamp(gala_date)]
        if len(before) == 0:
            rows.append(dict(year=year, gala_date=gala_date, included=False,
                              reason="basket/benchmark predate the gala (pre-VGK 2005)"))
            continue
        p = common.get_loc(before[-1])
        if p + MONTH_K >= len(common) or p + 1 >= len(common):
            rows.append(dict(year=year, gala_date=gala_date, included=False,
                              reason="insufficient trailing history"))
            continue
        d_m1, d_0 = common[p], common[p + 1]
        d_wk, d_mo = common[p + WEEK_K], common[p + MONTH_K]

        def ar(series, t0, t1):
            return float((series.loc[t1] / series.loc[t0] - 1.0)
                         - (bench.loc[t1] / bench.loc[t0] - 1.0))

        row = dict(year=year, gala_date=gala_date, included=True, reason="",
                   anchor_date=str(d_m1.date()), day0_date=str(d_0.date()))
        # basket signal (day(-1) -> day(-1)+k)
        row["ar_week"] = ar(basket, d_m1, d_wk)
        row["ar_month"] = ar(basket, d_m1, d_mo)
        # basket tradable capture (day(0) -> day(0)+k), zero look-ahead, net of costs
        for base, t_end in (("cap_week", d_wk), ("cap_month", d_mo)):
            gross = ar(basket, d_0, t_end)
            row[base + "_gross"] = gross
            row[base + "_net"] = gross - 2.0 * cost_bps / 1e4
        # per-name signal (for the "driven by one name?" third axis)
        for tk, name in dt.LUXURY.items():
            s = prices[tk]
            if d_m1 in s.index and d_mo in s.index and d_wk in s.index:
                row[f"name_{name}_wk"] = ar(s, d_m1, d_wk)
                row[f"name_{name}_mo"] = ar(s, d_m1, d_mo)
            else:
                row[f"name_{name}_wk"] = np.nan
                row[f"name_{name}_mo"] = np.nan
        rows.append(row)
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
    """Welch t of mean(a) - mean(b) (unequal variances)."""
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
# Random-window placebo: is the observed mean AR inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], col: str,
                   k: int, entry_offset: int = 0, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 719,
                   tail: str = "right") -> dict:
    """For each INCLUDED event, redraw a random (non-gala) k-session window on the SAME
    luxury basket vs VGK and recompute the abnormal return; average across the same n
    events; repeat n_seeds x n_draws_per_seed times.

    ``entry_offset`` matches the column being tested: 0 for the "ar_*" signal columns,
    1 for the "cap_*" tradable-capture columns (window anchored ON a random point,
    mirroring the zero-look-ahead day(0) entry). ``tail``: "right" (positive-bump claim)
    or "left" (observed negative).
    """
    bench = prices[dt.EUROPE_BENCHMARK]
    basket = dt.basket_close(prices)
    common = basket.index.intersection(bench.index).sort_values()
    inc = events[events["included"]]
    obs = float(inc[col].mean())
    n_ev = len(inc)

    lo = 0
    hi = len(common) - k - entry_offset - 1
    means = []
    if hi > lo:
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            for _ in range(n_draws_per_seed):
                pos = rng.integers(lo, hi, size=n_ev) + entry_offset
                vals = []
                for p in pos:
                    d0, d1 = common[p], common[p + k]
                    r = ((basket.loc[d1] / basket.loc[d0] - 1.0)
                         - (bench.loc[d1] / bench.loc[d0] - 1.0)) - 2.0 * cost_bps / 1e4
                    vals.append(r)
                means.append(np.mean(vals))
    means = np.asarray(means)
    if means.size == 0:
        return {"obs": obs, "placebo_mean": float("nan"), "placebo_sd": float("nan"),
                "p_value": float("nan"), "n_draws": 0}
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative basket AR by trading-day offset (for the chart)
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], pre: int = 5,
             post: int = MONTH_K) -> pd.Series:
    """Mean cumulative abnormal return (basket - VGK) at each offset -pre..+post from
    day(-1) (the pre-gala anchor), averaged across all INCLUDED events.

    The negative offsets show the RUN-UP (anticipation) into the gala; CAR is anchored so
    offset 0 (= day(-1), the last pre-gala close) is 0.
    """
    bench = prices[dt.EUROPE_BENCHMARK]
    basket = dt.basket_close(prices)
    common = basket.index.intersection(bench.index).sort_values()
    inc = events[events["included"]]
    offsets = list(range(-pre, post + 1))
    paths = []
    for _, row in inc.iterrows():
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        if p - pre < 0 or p + post >= len(common):
            continue
        d_anchor = common[p]
        vals = []
        for k in offsets:
            d_k = common[p + k]
            r = ((basket.loc[d_k] / basket.loc[d_anchor] - 1.0)
                 - (bench.loc[d_k] / bench.loc[d_anchor] - 1.0))
            vals.append(r)
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=offsets)


def per_name_stats(events: pd.DataFrame, horizon: str = "mo") -> pd.DataFrame:
    """One-sample t of each luxury name's abnormal return across events (the third-axis
    'driven by one name?' check). ``horizon`` is 'wk' or 'mo'."""
    inc = events[events["included"]]
    rows = []
    for name in dt.LUXURY.values():
        s = one_sample_t(inc[f"name_{name}_{horizon}"].values)
        rows.append(dict(name=name, n=s["n"], mean=s["mean"], t=s["t"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = MONTH_K) -> dict:
    """Run the one-sample-t detector on a synthetic paired (basket, benchmark) world
    with a planted bump on its synthetic event calendar."""
    a, b, events = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in events:
        if p + k >= len(a):
            continue
        ra = a.iloc[p:p + k + 1].sum()   # cumulative log-return, day(-1)->day(-1)+k
        rb = b.iloc[p:p + k + 1].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
