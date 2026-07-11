"""Strategy + inference for Study 708 — Eurovision-Effect.

The claim: **the winning (or hosting) country's stock market gets a feel-good bump**
around the Grand Final -- the folklore cousin of the Edmans-Garcia-Norli
elimination-shock effect, applied to a song contest instead of a football tournament.

The Grand Final airs on a **Saturday night**, a non-trading day, so nobody can act on
the result until markets reopen. That gives the study its single, unavoidable
execution lag for free:

* **day(-1)** = the last trading close *before* the final (ordinarily the Friday) --
  the last price that does NOT yet know the winner.
* **day(0)**  = the first trading close *after* the final (ordinarily the Monday) --
  the first price that DOES know the winner. By the time markets reopen the result is
  ~36-60 hours old and globally televised, so entering at day(0)'s close (not before)
  is zero-look-ahead by construction.

Two measurements per event, both abnormal (country ETF minus the VGK Europe
benchmark, both total-return):

* **Signal (the announcement effect).** Cumulative abnormal return from day(-1) to
  day(-1)+k, for k = 5 trading sessions (~1 week) and k = 21 (~1 month). This is the
  full price reaction to the result becoming public, including the un-tradable
  Friday-close -> Monday-close weekend jump -- the honest size of "the bump", not a
  strategy.
* **Tradability (can retail bank it?).** The same abnormal return but entered at
  day(0)'s close instead of day(-1)'s -- i.e. AFTER the result is already public and
  priced -- held k-1 further sessions, net of one round-trip's costs. This is what a
  retail investor reacting to Sunday-morning headlines could actually capture.

Because each Eurovision year is a single independent event (not a daily series), the
primary statistic is a **one-sample t** of the abnormal return across events (n is the
number of countries/years with BOTH a tradable ETF and benchmark coverage -- small by
construction, and said out loud). A random-window placebo (drawing the same number of
random, non-Eurovision anchor dates from each ticker's own history) checks whether the
observed mean is inside or outside the luck cloud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

WEEK_K = 5     # trading sessions ~ 1 week
MONTH_K = 21   # trading sessions ~ 1 month
COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar + country map -> per-event abnormal returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per (year, role) Eurovision event: abnormal returns + inclusion reason.

    ``role`` is "winner" or "host". A row is INCLUDED only if the country has a
    mapped ETF (``data.COUNTRY_ETF``) AND that ETF's cached history AND the VGK
    benchmark's cached history both cover [day(-1) .. day(-1)+MONTH_K] for that
    event. Excluded rows are kept (with a reason) so the funnel is auditable.
    """
    bench = prices[dt.EUROPE_BENCHMARK]
    rows = []
    for year, final_date, winner, host in dt.EVENTS:
        if final_date is None:
            rows.append(dict(year=year, role="winner", country=None, ticker=None,
                              included=False, reason="2020 cancelled (COVID-19)"))
            rows.append(dict(year=year, role="host", country=host, ticker=None,
                              included=False, reason="2020 cancelled (COVID-19)"))
            continue
        for role, country in (("winner", winner), ("host", host)):
            ticker = dt.COUNTRY_ETF.get(country)
            row = dict(year=year, role=role, country=country, ticker=ticker,
                       final_date=final_date)
            if ticker is None:
                row.update(included=False,
                           reason=f"no US-listed {country} ETF exists")
                rows.append(row)
                continue
            etf = prices[ticker]
            common = etf.index.intersection(bench.index).sort_values()
            before = common[common < pd.Timestamp(final_date)]
            if len(before) == 0:
                row.update(included=False, reason="ETF/benchmark predate the final")
                rows.append(row)
                continue
            p = common.get_loc(before[-1])
            if p + MONTH_K >= len(common):
                row.update(included=False, reason="insufficient trailing history")
                rows.append(row)
                continue
            d_m1, d_0 = common[p], common[p + 1]
            d_wk, d_mo = common[p + WEEK_K], common[p + MONTH_K]

            def ar(t_end):
                r_etf = etf.loc[t_end] / etf.loc[d_m1] - 1.0
                r_bench = bench.loc[t_end] / bench.loc[d_m1] - 1.0
                return float(r_etf - r_bench)

            def cap(t_end):
                # entered at day(0)'s close -- AFTER the result is public: zero look-ahead
                r_etf = etf.loc[t_end] / etf.loc[d_0] - 1.0
                r_bench = bench.loc[t_end] / bench.loc[d_0] - 1.0
                gross = float(r_etf - r_bench)
                return gross, gross - 2.0 * cost_bps / 1e4

            ar_week, ar_month = ar(d_wk), ar(d_mo)
            cap_week_gross, cap_week_net = cap(d_wk)
            cap_month_gross, cap_month_net = cap(d_mo)
            row.update(
                included=True, reason="",
                anchor_date=str(d_m1.date()), day0_date=str(d_0.date()),
                ar_week=ar_week, ar_month=ar_month,
                cap_week_gross=cap_week_gross, cap_week_net=cap_week_net,
                cap_month_gross=cap_month_gross, cap_month_net=cap_month_net,
            )
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
    """Welch t of mean(a) - mean(b) (unequal variances) -- used for the "winning vs
    hosting" third-axis comparison. NaN if either group has < 2 obs."""
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
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 708,
                   tail: str = "right") -> dict:
    """For each INCLUDED event, redraw a random (non-Eurovision) k-session window on
    the SAME ticker's own history and recompute the abnormal return; average across
    the same n events; repeat n_seeds x n_draws_per_seed times.

    ``entry_offset`` matches the column being tested: 0 for the "ar_*" signal columns
    (window anchored the session BEFORE a random point, mirroring day(-1)->day(-1)+k),
    1 for the "cap_*" tradable-capture columns (window anchored ON a random point,
    mirroring the zero-look-ahead day(0)->day(0)+k entry). ``tail``: "right" (the
    claim predicts a POSITIVE bump -> p = share of null means >= observed) or "left"
    (observed is negative -> p = share of null means <= observed).
    """
    bench = prices[dt.EUROPE_BENCHMARK]
    inc = events[events["included"]]
    obs = float(inc[col].mean())

    tickers = inc["ticker"].tolist()
    common_by_ticker = {t: prices[t].index.intersection(bench.index).sort_values()
                        for t in set(tickers)}
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for t in tickers:
                common = common_by_ticker[t]
                lo = 0
                hi = len(common) - k - entry_offset - 1
                if hi <= lo:
                    continue
                p = int(rng.integers(lo, hi)) + entry_offset
                d_start = common[p]
                d_end = common[p + k]
                etf = prices[t]
                r_etf = etf.loc[d_end] / etf.loc[d_start] - 1.0
                r_bench = bench.loc[d_end] / bench.loc[d_start] - 1.0
                draw_vals.append(float(r_etf - r_bench) - 2.0 * cost_bps / 1e4)
            if draw_vals:
                means.append(np.mean(draw_vals))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative AR by trading-day offset (for the notebook chart)
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], role: str,
            max_k: int = MONTH_K) -> pd.Series:
    """Mean cumulative abnormal return (country - VGK) at each offset 0..max_k from
    day(-1) (the pre-final anchor), averaged across all INCLUDED events of ``role``.
    """
    bench = prices[dt.EUROPE_BENCHMARK]
    inc = events[(events["included"]) & (events["role"] == role)]
    paths = []
    for _, row in inc.iterrows():
        etf = prices[row["ticker"]]
        common = etf.index.intersection(bench.index).sort_values()
        p = common.get_loc(pd.Timestamp(row["anchor_date"]))
        d_m1 = common[p]
        vals = []
        for k in range(0, max_k + 1):
            d_k = common[p + k]
            r_etf = etf.loc[d_k] / etf.loc[d_m1] - 1.0
            r_bench = bench.loc[d_k] / bench.loc[d_m1] - 1.0
            vals.append(float(r_etf - r_bench))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=range(0, max_k + 1))


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int, k: int = 5) -> dict:
    """Run the one-sample-t detector on a synthetic paired (asset, benchmark) world
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
