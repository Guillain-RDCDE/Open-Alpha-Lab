"""Reproducible headline run for Study 707 — Plane-Crash-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / airline-basket
tapes under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from plane_crash_effect import data, strategy as st  # noqa: E402

print("# Plane-Crash-Effect — does a major air disaster dent the market's mood?")
print("# (Kaplanski & Levy 2010: aviation-disaster sentiment; clinical, no trading")
print("#  recommendation is implied by the buy-the-dip test below.)")

events = data.disaster_table()
print(f"\ndisaster table: {len(events)} major commercial-aviation accidents "
      f"{events['date'].min().date()} -> {events['date'].max().date()} (hardcoded; "
      f"9/11 and MH17 excluded — already in 313-geopolitical-shock)")

if not data.have_real():
    print("(cache miss — fetching SPY / AAL / DAL / UAL / LUV once)")
    data.fetch()

spy, airlines = data.load_real()
print(data_stamp("SPY close", spy.to_frame("Close"), asof=data.AS_OF))
for t, s in airlines.items():
    print(data_stamp(f"{t} close", s.to_frame("Close"), asof=data.AS_OF))

ret_spy = st.daily_returns(spy)
ar_spy = st.abnormal_returns(ret_spy)
basket, coverage = st.airline_basket_returns(airlines)
ar_air = st.abnormal_returns(basket)

PRE, POST = 1, 5

print(f"\n# THE HEADLINE — SPY crash-day (day 0) abnormal return, event window [-{PRE}..+{POST}]")
d0 = st.day0_stats(ar_spy, events["date"], pre=PRE, post=POST)
hit_down = int((np.array([st.event_window(ar_spy, d, PRE, POST)[PRE]
                           for d in d0["kept_dates"]]) < 0).sum())


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


wlo, whi = wilson_interval(hit_down, d0["n"])
print(f"  n = {d0['n']} events on the tape")
print(f"  mean crash-day abnormal SPY return: {d0['mean']*1e4:+.2f} bps   "
      f"one-sample t = {d0['t']:+.3f}")
print(f"  hit rate (SPY down on the day)     : {hit_down}/{d0['n']} = "
      f"{hit_down/d0['n']*100:.1f}%  (Wilson 95% [{wlo*100:.1f}%, {whi*100:.1f}%])")

print(f"\n# Random-calendar placebo (20 seeds x 1,000 draws of {d0['n']} random non-crash days)")
draws = np.concatenate([
    st.placebo_distribution(ar_spy, d0["n"], pre=PRE, post=POST, n_draws=1000,
                             seed=707 + s, stat="day0")
    for s in range(20)
])
p_left = st.placebo_pvalue(d0["mean"], draws, tail="left")
print(f"  observed {d0['mean']*1e4:+.2f} bps vs placebo mean {draws.mean()*1e4:+.2f} bps "
      f"(sd {draws.std()*1e4:.2f}) over {len(draws):,} draws -> left-tail p = {p_left:.3f}")

print(f"\n# Event window — mean abnormal SPY return by offset [-{PRE}..+{POST}] (own one-sample t)")
cp = st.car_path_stats(ar_spy, events["date"], pre=PRE, post=POST)
for k, row in cp.iterrows():
    tag = "  <-- crash day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_ar']*1e4:+.2f} bps   CAR {row['car']*1e4:+.2f} bps  "
          f"(t={row['t']:+.2f}){tag}")

print(f"\n# Reversal — cumulative abnormal return [+1..+{POST}] (does the dip bounce back?)")
rev = st.reversal_stats(ar_spy, events["date"], pre=PRE, post=POST)
print(f"  mean post-crash CAR: {rev['mean']*1e4:+.2f} bps   one-sample t = {rev['t']:+.3f} "
      f"(n={rev['n']})")

print("\n# THE AIRLINE TEST — does the sector drop harder than the market on its own bad news?")
print("  (4-carrier equal-weight basket: AAL/DAL/UAL/LUV, whichever are trading that day)")
extra = st.airline_extra_drop(ar_spy, ar_air, events["date"], pre=PRE, post=POST)
print(f"  airline-basket day-0 AR {extra['air_mean']*1e4:+.2f} bps  vs  SPY day-0 AR "
      f"{extra['spy_mean']*1e4:+.2f} bps")
print(f"  paired (airline - SPY) difference: {extra['mean_diff']*1e4:+.2f} bps   "
      f"one-sample t = {extra['t']:+.3f}  (n={extra['n']})")
full_cov = events.loc[events["date"] >= "2008-01-01", "date"]
extra_full = st.airline_extra_drop(ar_spy, ar_air, full_cov, pre=PRE, post=POST)
print(f"  robustness — full 4-carrier-coverage subsample only (events >= 2008, n="
      f"{extra_full['n']}): diff {extra_full['mean_diff']*1e4:+.2f} bps   "
      f"t = {extra_full['t']:+.3f}")
print("  named honestly: the earliest 6 events (2000-2005) fall back to LUV alone "
      "(Southwest — the only current-ticker carrier trading that far back); 3 more "
      "(2006-2007) run on 3 of 4 carriers. Coverage never drops to zero.")

print("\n# THE TIMER — \"buy the dip\" after a crash (clinical statistical test, not advice;")
print("  the obvious ethical caveat: this measures a pattern in public market data, it is")
print("  not a recommendation to trade on tragedy). Enter SPY at the crash-day close,")
print("  hold N sessions, one round trip of one-way costs charged twice.")
for hold in (1, 3, 5, 10):
    ledger_g = st.buy_the_dip(spy, events["date"], hold=hold, cost_bps=0.0)
    g = st.summarize_dip(ledger_g, col="ret_gross")
    ledger_n = st.buy_the_dip(spy, events["date"], hold=hold, cost_bps=5.0)
    n_ = st.summarize_dip(ledger_n, col="ret_net")
    fwd = (spy.shift(-hold) / spy - 1.0)
    baseline_bps = float(fwd.mean() * 1e4)
    print(f"  hold {hold:>2d}d: gross {g['mean_bps']:+7.2f} bps  net(5bps) {n_['mean_bps']:+7.2f} bps  "
          f"t(net) = {n_['t']:+.2f}   unconditional SPY {hold}-day baseline: {baseline_bps:+.2f} bps")

print("\n# Synthetic positive control — deterministic, no network")
print("  the day0 detector must NOT fire on a null world (dip=0) and must recover a")
print("  planted crash-day dip. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    close, ev = data.synthetic_world(dip=0.0, seed=707 + s_)
    null_ts.append(st.synthetic_detect(close, ev)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (dip=0), 20 seeds: mean t = {null_ts.mean():+.2f}  (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
close, ev = data.synthetic_world(dip=-0.02, seed=707)
sy = st.synthetic_detect(close, ev)
print(f"  planted dip=-2.0% (seed 707): mean {sy['mean']*1e4:+.1f} bps   t = {sy['t']:+.2f}")
