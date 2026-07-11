"""Reproducible headline run for Study 653 — Dividend-Cut-Drift.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic given the cached tape under ``_cache/`` (fetching
once on a cache miss); the synthetic control always runs with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp, fingerprint  # noqa: E402

from dividend_cut_drift import data, strategy as st  # noqa: E402

print("# Dividend-Cut-Drift — does a stock that cuts or omits its dividend keep underperforming?")

if not data.have_real():
    print("(cache miss — fetching ~100 tickers + SPY once)")
    ok, skipped = data.fetch()
    print(f"  fetched {len(ok)}, skipped {skipped}")

px_map, div_map, spy = data.load_real()
print(f"universe: {len(data.UNIVERSE)} tickers named (a mix of stable payers and known "
      f"cutters/suspenders) -> {len(px_map)} loaded with a usable cache "
      f"({len(data.UNIVERSE) - len(px_map)} skipped: renamed/delisted symbols)")
print(data_stamp("SPY adjusted close", spy.to_frame("close"), asof=data.AS_OF))

print("\n# EVENT DETECTION — dividend cuts (<=70% of prior payment) and omissions (>=1.8x the "
      "ticker's own trailing typical interval), split-adjusted, specials/stubs stripped")
events = st.build_event_table(div_map)
n_cut_raw = int((events["type"] == "cut").sum())
n_om_raw = int((events["type"] == "omission").sum())
print(f"  detected: {len(events)} events ({n_cut_raw} cuts, {n_om_raw} omissions) across "
      f"{events['ticker'].nunique()} of {len(px_map)} tickers, "
      f"{events['event_date'].min().date()} -> {events['event_date'].max().date()}")

events_sorted = events.sort_values(["event_date", "ticker"]).set_index("event_date")
fp_events = fingerprint(events_sorted, cols=["prior_amt", "new_amt", "ratio"])
print(f"  event-table fingerprint={fp_events}")

car_mat, kept = st.build_cars(px_map, spy, events)
n_dropped = len(events) - len(kept)
kept_dates = pd.DatetimeIndex([k[1] for k in kept])
print(f"  {len(kept)}/{len(events)} events have a full [-20,+120]-trading-day window on the "
      f"tape by as-of ({n_dropped} dropped — too close to the tape's start or end); analyzed "
      f"window {kept_dates.min().date()} -> {kept_dates.max().date()}, "
      f"{len(set(k[0] for k in kept))} distinct tickers")

print("\n# THE HEADLINE — cross-sectional CAR (ticker log return - SPY log return), "
      "one execution lag: signal known at the event day's close, post-drift measured from "
      "day+1")
hs = st.horizon_stats(car_mat)
print(f"  pre-event drift  CAR[-20..-1] : {hs['pre_mean']*100:+.2f}%   "
      f"one-sample t = {hs['pre_t']:+.2f}   (n={hs['n']})")
print(f"  post-event drift CAR[+1..+20] : {hs['post20_mean']*100:+.2f}%   t = {hs['post20_t']:+.2f}")
print(f"  post-event drift CAR[+1..+60] : {hs['post60_mean']*100:+.2f}%   t = {hs['post60_t']:+.2f}")
print(f"  post-event drift CAR[+1..+120]: {hs['post120_mean']*100:+.2f}%   t = {hs['post120_t']:+.2f}")
print(f"  hit rate (CAR[+1..+120] < 0, i.e. 'kept falling'): {hs['hit_neg']}/{hs['n']} = "
      f"{hs['hit_rate']*100:.1f}%  (Wilson 95% [{hs['hit_lo']*100:.1f}%, {hs['hit_hi']*100:.1f}%])")

print("\n# Cut vs omission — same test, split by event type")
types = pd.Series([k[2] for k in kept])
post120 = (car_mat[120] - car_mat[0])
for typ in ("cut", "omission"):
    mask = (types == typ).to_numpy()
    x = post120[mask].to_numpy()
    print(f"  {typ:8s}: n={mask.sum():3d}  mean CAR[+1..+120] = {x.mean()*100:+.2f}%   "
          f"t = {st.one_sample_t(x):+.2f}")

print("\n# Calendar-time cross-check — equal-weight 'cutters' portfolio, Newey-West(5) t "
      "(overlap-robust: events share calendar time, so the cross-sectional t above treats "
      "correlated observations as independent)")
nw = st.calendar_time_nw_t(px_map, spy, kept)
print(f"  {nw['n_days']:,} portfolio-days, mean daily AR = {nw['mean_daily_bps']:+.2f} bps   "
      f"Newey-West t = {nw['nw_t']:+.2f}")

print("\n# Daily-level Welch cross-check — pooled ticker-day AR, treatment (post-event window) "
      "vs control (the same tickers' other days)")
dw = st.daily_welch(px_map, spy, kept)
print(f"  treatment mean {dw['mean_treat_bps']:+.2f} bps/day (n={dw['n_treat']:,})  vs  "
      f"control mean {dw['mean_control_bps']:+.2f} bps/day (n={dw['n_control']:,})   "
      f"Welch t = {dw['welch_t']:+.2f}")

print("\n# Random-date placebo (20 seeds x 200 draws of "
      f"{len(kept)} random ticker/date pairs)")
tickers_avail = list(px_map.keys())
pl = st.random_date_placebo(px_map, spy, tickers_avail, n_events=len(kept))
p_val = st.placebo_pvalue(hs["post120_mean"], pl, draws=pl["draws"])
print(f"  observed mean CAR[+1..+120] {hs['post120_mean']*100:+.2f}% vs placebo mean "
      f"{pl['placebo_mean']*100:+.2f}% (sd {pl['placebo_sd']*100:.2f}%) over "
      f"{pl['n_draws']:,} draws -> p = {p_val:.3f}")

print("\n# THIRD AXIS — two tradable expressions, one execution lag (enter the close the "
      "session AFTER the event day; exit 120 trading days later)")
for side in ("short", "long"):
    print(f"  -- {side}-the-cutter --")
    for cb in (5.0, 10.0):
        bt = st.backtest(px_map, spy, kept, side=side, cost_bps=cb)
        print(f"    cost={cb:>4.1f}bps: net/event {bt['mean_net']*100:+.2f}%  "
              f"(t={bt['t_net']:+.2f})   excess-of-matched-exposure {bt['mean_excess']*100:+.2f}%  "
              f"(t={bt['t_excess']:+.2f})")
    bt5 = st.backtest(px_map, spy, kept, side=side, cost_bps=5.0)
    print(f"    hit rate {bt5['hit_rate']*100:.1f}% (Wilson [{bt5['hit_lo']*100:.1f}%, "
          f"{bt5['hit_hi']*100:.1f}%])  worst {bt5['worst']*100:+.1f}%  best {bt5['best']*100:+.1f}%  "
          f"Sharpe(net) {bt5['sharpe_net']:+.2f}  Sharpe(excess) {bt5['sharpe_excess']:+.2f}  "
          f"~{bt5['events_per_year']:.1f} events/yr, n={bt5['n']}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the CAR detector must NOT fire on a null world (drift=0) and must recover a planted "
      "post-event drift. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    pxs, bench, evs = data.synthetic_world(drift=0.0, seed=653 + s_)
    null_ts.append(st.synthetic_detect(pxs, bench, evs)["post120_t"])
null_ts = np.asarray(null_ts)
print(f"  null (drift=0), 20 seeds: mean t = {null_ts.mean():+.2f}  (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
pxs, bench, evs = data.synthetic_world(drift=-0.001, seed=653)
sy = st.synthetic_detect(pxs, bench, evs)
print(f"  planted drift=-0.001/day (seed 653, n={sy['n']} events): mean post-120 CAR "
      f"{sy['post120_mean']*100:+.2f}%   t = {sy['post120_t']:+.2f}")
