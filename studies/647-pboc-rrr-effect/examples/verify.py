"""Reproducible headline run for Study 647 — PBoC RRR Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached FXI / MCHI tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from pboc_rrr_effect import data, strategy as st  # noqa: E402

print("# PBoC RRR Effect — do Chinese equities pop when the PBoC cuts the Reserve "
      "Requirement Ratio?")

rrr = data.rrr_frame()
n_cut = int((rrr["direction"] == "cut").sum())
n_hike = int((rrr["direction"] == "hike").sum())
print(f"calendar: {len(rrr)} broad-based PBoC RRR announcements "
      f"{rrr['date'].min().date()} -> {rrr['date'].max().date()} "
      f"({n_cut} cuts, {n_hike} hikes; hardcoded PBoC-archive table, broad-based system-wide "
      f"moves only)")

if not data.have_real():
    print("(cache miss — fetching FXI / MCHI once)")
    data.fetch()

fxi, mchi = data.load_real()
print(data_stamp("FXI raw OHLC", fxi, asof=data.AS_OF))
print(data_stamp("MCHI raw OHLC", mchi, asof=data.AS_OF))

matched_fxi = st.match_trading_days(fxi.index, rrr)
matched_mchi = st.match_trading_days(mchi.index, rrr)
df = st.day_frame(fxi, matched_fxi)
mdf = st.day_frame(mchi, matched_mchi)
print(f"events on the FXI tape: {int(df['event'].sum())} of {len(rrr)} "
      f"| other trading days: {int((~df['event']).sum())}")
print(f"events on the MCHI tape (inception {data.MCHI_INCEPTION}): "
      f"{int(mdf['event'].sum())} | other trading days: {int((~mdf['event']).sum())}")

print("\n# THE HEADLINE — FXI return on RRR-cut / RRR-hike days vs all other days")
s_cut = st.decision_day_stats(df, "ret", "cut")
s_hike = st.decision_day_stats(df, "ret", "hike")
for label, s in (("cut", s_cut), ("hike", s_hike)):
    print(f"  {label:>4} n={s['n_event']:>3}  event-day {s['event_pct']:+.4f}%  vs other-day "
          f"{s['rest_pct']:+.4f}%  gap {s['gap_pct']:+.4f}%   Welch t = {s['welch_t']:+.2f}  "
          f"NW(5) t = {s['nw_t']:+.2f}")
    print(f"        hit rate (return > 0): {s['hit_up']}/{s['n_event']} = "
          f"{s['hit_rate']*100:.1f}%  (Wilson 95% [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")

cvh = st.cuts_vs_hikes(df, "ret")
print(f"  cuts vs hikes, direct: cut {cvh['cut_pct']:+.4f}%  vs hike {cvh['hike_pct']:+.4f}%   "
      f"Welch t = {cvh['welch_t']:+.2f}")

print("\n# One-sided random-calendar placebo (20 seeds x 1,000 draws, excludes ALL RRR days "
      "from the null pool)")
pl_cut = st.placebo_pvalue(df, "ret", "cut", tail="right")
pl_hike = st.placebo_pvalue(df, "ret", "hike", tail="left")
print(f"  cut  (right tail): observed {pl_cut['obs']*100:+.4f}% vs placebo mean "
      f"{pl_cut['placebo_mean']*100:+.4f}% (sd {pl_cut['placebo_sd']*100:.4f}%) -> "
      f"p = {pl_cut['p_value']:.4f}")
print(f"  hike (left  tail): observed {pl_hike['obs']*100:+.4f}% vs placebo mean "
      f"{pl_hike['placebo_mean']*100:+.4f}% (sd {pl_hike['placebo_sd']*100:.4f}%) -> "
      f"p = {pl_hike['p_value']:.4f}")

print("\n# Event window [-5..+10] — mean FXI return (%) by session offset (Welch t vs far days)")
cut_days = matched_fxi.loc[matched_fxi["direction"] == "cut", "trading_date"]
cut_days = pd.DatetimeIndex(cut_days)
ev = st.event_study(df, "ret", cut_days)
for k, row in ev.iterrows():
    tag = "  <-- cut day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_pct']:+.4f}%  (n={int(row['n'])}, "
          f"Welch t={row['welch_t']:+.2f}){tag}")
ru = st.runup_stats(df, "ret", cut_days)
pe = st.postevent_stats(df, "ret", cut_days)
print(f"  pre-cut run-up [-5..-1] cumulative:  {ru['mean_runup_pct']:+.3f}%/event "
      f"(one-sample t = {ru['t']:+.2f}, n={ru['n_events']})")
print(f"  post-cut window [+1..+10] cumulative: {pe['mean_postrun_pct']:+.3f}%/event "
      f"(one-sample t = {pe['t']:+.2f}, n={pe['n_events']})")

print("\n# Realized range on ANY RRR event day vs other days ((H-L)/prev close, %)")
rg = st.range_stats(df, "range")
print(f"  RRR days {rg['event_pct']:.4f}%  vs other days {rg['rest_pct']:.4f}%   "
      f"Welch t = {rg['welch_t']:+.2f}")

print("\n# MCHI cross-check (modern era, inception 2011-03-29) — same headline split")
sm_cut = st.decision_day_stats(mdf, "ret", "cut")
sm_hike = st.decision_day_stats(mdf, "ret", "hike")
print(f"  cut  n={sm_cut['n_event']}: MCHI cut-day  {sm_cut['event_pct']:+.4f}% vs other "
      f"{sm_cut['rest_pct']:+.4f}%   Welch t = {sm_cut['welch_t']:+.2f}")
print(f"  hike n={sm_hike['n_event']}: MCHI hike-day {sm_hike['event_pct']:+.4f}% vs other "
      f"{sm_hike['rest_pct']:+.4f}%   Welch t = {sm_hike['welch_t']:+.2f}")
cvh_m = st.cuts_vs_hikes(mdf, "ret")
print(f"  cuts vs hikes, direct (MCHI): cut {cvh_m['cut_pct']:+.4f}% vs hike "
      f"{cvh_m['hike_pct']:+.4f}%   Welch t = {cvh_m['welch_t']:+.2f}")

print("\n# Era contrast (cut days only) — panic-easing 2008-2012 vs secular-grind 2015-2025 "
      f"(split {data.ZERO_RATE_ERA_SPLIT}, justified: the regime pivot itself)")
ec = st.era_contrast(df, "ret", data.ZERO_RATE_ERA_SPLIT)
print(f"  2008-2012: FXI cut-day {ec['early_pct']:+.4f}% (n={ec['n_early']}, within-era Welch t "
      f"= {ec['welch_t_early']:+.2f})  |  2015-2025: {ec['late_pct']:+.4f}% (n={ec['n_late']}, "
      f"within-era Welch t = {ec['welch_t_late']:+.2f})  |  diff t = {ec['welch_t_diff']:+.2f}")

print("\n# THIRD AXIS — 'buy the rumor, sell the news'? The buy-the-cut timer with costs")
print("  (enter FXI at the prior close, hold 1/3/5/10 trading days, 2 x one-way cost x NAV)")
for h in (1, 3, 5, 10):
    cap = st.capture_horizon(fxi["AdjClose"], matched_fxi, "cut", horizon=h, cost_bps=5.0)
    print(f"  hold {h:>2}d: gross {cap['gross_bps']:+.1f} bps -> net {cap['net_bps']:+.1f} bps  "
          f"(rest-of-tape mean {cap['rest_mean_bps']:+.1f} bps, Welch t = {cap['welch_t']:+.2f}, "
          f"hit {cap['hit_rate']*100:.1f}%, worst {cap['worst_pct']:+.1f}%, n={cap['n_events']})")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT (systematically) fire on a null world (effect=0) and must")
print("  recover a planted cut-day mean shift. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    close, dec = data.synthetic_world(effect=0.0, seed=647 + s_)
    null_ts.append(st.synthetic_detect(close, dec)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (effect=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
close, dec = data.synthetic_world(effect=0.003, seed=647)
sy = st.synthetic_detect(close, dec)
print(f"  planted effect=+0.30% log-return (seed 647): cut-day mean {sy['event_pct']:+.4f}% "
      f"vs rest {sy['rest_pct']:+.4f}%  Welch t = {sy['welch_t']:+.2f}")
