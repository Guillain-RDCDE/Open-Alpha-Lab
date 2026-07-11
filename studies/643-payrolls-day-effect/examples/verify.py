"""Reproducible headline run for Study 643 — Payrolls-Day-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from payrolls_day_effect import data, strategy as st  # noqa: E402

print("# Payrolls-Day-Effect — does SPY behave systematically on NFP release mornings?")

nfp = data.nfp_calendar()
print(f"calendar: {len(nfp)} actual NFP (Employment Situation) release days "
      f"{nfp.min().date()} -> {nfp.max().date()} (hardcoded BLS release-date table, "
      f"same source-verified table as sibling study 602)")

if not data.have_real():
    print("(cache miss — fetching SPY once)")
    data.fetch()

spy = data.load_real()
print(data_stamp("SPY OHLC", spy, asof=data.AS_OF))

sessions, n_mapped = data.map_to_sessions(spy.index, nfp)
print(f"release days mapped onto trading sessions: {len(sessions)} of {len(nfp)} "
      f"({n_mapped} forward-mapped off a non-trading day)")

df = st.day_frame(spy, sessions)
on_tape = int(df["nfp"].sum())
print(f"release days on the SPY tape: {on_tape} of {len(sessions)} "
      f"| other trading days: {int((~df['nfp']).sum())}")

print("\n# THE HEADLINE — NFP-day SPY return vs all other days")
s = st.nfp_day_stats(df)
print(f"  NFP-day return  : {s['nfp_bps']:+.2f} bps")
print(f"  other-day return: {s['rest_bps']:+.2f} bps")
print(f"  gap             : {s['gap_bps']:+.2f} bps   Welch t = {s['welch_t']:+.2f}")
print(f"  Newey-West t (dummy regression, 5 lags): {s['nw_t']:+.2f}")
print(f"  hit rate        : SPY rose on {s['hit_up']}/{s['n_nfp']} release days = "
      f"{s['hit_rate']*100:.1f}%  (Wilson 95% [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")

print("\n# Random-calendar placebo (20 seeds x 1,000 draws of "
      f"{s['n_nfp']} random non-NFP days, two-sided)")
pl = st.placebo_pvalue(df)
print(f"  observed NFP-day mean {pl['obs']*1e4:+.2f} bps vs placebo mean "
      f"{pl['placebo_mean']*1e4:+.2f} (sd {pl['placebo_sd']*1e4:.2f} bps) over "
      f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# Event window — mean return (bps) by session offset around the release "
      "(Welch t vs far days)")
ev = st.event_study(df, sessions)
for k, row in ev.iterrows():
    tag = "  <-- release day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_bps']:+.2f} bps  (n={int(row['n'])}, "
          f"Welch t={row['welch_t']:+.2f}){tag}")
ru = st.runup_stats(df, sessions)
print(f"  pre-release run-up [-3..-1] cumulative: {ru['mean_runup_bps']:+.2f} bps/event "
      f"(one-sample t = {ru['t']:+.2f}, n={ru['n_events']} events)")

print("\n# Realized SPY high-low range on the same days ((H-L)/prev close)")
rg = st.spy_range_stats(df)
print(f"  NFP days {rg['nfp_range_pct']:.3f}%  vs other days {rg['rest_range_pct']:.3f}%   "
      f"Welch t = {rg['welch_t']:+.2f}")

print("\n# Era contrast — pre vs post 2013-01-01 (justified split: taper-talk / "
      "post-crisis-normalization era, roughly the sample midpoint)")
ec = st.era_contrast(df, "2013-01-01")
print(f"  1997->2013: NFP-day return {ec['early_bps']:+.2f} bps (n={ec['n_early']}, "
      f"within-era Welch t = {ec['welch_t_early']:+.2f})")
print(f"  2013->2026: NFP-day return {ec['late_bps']:+.2f} bps (n={ec['n_late']}, "
      f"within-era Welch t = {ec['welch_t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['welch_t_diff']:+.2f}")

print("\n# THIRD AXIS — the naive timer: own SPY only on the NFP release day")
print("  (enter prior close — the calendar is public months ahead, zero look-ahead;")
print("   exit the release-day close; one round trip = 2 x one-way cost x NAV)")
for cb in (5.0, 10.0):
    tc = st.timer_capture(df, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: gross {tc['gross_bps']:+.2f} bps/event -> net "
          f"{tc['net_bps']:+.2f} bps/event  (~{tc['ann_net_pct']:+.2f}%/yr at 12 events)")
tc = st.timer_capture(df, cost_bps=5.0)
print(f"  hit rate {tc['hit_rate']*100:.1f}% | worst single release day "
      f"{tc['worst_day_pct']:+.1f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (edge=0) and must recover a")
print("  planted release-day effect. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    close, rel = data.synthetic_world(edge=0.0, seed=643 + s_)
    null_ts.append(st.synthetic_detect(close, rel)["welch_t"])
import numpy as np  # noqa: E402

null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
close, rel = data.synthetic_world(edge=0.0015, seed=643)
sy = st.synthetic_detect(close, rel)
print(f"  planted edge=+15 bps/day (seed 643): NFP-day mean {sy['nfp_bps']:+.2f} vs rest "
      f"{sy['rest_bps']:+.2f} bps  Welch t = {sy['welch_t']:+.2f}")
