"""Reproducible headline run for Study 652 — Index-Deletion-Bounce.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached per-ticker/SPY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from index_deletion_bounce import data, strategy as st  # noqa: E402

print("# Index-Deletion-Bounce — are stocks DELETED from the S&P 500 dumped, then rebound?")

events = data.deletions_frame()
print(f"deletion calendar: {len(events)} hardcoded S&P 500 'market capitalization change' "
      f"deletions {events['effective'].min().date()} -> {events['effective'].max().date()} "
      "(S&P Dow Jones Indices announcements, via the Wikipedia change log)")

if not data.have_real():
    print("(cache miss — fetching SPY + per-ticker tapes once; ~70 tickers)")
    data.fetch()

tapes, spy, missing = data.load_real()
print(data_stamp("SPY OHLC", spy, asof=data.AS_OF))
print(f"usable per-ticker tapes: {len(tapes)}/{len(events)}  |  NO tape at all (delisted off "
      f"Yahoo by a LATER, unrelated corporate death): {len(missing)}/{len(events)} "
      f"-> {sorted(missing)}")

panel, dropped = st.build_event_panel(tapes, spy, events)
print(f"event panel: {len(panel)} events with a full [-5..+40] window "
      f"({len(dropped)} dropped: no tape or incomplete window)")

print("\n# THE HEADLINE — market-adjusted CAR (vs SPY) around the effective date")
cb = st.car_by_offset(panel)
for k in (-5, -3, -1, 0, 1, 5, 10, 20, 30, 40):
    row = cb.loc[k]
    tag = "  <-- effective day" if k == 0 else ""
    print(f"  offset {k:+3d}: mean CAR {row['mean_car']*100:+.2f}%  "
          f"(t = {row['t']:+.2f}, n={int(row['n'])}){tag}")

dump = st.window_car(panel, -5, 0)
reb = st.window_car(panel, 1, 40)
dump_ci = st.bootstrap_ci(dump)
reb_ci = st.bootstrap_ci(reb)
print(f"\n  THE DUMP    [-5..0]  : mean {dump.mean()*100:+.2f}%  t = {st.one_sample_t(dump):+.2f}"
      f"  95% bootstrap CI [{dump_ci[0]*100:+.2f}%, {dump_ci[1]*100:+.2f}%]  (n={len(dump)})")
print(f"  THE REBOUND [1..40]  : mean {reb.mean()*100:+.2f}%  t = {st.one_sample_t(reb):+.2f}"
      f"  95% bootstrap CI [{reb_ci[0]*100:+.2f}%, {reb_ci[1]*100:+.2f}%]  (n={len(reb)})")

print("\n# Announce -> effective CAR (the informed-selling leg, mirrors 249-index-inclusion's ADD side)")
ae = st.announce_to_effective_car(tapes, spy, events)
print(f"  mean {ae['mean_car']*100:+.2f}%  t = {ae['t']:+.2f}  (n={ae['n']})")

print("\n# Random-day placebo — same tickers, random non-event anchor day, 300 draws")
pl = st.placebo_car(tapes, spy, events, n_draws=300)
print(f"  observed post-event [1..40] mean {pl['obs_mean']*100:+.2f}%  vs placebo mean "
      f"{pl['placebo_mean']*100:+.2f}% (sd {pl['placebo_sd']*100:.2f}%) over {pl['n_draws']} "
      f"draws -> p = {pl['p_value']:.3f}")
print("  (a random 40-day window for these SAME distressed names is far worse on average than "
      "the post-effective window — a real but SECONDARY finding: relative stabilization, not "
      "an absolute positive rebound; see docs/results.md for why this does not earn a REAL stamp)")

print("\n# Era contrast — first half vs second half of the sample (split "
      f"{data.ERA_SPLIT}, justified: bisects the 2012-2025 span)")
ec = st.era_contrast(panel, events, data.ERA_SPLIT)
print(f"  2012 -> {data.ERA_SPLIT}: post-event CAR {ec['early_car']*100:+.2f}% "
      f"(n={ec['n_early']}, t = {ec['t_early']:+.2f})")
print(f"  {data.ERA_SPLIT} -> 2025: post-event CAR {ec['late_car']*100:+.2f}% "
      f"(n={ec['n_late']}, t = {ec['t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['t_diff']:+.2f}")

print("\n# THIRD AXIS — can retail capture it with a long-the-deleted timer?")
print("  (enter at the effective-date close — public days ahead, zero look-ahead; hold 40")
print("   trading days; one round trip = 2 x one-way cost x NAV; already market-adjusted vs SPY)")
for cb_ in (5.0, 10.0):
    lt = st.long_timer(panel, hold_days=40, cost_bps=cb_)
    print(f"  cost={cb_:>4.1f} bps: gross {lt['gross']*100:+.2f}% -> net {lt['net']*100:+.2f}%  "
          f"(t_net = {lt['t_net']:+.2f}, hit rate {lt['hit_rate']*100:.1f}%)")
lt5 = st.long_timer(panel, hold_days=40, cost_bps=5.0)
print(f"  worst single event {lt5['worst']*100:+.1f}%  |  best single event "
      f"{lt5['best']*100:+.1f}%  (n={lt5['n']})")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT fire on a null world (dump=rebound=0) and must recover a")
print("  planted dump-then-rebound. Null checked over 20 seeds (never a single stream).")
null_dump_t, null_reb_t = [], []
for s_ in range(20):
    evs = data.synthetic_panel(n_events=len(panel), seed=652 + s_, dump=0.0, rebound=0.0)
    r = st.synthetic_detect(evs)
    null_dump_t.append(r["t_dump"])
    null_reb_t.append(r["t_rebound"])
null_dump_t = np.asarray(null_dump_t)
null_reb_t = np.asarray(null_reb_t)
print(f"  null dump   t: mean {null_dump_t.mean():+.2f} (sd {null_dump_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_dump_t) >= 2).sum()}/20 seeds")
print(f"  null rebound t: mean {null_reb_t.mean():+.2f} (sd {null_reb_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_reb_t) >= 2).sum()}/20 seeds")
evs = data.synthetic_panel(n_events=len(panel), seed=652, dump=0.08, rebound=0.08)
sy = st.synthetic_detect(evs)
print(f"  planted dump=-8%, rebound=+8% (seed 652): dump t = {sy['t_dump']:+.2f}  "
      f"rebound t = {sy['t_rebound']:+.2f}")
