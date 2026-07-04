"""Reproducible headline run for Study 637 — FOMC Vol Crush.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ^VIX / SPY / SVXY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from fomc_vol_crush import data, strategy as st  # noqa: E402

print("# FOMC Vol Crush — does the VIX collapse the afternoon of a scheduled Fed decision?")

fomc = data.fomc_calendar()
print(f"calendar: {len(fomc)} scheduled FOMC decision days "
      f"{fomc.min().date()} -> {fomc.max().date()} (hardcoded Fed-calendar table, "
      f"scheduled meetings only)")

if not data.have_real():
    print("(cache miss — fetching ^VIX / SPY / SVXY once)")
    data.fetch()

vix, spy, svxy = data.load_real()
print(data_stamp("^VIX OHLC", vix, asof=data.AS_OF))
print(data_stamp("SPY OHLC", spy, asof=data.AS_OF))
print(data_stamp("SVXY adj close", svxy, asof=data.AS_OF))

df = st.day_frame(vix, spy, fomc)
on_tape = int(df["fomc"].sum())
print(f"decision days on the ^VIX tape: {on_tape} of {len(fomc)} "
      f"| other trading days: {int((~df['fomc']).sum())}")

print("\n# THE HEADLINE — decision-day ΔVIX vs all other days")
s = st.decision_day_stats(df)
print(f"  FOMC-day ΔVIX  : {s['fomc_pts']:+.3f} pts  ({s['fomc_logpct']:+.2f}% log)")
print(f"  other-day ΔVIX : {s['rest_pts']:+.3f} pts  ({s['rest_logpct']:+.2f}% log)")
print(f"  gap            : {s['gap_pts']:+.3f} pts   Welch t = {s['welch_t_pts']:+.2f} "
      f"(points)  |  Welch t = {s['welch_t_log']:+.2f} (log)")
print(f"  Newey-West t (dummy regression, 5 lags): {s['nw_t_pts']:+.2f}")
print(f"  hit rate       : VIX fell on {s['hit_down']}/{s['n_fomc']} decision days = "
      f"{s['hit_rate']*100:.1f}%  (Wilson 95% [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")

print("\n# Random-calendar placebo (20 seeds x 1,000 draws of "
      f"{s['n_fomc']} random non-FOMC days)")
pl = st.placebo_pvalue(df)
print(f"  observed FOMC-day mean {pl['obs']:+.3f} pts vs placebo mean {pl['placebo_mean']:+.3f} "
      f"(sd {pl['placebo_sd']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# Event window — ΔVIX (points) by session offset around the decision (Welch t vs far days)")
ev = st.event_study(df, fomc)
for k, row in ev.iterrows():
    tag = "  <-- decision day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_pts']:+.3f} pts  (n={int(row['n'])}, "
          f"Welch t={row['welch_t']:+.2f}){tag}")
ru = st.runup_stats(df, fomc)
print(f"  pre-meeting run-up [-5..-1] cumulative: {ru['mean_runup_pts']:+.3f} pts/meeting "
      f"(one-sample t = {ru['t']:+.2f}, n={ru['n_meetings']} meetings)")

print("\n# Realized SPY high-low range on the same days ((H-L)/prev close)")
rg = st.spy_range_stats(df)
print(f"  FOMC days {rg['fomc_range_pct']:.3f}%  vs other days {rg['rest_range_pct']:.3f}%   "
      f"Welch t = {rg['welch_t']:+.2f}")

print("\n# Era contrast — pre vs post press-conference era (split 2011-04-27, justified: "
      "first post-meeting press conference)")
ec = st.era_contrast(df, data.PRESSER_SPLIT)
print(f"  1994->2011-04: FOMC-day ΔVIX {ec['early_pts']:+.3f} pts (n={ec['n_early']}, "
      f"within-era Welch t = {ec['welch_t_early']:+.2f})")
print(f"  2011-04->2026: FOMC-day ΔVIX {ec['late_pts']:+.3f} pts (n={ec['n_late']}, "
      f"within-era Welch t = {ec['welch_t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['welch_t_diff']:+.2f}")

print("\n# THIRD AXIS — can retail capture it with SVXY held for the decision day only?")
print("  (enter prior close — the calendar is public years ahead, zero look-ahead; exit the")
print("   decision-day close; one round trip = 2 x one-way cost x NAV; long-only, no borrow)")
for cb in (5.0, 10.0):
    sv = st.svxy_capture(svxy, fomc, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: gross {sv['gross_bps']:+.1f} bps/event -> net "
          f"{sv['net_bps']:+.1f} bps/event  (~{sv['ann_net_pct']:+.2f}%/yr at 8 events)")
sv = st.svxy_capture(svxy, fomc, cost_bps=5.0)
print(f"  FOMC-day SVXY {sv['gross_bps']:+.1f} bps vs other-day {sv['rest_bps']:+.1f} bps   "
      f"Welch t = {sv['welch_t']:+.2f}  (n={sv['n_fomc']} events since 2011-10)")
print(f"  hit rate {sv['hit_rate']*100:.1f}% | worst single decision day "
      f"{sv['worst_day_pct']:+.1f}%")
sv2 = st.svxy_capture(svxy, fomc, cost_bps=5.0, start=data.SVXY_DELEV)
print(f"  post-deleverage (-0.5x, since {data.SVXY_DELEV}): gross {sv2['gross_bps']:+.1f} "
      f"bps/event  Welch t = {sv2['welch_t']:+.2f}  (n={sv2['n_fomc']})")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (crush=0) and must recover a")
print("  planted decision-day crush. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    close, dec = data.synthetic_world(crush=0.0, seed=637 + s_)
    null_ts.append(st.synthetic_detect(close, dec)["welch_t_pts"])
import numpy as np  # noqa: E402

null_ts = np.asarray(null_ts)
print(f"  null (crush=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
close, dec = data.synthetic_world(crush=1.0, seed=637)
sy = st.synthetic_detect(close, dec)
print(f"  planted crush=+1.0 pts (seed 637): FOMC-day mean {sy['fomc_pts']:+.3f} vs rest "
      f"{sy['rest_pts']:+.3f}  Welch t = {sy['welch_t_pts']:+.2f}")
