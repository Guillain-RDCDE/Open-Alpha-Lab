"""Reproducible headline run for Study 645 — ECB Announcement Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached FEZ / EURUSD tapes under
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

from ecb_announcement_effect import data, strategy as st  # noqa: E402

print("# ECB Announcement Effect — do euro-area equities react systematically around ECB decisions?")

ecb = data.ecb_calendar()
print(f"calendar: {len(ecb)} scheduled ECB Governing Council decision days "
      f"{ecb.min().date()} -> {ecb.max().date()} (hardcoded ECB-calendar table, "
      f"scheduled meetings only)")

if not data.have_real():
    print("(cache miss — fetching FEZ / EURUSD=X once)")
    data.fetch()

fez, eurusd = data.load_real()
print(data_stamp("FEZ OHLC", fez, asof=data.AS_OF))
print(data_stamp("EURUSD=X OHLC", eurusd, asof=data.AS_OF))

df = st.day_frame(fez, eurusd, ecb)
on_tape = int(df["ecb"].sum())
print(f"decision days on the FEZ tape: {on_tape} of {len(ecb)} "
      f"| other trading days: {int((~df['ecb']).sum())}")

print("\n# THE HEADLINE — decision-day FEZ return vs all other days")
s = st.decision_day_stats(df)
print(f"  ECB-day FEZ return  : {s['ecb_pct']:+.3f}%")
print(f"  other-day FEZ return: {s['rest_pct']:+.3f}%")
print(f"  gap                 : {s['gap_pct']:+.3f} pts   Welch t = {s['welch_t']:+.2f}")
print(f"  Newey-West t (dummy regression, 5 lags): {s['nw_t']:+.2f}")
print(f"  hit rate (FEZ up on decision day): {s['hit_up']}/{s['n_ecb']} = "
      f"{s['hit_rate']*100:.1f}%  (Wilson 95% [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")

print("\n# Random-calendar placebo (20 seeds x 1,000 draws of "
      f"{s['n_ecb']} random non-ECB days, two-sided)")
pl = st.placebo_pvalue(df, column="fez_ret")
print(f"  observed ECB-day mean {pl['obs']*100:+.4f}% vs placebo mean {pl['placebo_mean']*100:+.4f}% "
      f"(sd {pl['placebo_sd']*100:.4f}%) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# Realized FEZ high-low range on the same days ((H-L)/prev close)")
rg = st.range_stats(df)
print(f"  ECB days {rg['ecb_range_pct']:.3f}%  vs other days {rg['rest_range_pct']:.3f}%   "
      f"ratio {rg['ratio']:.2f}x   Welch t = {rg['welch_t']:+.2f}   NW(5) t = {rg['nw_t']:+.2f}")
pl_rg = st.placebo_pvalue(df, column="fez_range")
print(f"  range placebo: observed {pl_rg['obs']*100:.4f}% vs placebo mean {pl_rg['placebo_mean']*100:.4f}% "
      f"(sd {pl_rg['placebo_sd']*100:.4f}%) over {pl_rg['n_draws']:,} draws -> p = {pl_rg['p_value']:.5f}")

print("\n# EURUSD |return| on the same days — does the FX leg react more than equities?")
fx = st.eurusd_stats(df)
print(f"  ECB days {fx['ecb_abs_pct']:.3f}%  vs other days {fx['rest_abs_pct']:.3f}%   "
      f"Welch t = {fx['welch_t']:+.2f}")

print("\n# Event window — FEZ return (%) by session offset around the decision "
      "(Lucca-Moench-style; Welch t vs far days)")
ev = st.event_study(df, ecb)
for k, row in ev.iterrows():
    tag = "  <-- decision day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_pct']:+.3f}%  (n={int(row['n'])}, "
          f"Welch t={row['welch_t']:+.2f}){tag}")
ru = st.runup_stats(df, ecb)
print(f"  pre-meeting run-up [-5..-1] cumulative: {ru['mean_runup_pct']:+.3f}%/meeting "
      f"(one-sample t = {ru['t']:+.2f}, n={ru['n_meetings']} meetings)")

print("\n# Era contrast — monthly era vs the 6-week-cycle era (split 2015-01-01, justified: "
      "the Governing Council's own structural change in meeting frequency)")
ec = st.era_contrast(df, data.SIXWEEK_SPLIT)
print(f"  2005->2014 (monthly): ECB-day FEZ return {ec['early_pct']:+.3f}% (n={ec['n_early']}, "
      f"within-era Welch t = {ec['welch_t_early']:+.2f})")
print(f"  2015->2026 (6-week) : ECB-day FEZ return {ec['late_pct']:+.3f}% (n={ec['n_late']}, "
      f"within-era Welch t = {ec['welch_t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['welch_t_diff']:+.2f}")

print("\n# Era contrast, realized range — is the vol bump stable across the cadence change?")
ecr = st.era_contrast(df, data.SIXWEEK_SPLIT, column="fez_range")
print(f"  2005->2014 (monthly): ECB-day range {ecr['early_pct']:.3f}% (n={ecr['n_early']}, "
      f"within-era Welch t = {ecr['welch_t_early']:+.2f})")
print(f"  2015->2026 (6-week) : ECB-day range {ecr['late_pct']:.3f}% (n={ecr['n_late']}, "
      f"within-era Welch t = {ecr['welch_t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ecr['welch_t_diff']:+.2f}")

print("\n# THIRD AXIS — costs on a timer: hold FEZ for the decision day only")
print("  (enter prior close — the calendar is public months ahead, zero look-ahead; exit the")
print("   decision-day close; one round trip = 2 x one-way cost x NAV; long-only, no borrow)")
for cb in (5.0, 10.0, 20.0):
    tc = st.timer_capture(df, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: gross {tc['gross_bps']:+.1f} bps/event -> net "
          f"{tc['net_bps']:+.1f} bps/event  (~{tc['ann_net_pct']:+.2f}%/yr at 8 events)")
tc = st.timer_capture(df, cost_bps=5.0)
print(f"  ECB-day FEZ {tc['gross_bps']:+.1f} bps vs other-day {tc['rest_bps']:+.1f} bps   "
      f"Welch t = {tc['welch_t']:+.2f}  (n={tc['n_ecb']} events)")
print(f"  hit rate {tc['hit_rate']*100:.1f}% | worst single decision day "
      f"{tc['worst_day_pct']:+.1f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (drift=0) and must recover a")
print("  planted decision-day effect. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    close, dec = data.synthetic_world(drift=0.0, seed=645 + s_)
    null_ts.append(st.synthetic_detect(close, dec)["welch_t"])
import numpy as np  # noqa: E402

null_ts = np.asarray(null_ts)
print(f"  null (drift=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
close, dec = data.synthetic_world(drift=0.005, seed=645)
sy = st.synthetic_detect(close, dec)
print(f"  planted drift=+0.5% (seed 645): ECB-day mean {sy['ecb_pct']:+.3f}% vs rest "
      f"{sy['rest_pct']:+.3f}%  Welch t = {sy['welch_t']:+.2f}")
