"""Reproducible headline run for Study 735 — Ryder-Cup-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / VGK tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

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

from ryder_cup_effect import data as dt, strategy as st  # noqa: E402

print("# Ryder-Cup-Effect — does the losing continent's market lag the winner's the Monday after?")

n_played = len(dt.EVENTS)
n_contested = sum(1 for *_, loser in dt.EVENTS if loser is not None)
print(f"calendar: {n_played} editions {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      f"{n_contested} with a losing side (1989 was a 14-14 tie), hardcoded from Wikipedia")

if not dt.have_real():
    print("(cache miss — fetching SPY + VGK once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("Ryder Cup panel (SPY + VGK)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=2.0)
ev5 = st.build_event_table(prices, cost_bps=5.0)
inc, inc5 = ev[ev["included"]], ev5[ev5["included"]]
usa_loss = inc[inc["loser"] == "USA"]
eur_loss = inc[inc["loser"] == "Europe"]

print(f"\nevents resolved: {len(inc)} of {n_contested} contested editions have SPY+VGK "
      f"coverage around the result ({len(usa_loss)} USA-loss, {len(eur_loss)} Europe-loss)")
print("excluded, by reason:")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  {n:2d}x {reason}")

print("\n# THE SIGNAL — paired loser-minus-winner spread, day(-1) [pre-result close] -> day(-1)+k")
print("#   folklore predicts a NEGATIVE spread (loser lags); hit = share of events with spread < 0")
for label, col in (("Monday (k=1)", "spread_mon"), ("1 week (k=5)", "spread_wk")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values, direction="neg")
    print(f"  {label:<14s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"loser-lagged {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event; left-tail: p = share of "
      "null means <= observed)")
for label, col, k in (("Monday spread", "spread_mon", 1), ("1-week spread", "spread_wk", 5)):
    pl = st.placebo_pvalue(ev, prices, col, k=k, entry_offset=0, tail="left")
    print(f"  {label:<14s}: observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) "
          f"over {pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# THIRD AXIS — is the loser SLUMPING (Edmans), or the winner just drifting up? "
      "(constant-mean absolute abnormal, Monday)")
lo_s = st.one_sample_t(inc["loser_ab_mon"].values)
wi_s = st.one_sample_t(inc["winner_ab_mon"].values)
print(f"  loser  leg abnormal: mean {lo_s['mean']*100:+.3f}%  t={lo_s['t']:+.3f}  (folklore: < 0)")
print(f"  winner leg abnormal: mean {wi_s['mean']*100:+.3f}%  t={wi_s['t']:+.3f}")
t_asym = st.welch_t(usa_loss["spread_mon"].values, eur_loss["spread_mon"].values)
print(f"  USA-loss vs Europe-loss Monday spread, Welch t = {t_asym:+.3f} "
      f"(USA-loss mean {usa_loss['spread_mon'].mean()*100:+.3f}%, "
      f"Europe-loss mean {eur_loss['spread_mon'].mean()*100:+.3f}%)")

print("\n# Event anatomy — mean cumulative loser-minus-winner spread by trading day (day(-1) anchor)")
cp = st.car_path(ev, prices, max_k=5)
for k in range(0, 6):
    print(f"  day {k}: {cp[k]*100:+.3f}%")

print("\n# TRADABILITY — long winner / short loser, entered day(0) [first public close], net of costs")
for base, label in (("cap_day", "hold 1 day"), ("cap_week", "hold 1 week")):
    g = st.one_sample_t(inc[base + "_gross"].values)
    n2 = st.one_sample_t(inc[base + "_net"].values)
    n5 = st.one_sample_t(inc5[base + "_net"].values)
    print(f"  {label:<11s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@2bps {n2['mean']*100:+.3f}% (t={n2['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})")
pl_cap = st.placebo_pvalue(ev, prices, "cap_week_net", k=5, entry_offset=1, cost_bps=2.0,
                           tail="right")
print(f"  1-week capture placebo (right-tail): observed {pl_cap['obs']*100:+.3f}% "
      f"vs placebo mean {pl_cap['placebo_mean']*100:+.3f}%  -> p = {pl_cap['p_value']:.4f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the paired one-sample-t detector must NOT fire on a null world (slump=0) and must "
      "recover a planted loser slump. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(slump=0.0, seed=735 + s, k=1)["t"] for s in range(20)])
print(f"  null (slump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for slump in (0.01, 0.02):
    pl = st.synthetic_detect(slump=slump, seed=735, k=1)
    print(f"  planted slump=-{slump*100:.0f}% (seed 735): mean spread {pl['mean']*100:+.3f}%  "
          f"t = {pl['t']:+.2f}  (n={pl['n']} synthetic events)")

print("\n# VERDICT (filled in after reading the numbers above)")
