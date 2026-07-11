"""Reproducible headline run for Study 677 — Market Facilitation Index (BW-MFI).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + 5-ETF-basket tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from market_facilitation_index import data, strategy as st  # noqa: E402

print("# Market Facilitation Index — do BW-MFI bar colors predict what happens next?")
print("  Green (MFI up, Vol up) -> claimed CONTINUATION | Squat (MFI down, Vol up) -> "
      "claimed REVERSAL")

if not data.have_real():
    print("(cache miss — fetching SPY + basket once)")
    data.fetch()

basket = data.load_basket()
for t, bars in basket.items():
    print(data_stamp(t, bars, cols=["High", "Low", "Close", "Volume"], asof=data.AS_OF))

spy = basket["SPY"]
df_spy = st.day_frame(spy)
print(f"\nSPY tape: {len(df_spy)} bars, {df_spy['state'].notna().sum()} classified "
      f"(state needs 2 consecutive bars)")

print("\n# THE HEADLINE (SPY) — forward return & continuation score by bar color")
s = st.state_stats(df_spy)
print(f"{'state':10s} {'n':>6s} {'fwd(bps)':>9s} {'rest(bps)':>9s} {'t_fwd':>7s}  "
      f"{'cont(bps)':>10s} {'rest(bps)':>9s} {'t_cont':>7s}  {'cont%':>6s}")
for state in st.STATES:
    v = s[state]
    print(f"{state:10s} {v['n']:6d} {v['fwd_bps']:9.2f} {v['rest_fwd_bps']:9.2f} "
          f"{v['welch_t_fwd']:+7.2f}  {v['cont_bps']:10.2f} {v['rest_cont_bps']:9.2f} "
          f"{v['welch_t_cont']:+7.2f}  {v['cont_hit']*100:5.1f}%")

print("\n# Label-shuffle placebo on the continuation score (2,000 draws each)")
for state in ("green", "squat"):
    pv = st.permutation_pvalue_state(df_spy, state, n_perm=2000)
    print(f"  {state:6s}: observed gap {pv['obs_gap_bps']:+.2f} bps -> p = {pv['p_value']:.4f} "
          f"(n={pv['n']}, {pv['n_perm']} shuffles)")

print("\n# POOLED across the basket (SPY, QQQ, DIA, IWM, XLE, GLD)")
pooled = st.pooled_frame(basket)
sp = st.state_stats(pooled)
print(f"{'state':10s} {'n':>6s} {'fwd(bps)':>9s} {'t_fwd':>7s}  "
      f"{'cont(bps)':>10s} {'t_cont':>7s}  {'cont%':>6s}")
for state in st.STATES:
    v = sp[state]
    print(f"{state:10s} {v['n']:6d} {v['fwd_bps']:9.2f} {v['welch_t_fwd']:+7.2f}  "
          f"{v['cont_bps']:10.2f} {v['welch_t_cont']:+7.2f}  {v['cont_hit']*100:5.1f}%")

print("\n# Per-ticker Green/Squat continuation score Welch t (does the sign hold up?)")
for t, bars in basket.items():
    d = st.day_frame(bars)
    st_ = st.state_stats(d)
    print(f"  {t:5s} green t={st_['green']['welch_t_cont']:+6.2f} (n={st_['green']['n']:5d})  "
          f"squat t={st_['squat']['welch_t_cont']:+6.2f} (n={st_['squat']['n']:5d})")

print("\n# THE TIMER — Green filter / Squat avoidance / SMA(50/200), NET 1bp, vs buy-hold "
      "(SPY)")
r = st.run_timer(df_spy, cost_bps=1.0)
for name in ("green", "squat_avoid", "sma"):
    d = r[name]
    print(f"  {name:12s} Sharpe={d['sharpe_strat']:5.2f}  BH={d['sharpe_bench']:5.2f}  "
          f"diff={d['mean_diff_bps']:+7.3f} bps  HAC t={d['hac_t_diff']:+6.2f}  "
          f"flips/yr={d['flips_per_yr']:5.1f}")

print("\n# Cost sweep — Green filter (SPY)")
for cb in (0.0, 1.0, 2.0, 5.0):
    rr = st.run_timer(df_spy, cost_bps=cb)
    print(f"  cost={cb:4.1f}bp  Sharpe={rr['green']['sharpe_strat']:5.2f}  "
          f"HAC t={rr['green']['hac_t_diff']:+6.2f}")

print("\n# Sign-permutation placebo — Green filter timer (SPY, 1bp, 1000 draws)")
pv = st.permutation_pvalue_timer(df_spy, cost_bps=1.0, n_perm=1000)
print(f"  real Sharpe gap = {pv['real_gap']:+.3f}   p = {pv['p_value']:.3f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT fire on a null world (planted=0) and must recover a")
print("  planted Green-continuation / Squat-reversal effect. Null checked over 20 seeds.")
null_green, null_squat = [], []
for s_ in range(20):
    bars, _ = data.synthetic_world(planted=0.0, seed=677 + s_)
    sy = st.synthetic_detect(bars)
    null_green.append(sy["green"]["welch_t_cont"])
    null_squat.append(sy["squat"]["welch_t_cont"])
null_green = np.asarray(null_green)
null_squat = np.asarray(null_squat)
print(f"  null (planted=0), 20 seeds: green mean t = {null_green.mean():+.2f} "
      f"(sd {null_green.std(ddof=1):.2f}), |t|>=2 in {(abs(null_green) >= 2).sum()}/20")
print(f"                              squat mean t = {null_squat.mean():+.2f} "
      f"(sd {null_squat.std(ddof=1):.2f}), |t|>=2 in {(abs(null_squat) >= 2).sum()}/20")
bars_p, _ = data.synthetic_world(planted=0.6, seed=677)
syp = st.synthetic_detect(bars_p)
print(f"  planted=+0.6 (seed 677): green cont {syp['green']['cont_bps']:+.2f} bps "
      f"t={syp['green']['welch_t_cont']:+.2f}  |  squat cont {syp['squat']['cont_bps']:+.2f} bps "
      f"t={syp['squat']['welch_t_cont']:+.2f}")
