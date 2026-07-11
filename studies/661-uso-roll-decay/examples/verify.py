"""Reproducible headline run for Study 661 — USO-Roll-Decay.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached USO / CL=F tapes under
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

from uso_roll_decay import data, strategy as st  # noqa: E402

print("# USO-Roll-Decay — does the headline oil ETF actually track oil?")

if not data.have_real():
    print("(cache miss — fetching USO / CL=F once)")
    data.fetch()

uso, clf = data.load_real()
print(data_stamp("USO adj close", uso, asof=data.AS_OF))
print(data_stamp("CL=F front-month close", clf, asof=data.AS_OF))

df = st.gap_frame(uso, clf)
print(f"daily gap observations: {len(df)}  ({df.index.min().date()} -> {df.index.max().date()})"
      f"  [negative-oil day {data.NEGATIVE_OIL_DAY} and its neighbor drop out automatically "
      "via undefined log-returns]")

print("\n# THE HEADLINE — cumulative divergence since USO's 2006-04-10 inception")
cs = st.cumulative_stats(df)
print(f"  {cs['years']:.2f} years, {cs['start'].date()} -> {cs['end'].date()}")
print(f"  USO total return  : {cs['total_return_uso']*100:+.1f}%   (CAGR {cs['cagr_uso']*100:+.2f}%/yr)")
print(f"  CL=F total return : {cs['total_return_clf']*100:+.1f}%   (CAGR {cs['cagr_clf']*100:+.2f}%/yr)")
print(f"  CAGR gap (USO - CL=F): {cs['cagr_gap']*100:+.2f} pp/yr")

print("\n# Daily roll-drag — the gap (USO - CL=F) per day")
hs = st.headline_drag_stats(df)
print(f"  n={hs['n']}  mean {hs['mean']*100:+.4f}%/day  ({hs['ann_pct']:+.2f}%/yr)")
print(f"  naive t = {hs['naive_t']:+.2f}   NW(5) t = {hs['nw_t_5']:+.2f}   "
      f"NW(21) t = {hs['nw_t_21']:+.2f}   NW(63) t = {hs['nw_t_63']:+.2f}")
print(f"  hit rate (USO underperforms) {hs['hit_down']}/{hs['n']} = {hs['hit_rate']*100:.1f}%  "
      f"(Wilson 95% [{hs['hit_lo']*100:.1f}%, {hs['hit_hi']*100:.1f}%])")

print("\n# Circular block-bootstrap (block=21 sessions, 5,000 draws) for the mean daily gap")
bb = st.block_bootstrap_mean_ci(df["gap"].values, block=21, n_boot=5000, seed=661)
print(f"  95% CI (ann.): [{bb['lo']*252*100:+.2f}%, {bb['hi']*252*100:+.2f}%]   "
      f"p(mean >= 0) = {bb['p_ge0']:.4f}")

print("\n# Contango-stress regimes — hardcoded 2009 / 2020 super-contango windows vs the rest")
stress = data.stress_mask(df.index)
rs = st.regime_stats(df, stress)
print(f"  stress windows: n={rs['n_stress']} ({rs['n_stress']/hs['n']*100:.1f}% of days), "
      f"mean {rs['stress_ann_pct']:+.1f}%/yr  (one-sample t = {rs['one_t_stress']:+.2f})")
print(f"  rest of tape  : n={rs['n_rest']}, mean {rs['rest_ann_pct']:+.2f}%/yr  "
      f"(one-sample t = {rs['one_t_rest']:+.2f})")
print(f"  Welch t (stress vs rest) = {rs['welch_t_stress_vs_rest']:+.2f}")
print(f"  share of the entire 20-year cumulative log-divergence from stress windows: "
      f"{rs['stress_share_of_cum']*100:.1f}%")
for row in st.per_window_stats(df, data.CONTANGO_STRESS_WINDOWS):
    print(f"    {row['lo']} -> {row['hi']}: n={row['n']}, mean {row['ann_pct']:+.1f}%/yr, "
          f"t = {row['t']:+.2f}  -- {row['label'][:70]}...")

print("\n# The April-2020 negative-oil episode (simple returns; log undefined across a sign flip)")
case = st.negative_oil_case(uso, clf)
for d, row in case.iterrows():
    uret = "" if np.isnan(row["uso_ret_pct"]) else f"{row['uso_ret_pct']:+7.2f}%"
    cret = "" if np.isnan(row["clf_ret_pct"]) else f"{row['clf_ret_pct']:+8.2f}%"
    print(f"  {d.date()}  USO {row['uso_close']:7.2f} ({uret})   "
          f"CL=F {row['clf_close']:8.2f} ({cret})")

print("\n# THIRD AXIS — the 'long spot / short USO' carry-capture book")
print("  (constant-notional, monthly rebalance; borrow named on the short leg; costs = 2 x "
      "one-way x NAV per rebalance)")
cc = st.carry_capture_summary(df, stress, borrow_annual=0.0075, cost_bps_sweep=(5.0, 10.0))
g = cc["gross"]
print(f"  gross: {g['ann_ret']*100:+.2f}%/yr  vol {g['ann_vol']*100:.1f}%  "
      f"Sharpe {g['sharpe']:.2f}  max DD {g['max_dd']*100:.1f}%  worst day {g['worst_day']*100:.1f}%")
print(f"  HAC(21) t on the daily book return = {cc['hac_t']:+.2f}   "
      f"sign-shuffle placebo p = {cc['placebo_p']:.4f}")
for cb in (5.0, 10.0):
    n = cc[f"net_{cb:g}"]
    print(f"  net @ {cb:>4.1f} bps: {n['ann_ret']*100:+.2f}%/yr  Sharpe {n['sharpe']:.2f}  "
          f"max DD {n['max_dd']*100:.1f}%")
ex = cc["net5_ex_stress"]
st_only = cc["net5_stress_only"]
print(f"  net@5bps EX the 2009/2020 stress windows: {ex['ann_ret']*100:+.2f}%/yr  "
      f"Sharpe {ex['sharpe']:.3f}   (n={ex['n']})")
print(f"  net@5bps INSIDE the stress windows only  : {st_only['ann_ret']*100:+.1f}%/yr "
      f"(illustrative, n={st_only['n']} days)   Sharpe {st_only['sharpe']:.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT fire on a null world (drag=0) and must recover a planted drag.")
null_ts = []
for s_ in range(20):
    spot_r, fund_r, _ = data.synthetic_world(seed=2024 + s_, drag_daily_bps=0.0,
                                              stress_extra_bps=0.0)
    null_ts.append(st.synthetic_detect(spot_r, fund_r)["nw_t"])
null_ts = np.asarray(null_ts)
print(f"  null (drag=0), 20 seeds: mean NW t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
spot_r, fund_r, _ = data.synthetic_world(seed=2024, drag_daily_bps=3.0, stress_extra_bps=300.0)
sy = st.synthetic_detect(spot_r, fund_r)
print(f"  planted drag (3 bps/day + 300 bps in the two synthetic stress blocks): "
      f"{sy['ann_pct']:+.1f}%/yr   naive t = {sy['naive_t']:+.2f}   NW(5) t = {sy['nw_t']:+.2f}")
