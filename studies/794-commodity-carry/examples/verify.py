"""Reproducible headline run for Study 794 — Commodity Carry.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EIA curve + ETF tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from commodity_carry import data, strategy as st  # noqa: E402

print("# Commodity Carry — do backwardated commodities out-earn contangoed ones (cross-sectional)?")
print("# PROXY: two-name energy cross-section (WTI, Henry Hub gas); carry from the real EIA")
print("#        futures term structure, returns from investable ETFs. Under-powered by design.\n")

if not data.have_real():
    print("(cache miss — fetching EIA curve + ETFs once)")
    data.fetch()

curve = data.load_curve()
etfs = data.load_etfs()
print(data_stamp("EIA energy curve (WTI+HH, C1..C4)", curve, asof=data.AS_OF))
print(data_stamp("energy ETFs (USO/USL/UNG/UNL)", etfs, asof=data.AS_OF))

panel = st.build_panel(curve, etfs, data.COMMODITIES)
print(f"\nmonthly panel: {len(panel)} (commodity,month) rows, "
      f"{panel['month'].nunique()} months, {panel['commodity'].nunique()} commodities "
      f"({panel['month'].min()} -> {panel['month'].max()})")

print("\n# THE WORKHORSE — pooled cross-sectional carry -> forward return (NW 6-lag)")
pt = st.pooled_carry_test(panel)
print(f"  n_obs={pt['n_obs']} over {pt['n_months']} two-name months")
print(f"  slope (return per unit annualized carry) = {pt['slope']:+.4f}  "
      f"corr = {pt['corr']:+.3f}")
print(f"  Newey-West t = {pt['nw_t']:+.2f}   <-- decisive real-tape number")

print("\n# Conditional split — forward front-ETF return, backwardation vs contango (pooled)")
cs = st.conditional_split(panel)
print(f"  backwardated (carry>0, n={cs['n_back']}): {cs['back_pct']:+.2f}%/mo  |  "
      f"contangoed (carry<0, n={cs['n_cont']}): {cs['cont_pct']:+.2f}%/mo")
print(f"  Welch t (back - cont) = {cs['welch_t']:+.2f}")
lo, hi = st.wilson_interval(cs['hit'], cs['hit_n'])
print(f"  backwardated hit rate (forward return>0): {cs['hit']}/{cs['hit_n']} = "
      f"{cs['hit']/cs['hit_n']*100:.1f}%  Wilson [{lo*100:.1f}%, {hi*100:.1f}%]")

print("\n# Roll-drag mechanism proxy — USO minus USL vs WTI carry (NW 6-lag)")
rd = st.roll_drag_proxy(panel, "WTI")
print(f"  n={rd['n']}  slope={rd['slope']:+.4f}  NW t={rd['nw_t']:+.2f}  "
      f"sign-agreement={rd['sign_agree']*100:.0f}%")
print(f"  mean USO-USL gap: backwardation {rd['mean_gap_backwardation_pct']:+.2f}%/mo  "
      f"vs contango {rd['mean_gap_contango_pct']:+.2f}%/mo")

print("\n# THE TIMER — two-name cross-sectional long-short, monthly, costed + borrow")
for cb in (5.0, 10.0):
    tm = st.timer_stats(panel, cost_bps=cb, borrow_bps_yr=100.0)
    print(f"  cost={cb:>4.1f} bps + 100 bps/yr borrow: gross {tm['gross_ann_pct']:+.2f}%/yr "
          f"(NW t={tm['gross_t']:+.2f}, Sharpe {tm['gross_sharpe']:.2f}) -> "
          f"net {tm['net_ann_pct']:+.2f}%/yr (NW t={tm['net_t']:+.2f}, "
          f"Sharpe {tm['net_sharpe']:.2f}, worst month {tm['worst_pct']:+.2f}%)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the pooled carry detector must NOT fire on a null panel (premium=0) and must")
print("  recover a planted cross-sectional carry premium. Null over 20 seeds.")
null_ts = []
for s_ in range(20):
    p = data.synthetic_panel(premium=0.0, seed=794 + s_)
    null_ts.append(st.synthetic_detect(p)["nw_t"])
null_ts = np.asarray(null_ts)
print(f"  null (premium=0), 20 seeds: mean NW t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
planted = st.synthetic_detect(data.synthetic_panel(premium=0.15, seed=794))
print(f"  planted premium=0.15 (seed 794): slope {planted['slope']:+.4f}, "
      f"NW t = {planted['nw_t']:+.2f}")
