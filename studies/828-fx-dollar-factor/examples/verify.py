"""Reproducible headline run for Study 828 — FX Dollar Factor (DOL) + dollar-timing.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached month-end FX panel under
``_cache/`` (fetching once on a cache miss through yfinance), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from dollar_factor import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# FX Dollar Factor (DOL) — does the dollar factor earn a premium, and does the "
      "average forward discount time it?")

if not data.have_real():
    print("(cache miss — fetching the FX basket once via yfinance)")
    data.fetch()

panel = data.load_panel()
print(f"[data] {panel.shape[1]} currencies, {len(panel)} month-end rows  "
      f"{panel.index.min().date()} -> {panel.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(panel)={fingerprint(panel)}")
print("  SURVIVORSHIP: a fixed *current* G10 basket (no delisted/pegged legs) — the "
      "magnitudes are an upper bound. Named on the Signal axis.")

dol = st.dol_series(panel, carry=data.CARRY_PROXY)
print(f"\n# THE HEADLINE — DOL premium (equal-weight foreign basket vs USD), "
      f"{st.dol_stats(dol,'spot')['n_months']} months")
for leg, lbl in [("spot", "spot     "), ("excess", "excess(+carry)")]:
    h = st.dol_stats(dol, leg)
    print(f"  DOL {lbl}: {h['mean_bps']:+.2f} bps/mo ({h['ann_pct']:+.2f}%/yr)  "
          f"NW(6) t = {h['t_nw']:+.2f}  one-sample t = {h['t_1s']:+.2f}  "
          f"Sharpe {h['sharpe']:+.2f}  vol {h['vol_ann_pct']:.1f}%/yr")

print("\n# DOLLAR-TIMING — predictive regression DOL_{t+1} = a + b*signal_t "
      "(signal = trailing-12m DOL, a spot-only AFD proxy)")
tm = st.timing_regression(dol, "spot", 12)
print(f"  slope b = {tm['beta']:+.4f}  NW(6) t = {tm['t_beta']:+.2f}  "
      f"R2 = {tm['r2']:.4f}  n = {tm['n']}")
pl = st.block_shuffle_placebo(dol, "spot", 12, n_perm=1000)
print(f"  block-shuffle placebo: |obs beta| {pl['obs_beta']:.4f} vs placebo sd "
      f"{pl['placebo_beta_sd']:.4f} over {pl['n_perm']} rotations -> p = {pl['p_value']:.4f}")

print("\n# ROBUSTNESS — two eras (split 2015-01-01)")
era = st.era_stats(dol, "2015-01-01", "spot")
for k, lbl in [("early", "2004-2014"), ("late", "2015-2026")]:
    e = era[k]
    print(f"  {lbl}: n={e['n_months']}  DOL {e['mean_bps']:+.2f} bps/mo "
          f"({e['ann_pct']:+.2f}%/yr)  NW t = {e['t_nw']:+.2f}")

print("\n# THE TIMER — costed static DOL and timed DOL (one-way 5 bps)")
tt = st.timer_stats(dol, "spot", 12, cost_bps=5.0)
print(f"  static long basket: gross {tt['static_gross_ann_pct']:+.2f}%/yr -> net "
      f"{tt['static_net_ann_pct']:+.2f}%/yr (Sharpe {tt['static_sharpe_net']:+.2f}, "
      f"t {tt['static_t_net']:+.2f})")
print(f"  timed (long when trend>0): gross {tt['timed_gross_ann_pct']:+.2f}%/yr -> net "
      f"{tt['timed_net_ann_pct']:+.2f}%/yr (Sharpe {tt['timed_sharpe_net']:+.2f}, "
      f"t {tt['timed_t_net']:+.2f}, {tt['switches_per_yr']:.2f} switches/yr, "
      f"invested {tt['frac_invested']*100:.0f}% of months)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_pt, null_tt = [], []
for s in range(20):
    p0 = data.synthetic_panel(edge=0.0, timing=0.0, seed=828 + s, n_months=480)
    d0 = st.synthetic_detect(p0)
    null_pt.append(d0["prem_t_nw"]); null_tt.append(d0["timing_t"])
null_pt = np.asarray(null_pt); null_tt = np.asarray(null_tt)
print(f"  null (edge=0, timing=0), 20 seeds: premium NW t mean {null_pt.mean():+.2f} "
      f"(sd {null_pt.std(ddof=1):.2f}), |t|>=2 in {(abs(null_pt)>=2).sum()}/20; "
      f"timing t |t|>=2 in {(abs(null_tt)>=2).sum()}/20")
dp = st.synthetic_detect(data.synthetic_panel(edge=0.004, timing=0.0, seed=828, n_months=480))
dt = st.synthetic_detect(data.synthetic_panel(edge=0.0, timing=0.02, seed=828, n_months=480))
print(f"  planted premium (edge=0.004): DOL {dp['prem_mean_bps']:+.2f} bps/mo, "
      f"NW t = {dp['prem_t_nw']:+.2f}")
print(f"  planted timing  (timing=0.02): slope {dt['timing_beta']:+.4f}, "
      f"t = {dt['timing_t']:+.2f}")
