"""Reproducible headline run for Study 879 — Weekly Economic Index.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached WEI + market panels under
``_cache/`` (fetching once on a cache miss from the Dallas Fed workbook + yfinance), and
always runs the synthetic control with no network.

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

from wei import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Weekly Economic Index — does a weekly growth nowcast time SPY / cyclical rotation?")

if not data.have_real():
    print("(cache miss — fetching the WEI workbook + SPY/XLY/XLP once)")
    data.fetch_wei()
    data.fetch_market()

f = data.build_real()
print(f"[data] {len(f)} weekly rows  {f.index.min().date()} -> {f.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint(wei,dwei)={fingerprint(f[['wei', 'dwei']])}")
print("  SIGNAL note: WEI is a *revised* nowcast; we use the current-vintage level with a "
      "one-week publication lag. Revisions are small vs the weekly change; the level "
      "caveat travels with every number.")

print("\n# THE HEADLINE — predictive regression (Newey-West HAC t), forward returns on "
      "standardized WEI level & weekly change")
for tgt, lbl in [("spy_h1", "SPY 1wk"), ("spy_h4", "SPY 4wk"),
                 ("rot_h1", "XLY-XLP 1wk"), ("rot_h4", "XLY-XLP 4wk")]:
    p = st.predict(f, tgt)
    print(f"  {lbl:12s} n={p['n']}  R2={p['r2']:+.4f}  "
          f"level t={p['t_level']:+.2f} (uni {p['t_level_uni']:+.2f})  "
          f"dwei t={p['t_dwei']:+.2f} (uni {p['t_dwei_uni']:+.2f})")

print("\n# ROBUSTNESS — two eras (split 2017-01-01), univariate HAC t")
for tgt, lbl in [("spy_h1", "SPY 1wk"), ("rot_h4", "XLY-XLP 4wk")]:
    e = st.era_split(f, tgt)
    print(f"  {lbl:12s} early n={e['early']['n']} lvl t={e['early']['t_level']:+.2f} "
          f"dwei t={e['early']['t_dwei']:+.2f}  |  late n={e['late']['n']} "
          f"lvl t={e['late']['t_level']:+.2f} dwei t={e['late']['t_dwei']:+.2f}")

print("\n# CONDITIONAL — forward return when the nowcast is 'on' vs the base rate")
for tgt, sig, rule in [("spy_h1", "wei", "above_median"), ("spy_h1", "dwei", "positive"),
                       ("rot_h4", "wei", "above_median")]:
    c = st.conditional(f, tgt, sig, rule)
    print(f"  {tgt:7s} | {sig:4s} {rule:12s}: cond {c['cond_mean']*100:+.2f}% "
          f"(win {c['cond_win']:.2f}) vs base {c['base_mean']*100:+.2f}%  "
          f"Welch t = {c['welch_t']:+.2f}")

print("\n# PLACEBO — permute the nowcast, re-run the HAC slope t (rot_h4 level, 2,000 draws)")
pl = st.placebo_pvalue(f, "rot_h4", "wei", n_draws=2000)
print(f"  observed |t| = {pl['obs_t']:.2f} vs placebo mean |t| = {pl['placebo_mean_abs_t']:.2f}"
      f"  -> two-sided p = {pl['p_value']:.4f}")

print("\n# THE TIMER — costed long-cyclical / short-defensive (XLY-XLP) overlay, weekly")
for sig, rule in [("wei", "above_median"), ("dwei", "positive")]:
    o = st.rotation_overlay(f, sig, rule, cost_bps=5.0, borrow_bps_yr=50.0)
    print(f"  {sig:4s}/{rule:12s}: gross Sh {o['gross']['sharpe']:+.2f}  net Sh "
          f"{o['net']['sharpe']:+.2f}  vs hold Sh {o['hold']['sharpe']:+.2f}  "
          f"(net {o['net']['ann_ret']*100:+.1f}%/yr, t_net {o['t_net']:+.2f}, "
          f"{o['n_turns']:.0f} turns)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([st.synthetic_detect(data.synthetic(edge=0.0, seed=879 + s, n=700))["t_level"]
                   for s in range(20)])
print(f"  null (edge=0), 20 seeds: level t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
fp = data.synthetic(edge=0.010, seed=879, n=700)
sy_spy = st.synthetic_detect(fp, "spy_h1")
sy_rot = st.synthetic_detect(fp, "rot_h1")
print(f"  planted (edge=0.010, seed 879): SPY level t = {sy_spy['t_level']:+.2f} "
      f"(R2 {sy_spy['r2']:.3f}), rotation level t = {sy_rot['t_level']:+.2f}")
