"""Reproducible headline run for Study 823 — Variance-Risk-Premium Return Predictor.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY+VIX tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from vrp_predictor import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# VRP Return Predictor — does implied-minus-realized variance predict the market?")

if not data.have_real():
    print("(cache miss — fetching SPY + ^VIX once)")
    data.fetch()

df = data.load_panel()
spy, vix = df["SPY"], df["VIX"]
m = st.build_monthly(spy, vix)
print(f"[data] {len(df)} daily rows -> {len(m)} month-ends  "
      f"{m.index.min().date()} -> {m.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(SPY)={data.fingerprint(df)}")
print(f"  VRP mean = {m['vrp'].mean():+.5f} monthly variance "
      f"(~{(m['vrp'].mean()*12)**0.5*100:.1f} annualised vol-points); "
      f"IV mean {m['iv'].mean():.5f} > RV mean {m['rv'].mean():.5f} (implied > realized).")
print("  RISK-FREE proxied at 0 (named on the Signal axis — shifts the intercept, not "
      "the slope).")

print("\n# THE HEADLINE — predictive regression of forward return on VRP_t")
for h in (1, 3, 6, 12):
    hd = st.headline(m, horizon=h)
    print(f"  {h:2d}-month: slope {hd['slope']:+6.2f}  NW t {hd['t_nw']:+5.2f}  "
          f"R2 {hd['r2']*100:5.2f}%  tercile(hi-lo) {hd['tercile_spread_pct']:+5.2f}%  "
          f"n={hd['n']}")

print("\n# ROBUSTNESS — two eras (split 2010-01-01), 3-month horizon")
print(st.era_cut(m, split="2010-01-01", horizon=3).round(4).to_string())
print("  (1-month cut for reference:)")
print(st.era_cut(m, split="2010-01-01", horizon=1).round(4).to_string())

print("\n# PLACEBO — block-rotate forward returns vs VRP (2,000 draws), 3-month horizon")
pl = st.placebo_pvalue(m, horizon=3, n_perm=2000)
print(f"  observed slope {pl['obs_slope']:+.2f} vs placebo mean {pl['placebo_mean']:+.3f} "
      f"(sd {pl['placebo_sd']:.3f}) over {pl['n_perm']:,} draws -> right-tail p = {pl['p_value']:.3f}")

print("\n# THE TIMER — own SPY when VRP > expanding median, else cash (5 bps/switch)")
tm = st.timer_stats(m, horizon=1, cost_bps=5.0)
print(f"  timer Sharpe {tm['timer_sharpe']:.2f} vs buy-and-hold {tm['bh_sharpe']:.2f}; "
      f"switches/yr {tm['switches_per_yr']:.1f}; invested {tm['invested_frac']*100:.0f}%")
print(f"  timer - buy&hold mean return {tm['spread_pct_mo']:+.2f}%/month (NW t {tm['spread_t']:+.2f})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network (3-month horizon)")
null = st.synthetic_mean_t(data, edge=0.0, n_seeds=20, horizon=3)
planted = st.synthetic_mean_t(data, edge=6.0, n_seeds=20, horizon=3)
print(f"  null    (edge=0), 20 seeds: slope {null['mean_slope']:+.2f}  mean NW t "
      f"{null['mean_t']:+.2f}  R2 {null['mean_r2']*100:.1f}%  fires {int(null['fire_frac']*20)}/20")
print(f"  planted (edge=6), 20 seeds: slope {planted['mean_slope']:+.2f}  mean NW t "
      f"{planted['mean_t']:+.2f}  R2 {planted['mean_r2']*100:.1f}%  fires {int(planted['fire_frac']*20)}/20")

print("\n# VERDICT: Signal WEAK (shape replicates, strong 1993-2009 t=+5.93, but full-tape "
      "HAC t=+0.75 and decays/inverts post-2010) | Tradability MIRAGE | edge DECAYED.")
