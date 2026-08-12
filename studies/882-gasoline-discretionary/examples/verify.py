"""Reproducible headline run for Study 882 — Gas-Price → Discretionary (the "pump tax").

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached RB=F + XLY + XLP + XLE + SPY
tape under ``_cache/`` (fetching once, retried, on a cache miss), and always runs the
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

from gas_discretionary import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Gas-Price -> Discretionary — does a gas spike forecast XLY underperforming XLP?")

if not data.have_real():
    print("(cache miss — fetching RB=F + XLY + XLP + XLE + SPY once, retried)")
    data.fetch()

s = data.load_series()
fr = st.regression_frame(s, target="disc_stap")
print(f"[data] {len(s)} daily rows  {s.index.min().date()} -> {s.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint(XLY)={data.fingerprint(s)}")
print(f"       {len(fr)} monthly obs  {fr.index.min().date()} -> {fr.index.max().date()}")
print("  SURVIVORSHIP: RB=F/XLY/XLP/XLE/SPY are continuously-listed liquid futures/ETFs — "
      "no delisting bias; RB=F is a front-month RBOB roll proxy for the pump price "
      "(named on the Signal axis).")

h = st.regression_stats(s, target="disc_stap")
print("\n# THE HEADLINE — predictive regression  r_(XLY-XLP)[t+1] = a + b * r_gas[t]")
print(f"  slope beta = {h['beta']:+.4f}  (fwd XLY-XLP spread per unit monthly gas return)")
print(f"  Newey-West(6) t = {h['t_nw']:+.2f}   OLS t = {h['t_ols']:+.2f}   R2 = {h['r2_pct']:.2f}%")
print(f"  alpha = {h['alpha']*100:+.2f}%/mo   n = {h['n']} months")
print(f"  fwd XLY-XLP after gas-DOWN tercile {h['fwd_after_gas_down_pct']:+.2f}%  vs "
      f"gas-UP tercile {h['fwd_after_gas_up_pct']:+.2f}%  (Welch t = {h['welch_t']:+.2f})")
print("  => pump-tax predicts beta<0 (gas up -> discretionary lags staples).")

he = st.regression_stats(s, target="enr_mkt")
print("\n# ENERGY TILT — predictive regression  r_(XLE-SPY)[t+1] = a + b * r_gas[t]  (expect b>0)")
print(f"  slope beta = {he['beta']:+.4f}  NW(6) t = {he['t_nw']:+.2f}  R2 = {he['r2_pct']:.2f}%  n = {he['n']}")
print(f"  fwd XLE-SPY after gas-DOWN {he['fwd_after_gas_down_pct']:+.2f}%  vs "
      f"gas-UP {he['fwd_after_gas_up_pct']:+.2f}%  (Welch t = {he['welch_t']:+.2f})")

print("\n# PLACEBO — permute the target, keep the predictor (2,000 draws)")
pl = st.placebo_pvalue(s, target="disc_stap", n_draws=2000)
print(f"  observed beta {pl['obs_beta']:+.4f} vs placebo mean {pl['placebo_mean_beta']:+.4f} "
      f"(sd {pl['placebo_sd_beta']:.4f}) -> two-sided p = {pl['p_value']:.3f}")

print("\n# ROBUSTNESS — two eras (split 2016-01-01), XLY-XLP target")
for lo, hi, lbl in [("2005-01-01", "2016-01-01", "2005-2015"),
                    ("2016-01-01", "2026-07-01", "2016-2026")]:
    sub = fr[(fr.index >= lo) & (fr.index < hi)]
    reg = st.newey_west_ols(sub["x"].to_numpy(), sub["y"].to_numpy())
    print(f"  {lbl}: n={reg['n']:>3}  beta {reg['beta']:+.4f}  NW t = {reg['t_nw']:+.2f}  "
          f"R2 = {reg['r2']*100:.2f}%")

print("\n# THE TIMER — trade -sign(gas_ret) of the XLY-XLP spread next month, costed")
for cb in (1.0, 5.0):
    tm = st.timer_stats(s, target="disc_stap", cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  spread timer cost={cb:>3.0f}bp: gross {tm['gross_pct_mo']:+.3f}%/mo -> net "
          f"{tm['net_pct_mo']:+.3f}%/mo (t={tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, "
          f"~{tm['ann_net_pct']:+.1f}%/yr, hit {tm['hit_rate']:.3f})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([st.synthetic_detect(data.synthetic_series(edge=0.0, seed=882 + i))["t_nw"]
                   for i in range(20)])
print(f"  null (edge=0), 20 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_t) >= 2).sum()}/20 seeds")
sy = st.synthetic_detect(data.synthetic_series(edge=0.35, seed=882))
print(f"  planted (edge=0.35, seed 882): beta {sy['beta']:+.4f}, NW t = {sy['t_nw']:+.2f}, "
      f"R2 = {sy['r2_pct']:.2f}% (recovers the planted NEGATIVE pump-tax slope)")
