"""Reproducible headline run for Study 825 — Oil Predicts Equities.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached USO+SPY tape under
``_cache/`` (fetching once, retried, on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from oil_equities import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Oil Predicts Equities — does this month's oil change forecast next month's stocks?")

if not data.have_real():
    print("(cache miss — fetching USO + SPY once, retried)")
    data.fetch()

s = data.load_series()
fr = st.regression_frame(s)
print(f"[data] {len(s)} daily rows  {s.index.min().date()} -> {s.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint(SPY)={data.fingerprint(s)}")
print(f"       {len(fr)} monthly obs  {fr.index.min().date()} -> {fr.index.max().date()}")
print("  SURVIVORSHIP: USO/SPY are continuously-listed liquid ETFs — no delisting bias on "
      "this pair; named on the Signal axis (ETF proxies, not the front-month contract).")

h = st.regression_stats(s)
print("\n# THE HEADLINE — predictive regression  r_equity[t+1] = a + b * r_oil[t]")
print(f"  slope beta = {h['beta']:+.4f}  (monthly SPY return per unit monthly oil return)")
print(f"  Newey-West(6) t = {h['t_nw']:+.2f}   OLS t = {h['t_ols']:+.2f}   R2 = {h['r2_pct']:.2f}%")
print(f"  alpha = {h['alpha']*100:+.2f}%/mo   n = {h['n']} months")
print(f"  fwd equity after oil-DOWN tercile {h['fwd_after_oil_down_pct']:+.2f}%  vs "
      f"oil-UP tercile {h['fwd_after_oil_up_pct']:+.2f}%  (Welch t = {h['welch_t']:+.2f})")
print("  => Driesprong predicts beta<0; here beta is ~0 and (weakly) POSITIVE — claim absent.")

print("\n# PLACEBO — permute the target, keep the predictor (2,000 draws)")
pl = st.placebo_pvalue(s, n_draws=2000)
print(f"  observed beta {pl['obs_beta']:+.4f} vs placebo mean {pl['placebo_mean_beta']:+.4f} "
      f"(sd {pl['placebo_sd_beta']:.4f}) -> two-sided p = {pl['p_value']:.3f}")

print("\n# ROBUSTNESS — two eras (split 2016-01-01)")
for lo, hi, lbl in [("2006-01-01", "2016-01-01", "2006-2015"),
                    ("2016-01-01", "2026-07-01", "2016-2026")]:
    sub = fr[(fr.index >= lo) & (fr.index < hi)]
    reg = st.newey_west_ols(sub["x"].to_numpy(), sub["y"].to_numpy())
    print(f"  {lbl}: n={reg['n']:>3}  beta {reg['beta']:+.4f}  NW t = {reg['t_nw']:+.2f}  "
          f"R2 = {reg['r2']*100:.2f}%")

print("\n# THE TIMER — trade -sign(oil_ret) of SPY next month, costed")
for cb in (1.0, 5.0):
    tm = st.timer_stats(s, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  long/short  cost={cb:>3.0f}bp: gross {tm['gross_pct_mo']:+.3f}%/mo -> net "
          f"{tm['net_pct_mo']:+.3f}%/mo (t={tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, "
          f"~{tm['ann_net_pct']:+.1f}%/yr, hit {tm['hit_rate']:.3f})")
for cb in (1.0,):
    tm = st.timer_stats(s, cost_bps=cb, borrow_bps_yr=50.0, long_only=True)
    print(f"  long/flat   cost={cb:>3.0f}bp: net {tm['net_pct_mo']:+.3f}%/mo (t={tm['t_net']:+.2f}, "
          f"Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")
m = st.monthly_returns(s)
sp = m["equity"].to_numpy()
print(f"  SPY buy&hold          : {sp.mean()*100:+.3f}%/mo (t={st.one_sample_t(sp):+.2f}, "
      f"Sharpe {sp.mean()/sp.std(ddof=1)*np.sqrt(12):.2f}) -- the long/flat timer LOSES to this.")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([st.synthetic_detect(data.synthetic_series(edge=0.0, seed=825 + i))["t_nw"]
                   for i in range(20)])
print(f"  null (edge=0), 20 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_t) >= 2).sum()}/20 seeds")
sy = st.synthetic_detect(data.synthetic_series(edge=0.35, seed=825))
print(f"  planted (edge=0.35, seed 825): beta {sy['beta']:+.4f}, NW t = {sy['t_nw']:+.2f}, "
      f"R2 = {sy['r2_pct']:.2f}% (recovers the planted NEGATIVE Driesprong slope)")
