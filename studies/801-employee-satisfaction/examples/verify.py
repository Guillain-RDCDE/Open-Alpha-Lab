"""Reproducible headline run for Study 801 — Employee Satisfaction ("100 Best" alpha).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached price tape under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from employee_satisfaction import data, strategy as st  # noqa: E402

print("# Employee Satisfaction — does a '100 Best Companies to Work For' basket beat the market?")

if not data.have_real():
    print("(cache miss — fetching prices once)")
    data.fetch_prices()

monthly, basket_present, control_present = data.load_real()
fp = data.fingerprint(monthly, data.BASKET_TICKERS + [data.MARKET])
print(f"[data] monthly total returns: {len(monthly)} months  "
      f"{monthly.index.min().date()} -> {monthly.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={fp}")
print(f"  basket names present: {len(basket_present)}/{len(data.BASKET_TICKERS)}  "
      f"{basket_present}")
print(f"  control (survivor placebo) names present: {len(control_present)}/{len(data.CONTROL)}")

basket = st.equal_weight_basket(monthly, basket_present)
market = monthly[data.MARKET]

print("\n# THE HEADLINE — market-model (CAPM) alpha of the equal-weight basket vs SPY")
h = st.headline_stats(basket, market)
print(f"  months: {h['n_months']}")
print(f"  basket ann return {h['basket_ann_pct']:+.2f}% (vol {h['basket_vol_pct']:.1f}%)  "
      f"vs market {h['market_ann_pct']:+.2f}% (vol {h['market_vol_pct']:.1f}%)")
print(f"  CAPM alpha: {h['alpha_monthly_bps']:+.1f} bps/mo = {h['alpha_ann_pct']:+.2f}%/yr, "
      f"beta {h['beta']:.2f}")
print(f"  alpha t: Newey-West = {h['t_alpha_nw']:+.2f}   OLS = {h['t_alpha_ols']:+.2f}   "
      f"(bar = 2)")
print(f"  excess-over-market: {h['excess_mean_bps']:+.1f} bps/mo, NW t = {h['excess_t_nw']:+.2f}, "
      f"OLS t = {h['excess_t_ols']:+.2f}, info ratio {h['ir_ann']:.2f}/yr")
print(f"  monthly win-rate vs market: {h['hit']}/{h['n_months']} = {h['hit_rate']*100:.1f}% "
      f"(Wilson [{h['hit_lo']*100:.1f}%, {h['hit_hi']*100:.1f}%])")

print("\n# SURVIVORSHIP PLACEBO — random equal-weight baskets of large-cap SURVIVORS (not")
print("  selected for workplace prestige). Does a random survivor basket earn alpha too?")
pl = st.survivor_placebo(monthly, control_present, market,
                         basket_size=len(basket_present), n_draws=5000)
p = st.placebo_pvalue(h["alpha_ann_pct"], pl["alphas_ann_pct"])
print(f"  random survivor-basket CAPM alpha: mean {pl['mean']:+.2f}%/yr (sd {pl['sd']:.2f}), "
      f"n_draws={pl['n']}, basket size {pl['k']}")
print(f"  p (share of random survivor baskets with alpha >= {h['alpha_ann_pct']:+.2f}%/yr) = {p:.3f}")

print("\n# SUB-PERIOD SPLIT — first half vs second half")
sp = st.subperiod_split(basket, market)
for tag in ("early", "late"):
    s = sp[tag]
    print(f"  {tag}: n={s['n']}  alpha {s['alpha_ann_pct']:+.2f}%/yr  NW t = {s['t_alpha_nw']:+.2f}")

print("\n# THE TIMER — hold the equal-weight basket, rebalance monthly; costs one-way x NAV x turnover")
for cb in (5.0, 10.0):
    tm = st.timer_stats(basket, market, monthly, basket_present, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: turnover {tm['turnover_1way_pct']:.1f}%/mo one-way -> "
          f"cost drag {tm['cost_drag_ann_pct']:.2f}%/yr; gross alpha {tm['gross_alpha_ann_pct']:+.2f}%/yr "
          f"(t={tm['gross_t_nw']:+.2f}) -> net alpha {tm['net_alpha_ann_pct']:+.2f}%/yr (t={tm['net_t_nw']:+.2f})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  the CAPM-alpha detector must NOT fire on a null world (alpha=0) and must recover a")
print("  planted monthly alpha. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    w = data.synthetic_world(alpha_bps=0.0, seed=801 + s_)
    null_ts.append(st.synthetic_detect(w)["t_alpha_nw"])
null_ts = np.asarray(null_ts)
print(f"  null (alpha=0), 20 seeds: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
w = data.synthetic_world(alpha_bps=40.0, seed=801)
sy = st.synthetic_detect(w)
print(f"  planted alpha=+40 bps/mo (seed 801): recovered {sy['alpha_ann_pct']:+.2f}%/yr, "
      f"beta {sy['beta']:.2f}, NW t = {sy['t_alpha_nw']:+.2f}")
