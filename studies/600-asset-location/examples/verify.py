"""Reproducible headline run for Study 600 — Asset Location.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached Shiller long tape and the cached
SPY/IEF modern tape under ``_cache/`` if present (the real-tape numbers), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd

from asset_location import data, strategy as st
from quantlab import repro

AS_OF = "2026-07-03"   # run date; the study uses complete CALENDAR YEARS only

print("# Asset Location — bonds in the IRA vs stocks in the IRA vs pro-rata (60/40 household)")
print(f"baseline household: ordinary {st.ORDINARY:.0%} / LTCG+QDI {st.LTCG:.0%}, "
      f"equity turnover {st.TURNOVER:.0%}/yr, taxable/IRA split {st.SPLIT:.0%}/{1-st.SPLIT:.0%}, "
      f"{st.HORIZON}-year horizon, HAC lag {st.HAC_LAGS}")

if data.have_shiller():
    sh = data.load_shiller()
    sh = repro.as_of(sh, AS_OF)
    comp = data.annual_components(sh)
    print("\n" + repro.data_stamp("Shiller monthly (price/div/GS10)", sh, asof=AS_OF))
    print(f"[data] annual components: {len(comp)} complete calendar years "
          f"{comp.index.min()} -> {comp.index.max()} (year t bond coupon = prior-Dec 10y yield "
          f"— the one documented decision lag)")

    tab = st.cohort_table(comp)
    s = st.summarize_cohorts(tab)

    print(f"\n# Long tape — every overlapping {st.HORIZON}-year cohort "
          f"({s['n_cohorts']} cohorts, starts {tab.index.min()}-{tab.index.max()})")
    print(f"  after-tax annualised, mean across cohorts:")
    print(f"    bonds-in-IRA : {s['mean_ann_bonds_ira']:.2f}%/yr")
    print(f"    stocks-in-IRA: {s['mean_ann_stocks_ira']:.2f}%/yr")
    print(f"    pro-rata     : {s['mean_ann_pro_rata']:.2f}%/yr")
    print(f"  DELTA bonds-in-IRA vs stocks-in-IRA: {s['mean_delta_bs']:+.2f} bps/yr   "
          f"HAC t (lag {st.HAC_LAGS}) = {s['t_bs']:+.2f}   win rate {s['win_bs']:.1f}%")
    print(f"  DELTA bonds-in-IRA vs pro-rata     : {s['mean_delta_bp']:+.2f} bps/yr   "
          f"HAC t (lag {st.HAC_LAGS}) = {s['t_bp']:+.2f}   win rate {s['win_bp']:.1f}%")
    print(f"  cohort delta range: {s['min_delta_bs']:+.2f} to {s['max_delta_bs']:+.2f} bps/yr")
    print(f"  mean terminal after-tax wealth gap (bonds-in-IRA / stocks-in-IRA - 1): "
          f"{s['mean_wealth_gap_pct']:+.2f}%")

    print("\n# Costs — the location choice itself is free; only rebalancing trades differ")
    print(f"  avg taxable trade volume (%NAV/yr): bonds-in-IRA {s['mean_trade_bonds_ira']:.2f}  "
          f"stocks-in-IRA {s['mean_trade_stocks_ira']:.2f}  pro-rata {s['mean_trade_pro_rata']:.2f}")
    diff_cost = 5.0 * (s['mean_trade_bonds_ira'] - s['mean_trade_stocks_ira']) / 100.0
    print(f"  differential cost at 5 bps one-way x NAV: {diff_cost:+.3f} bps/yr "
          f"(negligible next to the {s['mean_delta_bs']:+.1f} bps/yr delta)")

    print("\n# The benefit scales with the yield level (cohort-average locked coupon)")
    q = st.yield_quartiles(tab)
    for lbl, row in q.iterrows():
        print(f"  {lbl:<20s}: mean coupon {row['mean_cpn_pct']:.2f}%  "
              f"delta {row['mean_delta_bs']:+.2f} bps/yr  (n={int(row['n'])})")
    print(f"  regression delta ~ coupon: {s['slope_bps_per_pp']:+.2f} bps/yr per 1pp of yield   "
          f"HAC t = {s['t_slope']:+.2f}")

    print("\n# Tax-rate grid — mean delta (bonds-in-IRA vs stocks-in-IRA), bps/yr")
    print("  (realistic 2026 US pairs sit on the diagonal: 22/15, 24/15, 32/15, 37/20 —")
    print("   the 20% LTCG rate only starts inside the top ordinary brackets)")
    g = st.rate_grid(comp)
    for _, row in g.iterrows():
        print(f"  ordinary {row['ordinary']:.0%} / LTCG {row['ltcg']:.0%}: "
              f"delta {row['mean_delta_bs']:+.2f} bps/yr   HAC t = {row['t_bs']:+.2f}   "
              f"win {row['win_pct']:.1f}%")

    print("\n# Third axis — when does the rule FLIP? (equity-turnover sweep, long tape)")
    tf = st.turnover_flip(comp)
    for _, row in tf.iterrows():
        print(f"  turnover {row['turnover']:.0%}: delta {row['mean_delta_bs']:+.2f} bps/yr   "
              f"HAC t = {row['t_bs']:+.2f}   win {row['win_pct']:.1f}%   "
              f"lowest-yield-quartile delta {row['lowq_delta_bs']:+.2f} bps/yr")
else:
    print("(no _cache/al_shiller.csv — run data.fetch_shiller() once to build the cache)")

if data.have_modern():
    px, dv = data.load_modern()
    px = repro.as_of(px, AS_OF)
    mc = data.modern_components(px, dv)
    stamp_df = mc.copy()
    stamp_df.index = pd.to_datetime(stamp_df.index, format="%Y")
    print("\n" + repro.data_stamp("modern SPY/IEF annual components", stamp_df, asof=AS_OF))
    r = st.single_cohort(mc)
    print(f"# Modern low-rate cohort — SPY (stock) + IEF (bond), "
          f"{mc.index.min()}-{mc.index.max()} ({r['n_years']} complete years, "
          f"mean bond income yield {r['mean_cpn_pct']:.2f}%)")
    print(f"  after-tax annualised: bonds-in-IRA {r['ann_bonds_ira']:.2f}%  "
          f"stocks-in-IRA {r['ann_stocks_ira']:.2f}%  pro-rata {r['ann_pro_rata']:.2f}%")
    print(f"  delta bonds-in-IRA vs stocks-in-IRA: {r['delta_bs']:+.2f} bps/yr   "
          f"vs pro-rata: {r['delta_bp']:+.2f} bps/yr  (single cohort — no t; the long tape "
          f"carries the inference)")
    print("  modern turnover sweep (single cohort):")
    for tv in (0.0, 0.10, 0.25, 0.50, 1.00):
        rr = st.single_cohort(mc, turnover=tv)
        print(f"    turnover {tv:.0%}: delta_bs {rr['delta_bs']:+.2f} bps/yr")
else:
    print("(no _cache/al_modern_prices.csv — run data.fetch_modern() once to build the cache)")

print("\n# Synthetic control — deterministic, no network, averaged over 20 seeds")
print("  null: zero tax rates -> the account boundary is meaningless; every policy must return")
print("  the SAME after-tax wealth (delta exactly 0). Planted effect: the bond coupon knob.")
syn = st.synthetic_scaling()
for _, row in syn.iterrows():
    print(f"  {row['label']:<22s}: mean delta {row['mean_delta_bs']:+.3f} bps/yr  "
          f"(sd {row['sd_delta_bs']:.3f}, n_seeds={int(row['n_seeds'])})")
print("  (machinery proof only — never cited in support of the Signal stamp)")
