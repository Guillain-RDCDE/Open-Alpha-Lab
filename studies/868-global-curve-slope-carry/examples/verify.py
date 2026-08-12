"""Reproducible headline run for Study 868 — Global Curve-Slope Carry.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached month-end ETF panel under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from curve_slope_carry import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Global Curve-Slope Carry — does a high-carry / steep-curve sort pay a duration holder?")

if not data.have_real():
    print("(cache miss — fetching the sovereign-bond ETF tape once via yfinance)")
    data.fetch()

panel = data.load_panel()
print(f"[data] {panel.shape[1]} ETFs {list(panel.columns)}, {len(panel)} months  "
      f"{panel.index.min().date()} -> {panel.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(panel)}")
print("  DURATIONS (yrs, published): " + ", ".join(f"{k}={v}" for k, v in data.DURATIONS.items()))
print("  SURVIVORSHIP: currently-listed global-govvie ETFs only; BNDX lists from 2013 so the "
      "full 6-market cross-section is short. Named on the Signal axis; a price-only carry proxy.")

DUR = data.DURATIONS
bs = st.benchmark_stats(panel)
print(f"\n# BENCHMARK — naive equal-weight buy-and-hold of the six ETFs")
print(f"  {bs['mean_bps']:+.2f} bps/mo ({bs['ann_pct']:+.2f}%/yr), Sharpe {bs['sharpe']:.3f}, "
      f"NW t = {bs['t_nw']:+.2f}  (n={bs['n_months']})")

for tag, dur in [("YIELD-TO-DURATION carry (primary)", DUR), ("RAW realized-yield carry", None)]:
    print(f"\n# THE HEADLINE — cross-sectional {tag}, long high-carry / short low-carry")
    bt = st.carry_book(panel, durations=dur)
    s = st.carry_stats(bt)
    print(f"  {s['mean_bps']:+.2f} bps/mo ({s['ann_pct']:+.2f}%/yr, vol {s['vol_ann_pct']:.2f}%), "
          f"n={s['n_months']} months  (long {s['long_bps']:+.2f} / short {s['short_bps']:+.2f} bps)")
    print(f"  Sharpe {s['sharpe']:.3f}  NW(6) t = {s['t_nw']:+.2f}  one-sample t = {s['t_1s']:+.2f}  "
          f"hit {s['hit_rate']:.3f}")
    pl = st.placebo_pvalue(panel, durations=dur, n_perm=3000)
    print(f"  placebo (column-permutation, {pl['n_draws']:,}): observed {pl['obs_bps']:+.2f} bps vs "
          f"null mean {pl['placebo_mean_bps']:+.2f} (sd {pl['placebo_sd_bps']:.2f}) -> p = {pl['p_value']:.4f}")
    edges = [("2010-2016", "2010-01-01", "2016-01-01"),
             ("2016-2021", "2016-01-01", "2021-01-01"),
             ("2021-2026", "2021-01-01", "2026-07-01")]
    sw = st.subperiod_sweep(panel, edges, durations=dur)
    for per, row in sw.iterrows():
        print(f"    {per}: {row['mean_bps']:+.2f} bps  NW t = {row['t_nw']:+.2f}  "
              f"Sharpe {row['sharpe']:+.3f}  (n={int(row['n'])})")
    for cb in (5.0, 10.0):
        tm = st.timer_stats(panel, durations=dur, cost_bps=cb, borrow_bps_yr=75.0)
        print(f"  cost {cb:>4.1f} bps one-way: gross {tm['gross_bps']:+.2f} -> net {tm['net_bps']:+.2f} "
              f"bps/mo (cost {tm['cost_bps_per_mo']:.2f}/mo, turnover {tm['avg_turnover']:.2f}, "
              f"net Sharpe {tm['net_sharpe']:+.3f}, t = {tm['t_net']:+.2f})")

print("\n# WINDOW ROBUSTNESS — yield-to-duration carry NW t across formation windows")
sw = st.window_sweep(panel, (24, 36, 48, 60), durations=DUR)
for w, row in sw.iterrows():
    print(f"  window {int(w):>2}m: {row['mean_bps']:+.2f} bps  Sharpe {row['sharpe']:+.3f}  "
          f"NW t = {row['t_nw']:+.2f}  (n={int(row['n'])})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null = st.synthetic_mean_t(data, edge=0.0, n_seeds=20, base_seed=868)
planted = st.synthetic_mean_t(data, edge=0.010, n_seeds=20, base_seed=868)
print(f"  null (edge=0), 20 seeds:      mean NW t {null['mean_t']:+.2f}, |t|>=2 in "
      f"{null['fire_frac']*20:.0f}/20 seeds")
print(f"  planted (edge=0.010), 20 seeds: mean NW t {planted['mean_t']:+.2f}, mean Sharpe "
      f"{planted['mean_sharpe']:.2f}, |t|>=2 in {planted['fire_frac']*20:.0f}/20 seeds")
