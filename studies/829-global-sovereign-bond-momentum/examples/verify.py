"""Reproducible headline run for Study 829 — Global Sovereign-Bond Momentum.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached month-end ETF panel under
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

from sovereign_momentum import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Global Sovereign-Bond Momentum — does a bond market's own 12-1 trend predict it?")

if not data.have_real():
    print("(cache miss — fetching the sovereign-bond ETF tape once via yfinance)")
    data.fetch()

panel = data.load_panel()
print(f"[data] {panel.shape[1]} ETFs {list(panel.columns)}, {len(panel)} months  "
      f"{panel.index.min().date()} -> {panel.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(panel)}")
print("  SURVIVORSHIP: currently-listed global-govvie ETFs only (wound-up funds absent) "
      "-> a mild upper bound; the tiny 5-ETF panel also limits power. Named on the Signal axis.")

r = st.monthly_returns(panel)
bh = r.mean(axis=1).dropna().to_numpy()
print(f"\n# BENCHMARK — naive equal-weight buy-and-hold of the five ETFs")
print(f"  {bh.mean()*1e4:+.2f} bps/mo ({bh.mean()*12*100:+.2f}%/yr), Sharpe "
      f"{bh.mean()/bh.std(ddof=1)*np.sqrt(12):.3f}, NW t = {st.newey_west_t(bh,6):+.2f}")

for long_only in (False, True):
    tag = "LONG-ONLY trend timing (+1/0)" if long_only else "LONG-SHORT time-series momentum (+1/-1)"
    print(f"\n# THE HEADLINE — {tag}")
    bt = st.tsmom_returns(panel, long_only=long_only)
    s = st.tsmom_stats(bt)
    print(f"  {s['mean_bps']:+.2f} bps/mo ({s['ann_pct']:+.2f}%/yr, vol {s['vol_ann_pct']:.2f}%), "
          f"n={s['n_months']} months")
    print(f"  Sharpe {s['sharpe']:.3f}  NW(6) t = {s['t_nw']:+.2f}  one-sample t = {s['t_1s']:+.2f}  "
          f"hit {s['hit_rate']:.3f}")
    pl = st.placebo_pvalue(panel, long_only=long_only, n_perm=3000)
    print(f"  placebo (block-rotation, {pl['n_draws']:,}): observed {pl['obs_bps']:+.2f} bps vs null "
          f"mean {pl['placebo_mean_bps']:+.2f} (sd {pl['placebo_sd_bps']:.2f}) -> p = {pl['p_value']:.4f}")
    edges = [("2007-2014", "2007-01-01", "2014-01-01"),
             ("2014-2020", "2014-01-01", "2020-01-01"),
             ("2020-2026", "2020-01-01", "2026-07-01")]
    sw = st.subperiod_sweep(panel, edges, long_only=long_only)
    for per, row in sw.iterrows():
        print(f"    {per}: {row['mean_bps']:+.2f} bps  NW t = {row['t_nw']:+.2f}  "
              f"Sharpe {row['sharpe']:+.3f}  (n={int(row['n'])})")
    for cb in (5.0, 10.0):
        tm = st.timer_stats(panel, long_only=long_only, cost_bps=cb, borrow_bps_yr=75.0)
        print(f"  cost {cb:>4.1f} bps one-way: gross {tm['gross_bps']:+.2f} -> net {tm['net_bps']:+.2f} "
              f"bps/mo (cost {tm['cost_bps_per_mo']:.2f}/mo, turnover {tm['avg_turnover']:.2f}, "
              f"net Sharpe {tm['net_sharpe']:+.3f}, t = {tm['t_net']:+.2f})")

print("\n# LOOKBACK ROBUSTNESS — long-only NW t across formation windows")
for lb in (6, 9, 12, 15):
    s = st.tsmom_stats(st.tsmom_returns(panel, lookback=lb, long_only=True))
    print(f"  lookback {lb:>2}m: {s['mean_bps']:+.2f} bps  Sharpe {s['sharpe']:.3f}  NW t = {s['t_nw']:+.2f}")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null = st.synthetic_mean_t(data, edge=0.0, n_seeds=20, base_seed=829)
planted = st.synthetic_mean_t(data, edge=0.010, n_seeds=20, base_seed=829)
print(f"  null (edge=0), 20 seeds:      mean NW t {null['mean_t']:+.2f}, |t|>=2 in "
      f"{null['fire_frac']*20:.0f}/20 seeds")
print(f"  planted (edge=0.010), 20 seeds: mean NW t {planted['mean_t']:+.2f}, mean Sharpe "
      f"{planted['mean_sharpe']:.2f}, |t|>=2 in {planted['fire_frac']*20:.0f}/20 seeds")
