"""Reproducible headline run for Study 849 — Dry January / Veganuary.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached daily panel under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from dry_january import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Dry January / Veganuary — does the January cultural wave move the stocks?")

if not data.have_real():
    print("(cache miss — fetching the daily tape once)")
    data.fetch()

panel = data.load_panel()
closes = data.load_closes()
print(f"[data] {panel.shape[1]} tickers, {len(panel)} rows  "
      f"{panel.index.min().date()} -> {panel.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(panel)}")
print(f"  groups: alcohol={data.ALCOHOL}  plant={data.PLANT}  staples={data.STAPLES}  "
      f"bench={data.BENCH}")

abn = st.build_abnormals(closes)

print("\n# THE HEADLINE — January & February abnormal return (group - SPY), one obs/year")
for grp in ("alcohol", "plant", "staples", "plant_minus_alcohol"):
    ar = abn[grp]
    jan = st.month_stats(ar, month=1)
    feb = st.month_stats(ar, month=2)
    print(f"  {grp:20s} JAN n={jan['n']:2d} mean={jan['mean_pct']:+.2f}%  "
          f"t_1s={jan['t_1s']:+.2f}  NW t(dummy)={jan['t_nw']:+.2f}  "
          f"hit={jan['hit_k']}/{jan['hit_n']}")
    print(f"  {'':20s} FEB n={feb['n']:2d} mean={feb['mean_pct']:+.2f}%  "
          f"t_1s={feb['t_1s']:+.2f}  NW t(dummy)={feb['t_nw']:+.2f}")

print("\n# PLACEBO — is January special or one of twelve calendar months?")
for grp, tail in (("alcohol", "left"), ("plant", "right"), ("plant_minus_alcohol", "right")):
    pl = st.month_placebo(abn[grp], target=1, tail=tail)
    print(f"  {grp:20s} Jan mean {pl['target_mean_pct']:+.2f}%  rank {pl['rank']}/12  "
          f"({tail}-tail p = {pl['p_value']:.3f})")

print("\n# ROBUSTNESS — alcohol January abnormal return, two eras (split 2013)")
jan = abn["alcohol"]
jan = jan[jan.index.month == 1]
for lo, hi, lbl in [(0, 2013, "1999-2012"), (2013, 9999, "2013-2026")]:
    x = jan[(jan.index.year >= lo) & (jan.index.year < hi)].to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    print(f"  {lbl}: n={len(x):2d}  mean {np.mean(x)*100:+.2f}%  t = {st.one_sample_t(x):+.2f}")

print("\n# THE TIMER — long-plant / short-alcohol every January, costed")
for cb in (5.0, 10.0):
    tm = st.timer_stats(closes, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/leg: n={tm['n_years']}  gross {tm['gross_pct']:+.2f}% -> "
          f"net {tm['net_pct']:+.2f}% (cost {tm['cost_pct']:.2f}%, t_net = {tm['t_net']:+.2f}, "
          f"hit {tm['hit_k']}/{tm['hit_n']})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network (>=20 seeds)")
for edge in (0.0, 0.02, 0.05):
    r = st.synthetic_mean_t(data, edge=edge, n_seeds=20)
    print(f"  edge={edge:.2f}: mean NW t = {r['mean_t_nw']:+.2f}  "
          f"mean beta = {r['mean_beta_pct']:+.2f}%  |t|>=2 in {r['fire_frac']*100:.0f}% of seeds")
