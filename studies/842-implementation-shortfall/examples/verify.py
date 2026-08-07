"""Reproducible headline run for Study 842 — Implementation Shortfall (the cost gap).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic and fully offline — a synthetic-only
research-method demo (no real tape, no network, no cache).

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from cost_gap import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

# The frozen configuration (the Fingerprint pins it).
EDGE = 0.0005
PHI = 0.96
N_ASSETS = 30
N_DAYS = 2520
FRAC = 0.2
SEED = 842

print("# Implementation Shortfall — the paper-vs-live cost gap")
print(f"[config] synthetic panel: {N_ASSETS} names x {N_DAYS} days, edge={EDGE}, "
      f"persistence(phi)={PHI}, frac={FRAC}, seed={SEED}  as-of {data.AS_OF}")

rets, sig, truth = data.synthetic_panel(edge=EDGE, persistence=PHI, n_assets=N_ASSETS,
                                        n_days=N_DAYS, seed=SEED)
null_rets, null_sig, _ = data.synthetic_panel(edge=0.0, persistence=PHI, n_assets=N_ASSETS,
                                              n_days=N_DAYS, seed=SEED)
print(f"[data] returns fingerprint = {data.fingerprint(rets)}  "
      f"null fingerprint = {data.fingerprint(null_rets)}")

book = st.book_returns(rets, sig, frac=FRAC)
base = st.book_stats(book, cost_bps=0.0, impact_coef_bps=0.0)
print(f"\n# THE PAPER PORTFOLIO — 0 cost (n={base['n_days']} days, "
      f"mean turnover {base['mean_turnover']:.3f}/day)")
print(f"  gross spread : {base['gross_bps']:+.2f} bps/day  NW(10) t = {base['gross_t']:+.2f}  "
      f"gross Sharpe = {base['gross_sharpe']:.2f}  (~{base['ann_gross_pct']:+.1f}%/yr)")

print("\n# THE COST LADDER — same strategy, 0 / realistic / stressed")
ladder = st.cost_ladder(book)
for scen, row in ladder.iterrows():
    print(f"  {scen:<22}: gross Sharpe {row['gross_sharpe']:5.2f}  net Sharpe "
          f"{row['net_sharpe']:6.2f}  net {row['net_bps']:+6.2f} bps/day  "
          f"(cost {row['cost_bps_per_day']:.2f}/day, t={row['net_t']:+.2f}, "
          f"{row['ann_net_pct']:+.1f}%/yr)")

be = st.breakeven_cost_bps(book)
print(f"\n# BREAK-EVEN one-way cost (linear only): {be:.2f} bps  "
      f"(the deceptive 'headroom' a naive backtest sees; impact is NOT in it)")

print("\n# THE TURNOVER CURVE — alpha dies as a FUNCTION of turnover (gross fixed)")
print("  (persistence phi down => turnover up; net Sharpe at realistic 10bp + impact 50)")
tc = st.turnover_curve(data, edge=EDGE, n_assets=N_ASSETS, n_days=N_DAYS,
                       frac=FRAC, cost_bps=10.0, impact_coef_bps=50.0, seed=SEED)
for phi, row in tc.iterrows():
    print(f"  phi={phi:<5}: turnover {row['mean_turnover']:.3f}/day  gross Sharpe "
          f"{row['gross_sharpe']:4.2f}  net Sharpe {row['net_sharpe']:6.2f}  "
          f"break-even {row['breakeven_bps']:5.2f} bps")

print("\n# SYNTHETIC CONTROL — the machinery is unbiased (gross-book HAC t)")
null_c = st.seed_robust_control(data, edge=0.0, persistence=PHI, n_seeds=20,
                                n_assets=N_ASSETS, n_days=1500)
edge_c = st.seed_robust_control(data, edge=EDGE, persistence=PHI, n_seeds=20,
                                n_assets=N_ASSETS, n_days=1500)
print(f"  null  (edge=0)   : gross t mean {null_c['mean_t']:+.2f} (sd {null_c['sd_t']:.2f}), "
      f"|t|>=2 in {null_c['fire_count']}/{null_c['n_seeds']} seeds")
print(f"  planted (edge={EDGE}): gross t mean {edge_c['mean_t']:+.2f} (sd {edge_c['sd_t']:.2f}), "
      f"|t|>=2 in {edge_c['fire_count']}/{edge_c['n_seeds']} seeds")
