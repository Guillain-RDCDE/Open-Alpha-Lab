"""Reproducible headline run for Study 352 — Opening-Range Breakout (ORB).

Prints every number quoted in docs/results.md and frozen into notebooks/build_notebooks.py
(the ``R`` dict). The real-tape block reads the cached yfinance 5-minute bars under ``_cache/``;
the synthetic positive-control block runs anywhere with no network. Deterministic (fixed seeds).

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opening_range_breakout import data, strategy as st

SYM = "QQQ"
N_ITER = 2000

days = data.build_days(SYM)
o, c = days[:, 0, st.O], days[:, -1, st.C]
print(f"# Real tape — yfinance {SYM} 5-minute bars, regular session, folded into days")
print(f"symbol         : {SYM}")
print(f"days           : {len(days)}")
print(f"bars/day       : {days.shape[1]}")
print(f"buy&hold mean  : {np.mean(c / o - 1) * 1e4:+.1f} bps/day")
print(f"up-days        : {np.mean(c > o) * 100:.0f}%")

print("\n# ORB vs same-exposure RANDOM entry (gross, no cost)")
print("# null modes: matched_dir (same side, random time) | coin (random side+time) | drift (always long)")
for orb in (5, 15):
    for mode in ("matched_dir", "coin", "drift"):
        r = st.evaluate(days, opening_range_min=orb, cost_bps=0.0,
                        null_mode=mode, n_iter=N_ITER)
        print(f"  ORB{orb:>2}m null={mode:11s} trades={r['n_trades']:>2} "
              f"orb={r['orb_mean_bps']:+6.1f}bps null={r['null_mean_bps']:+6.1f}bps "
              f"edge={r['edge_bps']:+6.1f}bps  t_vs0={r['t_vs_zero']:+5.2f} "
              f"t_paired={r['t_paired']:+5.2f} hit={r['orb_hit']*100:3.0f}% perm_p={r['perm_p']:.3f}")

print("\n# Cost sweep — ORB net edge over the matched-direction null (one-way bps, ORB15m)")
for r in st.cost_sweep(days, opening_range_min=15, costs=(0.0, 1.0, 2.0, 3.0, 5.0),
                       null_mode="matched_dir"):
    print(f"  cost={r['cost_bps']:>3.0f}bps  orb_net={r['orb_mean_bps']:+6.1f}bps  "
          f"edge={r['edge_bps']:+6.1f}bps  t_vs0={r['t_vs_zero']:+5.2f}")
be = st.breakeven_cost(days, opening_range_min=15, null_mode="matched_dir")
print(f"  break-even one-way cost vs null: {be:+.2f} bps  (negative => no cost makes ORB win)")

print("\n# Synthetic positive control — planted intraday trend-persistence (ORB15m, matched-dir null)")
print("# the harness MUST recover a positive, significant ORB-minus-random edge as persistence rises")
for pers in (0.0, 0.25, 0.40):
    S = data.synthetic_days(n_days=800, persistence=pers, seed=352)
    r = st.evaluate(S, opening_range_min=15, cost_bps=0.0, null_mode="matched_dir", n_iter=N_ITER)
    tag = "random walk (edge ~0 expected)" if pers == 0 else "trend planted (edge > 0 expected)"
    print(f"  persistence={pers:.2f}  n={r['n_trades']:>3}  orb={r['orb_mean_bps']:+7.1f}bps  "
          f"null={r['null_mean_bps']:+6.1f}bps  edge={r['edge_bps']:+7.1f}bps  "
          f"t_paired={r['t_paired']:+5.2f}  perm_p={r['perm_p']:.3f}   {tag}")

print("\n# Cross-check on SPY (same conclusion expected)")
spy = data.build_days("SPY")
r = st.evaluate(spy, opening_range_min=15, cost_bps=0.0, null_mode="matched_dir", n_iter=N_ITER)
print(f"  SPY ORB15m: trades={r['n_trades']} orb={r['orb_mean_bps']:+.1f}bps "
      f"edge={r['edge_bps']:+.1f}bps t_vs0={r['t_vs_zero']:+.2f} t_paired={r['t_paired']:+.2f}")
