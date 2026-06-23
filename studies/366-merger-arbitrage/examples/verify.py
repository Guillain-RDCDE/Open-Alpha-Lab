"""Reproducible headline run for Study 366 — Merger-Arbitrage.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; the real deal book is reproducible offline (the
documented entry/pre-deal closes live in the deal table), and the synthetic control always
runs with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from merger_arbitrage import data, strategy as st

print("# Merger-Arbitrage — realized arb returns from a book of real all-cash US M&A deals")

f = data.load_real()
n_break = int((~f["completed"]).sum())
print(f"deals          : {len(f)}  (closed {len(f)-n_break}, broke {n_break})")
print(f"window         : announces {f['announce'].min().date()} -> "
      f"resolutions through {f['resolve'].max().date()}")
print(f"price confirm  : yfinance cache present = {data.have_real()} "
      f"(still-listed targets confirm documented marks)")

print("\n# The deal book")
print(f"  {'target':>6} {'done':>5} {'days':>5} {'entry':>9} {'offer':>9} "
      f"{'spread%':>8} {'ret%':>8}")
for _, d in f.iterrows():
    print(f"  {d['target']:>6} {str(bool(d['completed'])):>5} {int(d['days']):>5} "
          f"{d['entry']:>9.2f} {d['offer']:>9.2f} {d['spread']*100:>7.2f}% {d['ret']*100:>7.2f}%")

s = st.summarize(f)
comp = f[f["completed"]]
print("\n# The signal — what you 'see' vs what the tape pays")
print(f"  gross spread at entry (all)      : {f['spread'].mean()*100:>6.2f}%")
print(f"  spread on the deals that CLOSED  : {comp['spread'].mean()*100:>6.2f}%")
print(f"  mean realized return on BREAKS   : {s['mean_loss']*100:>6.2f}%")
print(f"  MEAN realized arb return (all)   : {s['mean']*100:>6.2f}%")
print(f"  median realized arb return       : {s['median']*100:>6.2f}%")
print(f"  win-rate (deals that paid)       : {s['win_rate']*100:>6.0f}%")
print(f"  return skew (fat left tail < 0)  : {s['skew']:>6.2f}")

print("\n# Is +1.79% real? — small-sample, fat-tail test")
print(f"  one-sample Welch t (mean vs 0)   : {s['t']:>6.2f}")
print(f"  bootstrap p(mean <= 0)           : {s['p_le0']:>6.3f}")
print(f"  bootstrap 90% CI of the mean     : [{s['boot_lo']*100:>+.2f}%, {s['boot_hi']*100:>+.2f}%]")

sa = st.summarize(f, annualize=True)
print(f"\n  (annualized mean {sa['mean']*100:.2f}%/yr, t={sa['t']:.2f} -- NOT a Signal-axis "
      f"statistic; annualizing short overlapping bets inflates t)")

print("\n# Costs barely move it (one-way bps on entry AND exit)")
for c in (5, 10, 25):
    nc = st.net_of_costs(f, cost_bps=c)
    print(f"  {c:>2}bps  gross={nc['gross_mean']*100:>5.2f}%  net={nc['net_mean']*100:>5.2f}%  "
          f"net_t={nc['net_t']:>5.2f}")

print("\n# Synthetic positive control — fair bet (edge=0) vs planted edge")
print("  break_prob=0.08, break_loss=-0.28, spread pinned to fair-insurance value")
for edge in (0.0, 0.02, 0.05):
    syn = data.synthetic_book(n_deals=250, edge=edge, seed=366)
    ss = st.summarize(syn)
    print(f"  edge={edge:+.2f}: mean={ss['mean']*100:>5.2f}%  t={ss['t']:>5.2f}  "
          f"p(<=0)={ss['p_le0']:.3f}  breaks={ss['n_breaks']:>2}")
