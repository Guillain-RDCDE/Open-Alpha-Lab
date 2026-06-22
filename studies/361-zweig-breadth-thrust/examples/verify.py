"""Reproducible headline run for Study 361 — Zweig Breadth Thrust.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zweig_breadth_thrust import data, strategy as st

print("# Zweig Breadth Thrust — breadth proxy from a 40-name large-cap basket + SPY")
if data.have_real():
    f = data.load_real()
    span_years = (f.index.max() - f.index.min()).days / 365.25
    print(f"proxy days     : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span_years:.1f} years)")
    print(f"basket names   : breadth = fraction of basket advancing each day (PROXY for NYSE A/D)")

    ev = st.detect_thrusts(f)
    print(f"\n# Thrust events detected (10d EMA of adv-ratio: <0.40 -> >0.615 within 10d)")
    print(f"n events       : {len(ev)}  (~one per {span_years/len(ev):.1f} years)")
    for d in ev:
        print(f"  {d.date()}")

    print("\n# Forward SPY returns after each thrust vs unconditional (1-day entry lag)")
    print(f"  {'H':>4} {'n':>3} {'cond_mean':>10} {'cond_win':>9} {'base_mean':>10} "
          f"{'base_win':>9} {'Welch_t':>8} {'p_placebo':>10}")
    for m in (3, 6, 12):
        s = st.summarize(f, ev, m)
        print(f"  {str(m)+'m':>4} {s['n']:>3} {s['cond_mean']*100:>9.2f}% "
              f"{s['cond_win']*100:>8.0f}% {s['base_mean']*100:>9.2f}% "
              f"{s['base_win']*100:>8.0f}% {s['t']:>8.2f} {s['p_placebo']:>10.3f}")

    print("\n# Net of one round-trip cost (buy-on-thrust, hold H months)")
    for m in (3, 6, 12):
        c = st.net_of_costs(f, ev, m, cost_bps=5.0)
        print(f"  {m:>2}m  n_trades={c['n_trades']:>2}  gross={c['gross_mean']*100:>6.2f}%  "
              f"net@5bps={c['net_mean']*100:>6.2f}%")

    print("\n# Robustness — shift the upper threshold")
    for hi in (0.61, 0.615, 0.65):
        e2 = st.detect_thrusts(f, hi=hi)
        s = st.summarize(f, e2, 12)
        print(f"  hi={hi:.3f}: n={len(e2):>2}  cond12={s['cond_mean']*100:>5.1f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/basket_prices.csv — run data.fetch_basket() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must find injected thrusts; inference must recover a PLANTED edge and")
print("  must NOT manufacture significance from ~a dozen events when the true edge is 0.")
for edge in (0.0, 0.10):
    syn = data.synthetic_breadth(n_events=12, edge_6m=edge, seed=361)
    ev_s = st.detect_thrusts(syn, cooldown=120)
    s6 = st.summarize(syn, ev_s, 6)
    print(f"  planted edge_6m={edge:+.2f}: detected={len(ev_s):>2}  "
          f"cond6={s6['cond_mean']*100:>6.2f}%  base6={s6['base_mean']*100:>5.2f}%  "
          f"t={s6['t']:>5.2f}  p_placebo={s6['p_placebo']:.3f}")
