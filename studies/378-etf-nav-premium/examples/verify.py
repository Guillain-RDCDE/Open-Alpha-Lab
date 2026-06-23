"""Reproducible headline run for Study 378 — ETF NAV Premium/Discount.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF prices under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etf_nav_premium import data, strategy as st

print("# ETF NAV premium/discount — does buying below NAV earn the discount back?")
print("# NAV proxy = per-ETF hedge-basket fair value; basis = detrended cumulative residual")

if data.have_real():
    P = data.load_prices()
    frames = data.load_real()
    yrs = (P.index.max() - P.index.min()).days / 365.25
    print(f"\nwindow         : {P.index.min().date()} -> {P.index.max().date()}  "
          f"({len(P)} trading days, {yrs:.1f} years)")
    print(f"targets        : {list(frames)}  (proxy NAV from sister ETF + rate leg)")

    print("\n# Pooled hedged harvest after a discount (z<-1, 1-day lag)")
    print(f"  {'h':>3} {'n':>5} {'harvest':>9} {'win':>5} {'HAC_t':>7}")
    for h in (5, 10, 20):
        p = st.pooled_harvest(frames, h)
        print(f"  {h:>3} {p['n']:>5} {p['harvest_mean']*100:>8.4f}% "
              f"{p['win']*100:>4.0f}% {p['t']:>7.2f}")

    print("\n# Per-ETF, 10-day hold")
    print(f"  {'ETF':<4} {'n':>4} {'depth':>8} {'harvest':>9} {'win':>5} {'HAC_t':>7} {'placebo_p':>10}")
    for tg, f in frames.items():
        ev = st.detect_discounts(f)
        s = st.summarize(f, ev, 10)
        print(f"  {tg:<4} {s['n']:>4} {s['depth_mean']*100:>7.2f}% "
              f"{s['harvest_mean']*100:>8.4f}% {s['win']*100:>4.0f}% "
              f"{s['t']:>7.2f} {s['p_placebo']:>10.3f}")

    print("\n# Costs vs pooled gross (5-day) — round-trip on ETF + hedge leg")
    p5 = st.pooled_harvest(frames, 5)
    for cb in (1.0, 2.0, 3.0):
        c = st.net_of_costs(p5["harvest_mean"], cost_bps_per_leg=cb)
        print(f"  half-spread {cb:.0f}bp/leg: gross {c['gross']*100:>7.4f}%  "
              f"cost {c['cost']*100:.4f}%  net {c['net']*100:>+7.4f}%")

    print("\n# Robustness — z-threshold sweep, pooled 5-day")
    for zt in (-0.5, -1.0, -1.5, -2.0):
        p = st.pooled_harvest(frames, 5, z_thr=zt)
        print(f"  z<{zt:>4.1f}: n={p['n']:>5}  gross={p['harvest_mean']*100:>7.4f}%  "
              f"win={p['win']*100:>3.0f}%  t={p['t']:>5.2f}")

    print("\n# Stress clustering — share of deepest-5% discount-days in 2020 (uniform ~7%)")
    for tg, f in frames.items():
        deep = f["basis"][f["basis"] < f["basis"].quantile(0.05)]
        sh = (deep.index.year == 2020).mean()
        print(f"  {tg:<4} {sh*100:>3.0f}% in 2020  worst {deep.min()*100:>6.2f}% "
              f"on {deep.idxmin().date()}")
else:
    print("(no _cache/etf_prices.csv — run data.fetch_etfs() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must find discounts; inference must recover a PLANTED per-day earn-back")
print("  and must NOT manufacture significance when the true edge is 0.")
for edge in (0.0, 5.0):
    syn = data.synthetic_basis(edge_bps=edge, seed=378)
    ev = st.detect_discounts(syn)
    s = st.summarize(syn, ev, 10)
    print(f"  planted edge={edge:>3.0f} bp/day: detected={s['n']:>3}  "
          f"harvest={s['harvest_mean']*100:>+7.4f}%  t={s['t']:>5.2f}  "
          f"p_placebo={s['p_placebo']:.3f}")
