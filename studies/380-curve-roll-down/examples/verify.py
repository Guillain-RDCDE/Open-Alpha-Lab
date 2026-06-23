"""Reproducible headline run for Study 380 — Curve-Roll-Down.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached CMT yields under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from curve_roll_down import data, strategy as st

print("# Curve-Roll-Down — 5y duration sleeve vs cash on the real Treasury curve")
if data.have_real():
    f = data.load_real(sleeve_mat=5.0)
    span = (f.index.max() - f.index.min()).days / 365.25
    print(f"curve days     : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span:.1f} years)")
    res = st.realized_excess(f, sleeve_mat=5.0)

    s = st.summarize_excess(res)
    print("\n# Realized excess-over-cash of the always-on 5y sleeve (1-day lag)")
    print(f"n entries      : {s['n']}")
    print(f"mean excess    : {s['mean_excess']*100:+.2f}% / yr   win-rate {s['win']*100:.0f}%   "
          f"vol {s['vol']*100:.2f}%   Sharpe {s['sharpe']:.2f}")
    print(f"HAC (NW) t      : {s['t_hac']:.2f}   |   naive t (wrong, overlap): {s['t_naive']:.1f}")

    p = st.promise_vs_reality(res)
    print("\n# Promise (textbook roll+carry) vs realized total return")
    print(f"promised roll+carry : {p['mean_promised']*100:+.2f}% / yr")
    print(f"realized total      : {p['mean_realized_total']*100:+.2f}% / yr")
    print(f"gap (curve moving)  : {p['mean_gap']*100:+.2f}pp")

    print("\n# Sub-period cut — the edge IS the secular bond bull")
    for lo, hi, lab in (("1990", "2010", "1990-2009"),
                        ("2010", "2027", "2010-2026"),
                        ("1990", "2027", "full     ")):
        sub = res.loc[lo:hi]
        ss = st.summarize_excess(sub)
        print(f"  {lab}: n={ss['n']:>4}  mean={ss['mean_excess']*100:>+5.2f}%  "
              f"Sharpe={ss['sharpe']:>5.2f}  meanDy={ss['mean_dy']:>+5.2f}  HAC_t={ss['t_hac']:>5.2f}")

    rg = st.regime_split(res)
    print("\n# Regime split — roll-down is a static-curve fiction")
    print(f"  yields FELL  (n={rg['n_down']:>4}): mean excess {rg['mean_down']*100:+.2f}%")
    print(f"  yields ROSE  (n={rg['n_up']:>4}): mean excess {rg['mean_up']*100:+.2f}%")

    pl = st.slope_timing_placebo(res)
    print("\n# Steep-curve placebo — steep beats random only via more carry")
    print(f"  steep-tercile entries (k={pl['k']}): {pl['steep_mean']*100:+.2f}%   "
          f"placebo mean {pl['placebo_mean']*100:+.2f}%   p={pl['p_value']:.3f}")

    c = st.net_of_costs(res)
    print("\n# Net of 2 bps on the annual roll")
    print(f"  gross {c['gross_mean']*100:+.2f}%   net {c['net_mean']*100:+.2f}%")

    print("\n# Sleeve-maturity robustness (full sample)")
    for sm in (2.0, 5.0, 10.0):
        fr = data.load_real(sleeve_mat=sm)
        r2 = st.realized_excess(fr, sleeve_mat=sm)
        s2 = st.summarize_excess(r2)
        print(f"  {sm:>4.0f}y: mean={s2['mean_excess']*100:>+5.2f}%  Sharpe={s2['sharpe']:.2f}  "
              f"HAC_t={s2['t_hac']:.2f}")
else:
    print("(no _cache/curve_yields.csv — run data.fetch_curve() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  edge=0 (carry baseline only) must NOT fire; a planted slope edge must light up.")
for edge in (0.0, 0.05):
    syn = data.synthetic_curve(n_days=6000, edge=edge, seed=380)
    r = st.synthetic_edge_test(syn)
    print(f"  planted edge={edge:+.2f}: recovered beta={r['beta']:+.4f}  HAC_t={r['t_hac']:>6.2f}")
