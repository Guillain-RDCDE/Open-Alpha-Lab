"""Reproducible headline run for Study 384 — ISM-PMI-Regime.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached industrial-basket prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ism_pmi_regime import data, strategy as st

print("# ISM-PMI-Regime — PMI proxy = diffusion index of a 37-name industrial basket + SPY")
if data.have_real():
    f = data.load_real()
    span_years = (f.index.max() - f.index.min()).days / 365.25
    above_frac = (f["pmi"] > 50).mean()
    print(f"proxy months   : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span_years:.1f} years)")
    print(f"PMI proxy       : monthly share of basket with positive 3m momentum x100 "
          f"(PROXY for ISM Manufacturing PMI)")
    print(f"months PMI>50   : {above_frac*100:.1f}%  (manufacturing expands most of the time)")

    print("\n# Next-month SPY returns by PMI regime (1-month entry lag)")
    s = st.summarize(f)
    print(f"  above-50: n={s['n_above']:>3}  mean={s['above_mean']*100:>6.3f}%  "
          f"win={s['above_win']*100:>3.0f}%")
    print(f"  below-50: n={s['n_below']:>3}  mean={s['below_mean']*100:>6.3f}%  "
          f"win={s['below_win']*100:>3.0f}%")
    print(f"  base    : n={len(f)-1:>3}  mean={s['base_mean']*100:>6.3f}%  "
          f"win={s['base_win']*100:>3.0f}%")
    print(f"  spread (above-below) = {s['spread']*100:+.3f}%/mo  "
          f"Welch t = {s['t']:+.2f}  block-placebo p = {s['p_placebo']:.3f}")

    print("\n# Deployable timing rule: own SPY only when prior-month PMI>50, else cash")
    ts = st.timing_strategy(f, cost_bps=10.0)
    print(f"  switches        : {ts['switches']}  (one-way 10 bps each)")
    print(f"  timing net      : ann={ts['net_ann']*100:>6.2f}%  vol={ts['net_vol']*100:>5.1f}%  "
          f"Sharpe={ts['net_sharpe']:.2f}")
    print(f"  buy-and-hold    : ann={ts['bh_ann']*100:>6.2f}%  vol={ts['bh_vol']*100:>5.1f}%  "
          f"Sharpe={ts['bh_sharpe']:.2f}")

    print("\n# Robustness — shift the regime threshold")
    for thr in (45.0, 50.0, 55.0):
        s2 = st.summarize(f, threshold=thr)
        print(f"  thr={thr:.0f}: n_above={s2['n_above']:>3} n_below={s2['n_below']:>3}  "
              f"spread={s2['spread']*100:>+6.3f}%  t={s2['t']:>+5.2f}  p={s2['p_placebo']:.3f}")
else:
    print("(no _cache/manuf_prices.csv — run data.fetch_basket() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must NOT manufacture a regime signal from noise (edge=0 -> t<2), and")
print("  must recover a PLANTED above-50-only return edge (edge=+1.5%/mo -> t>2).")
for edge in (0.0, 0.015):
    syn = data.synthetic_regime(n_months=600, edge=edge, seed=384)
    s = st.summarize(syn)
    print(f"  planted edge={edge:+.3f}/mo: above={s['above_mean']*100:>6.3f}%  "
          f"below={s['below_mean']*100:>6.3f}%  spread={s['spread']*100:>+6.3f}%  "
          f"t={s['t']:>+5.2f}  p={s['p_placebo']:.3f}")
