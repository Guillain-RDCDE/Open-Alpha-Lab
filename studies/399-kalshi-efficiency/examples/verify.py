"""Reproducible headline run for Study 399 — Kalshi-Efficiency.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached illustrative contract book under
``_cache/`` if present (the "real"-tape numbers — an ILLUSTRATION, not live Kalshi), and always
runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalshi_efficiency import data, strategy as st

print("# Kalshi-Efficiency — calibration of a binary event-contract book")
print("# NOTE: the 'real' tape is an ILLUSTRATION (Kalshi resolved history is not free).")

if data.have_real():
    f = data.load_real()
    print(f"\ncontracts      : {len(f)}  (YES base rate {f['outcome'].mean()*100:.1f}%, "
          f"baked favourite-longshot edge = {data.ILLUSTRATIVE_EDGE})")

    print("\n# Calibration curve (mean price vs realized YES frequency per decile)")
    cc = data.__dict__  # noqa: F841  (keep import side-effects explicit)
    curve = st.calibration_curve(f)
    print(f"  {'bucket':>8} {'mean_price':>10} {'freq_yes':>9} {'n':>5} {'gap':>8}")
    for _, r in curve.iterrows():
        print(f"  {r['lo']:.1f}-{r['hi']:.1f}  {r['mean_price']:>9.3f} "
              f"{r['freq_yes']:>9.3f} {int(r['n']):>5} {r['freq_yes']-r['mean_price']:>+8.3f}")

    bd = st.brier_decomposition(f)
    print(f"\n# Brier decomposition: brier={bd['brier']:.4f}  reliability={bd['reliability']:.5f}  "
          f"resolution={bd['resolution']:.4f}  uncertainty={bd['uncertainty']:.4f}")

    print("\n# Harvestable spread (long favourites >=0.80, short longshots <=0.20) vs fee")
    print(f"  {'fee':>5} {'n':>5} {'gross_c':>8} {'net_c':>7} {'win':>5} {'t':>6} {'p_rand':>7}")
    for fee in (0.0, 0.01, 0.02, 0.03):
        s = st.summarize(f, fee=fee)
        print(f"  {fee:>5.2f} {s['n']:>5} {s['gross_mean']*100:>+7.2f} "
              f"{s['net_mean']*100:>+6.2f} {s['win_rate']*100:>4.0f}% {s['t']:>6.2f} {s['p_rand']:>7.3f}")
else:
    print("(no _cache/kalshi_contracts.csv — run data.fetch_real() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must be faithful: edge=0 stays quiet (no false positive, fee=pure cost);")
print("  a large planted favourite-longshot edge must light up.")
print(f"  {'edge':>5} {'reliab_e4':>10} {'n':>5} {'gross_c':>8} {'net_c':>7} {'win':>5} {'t':>6} {'p_rand':>7}")
for edge in (0.0, 0.08):
    syn = data.synthetic_book(edge=edge, seed=399)
    s = st.summarize(syn)
    print(f"  {edge:>5.2f} {s['reliability']*1e4:>10.2f} {s['n']:>5} "
          f"{s['gross_mean']*100:>+7.2f} {s['net_mean']*100:>+6.2f} "
          f"{s['win_rate']*100:>4.0f}% {s['t']:>6.2f} {s['p_rand']:>7.3f}")
