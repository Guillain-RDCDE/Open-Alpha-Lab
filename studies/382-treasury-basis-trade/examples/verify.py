"""Reproducible headline run for Study 382 — Treasury cash-futures Basis Trade.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basis inputs under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from treasury_basis_trade import data, strategy as st

print("# Treasury cash-futures Basis Trade — transparent carry / implied-repo MODEL")
print("# net carry = 10y yield (^TNX) - funding (^IRX, repo proxy); MtM = residual basis wobble")

if data.have_real():
    F = data.load_real()
    s = st.summarize(F)
    print(f"\nmodel days   : {s['n']}  ({F.index.min().date()} -> {F.index.max().date()}, "
          f"{s['years']:.1f} years)")
    print(f"mean 10y/funding/carry : {F['yield10'].mean():.2f}% / {F['funding'].mean():.2f}% / "
          f"{F['net_carry'].mean():.2f}%   (carry>0 on {(F['net_carry']>0).mean()*100:.1f}% of days)")

    print("\n# Signal (leverage-invariant) — is the carry real?")
    print(f"  unlevered ann return : {s['ann_return']*100:.2f}%")
    print(f"  HAC (Newey-West, lag=21) t of the mean : {s['hac_t']:.2f}   (need >=2 for REAL)")
    print(f"  annualised Sharpe : {s['sharpe']:.2f}   (Sharpe-t ~ {s['sharpe_t']:.1f})")
    pl = st.placebo_pvalue(F, n_draws=5000)
    print(f"  sign-flip placebo : obs Sharpe {pl['obs_sharpe']:.2f}, "
          f"placebo mean {pl['placebo_sharpe']:.2f}, p = {pl['p_value']:.3f}")

    print("\n# Tradability — the SAME Sharpe at the leverage the trade actually runs")
    print(f"  {'lev':>4} {'ann_ret':>9} {'Sharpe':>7} {'maxDD':>9} {'worst_day':>10} {'CVaR1%':>8}")
    for row in st.leverage_table(F, leverages=(1, 50, 100)):
        print(f"  {row['leverage']:>4} {row['ann_return']*100:>8.1f}% {row['sharpe']:>7.2f} "
              f"{row['max_dd']*100:>8.1f}% {row['worst_day']*100:>9.1f}% {row['cvar']*100:>7.1f}%")

    print("\n# Tail diagnostics (unlevered)")
    t = st.tail_stats(F["ret"].values)
    print(f"  skew {t['skew']:.2f}  excess-kurtosis {t['exkurt']:.1f}  "
          f"worst day {t['worst_day']*100:.3f}%  worst 5-day {t['worst_5day']*100:.3f}%")

    print("\n# Funding-shock stress — March 2020 (repo spike / basis dislocation)")
    fs = st.funding_shock(F, "2020-03-09", "2020-03-20", leverages=(1, 50, 100))
    print(f"  {fs['start']} -> {fs['end']} ({fs['n_days']}d): unlev {fs['unlev']*100:.2f}%  "
          f"| L50 {fs['L50']*100:.1f}%  | L100 {fs['L100']*100:.1f}%")

    print("\n# Costs barely matter (quarterly roll, 1 bp one-way) — the tail kills it, not cost")
    c = st.net_of_costs(F, leverage=50.0)
    print(f"  L=50: gross {c['gross_ann']*100:.1f}%/yr  net {c['net_ann']*100:.1f}%/yr  "
          f"(roll cost {c['cost_ann']*100:.1f}%/yr)")
else:
    print("(no _cache/basis_inputs.csv — run data.fetch_inputs() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  harness must recover a PLANTED unlevered Sharpe, find NOTHING when the edge is 0,")
print("  and show Sharpe is leverage-invariant while the drawdown explodes.")
for edge in (0.0, 2.5):
    syn = data.synthetic_basis(edge_sharpe=edge, seed=382)
    r = syn["ret"].values
    print(f"  planted Sharpe={edge:.1f}: measured={st.sharpe(r):.2f}  HAC_t={st.hac_t(r):.2f}  "
          f"| Sharpe(L1)={st.sharpe(r):.2f}=Sharpe(L50)={st.sharpe(50*r):.2f}  "
          f"| maxDD(L1)={st.max_drawdown(r)*100:.0f}% -> maxDD(L50)={st.max_drawdown(50*r)*100:.0f}%")
