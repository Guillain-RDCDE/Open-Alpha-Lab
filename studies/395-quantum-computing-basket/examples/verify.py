"""Reproducible headline run for Study 395 — Quantum-Computing-Basket.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached quantum panel under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_computing_basket import data, strategy as st

print("# Quantum-Computing-Basket — 4 pure-plays (IONQ/RGTI/QBTS/QUBT) vs SPY/QQQ/QTUM")
if data.have_real():
    rets, bench = data.load_real(allow_survivorship=True)
    fp = data.fingerprint(rets)
    span_years = len(rets) / 12.0
    print(f"panel months   : {len(rets)}  ({rets.index.min().date()} -> {rets.index.max().date()}, "
          f"{span_years:.1f} years)")
    print(f"panel names    : {list(rets.columns)}   fingerprint: {fp}")

    res = st.race(rets, bench, data.BASKET, cost_bps=10.0, allow_survivorship=True)
    s = res["summ"]
    print("\n# Risk-adjusted race — equal-weight, monthly-rebalanced (the hype signature)")
    print(f"  {'leg':24s} {'CAGR':>8} {'Sharpe':>7} {'vol':>7} {'maxDD':>8}")
    for key, label in [("basket", "Quantum basket (4 pure)"), ("etf", "QTUM ETF (diversified)"),
                       ("spy", "S&P 500 (SPY)"), ("qqq", "Nasdaq-100 (QQQ)")]:
        if key in s:
            d = s[key]
            print(f"  {label:24s} {d['cagr']*100:>7.1f}% {d['sharpe']:>7.2f} "
                  f"{d['vol']*100:>6.0f}% {d['max_dd']*100:>7.1f}%")
    dn = s["basket_net"]
    print(f"  {'  net @10bps one-way':24s} {dn['cagr']*100:>7.1f}% {dn['sharpe']:>7.2f}")

    print("\n# HAC (Newey-West) t-stat of the basket's monthly spread")
    for lab, key in [("vs SPY", "test_vs_spy"), ("vs QQQ", "test_vs_qqq"), ("vs QTUM", "test_vs_etf")]:
        t = res[key]
        print(f"  basket {lab:8s}: spread={t['mean_diff_ann']*100:>7.1f}%/yr  HAC t={t['tstat']:>5.2f}  n={t['n']}")

    print("\n# Single-name decomposition — broad theme or a few moon-shots?")
    print(f"  {'name':6s} {'CAGR':>9} {'vol':>6} {'maxDD':>7} {'excess/yr':>10} {'HAC t':>6}")
    for nm, d in res["single_names"].items():
        print(f"  {nm:6s} {d['cagr']*100:>8.1f}% {d['vol']*100:>5.0f}% {d['max_dd']*100:>6.0f}% "
              f"{d['mean_ann']*100:>9.1f}% {d['tstat']:>6.2f}")
else:
    print("(no _cache/quantum_panel.parquet — run "
          "data.load_real(fetch=True, allow_survivorship=True) once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must NOT manufacture a significant Sharpe/HAC t from a hype basket with NO")
print("  risk-adjusted edge (edge_sharpe=0), and MUST light up when a real edge is planted.")
print(f"  {'planted Sharpe edge':22s} {'basket CAGR':>12} {'Sharpe':>7} {'spread/yr':>10} {'HAC t':>6}")
for edge in (0.0, 1.0):
    rdf, b, truth = data.synthetic_panel(edge_sharpe=edge, seed=395)
    bench_df = b.to_frame("spy").assign(qqq=b.values)
    basket = st.basket_returns(rdf, truth["basket_cols"])
    sm = st.summarize(basket)
    t = st.hac_tstat_diff(basket, b)
    print(f"  edge_sharpe={edge:+.2f}        {sm['cagr']*100:>11.1f}% {sm['sharpe']:>7.2f} "
          f"{t['mean_diff_ann']*100:>9.1f}% {t['tstat']:>6.2f}")
