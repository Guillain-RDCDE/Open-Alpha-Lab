"""Reproducible headline run for Study 356 — the GLP-1 / Ozempic basket.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict at the top of
notebooks/build_notebooks.py. Reads the cached yfinance panel (offline); deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from glp1_basket import data, strategy as st

P = data.load_real()
R = data.daily_returns(P)
basket = data.basket_returns(R)

print("# Real tape — yfinance adjusted close (total return), 2018-2026")
print(f"window        : {P.index.min().date()} -> {P.index.max().date()}  ({len(R)} daily returns)")
print(f"NVO peak      : {P['NVO'].max():.1f} on {P['NVO'].idxmax().date()} "
      f"-> {P['NVO'].iloc[-1]:.1f}  ({(P['NVO'].iloc[-1]/P['NVO'].max()-1)*100:.0f}% from peak)")

print("\n# Performance (total return; rf=0); t_naive is NOT the Signal test")
for name, x in [("LLY", R["LLY"]), ("NVO", R["NVO"]), ("basket 50/50", basket),
                ("XLV", R["XLV"]), ("SPY", R["SPY"])]:
    s = st.perf(x)
    print(f"  {name:12s} CAGR={s['cagr']*100:6.1f}%  vol={s['vol']*100:5.1f}%  "
          f"Sharpe={s['sharpe']:5.2f}  total={s['total']*100:8.1f}%  maxDD={st.max_drawdown(x)*100:6.1f}%")

print("\n# SIGNAL — mean EXCESS return vs benchmark, HAC (Newey-West) t  [>=2 to be REAL]")
for bench in ("SPY", "XLV"):
    e = st.excess_test(basket, R[bench])
    print(f"  basket - {bench}: ann excess={e['ann_excess']*100:6.2f}%  HAC-t={e['t_hac']:5.2f}  (lag={e['lag']})")
for bench in ("SPY", "XLV"):
    mm = st.market_model(basket, R[bench])
    print(f"  market-model basket~{bench}: alpha={mm['alpha_ann']*100:6.2f}%/yr  "
          f"HAC-t(alpha)={mm['t_alpha']:5.2f}  beta={mm['beta']:4.2f}  R2={mm['r2']:.2f}")

bb = st.block_bootstrap_excess(basket, R["XLV"])
print(f"  block-bootstrap basket-XLV excess: {bb['point']*100:.1f}%  "
      f"95% CI [{bb['lo']*100:.1f}%, {bb['hi']*100:.1f}%]  P(>0)={bb['p_pos']:.3f}")

print("\n# MYTH-CHECK — where does the alpha live? (single-name market model vs SPY)")
for nm in ("LLY", "NVO"):
    mm = st.market_model(R[nm], R["SPY"])
    print(f"  {nm} ~ SPY: alpha={mm['alpha_ann']*100:6.2f}%/yr  HAC-t(alpha)={mm['t_alpha']:5.2f}  beta={mm['beta']:4.2f}")

print("\n# RECENCY — split at 2023-01-01 (the GLP-1 mania starts)")
for lbl, mask in [("2018-2022", R.index < "2023-01-01"),
                  ("2023-2026", R.index >= "2023-01-01")]:
    b = basket[mask]
    e = st.excess_test(b, R["SPY"][mask])
    print(f"  {lbl}: basket CAGR={st.perf(b)['cagr']*100:6.1f}%  "
          f"SPY={st.perf(R['SPY'][mask])['cagr']*100:6.1f}%  "
          f"ann excess={e['ann_excess']*100:6.1f}%  HAC-t={e['t_hac']:5.2f}  n={e['n']}")

print("\n# TRADABILITY — one-way cost sweep on rebalancing turnover (vs XLV)")
print(f"  mean one-way turnover/day = {st.turnover_per_day(R)*100:.3f}% of NAV")
for row in st.cost_sweep(basket, R["XLV"], R):
    print(f"  {row['bps']:>2} bps: net ann excess={row['ann_excess']*100:6.2f}%  HAC-t={row['t_hac']:5.2f}")

print("\n# SYNTHETIC positive control — engine recovers planted alpha (and its absence)")
S = data.synthetic_panel()
for nm, planted in [("LLY", 0.0010 * 252), ("NVO", 0.0), ("XLV", 0.0002 * 252)]:
    mm = st.market_model(S[nm], S["SPY"])
    print(f"  {nm}: planted alpha={planted*100:5.1f}%/yr  recovered={mm['alpha_ann']*100:5.1f}%/yr  "
          f"HAC-t={mm['t_alpha']:5.2f}")
