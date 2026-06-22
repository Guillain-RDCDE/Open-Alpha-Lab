"""Reproducible headline run for Study 351 — prints every number quoted in docs/results.md
and frozen into notebooks/build_notebooks.py (the ``R`` dict). Offline, deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from btc5m_polymarket import data, strategy as st

W = data.load_real()
o, cf = W[:, 0], W[:, 3]
print(f"# Real tape — Binance BTCUSDT 1m, folded into 5m windows")
print(f"windows        : {len(W)}")
print(f"days           : {len(W) * 5 / 60 / 24:.1f}")
print(f"P(up)          : {np.mean(cf > o):.4f}")
print(f"P(flat)        : {np.mean(cf == o):.5f}")
print(f"BTC range      : {o.min():.0f} - {o.max():.0f}")
print(f"median |2L move|: {np.median(np.abs(W[:,1]-o)):.1f}")
print(f"90pct |2L move|: {np.percentile(np.abs(W[:,1]-o),90):.1f}")

print("\n# Continuation win-rate w (2 minutes left)")
for r in st.sweep(W, [30, 50, 70, 85, 100, 120, 150], snap=st.SNAP_2MIN):
    print(f"  D>={r['threshold']:>3}$  n={r['n']:>5}  frac={r['frac']*100:4.1f}%  "
          f"w={r['w']:.4f}  z={r['z']:6.1f}  EV@0.85={st.ev_per_dollar(r['w'],0.85):+.3f}")
print("\n# Continuation win-rate w (1 minute left)")
for r in st.sweep(W, [70, 100], snap=st.SNAP_1MIN):
    print(f"  D>={r['threshold']:>3}$  n={r['n']:>5}  w={r['w']:.4f}  z={r['z']:6.1f}")

r70 = st.continuation(W, 70, st.SNAP_2MIN)
w70 = r70["w"]
print(f"\n# Monte-Carlo of '$300 -> $14,000' (w={w70:.4f} at D>=70, 2min left)")
scenarios = [
    ("fair price, bet 50%", round(w70, 3), 0.50),
    ("pay +0.05 over fair, bet 50%", round(min(w70 + 0.05, 0.99), 3), 0.50),
    ("buy 0.85 (true w), bet 50%", 0.85, 0.50),
    ("buy 0.85, bet 5% (YAML)", 0.85, 0.05),
]
for label, p, frac in scenarios:
    m = st.monte_carlo(w70, p, frac=frac)
    print(f"  [{label}] p={p:.3f} edge={m['edge']:+.3f}  "
          f"target={m['p_target']*100:5.2f}%  ruin={m['p_ruin']*100:5.2f}%  median=${m['median']:.2f}")

print("\n# Synthetic positive control — measured vs exact conditional expectation")
for sig in (30.0, 45.0, 60.0):
    S = data.synthetic_windows(n=200_000, sigma_per_min=sig, seed=351)
    meas = st.continuation(S, 70, st.SNAP_2MIN)["w"]
    exact = st.expected_continuation(70, sig)
    print(f"  sigma/min=${sig:.0f}  measured w={meas:.4f}  exact E[Phi]={exact:.4f}  "
          f"fair-price EV(p=w)={st.ev_per_dollar(meas, meas):+.4f}")

print("\n# Live Polymarket capture (real CLOB asks vs realised outcomes)")
rows = data.load_live()
b = st.live_buckets(rows)
for k in ("all", "signal", "strong"):
    s = b[k]
    if s["n"]:
        print(f"  {k:7s} n={s['n']:>3}  w={s['w']:.4f}  p={s['p']:.4f}  edge={s['edge']:+.4f}  EV={s['ev']:+.4f}")
    else:
        print(f"  {k:7s} n=0")
