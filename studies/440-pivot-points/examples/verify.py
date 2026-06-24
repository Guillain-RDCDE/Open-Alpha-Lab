"""Reproducible headline run for Study 440 — Floor-Trader Pivot Points.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached yfinance 5-minute basket bars
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic positive
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pivot_points import data, strategy as st  # noqa: E402

TOL_BPS = 5.0     # touch tolerance (bps of the level)
K = 6             # bounce horizon in 5-minute bars (= 30 min)
COST_BPS = 1.0    # one-way intraday cost; round trip = 2x

print("# Floor-trader pivots — do they act as support/resistance? (yfinance 5m, ~60d basket)")

if data.have_real():
    cache = data.load_real()
    piv_all, ctl_all = [], []
    spans = {}
    for t, b in cache.items():
        b = st.drop_partial_sessions(b)
        spans[t] = (b["date"].min().date(), b["date"].max().date(),
                    b["date"].nunique(), data.fingerprint(b))
        piv_all.append(st.pivot_touch_events(b, tol_bps=TOL_BPS, k=K))
        ctl_all.append(st.control_touch_events(b, tol_bps=TOL_BPS, k=K, seed=440))
    P = pd.concat(piv_all, ignore_index=True)
    C = pd.concat(ctl_all, ignore_index=True)

    print("\n# Per-name 5-minute tape stamp (partial/live session dropped)")
    for t, (lo, hi, n, fp) in spans.items():
        print(f"  {t:>4}: {lo} -> {hi}  ({n} sessions)  fingerprint={fp}")

    pt = st.trade_stats(P, cost_bps=COST_BPS)
    ct = st.trade_stats(C, cost_bps=COST_BPS)
    pr, cr = st.bounce_rate(P), st.bounce_rate(C)

    print("\n# POOLED across the basket — pivot touches vs randomly-placed control lines")
    print(f"  pivot   : touches={pt['n']:5d}  bounce_rate={pr:.3f}  "
          f"mean={pt['mean_bps']:+.2f} bps  HAC t={pt['t']:+.2f}  win={pt['win']:.3f}")
    print(f"  control : touches={ct['n']:5d}  bounce_rate={cr:.3f}  "
          f"mean={ct['mean_bps']:+.2f} bps  HAC t={ct['t']:+.2f}")
    print(f"  pivot - control bounce-rate diff: {pr - cr:+.3f}")
    print(f"  pivot fade trade net of 2x{COST_BPS}bps round trip: {pt['net_bps']:+.2f} bps")

    print("\n# Bounce rate by pivot level (low support -> high resistance)")
    for nm in st.ALL_LEVELS:
        sub = P[P["level_name"] == nm]
        if len(sub):
            print(f"  {nm:>3}: n={len(sub):4d}  rate={st.bounce_rate(sub):.3f}  "
                  f"mean={sub['bounce'].mean() * 1e4:+.2f} bps")

    print("\n# Per-name run_experiment (pivot vs control, with permutation placebo)")
    for t, b in cache.items():
        b = st.drop_partial_sessions(b)
        r = st.run_experiment(b, tol_bps=TOL_BPS, k=K, cost_bps=COST_BPS, n_placebo=200)
        print(f"  {t:>4}: pivot_rate={r['pivot_bounce_rate']:.3f}  "
              f"ctrl_rate={r['control_bounce_rate']:.3f}  diff={r['rate_diff']:+.3f}  "
              f"mean={r['pivot_mean_bps']:+.2f}bps  t={r['pivot_t']:+.2f}  "
              f"net={r['pivot_net_bps']:+.2f}bps  placebo_p={r['placebo_p']:.3f}")

    print("\n# Pooled permutation placebo — could a RANDOM line match the pivots' bounce rate?")
    bks = [st.drop_partial_sessions(b) for b in cache.values()]
    ND = 300
    rates = np.empty(ND)
    for i in range(ND):
        evs = [st.control_touch_events(b, tol_bps=TOL_BPS, k=K,
                                       seed=10_000 + i + 13 * j) for j, b in enumerate(bks)]
        rates[i] = st.bounce_rate(pd.concat(evs, ignore_index=True))
    p = float((rates >= pr).mean())
    print(f"  control bounce-rate over {ND} redraws: mean={rates.mean():.3f} sd={rates.std():.4f}")
    print(f"  p(control_rate >= pivot_rate {pr:.3f}) = {p:.3f}   "
          f"(>0.5 means a random line is at least as good as the pivots)")
else:
    print("(no _cache/bars_*_5m.parquet — run data.fetch_real(tk, fetch=True) once per name)")

print("\n# Synthetic positive control — deterministic, no network")
print("  edge=0 must NOT beat the random control; a planted bounce must light up.")
for edge in (0.0, 1.5):
    b, _ = data.synthetic_panel(edge=edge, seed=440)
    r = st.run_experiment(b, tol_bps=TOL_BPS, k=K, cost_bps=COST_BPS, n_placebo=150)
    print(f"  planted edge={edge:>4}: touches={r['n_pivot_touches']:4d}  "
          f"pivot_rate={r['pivot_bounce_rate']:.3f}  ctrl_rate={r['control_bounce_rate']:.3f}  "
          f"mean={r['pivot_mean_bps']:+.2f}bps  t={r['pivot_t']:+.2f}  "
          f"win={r['pivot_win']:.3f}  placebo_p={r['placebo_p']:.3f}")
