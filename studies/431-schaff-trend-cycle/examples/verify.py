"""Reproducible headline run for Study 431 — Schaff Trend Cycle (STC).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + panel daily tapes under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schaff_trend_cycle import data, strategy as st

print("# Schaff Trend Cycle — daily long/flat timing rule, net of 1 bp one-way x NAV, 1-day lag")

if data.have_real("SPY"):
    spy = data.load_real("SPY")
    print(f"\n# SPY data stamp: {spy.index.min().date()} -> {spy.index.max().date()}  "
          f"n={len(spy)}  fingerprint={data.fingerprint(spy)}")

    r = st.run_experiment(spy, cost_bps=1.0, long_short=False)
    print("\n# SPY long/flat (1 bp)")
    print(f"  STC net Sharpe   : {r['stc_sharpe']:.4f}")
    print(f"  buy&hold Sharpe  : {r['bh_sharpe']:.4f}   (delta {r['stc_minus_bh']:+.4f})")
    print(f"  plain MACD Sharpe: {r['macd_sharpe']:.4f}")
    print(f"  STC CAGR / B&H   : {r['stc_cagr']*100:.2f}% / {r['bh_cagr']*100:.2f}%")
    print(f"  STC maxDD / B&H  : {r['stc_maxdd']*100:.1f}% / {r['bh_maxdd']*100:.1f}%")
    print(f"  STC HAC t        : {r['stc_hac_t']:.4f}   (bar = 2.0)")
    print(f"  turnover/yr      : {r['turnover_per_yr']:.2f}   exposure {r['exposure']*100:.1f}%")

    rls = st.run_experiment(spy, cost_bps=1.0, long_short=True)
    print("\n# SPY long/short (1 bp)")
    print(f"  STC Sharpe {rls['stc_sharpe']:.4f}  MACD Sharpe {rls['macd_sharpe']:.4f}  "
          f"HAC t {rls['stc_hac_t']:.4f}  CAGR {rls['stc_cagr']*100:.2f}%  maxDD {rls['stc_maxdd']*100:.1f}%")

    print("\n# Block-permutation placebo (long/flat, 2000 draws)")
    stc = st.schaff_trend_cycle(spy["close"])
    pos = st.stc_positions(stc)
    pp = st.block_permutation_p(spy, pos, n_draws=2000, cost_bps=1.0, seed=431)
    print(f"  observed mean {pp['obs_mean']*1e4:+.3f} bps/day   placebo p = {pp['p_value']:.3f}")

    print("\n# Cost sweep (long/flat)")
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        rr = st.run_experiment(spy, cost_bps=c)
        print(f"  {c:>4.1f} bp: STC Sharpe {rr['stc_sharpe']:.3f}  HAC t {rr['stc_hac_t']:.3f}")

    print("\n# Robustness panel (long/flat, 1 bp)")
    print(f"  {'ticker':>6} {'n':>6} {'STC':>6} {'B&H':>6} {'MACD':>6} {'HAC t':>7}")
    for t in data.TICKERS:
        if not data.have_real(t):
            continue
        b = data.load_real(t)
        rr = st.run_experiment(b, cost_bps=1.0)
        print(f"  {t:>6} {rr['n_days']:>6} {rr['stc_sharpe']:>6.2f} {rr['bh_sharpe']:>6.2f} "
              f"{rr['macd_sharpe']:>6.2f} {rr['stc_hac_t']:>7.2f}")
else:
    print("(no _cache/stc_SPY_1d.parquet — run data.load_real('SPY', fetch=True) once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  no trend  -> STC must NOT beat buy-and-hold (no false positive)")
print("  planted trend -> STC must beat buy-and-hold and clear t=2 (engine can bank an edge)")
for tr in (0.0, 0.003):
    b, truth = data.synthetic_panel(trend=tr)
    rr = st.run_experiment(b, cost_bps=1.0)
    print(f"  trend={tr:>5}: STC Sharpe {rr['stc_sharpe']:>6.2f}  B&H {rr['bh_sharpe']:>6.2f}  "
          f"delta {rr['stc_minus_bh']:+.2f}  HAC t {rr['stc_hac_t']:>5.2f}")
