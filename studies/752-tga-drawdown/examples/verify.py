"""Reproducible headline run for Study 752 — TGA-Drawdown.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the hardcoded TGA proxy (always
available) and the cached SPY prices under ``_cache/`` if present; always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tga_drawdown import data, strategy as st

print("# TGA-Drawdown — Treasury General Account (hardcoded monthly PROXY) + SPY (yfinance)")

if data.have_real():
    f = data.load_real()
    yrs = (f.index.max() - f.index.min()).days / 365.25
    print(f"months         : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{yrs:.1f} years)")
    print("signal         : TGA DRAWING DOWN = 1-month change in balance < 0 (1m exec lag)")

    print("\n# Forward SPY returns when TGA DRAWING DOWN vs base rate")
    print(f"  {'H':>4} {'n_draw':>6} {'draw_mn':>8} {'build_mn':>8} {'base_mn':>8} "
          f"{'draw_up':>7} {'base_up':>7} {'Welch_t':>8} {'HAC_t':>7} {'p_plac':>7}")
    for m in (1, 2, 3, 6):
        s = st.summarize(f, m)
        print(f"  {str(m)+'m':>4} {s['n_draw']:>6} {s['draw_mean']*100:>7.2f}% "
              f"{s['build_mean']*100:>7.2f}% {s['base_mean']*100:>7.2f}% "
              f"{s['draw_uprate']*100:>6.0f}% {s['base_uprate']*100:>6.0f}% "
              f"{s['t']:>8.2f} {s['t_hac']:>7.2f} {s['p_placebo']:>7.3f}")

    print("\n# HAC (Newey-West) predictive regression: fwd return ~ injection (per +$100B drawn)")
    for m in (1, 2, 3, 6):
        r = st.hac_regression(f, m)
        print(f"  H={m}m: beta={r['beta']*100:>+6.2f}%/+100B  HAC_t={r['t_hac']:>+5.2f}  "
              f"R2={r['r2']*100:>5.2f}%  n={r['n']}")

    print("\n# Lead/lag: corr(injection@t, SPY ret over [t+L, t+L+1])  (lever would be L>0, positive)")
    ll = st.lead_lag(f)
    for L in (-3, -1, 0, 1, 3, 6):
        print(f"  L={L:>2}: {ll[L]:+.3f}")

    print("\n# Tradability — hold-when-drawing-down overlay (1m lag, 10 bps/switch)")
    o = st.timing_overlay(f, cost_bps=10.0)
    print(f"  buy&hold   : mean={o['bh_mean']*100:>5.1f}%  sharpe={o['bh_sharpe']:.2f}")
    print(f"  overlay gr : mean={o['overlay_gross_mean']*100:>5.1f}%  sharpe={o['overlay_gross_sharpe']:.2f}")
    print(f"  overlay net: mean={o['overlay_net_mean']*100:>5.1f}%  sharpe={o['overlay_net_sharpe']:.2f}  "
          f"switches={o['n_switches']}")

    print("\n# Robustness — change window k, threshold, ex-COVID (1-month fwd)")
    for k in (1, 2, 3):
        s = st.summarize(f, 1, k=k)
        print(f"  k={k}: n_draw={s['n_draw']:>3}  draw1={s['draw_mean']*100:>5.2f}%  "
              f"base1={s['base_mean']*100:>5.2f}%  Welch_t={s['t']:>5.2f}  HAC_t={s['t_hac']:>5.2f}")
    f2 = f[(f.index < "2020-01-01") | (f.index >= "2021-07-01")]
    s = st.summarize(f2, 1)
    print(f"  ex-COVID 1m: n_draw={s['n_draw']:>3}  draw1={s['draw_mean']*100:>5.2f}%  "
          f"base1={s['base_mean']*100:>5.2f}%  Welch_t={s['t']:>5.2f}  HAC_t={s['t_hac']:>5.2f}")
else:
    print("(no _cache/spy_prices.csv — run data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must recover a PLANTED drawdown->returns link and must NOT manufacture")
print("  significance from a no-link series (edge=0).")
for edge in (0.0, 0.04):
    syn = data.synthetic_tga(n_months=252, edge=edge, seed=752)
    s = st.summarize(syn, 1, k=1)
    print(f"  planted edge={edge:+.2f}: n_draw={s['n_draw']:>3}  draw1m={s['draw_mean']*100:>6.2f}%  "
          f"base1m={s['base_mean']*100:>6.2f}%  Welch_t={s['t']:>6.2f}  HAC_t={s['t_hac']:>6.2f}  p={s['p_placebo']:.3f}")
