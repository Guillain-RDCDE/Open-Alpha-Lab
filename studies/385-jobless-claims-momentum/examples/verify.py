"""Reproducible headline run for Study 385 — Jobless-Claims-Momentum.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the hardcoded claims snapshot (always
available) and the cached SPY prices under ``_cache/`` if present; always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jobless_claims_momentum import data, strategy as st

print("# Jobless-Claims-Momentum — FRED IC4WSA (hardcoded snapshot) + SPY (yfinance)")

if data.have_real():
    f = data.load_real()
    yrs = (f.index.max() - f.index.min()).days / 365.25
    print(f"months         : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{yrs:.1f} years)")
    print(f"signal         : claims-momentum RISING = 4wk-MA level > 3 months prior (1m exec lag)")

    print("\n# Forward SPY returns when claims RISING vs base rate")
    print(f"  {'H':>4} {'n_ris':>5} {'ris_mean':>9} {'fall_mean':>9} {'base_mean':>9} "
          f"{'ris_dn':>6} {'base_dn':>7} {'Welch_t':>8} {'p_plac':>7}")
    for m in (1, 3, 6, 12):
        s = st.summarize(f, m)
        print(f"  {str(m)+'m':>4} {s['n_rising']:>5} {s['rising_mean']*100:>8.2f}% "
              f"{s['falling_mean']*100:>8.2f}% {s['base_mean']*100:>8.2f}% "
              f"{s['rising_downrate']*100:>5.0f}% {s['base_downrate']*100:>6.0f}% "
              f"{s['t']:>8.2f} {s['p_placebo']:>7.3f}")

    print("\n# Lead/lag: corr(claims-mom@t, SPY ret over [t+L, t+L+1])  (early-warning would be L>0, negative)")
    ll = st.lead_lag(f)
    for L in (-3, -1, 0, 1, 3, 6):
        print(f"  L={L:>2}: {ll[L]:+.3f}")

    print("\n# Tradability — cash-when-claims-rising overlay (1m lag, 10 bps/switch)")
    o = st.timing_overlay(f, cost_bps=10.0)
    print(f"  buy&hold   : mean={o['bh_mean']*100:>5.1f}%  sharpe={o['bh_sharpe']:.2f}")
    print(f"  overlay gr : mean={o['overlay_gross_mean']*100:>5.1f}%  sharpe={o['overlay_gross_sharpe']:.2f}")
    print(f"  overlay net: mean={o['overlay_net_mean']*100:>5.1f}%  sharpe={o['overlay_net_sharpe']:.2f}  "
          f"switches={o['n_switches']}")

    print("\n# Robustness — momentum window k, threshold, ex-COVID (12-month)")
    for k in (1, 3, 6):
        s = st.summarize(f, 12, k=k)
        print(f"  k={k}: n_ris={s['n_rising']:>3}  ris12={s['rising_mean']*100:>5.1f}%  "
              f"base12={s['base_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    for thr in (0.05, 0.10):
        s = st.summarize(f, 12, thresh=thr)
        print(f"  thr=+{thr:.0%}: n_ris={s['n_rising']:>3}  ris12={s['rising_mean']*100:>5.1f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    f2 = f[(f.index < "2020-01-01") | (f.index >= "2022-01-01")]
    s = st.summarize(f2, 12)
    print(f"  ex-COVID 12m: n_ris={s['n_rising']:>3}  ris12={s['rising_mean']*100:>5.1f}%  "
          f"base12={s['base_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/spy_prices.csv — run data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must recover a PLANTED claims->returns link and must NOT manufacture")
print("  significance from a no-link series (edge=0).")
for edge in (0.0, 0.04):
    syn = data.synthetic_claims(n_months=360, edge=edge, seed=385)
    s = st.summarize(syn, 1, k=3)
    print(f"  planted edge={edge:+.2f}: n_ris={s['n_rising']:>3}  ris1m={s['rising_mean']*100:>6.2f}%  "
          f"base1m={s['base_mean']*100:>6.2f}%  t={s['t']:>6.2f}  p_placebo={s['p_placebo']:.3f}")
