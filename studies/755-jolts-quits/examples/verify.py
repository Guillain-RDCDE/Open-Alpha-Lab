"""Reproducible headline run for Study 755 — JOLTS-Quits.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the hardcoded quits snapshot (always
available) and the cached SPY/XLY/XLP prices under ``_cache/`` if present; always runs
the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jolts_quits import data, strategy as st

print("# JOLTS-Quits — FRED JTSQUR (hardcoded snapshot) + SPY/XLY/XLP (yfinance)")

if data.have_real():
    f = data.load_real()
    yrs = (f.index.max() - f.index.min()).days / 365.25
    print(f"months         : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{yrs:.1f} years)")
    print(f"signal         : quits-momentum FALLING = quits rate < value 3 months prior "
          f"(2-month JOLTS release lag)")

    print("\n# Forward SPY returns when quits FALLING vs base rate")
    print(f"  {'H':>4} {'n_fall':>6} {'fall_mean':>9} {'rise_mean':>9} {'base_mean':>9} "
          f"{'fall_dn':>7} {'base_dn':>7} {'Welch_t':>8} {'p_plac':>7}")
    for m in (1, 3, 6, 12):
        s = st.summarize(f, m)
        print(f"  {str(m)+'m':>4} {s['n_falling']:>6} {s['falling_mean']*100:>8.2f}% "
              f"{s['rising_mean']*100:>8.2f}% {s['base_mean']*100:>8.2f}% "
              f"{s['falling_downrate']*100:>6.0f}% {s['base_downrate']*100:>6.0f}% "
              f"{s['t']:>8.2f} {s['p_placebo']:>7.3f}")

    print("\n# Lead/lag: corr(quits-mom@t, SPY ret over [t+L, t+L+1])  (leading gauge would be L>0, positive)")
    ll = st.lead_lag(f)
    for L in (-3, -1, 0, 1, 3, 6):
        print(f"  L={L:>2}: {ll[L]:+.3f}")

    print("\n# Tradability — cash-when-quits-falling overlay (2-month lag, 10 bps/switch)")
    o = st.timing_overlay(f, cost_bps=10.0)
    print(f"  buy&hold   : mean={o['bh_mean']*100:>5.1f}%  sharpe={o['bh_sharpe']:.2f}")
    print(f"  overlay gr : mean={o['overlay_gross_mean']*100:>5.1f}%  sharpe={o['overlay_gross_sharpe']:.2f}")
    print(f"  overlay net: mean={o['overlay_net_mean']*100:>5.1f}%  sharpe={o['overlay_net_sharpe']:.2f}  "
          f"switches={o['n_switches']}")

    if "cyc" in f.columns:
        print("\n# Cyclicals leg — forward XLY-minus-XLP (risk-appetite) returns when quits FALLING")
        for m in (1, 6, 12):
            s = st.summarize(f, m, price="cyc")
            print(f"  {str(m)+'m':>3}: n_fall={s['n_falling']:>3}  fall={s['falling_mean']*100:>6.2f}%  "
                  f"base={s['base_mean']*100:>6.2f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")

    print("\n# Robustness — momentum window k, threshold, ex-COVID (12-month)")
    for k in (1, 3, 6):
        s = st.summarize(f, 12, k=k)
        print(f"  k={k}: n_fall={s['n_falling']:>3}  fall12={s['falling_mean']*100:>5.1f}%  "
              f"base12={s['base_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    for thr in (0.1, 0.2):
        s = st.summarize(f, 12, thresh=thr)
        print(f"  thr>{thr:.1f}pp: n_fall={s['n_falling']:>3}  fall12={s['falling_mean']*100:>5.1f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    f2 = f[(f.index < "2020-01-01") | (f.index >= "2022-01-01")]
    s = st.summarize(f2, 12)
    print(f"  ex-COVID 12m: n_fall={s['n_falling']:>3}  fall12={s['falling_mean']*100:>5.1f}%  "
          f"base12={s['base_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    # lag sensitivity: a look-ahead 0-month vs the honest 2-month release lag (12m)
    for lg in (0, 1, 2):
        s = st.summarize(f, 12, lag=lg)
        print(f"  lag={lg}m: n_fall={s['n_falling']:>3}  fall12={s['falling_mean']*100:>5.1f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/spy_prices.csv — run data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must recover a PLANTED quits->returns link and must NOT manufacture")
print("  significance from a no-link series (edge=0).")
for edge in (0.0, 0.04):
    syn = data.synthetic_quits(n_months=300, edge=edge, seed=755)
    s = st.summarize(syn, 1, k=3, lag=1)
    print(f"  planted edge={edge:+.2f}: n_fall={s['n_falling']:>3}  fall1m={s['falling_mean']*100:>6.2f}%  "
          f"base1m={s['base_mean']*100:>6.2f}%  t={s['t']:>6.2f}  p_placebo={s['p_placebo']:.3f}")
