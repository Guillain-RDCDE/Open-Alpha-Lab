"""Reproducible headline run for Study 759 — Redbook-Retail.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the hardcoded Redbook proxy (always
available) and the cached XRT/SPY prices under ``_cache/`` if present; always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbook_retail import data, strategy as st

print("# Redbook-Retail — Redbook same-store YoY (hardcoded LABELLED PROXY) + XRT/SPY (yfinance)")

if data.have_real():
    f = data.load_real()
    yrs = (f.index.max() - f.index.min()).days / 365.25
    print(f"months         : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{yrs:.1f} years)")
    print(f"signal         : Redbook ACCELERATING = YoY level > 3 months prior (1m exec lag)")

    print("\n# Forward XRT returns when Redbook ACCELERATING vs base rate (absolute)")
    print(f"  {'H':>4} {'n_acc':>5} {'acc_mean':>9} {'dec_mean':>9} {'base_mean':>9} "
          f"{'acc_up':>6} {'base_up':>7} {'Welch_t':>8} {'p_plac':>7}")
    for m in (1, 3, 6, 12):
        s = st.summarize(f, m)
        print(f"  {str(m)+'m':>4} {s['n_accel']:>5} {s['accel_mean']*100:>8.2f}% "
              f"{s['decel_mean']*100:>8.2f}% {s['base_mean']*100:>8.2f}% "
              f"{s['accel_uprate']*100:>5.0f}% {s['base_uprate']*100:>6.0f}% "
              f"{s['t']:>8.2f} {s['p_placebo']:>7.3f}")

    print("\n# Retail-vs-market (XRT-minus-SPY) forward returns when ACCELERATING vs base")
    for m in (1, 3, 6, 12):
        s = st.summarize(f, m, relative=True)
        print(f"  {str(m)+'m':>4} n_acc={s['n_accel']:>3}  acc={s['accel_mean']*100:>6.2f}%  "
              f"base={s['base_mean']*100:>6.2f}%  t={s['t']:>6.2f}  p={s['p_placebo']:.3f}")

    print("\n# Lead/lag: corr(Redbook-mom@t, XRT ret over [t+L, t+L+1])  (nowcast would be L>0, positive)")
    ll = st.lead_lag(f)
    for L in (-4, -3, -2, -1, 0, 1, 3, 6):
        print(f"  L={L:>2}: {ll[L]:+.3f}")
    print(f"  argmax at L={int(ll.idxmax())} (rho={ll.max():+.3f}) -> Redbook LAGS the stocks")

    print("\n# Level regime — forward XRT returns in STRONG vs WEAK same-store months (median split)")
    for m in (1, 12):
        g = st.regime_summary(f, m)
        print(f"  {str(m)+'m':>4} strong={g['strong_mean']*100:>6.2f}%(n{g['n_strong']})  "
              f"weak={g['weak_mean']*100:>6.2f}%(n{g['n_weak']})  spread={g['spread']*100:>6.2f}%  t={g['t']:>6.2f}")

    print("\n# Tradability — own-XRT-when-accelerating overlay (1m lag, 10 bps/switch)")
    o = st.timing_overlay(f, cost_bps=10.0)
    print(f"  buy&hold   : mean={o['bh_mean']*100:>5.1f}%  sharpe={o['bh_sharpe']:.2f}")
    print(f"  overlay gr : mean={o['overlay_gross_mean']*100:>5.1f}%  sharpe={o['overlay_gross_sharpe']:.2f}")
    print(f"  overlay net: mean={o['overlay_net_mean']*100:>5.1f}%  sharpe={o['overlay_net_sharpe']:.2f}  "
          f"switches={o['n_switches']}")

    print("\n# Robustness — window k, smoothing, relative, ex-COVID (6-month)")
    for k in (1, 3, 6):
        s = st.summarize(f, 6, k=k)
        print(f"  k={k}: n_acc={s['n_accel']:>3}  acc6={s['accel_mean']*100:>5.1f}%  "
              f"base6={s['base_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    s = st.summarize(f, 6, smooth=3)
    print(f"  smooth3: n_acc={s['n_accel']:>3}  acc6={s['accel_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    s = st.summarize(f, 6, relative=True)
    print(f"  relative: n_acc={s['n_accel']:>3}  acc6={s['accel_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    f2 = f[(f.index < "2020-01-01") | (f.index >= "2022-07-01")]
    s = st.summarize(f2, 6)
    print(f"  ex-COVID 6m: n_acc={s['n_accel']:>3}  acc6={s['accel_mean']*100:>5.1f}%  "
          f"base6={s['base_mean']*100:>5.1f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/xrt_spy_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must recover a PLANTED Redbook->returns link and must NOT manufacture")
print("  significance from a no-link series (edge=0).")
for edge in (0.0, 0.05):
    syn = data.synthetic_redbook(n_months=360, edge=edge, seed=759)
    s = st.summarize(syn, 1, k=3)
    print(f"  planted edge={edge:+.2f}: n_acc={s['n_accel']:>3}  acc1={s['accel_mean']*100:>6.2f}%  "
          f"base1={s['base_mean']*100:>6.2f}%  t={s['t']:>6.2f}  p_placebo={s['p_placebo']:.3f}")
