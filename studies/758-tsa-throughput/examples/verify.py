"""Reproducible headline run for Study 758 — TSA-Throughput.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the hardcoded TSA snapshot (always
available) and the cached JETS/MAR/HLT/SPY prices under ``_cache/`` if present; always runs
the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tsa_throughput import data, strategy as st

print("# TSA-Throughput — TSA checkpoint volumes (hardcoded proxy) + travel basket (yfinance)")

if data.have_real():
    f = data.load_real()
    yrs = (f.index.max() - f.index.min()).days / 365.25
    print(f"months         : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{yrs:.1f} years)")
    print(f"TSA range      : {f['tsa'].min():.2f}M ({f['tsa'].idxmin().date()}) -> "
          f"{f['tsa'].max():.2f}M ({f['tsa'].idxmax().date()})")
    print(f"signal         : TSA-momentum ACCELERATING = throughput > 12 months prior (1m exec lag)")

    print("\n# Forward basket returns when TSA ACCELERATING vs base rate")
    print(f"  {'H':>4} {'n_acc':>5} {'acc_mean':>9} {'dec_mean':>9} {'base_mean':>9} "
          f"{'acc_up':>6} {'base_up':>7} {'Welch_t':>8} {'p_plac':>7}")
    for m in (1, 3, 6, 12):
        s = st.summarize(f, m)
        print(f"  {str(m)+'m':>4} {s['n_accel']:>5} {s['accel_mean']*100:>8.2f}% "
              f"{s['decel_mean']*100:>8.2f}% {s['base_mean']*100:>8.2f}% "
              f"{s['accel_uprate']*100:>5.0f}% {s['base_uprate']*100:>6.0f}% "
              f"{s['t']:>8.2f} {s['p_placebo']:>7.3f}")

    print("\n# Lead/lag: corr(TSA-mom@t, basket ret over [t+L, t+L+1])  (nowcast would be L>0, positive)")
    ll = st.lead_lag(f)
    for L in (-6, -3, 0, 1, 3, 6):
        print(f"  L={L:>2}: {ll[L]:+.3f}")

    print("\n# Beta control: forward basket ~ const + fwd SPY + ACCEL dummy")
    for m in (1, 3, 6, 12):
        b = st.beta_control(f, m)
        note = "  (overlapping-returns inflated t — see results.md)" if m == 12 else ""
        print(f"  {str(m)+'m':>4}: dummy={b['adj_coef']*100:>6.2f}%  t={b['adj_t']:>6.2f}  "
              f"beta={b['beta']:.2f}{note}")

    print("\n# Tradability — long-travel-when-accelerating overlay (1m lag, 10 bps/switch)")
    o = st.timing_overlay(f, cost_bps=10.0)
    print(f"  buy&hold    : mean={o['bh_mean']*100:>5.1f}%  sharpe={o['bh_sharpe']:.2f}")
    print(f"  overlay gr  : mean={o['overlay_gross_mean']*100:>5.1f}%  sharpe={o['overlay_gross_sharpe']:.2f}")
    print(f"  overlay net : mean={o['overlay_net_mean']*100:>5.1f}%  sharpe={o['overlay_net_sharpe']:.2f}  "
          f"switches={o['n_switches']}  exposure={o['exposure']:.2f}")
    o2 = st.timing_overlay(f, cost_bps=10.0, borrow_bps=100.0, allow_short=True)
    print(f"  long/short  : net mean={o2['overlay_net_mean']*100:>5.1f}%  sharpe={o2['overlay_net_sharpe']:.2f}")

    print("\n# Robustness — momentum window k, threshold, ex-COVID (12-month)")
    for k in (3, 6, 12):
        s = st.summarize(f, 12, k=k)
        print(f"  k={k:>2}: n_acc={s['n_accel']:>3}  acc12={s['accel_mean']*100:>6.2f}%  "
              f"base12={s['base_mean']*100:>6.2f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    for thr in (0.05, 0.10):
        s = st.summarize(f, 12, thresh=thr)
        print(f"  thr=+{thr:.0%}: n_acc={s['n_accel']:>3}  acc12={s['accel_mean']*100:>6.2f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    f2 = f[(f.index < "2020-06-01") | (f.index >= "2022-06-01")]
    print(f"  ex-COVID months: {len(f2)}")
    for m in (1, 3, 6, 12):
        s = st.summarize(f2, m)
        print(f"  ex-COVID {str(m)+'m':>3}: n_acc={s['n_accel']:>3}  acc={s['accel_mean']*100:>6.2f}%  "
              f"base={s['base_mean']*100:>6.2f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/travel_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must recover a PLANTED TSA-momentum->returns link and must NOT manufacture")
print("  significance from a no-link series (edge=0).")
for edge in (0.0, 0.08):
    syn = data.synthetic_tsa(n_months=240, edge=edge, seed=758)
    s = st.summarize(syn, 1, k=12)
    print(f"  planted edge={edge:+.2f}: n_acc={s['n_accel']:>3}  acc1m={s['accel_mean']*100:>6.2f}%  "
          f"base1m={s['base_mean']*100:>6.2f}%  t={s['t']:>6.2f}  p_placebo={s['p_placebo']:.3f}")
