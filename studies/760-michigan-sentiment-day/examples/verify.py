"""Reproducible headline run for Study 760 — Michigan-Sentiment-Day.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the hardcoded UMich sentiment snapshot
(always available) and the cached SPY prices under ``_cache/`` if present; always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from michigan_sentiment_day import data, strategy as st

print("# Michigan-Sentiment-Day — UMich UMCSENT (hardcoded snapshot) + SPY (yfinance)")

if data.have_real():
    F = data.load_monthly()
    spy = data.load_spy()
    dates = data.release_dates(1993, 2026)
    yrs = (F.index.max() - F.index.min()).days / 365.25
    print(f"monthly frame  : {len(F)} months  ({F.index.min().date()} -> {F.index.max().date()}, "
          f"{yrs:.1f} years)")
    print(f"daily SPY       : {len(spy)} days   ({spy.index.min().date()} -> {spy.index.max().date()})")

    print("\n# A. Release-day drift — is the print a market mover?")
    rd = st.release_day_summary(spy, dates)
    print(f"  release day : n={rd['n_release']}  mean={rd['release_mean']*1e4:5.1f}bp  "
          f"all-day mean={rd['all_mean']*1e4:5.1f}bp  Welch t={rd['t_vs_all']:+.2f}")
    for lag in (1, 2):
        d = st.drift_by_surprise(spy, data.sentiment_series(), dates, lag=lag)
        print(f"  drift {lag}d    : beat={d['beat_mean']*1e4:5.1f}bp (n={d['n_beat']})  "
              f"miss={d['miss_mean']*1e4:5.1f}bp (n={d['n_miss']})  "
              f"t(beat-miss)={d['t_beat_minus_miss']:+.2f}")

    print("\n# B. Level/regime — low-then-rising forward SPY returns (0 look-ahead)")
    print(f"  {'H':>4} {'base':>7} {'low':>7} {'t_low':>6} {'low&rise':>9} {'n':>4} "
          f"{'t_lr':>6} {'p_block':>8}")
    for h in (1, 3, 6, 12):
        s = st.summarize_regime(F, h)
        print(f"  {str(h)+'m':>4} {s['base_mean']*100:>6.2f}% {s['low_mean']*100:>6.2f}% "
              f"{s['t_low']:>6.2f} {s['low_rising_mean']*100:>8.2f}% {s['n_low_rising']:>4} "
              f"{s['t_low_rising']:>6.2f} {s['p_block']:>8.3f}")
    print(f"  independent low-and-rising episodes (gap>2mo): {st.n_episodes(F)}")

    print("\n# Tradability — long only in LOW&RISING months, else cash (1m lag, 10 bps/switch)")
    o = st.timing_overlay(F, cost_bps=10.0)
    print(f"  buy&hold    : mean={o['bh_mean']*100:>5.1f}%  sharpe={o['bh_sharpe']:.2f}")
    print(f"  overlay gr  : mean={o['overlay_gross_mean']*100:>5.1f}%  sharpe={o['overlay_gross_sharpe']:.2f}")
    print(f"  overlay net : mean={o['overlay_net_mean']*100:>5.1f}%  sharpe={o['overlay_net_sharpe']:.2f}  "
          f"switches={o['n_switches']}  exposure={o['exposure']*100:.0f}%")

    print("\n# Robustness — 12-month low-and-rising across specs (naive t vs block-boot p)")
    for q in (0.20, 0.30, 0.40):
        s = st.summarize_regime(F, 12, low_q=q)
        print(f"  low_q={q:.2f} : n={s['n_low_rising']:>3}  lr={s['low_rising_mean']*100:>5.1f}%  "
              f"t={s['t_low_rising']:>5.2f}  p_block={s['p_block']:.3f}")
    for k in (1, 6):
        s = st.summarize_regime(F, 12, k=k)
        print(f"  k={k}      : n={s['n_low_rising']:>3}  lr={s['low_rising_mean']*100:>5.1f}%  "
              f"t={s['t_low_rising']:>5.2f}  p_block={s['p_block']:.3f}")
    F2 = F[(F.index < "2008-07-01") | (F.index >= "2009-07-01")]
    s = st.summarize_regime(F2, 12)
    print(f"  ex-GFC    : n={s['n_low_rising']:>3}  lr={s['low_rising_mean']*100:>5.1f}%  "
          f"base={s['base_mean']*100:>5.1f}%  t={s['t_low_rising']:>5.2f}  p_block={s['p_block']:.3f}")
    F3 = F[(F.index < "2020-01-01") | (F.index >= "2021-01-01")]
    s = st.summarize_regime(F3, 12)
    print(f"  ex-COVID  : n={s['n_low_rising']:>3}  lr={s['low_rising_mean']*100:>5.1f}%  "
          f"base={s['base_mean']*100:>5.1f}%  t={s['t_low_rising']:>5.2f}  p_block={s['p_block']:.3f}")
else:
    print("(no _cache/spy.csv — run data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the regime detector + block bootstrap must recover a PLANTED bottom-timer edge")
print("  and must NOT manufacture significance from a no-link series (edge=0).")
for edge in (0.0, 0.10):
    syn = data.synthetic(n_months=396, edge=edge, seed=760)
    s = st.summarize_regime(syn["frame"], 12, min_periods=12)
    print(f"  planted edge={edge:.2f}: n_lr={s['n_low_rising']:>3}  lr={s['low_rising_mean']*100:>6.2f}%  "
          f"base={s['base_mean']*100:>6.2f}%  t={s['t_low_rising']:>6.2f}  p_block={s['p_block']:.3f}")
