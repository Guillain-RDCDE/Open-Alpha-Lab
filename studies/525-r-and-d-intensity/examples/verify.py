"""Reproducible headline run for Study 525 — R-And-D-Intensity (Chan-Lakonishok-Sougiannis).

Prints every number quoted in docs/results.md and frozen into notebooks/build_notebooks.py
(the ``R`` dict). Offline & deterministic once the cache is built: the real-tape panels are read
from ``_cache/`` (built once via data.fetch_fundamentals() + data.fetch_prices()); the placebo
label-shuffle null and the synthetic positive control are fixed-seed.

    python examples/verify.py            # run on the cache
    python examples/verify.py --fetch    # (re)build the cache from EDGAR + yfinance, then run
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from r_and_d_intensity import data, strategy as st

if "--fetch" in sys.argv:
    print("# fetching EDGAR fundamentals + yfinance prices into _cache/ ...")
    data.fetch_fundamentals()
    data.fetch_prices()
    print("# cache built.\n")

print("# R&D-Intensity — do high R&D-to-market-cap firms earn the CLS premium?\n")

if data.have_real():
    real = data.load_real(allow_survivorship_bias=True)
    rets = real["returns"]
    sigs = data.build_signals(real, report_lag=1)
    span_years = len(rets) / 12.0
    print("# Real tape — EDGAR R&D/revenue/shares + yfinance monthly price/total-return")
    print(f"window         : {rets.index.min().strftime('%Y-%m')} -> "
          f"{rets.index.max().strftime('%Y-%m')}  ({len(rets)} months, {span_years:.1f} years)")
    print(f"field          : {rets.shape[1]} names  "
          f"({real['rd'].shape[1]} with an EDGAR R&D series)")
    print(f"fingerprint    : "
          f"{data.fingerprint(real['rd'], real['rev'], real['shares'], rets, real['prices'])}")

    R = st.race(real, sigs, frac=1 / 3, cost_bps=10.0, borrow_bps=100.0, n_shuffles=400)

    # final-month tertiles (the signal that selects the last held book)
    last_sig = sigs["rd_me"].dropna(how="all").index[-2]
    lo, sh = st.tertile_members(sigs["rd_me"].loc[last_sig], frac=1 / 3)
    print(f"\n# Final-month tertiles (top/bottom by R&D / market-cap)")
    print(f"  long  (high R&D/ME): {sorted(lo)}")
    print(f"  short (low  R&D/ME): {sorted(sh)}")
    print(f"  avg one-way turnover/month: {R['avg_turnover']*100:.1f}%")

    print("\n# The race — high-R&D/ME long, low-R&D/ME short, vs SPY (CAGR/Sharpe/maxDD)")
    for name, key in [("Long (high R&D/ME)", "long"), ("Short leg (low R&D/ME)", "short"),
                      ("Long-short spread", "long_short"), ("S&P 500 (SPY)", "spy")]:
        s = st.summarize(R[key])
        print(f"  {name:26s} CAGR={s['cagr']*100:6.2f}%  Sharpe={s['sharpe']:5.2f}  "
              f"maxDD={s['max_dd']*100:6.1f}%  mean_ann={s['mean_ann']*100:6.2f}%")

    print("\n# Signal-axis test — HAC t-stat of the spreads (REAL needs t >= 2)")
    for label, key in [("Long - short (R&D/ME)   ", "test_ls"),
                       ("Long - SPY              ", "test_long_vs_spy"),
                       ("Short - SPY             ", "test_short_vs_spy")]:
        t = R[key]
        print(f"  {label}: mean {t['mean_ann']*100:+6.2f}%/yr   HAC t = {t['tstat']:5.2f}  (n={t['n']})")

    print("\n# Placebo — label-shuffle null (permute signal across names, 400 draws)")
    null = R["placebo_null"]
    print(f"  R&D/ME long-short spread       : {R['test_ls']['mean_ann']*100:+.2f}%/yr")
    print(f"  shuffled null (mean / sd)      : {null.mean()*100:+.2f}% / {null.std()*100:.2f} per yr")
    print(f"  R&D/ME percentile in null      : {R['placebo_pctile']:.1f}")
    print(f"  placebo two-sided p            : {R['placebo_p']:.3f}")

    print("\n# The CLS denominator contrast — R&D/ME vs R&D/sales (does scaling by price win?)")
    print(f"  R&D / MARKET-CAP long-short : {R['test_ls']['mean_ann']*100:+.2f}%/yr  "
          f"(HAC t={R['test_ls']['tstat']:.2f})")
    print(f"  R&D / SALES      long-short : {R['test_sales_ls']['mean_ann']*100:+.2f}%/yr  "
          f"(HAC t={R['test_sales_ls']['tstat']:.2f})")

    print("\n# Robustness — long/short fraction")
    for frac, lbl in [(0.50, "halves   "), (1 / 3, "tertiles "), (0.25, "quartiles"), (0.20, "quintiles")]:
        b = st.signal_books(sigs["rd_me"], rets, frac=frac)
        t = st.hac_tstat(b["long_short"])
        print(f"  frac={frac:.2f} ({lbl})  LS={t['mean_ann']*100:+5.2f}%/yr  HAC t={t['tstat']:5.2f}")

    print("\n# Costs + short-borrow")
    print(f"  long-short gross : {R['test_ls']['mean_ann']*100:+.2f}%/yr  (t={R['test_ls']['tstat']:.2f})")
    print(f"  long-short net   : {R['test_ls_net']['mean_ann']*100:+.2f}%/yr  "
          f"(t={R['test_ls_net']['tstat']:.2f})  "
          f"[10 bps turnover + 100 bps/yr borrow on short]")
else:
    print("(no _cache/ panels — run `python examples/verify.py --fetch` once to build them)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the harness must NOT manufacture significance with edge=0, and must light up with a large")
print("  planted R&D/ME premium.")
for edge in (0.0, 0.06):
    s2, r2, b2, truth = data.synthetic_panel(edge=edge)
    bk = st.signal_books(s2, r2)
    t = st.hac_tstat(bk["long_short"])
    tag = "NULL (no true premium)" if edge == 0 else "large planted premium"
    print(f"  edge={edge:.2f} [{tag:22s}]: long-short spread {t['mean_ann']*100:+6.2f}%/yr  "
          f"HAC t={t['tstat']:+.2f}")
