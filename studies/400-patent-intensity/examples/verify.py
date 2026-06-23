"""Reproducible headline run for Study 400 — Patent-Intensity (the "innovation premium").

Prints every number quoted in docs/results.md and frozen into notebooks/build_notebooks.py
(the ``R`` dict). Offline & deterministic: the real-tape panels are read from ``_cache/`` (built
once via data.fetch_intensity() + data.fetch_returns()); the random-basket control and the
synthetic positive control are fixed-seed.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from patent_intensity import data, strategy as st

print("# Patent-Intensity — do the most R&D-intensive firms earn an innovation premium?\n")

if data.have_real():
    inten, rets, spy = data.load_real(allow_survivorship_bias=True)
    span_years = len(rets) / 12.0
    print("# Real tape — EDGAR R&D intensity (proxy) + yfinance monthly total return")
    print(f"window         : {rets.index.min().strftime('%Y-%m')} -> "
          f"{rets.index.max().strftime('%Y-%m')}  ({len(rets)} months, {span_years:.1f} years)")
    print(f"field          : {rets.shape[1]} names  ({inten.shape[1]} with an EDGAR intensity series)")
    print(f"fingerprint    : {data.fingerprint(inten, rets)}")

    R = st.race(inten, rets, spy, frac=1 / 3, report_lag=1, cost_bps=10.0,
                borrow_bps=100.0, n_draws=2000)
    last = max(R["members"])
    lo, sh = R["members"][last]
    print(f"\n# Final-rebalance tertiles (top/bottom by R&D intensity, 1-year reporting lag)")
    print(f"  long  (high-intensity): {sorted(lo)}")
    print(f"  short (low-intensity) : {sorted(sh)}")

    print("\n# The race — high-intensity long, low-intensity short, vs SPY (CAGR/Sharpe/maxDD)")
    for name, key in [("Long (high-intensity)", "long"), ("Short leg (low-intensity)", "short"),
                      ("Long-short spread", "long_short"), ("S&P 500 (SPY)", "spy")]:
        s = st.summarize(R[key])
        print(f"  {name:26s} CAGR={s['cagr']*100:6.2f}%  Sharpe={s['sharpe']:5.2f}  "
              f"maxDD={s['max_dd']*100:6.1f}%  mean_ann={s['mean_ann']*100:6.2f}%")
    print(f"  {'Long net (10 bps)':26s} CAGR={st.summarize(R['long_net'])['cagr']*100:6.2f}%")

    print("\n# Signal-axis test — HAC t-stat of the spreads (REAL needs t >= 2)")
    for label, key in [("Long - short (innovation)", "test_ls"),
                       ("Long - SPY              ", "test_long_vs_spy"),
                       ("Short - SPY             ", "test_short_vs_spy")]:
        t = R[key]
        print(f"  {label}: mean {t['mean_ann']*100:+6.2f}%/yr   HAC t = {t['tstat']:5.2f}  (n={t['n']})")

    print("\n# Intensity split vs a BLIND random long/short of the same field (the sector control)")
    rs = R["rand_ls_spread"]
    print(f"  intensity long-short spread        : {R['test_ls']['mean_ann']*100:+.2f}%/yr")
    print(f"  random blind LS (mean / median)    : {rs.mean()*100:+.2f}% / {np.median(rs)*100:+.2f}% per yr"
          f"  (sd {rs.std()*100:.2f})")
    print(f"  intensity percentile in 2000 random: {R['ls_pctile']:.1f}")

    print("\n# Robustness — long/short fraction (the spread is fragile to specification)")
    for frac, lbl in [(0.50, "halves   "), (1 / 3, "tertiles "), (0.25, "quartiles"), (0.20, "quintiles")]:
        b = st.intensity_books(inten, rets, frac=frac)
        t = st.hac_tstat(b["long_short"])
        k = len(b["members"][max(b["members"])][0])
        print(f"  frac={frac:.2f} ({lbl}) k={k:2d}/leg  LS={t['mean_ann']*100:+5.2f}%/yr  HAC t={t['tstat']:5.2f}")

    print("\n# Costs + short-borrow")
    ln = st.hac_tstat(R["ls_net"])
    print(f"  long-short gross : {R['test_ls']['mean_ann']*100:+.2f}%/yr  (t={R['test_ls']['tstat']:.2f})")
    print(f"  long-short net   : {ln['mean_ann']*100:+.2f}%/yr  (t={ln['tstat']:.2f})  "
          f"[10 bps turnover + 100 bps/yr borrow on short]")
else:
    print("(no _cache/ panels — run data.fetch_intensity() + data.fetch_returns() once to build them)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the harness must NOT manufacture significance with edge=0, and must light up with a large")
print("  planted innovation premium.")
for edge in (0.0, 0.06):
    i2, r2, b2, truth = data.synthetic_panel(edge=edge)
    bk = st.intensity_books(i2, r2)
    t = st.hac_tstat(bk["long_short"])
    tag = "NULL (no true premium)" if edge == 0 else "large planted premium"
    print(f"  edge={edge:.2f} [{tag:22s}]: long-short spread {t['mean_ann']*100:+6.2f}%/yr  "
          f"HAC t={t['tstat']:+.2f}")
