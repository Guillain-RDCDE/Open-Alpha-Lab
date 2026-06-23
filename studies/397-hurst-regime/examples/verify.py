"""Reproducible headline run for Study 397 — Hurst-Regime.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached market prices under ``_cache/``
if present (the real-tape numbers) and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hurst_regime import data, strategy as st

print("# Hurst-Regime — rolling R/S Hurst gating a trend vs mean-revert switch")
if data.have_real():
    f = data.load_real("SPY")
    span_years = (f.index.max() - f.index.min()).days / 365.25
    rets = st.log_returns(f["price"])
    hurst = data.rolling_hurst(rets, window=252)
    print(f"SPY days       : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span_years:.1f} years)")
    print(f"rolling Hurst  : mean {hurst.mean():.3f}  median {hurst.median():.3f}  "
          f"std {hurst.std():.3f}  range [{hurst.min():.2f}, {hurst.max():.2f}]")
    hv = hurst.dropna()
    print(f"share H>0.5    : {(hv > 0.5).mean()*100:.1f}%  (of valid days; the gate calls 'trend regime')")

    books = st.build_books(f["price"], hurst, cost_bps=1.0)
    s = st.summarize_books(books)
    print("\n# The four books (1-day lag, 1 bp one-way costs x turnover)")
    print(f"  {'book':9s} {'Sharpe':>7} {'ann_mean':>9} {'ann_vol':>8}")
    for k in ("buyhold", "trend", "revert", "gated"):
        print(f"  {k:9s} {s[k]['sharpe']:>7.2f} {s[k]['ann_mean']*100:>8.1f}% "
              f"{s[k]['ann_vol']*100:>7.1f}%")

    bench = books["revert"] if s["revert"]["sharpe"] >= s["trend"]["sharpe"] else books["trend"]
    bname = "revert" if s["revert"]["sharpe"] >= s["trend"]["sharpe"] else "trend"
    bb = st.block_bootstrap_sharpe_diff(books["gated"], bench, n_boot=5000, seed=397)
    g, b = books["gated"].align(bench, join="inner")
    diff = (g - b).dropna()
    print(f"\n# Does the Hurst GATE beat the best static style ({bname})?")
    print(f"  gated - {bname} Sharpe diff = {bb['obs']:+.3f}  "
          f"95% CI [{bb['lo']:+.3f}, {bb['hi']:+.3f}]  p(diff<=0) = {bb['p_le0']:.3f}")
    print(f"  paired daily t (gated - best static) = {st.paired_t(diff.values):+.3f}")
    bbh = st.block_bootstrap_sharpe_diff(books["gated"], books["buyhold"], n_boot=5000, seed=397)
    print(f"  gated - buyhold Sharpe diff = {bbh['obs']:+.3f}  p(diff<=0) = {bbh['p_le0']:.3f}")
    pl = st.placebo_shuffled_regime(f["price"], hurst, bench, n_draws=2000, seed=397)
    print(f"  placebo (block-shuffle the regime label): real_edge = {pl['real_edge']:+.3f}  "
          f"placebo_mean = {pl['placebo_mean']:+.3f}  p = {pl['p_value']:.3f}")

    print("\n# Per-market (gated vs best static vs buy&hold)")
    for tk in data.MARKETS:
        fm = data.load_real(tk)
        h = data.rolling_hurst(st.log_returns(fm["price"]), window=252)
        bk = st.build_books(fm["price"], h, cost_bps=1.0)
        sm = st.summarize_books(bk)
        best = max(sm["trend"]["sharpe"], sm["revert"]["sharpe"])
        hv = h.dropna()
        print(f"  {tk}: H_mean={h.mean():.3f} share_trend={(hv>0.5).mean()*100:>4.0f}%  "
              f"gated={sm['gated']['sharpe']:+.2f}  best_static={best:+.2f}  "
              f"buyhold={sm['buyhold']['sharpe']:+.2f}")

    print("\n# Robustness — the Hurst window (SPY)")
    for win in (126, 252, 504):
        h = data.rolling_hurst(rets, window=win)
        bk = st.build_books(f["price"], h, cost_bps=1.0)
        sm = st.summarize_books(bk)
        best = max(sm["trend"]["sharpe"], sm["revert"]["sharpe"])
        print(f"  window={win}: gated={sm['gated']['sharpe']:+.2f}  best_static={best:+.2f}  "
              f"buyhold={sm['buyhold']['sharpe']:+.2f}")
else:
    print("(no _cache/prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic control 1 — the R/S estimator recovers a PLANTED Hurst (no network)")
for H in (0.30, 0.50, 0.70):
    syn = data.synthetic_prices(n_days=6000, H=H, seed=397)
    r = st.log_returns(syn["price"]).dropna().values
    print(f"  planted H={H:.2f} -> R/S estimate {data.hurst_rs(r):.3f}")

print("\n# Synthetic control 2 — the GATE power test (alternating-regime path)")
print("  edge=0 (both regimes random walk) must NOT beat best static; edge=1 must light up.")
for edge in (0.0, 1.0):
    syn = data.synthetic_regimes(n_days=6000, edge=edge, seed=397)
    h = data.rolling_hurst(st.log_returns(syn["price"]), window=252, stride=1)
    bk = st.build_books(syn["price"], h, cost_bps=1.0)
    sm = st.summarize_books(bk)
    bench = bk["trend"] if sm["trend"]["sharpe"] >= sm["revert"]["sharpe"] else bk["revert"]
    bb = st.block_bootstrap_sharpe_diff(bk["gated"], bench, n_boot=3000, seed=397)
    pl = st.placebo_shuffled_regime(syn["price"], h, bench, n_draws=1500, seed=397)
    print(f"  planted edge={edge:.1f}: gated={sm['gated']['sharpe']:+.2f}  "
          f"trend={sm['trend']['sharpe']:+.2f}  revert={sm['revert']['sharpe']:+.2f}  "
          f"diff={bb['obs']:+.3f}  p_boot={bb['p_le0']:.3f}  p_placebo={pl['p_value']:.3f}")
