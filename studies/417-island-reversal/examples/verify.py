"""Reproducible headline run for Study 417 — Island Reversal.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily OHLC panel under
``_cache/`` (the real-tape numbers, pinned to an explicit as-of) and always runs the
synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from island_reversal import data, strategy as st

ASOF = "2026-05-31"          # pinned as-of; last full bar used is the trading day on/before
GAP = 0.01                   # headline gap threshold (1%): the mechanical "clear gap"
NDRAWS = 5000

print("# Island Reversal — objective gap-island detector on SPY + 29 large-caps (yfinance daily OHLC)")
if data.have_real():
    panel = data.load_real()
    asof = pd.Timestamp(ASOF)
    for f in panel:
        panel[f] = panel[f].loc[:asof]
    closes = panel["close"]
    span_days = (closes.index.max() - closes.index.min()).days / 365.25
    print(f"as-of          : {ASOF}  (last bar {closes.index.max().date()}, "
          f"{closes.index.min().date()} -> {closes.index.max().date()}, {span_days:.1f} years)")
    print(f"panel          : {closes.shape[1]} names, {len(closes)} daily bars")
    print(f"fingerprint    : {data.fingerprint(panel)}")

    bt = st.collect_islands(panel, side="top", gap_pct=GAP)
    bb = st.collect_islands(panel, side="bottom", gap_pct=GAP)
    tot = sum(len(v) for v in bt.values())
    totb = sum(len(v) for v in bb.values())
    print(f"islands        : {tot} confirmed island-TOPS (SPY alone: {len(bt['SPY'])}); "
          f"{totb} island-BOTTOMS (myth check, SPY: {len(bb['SPY'])})")

    print("\n# ISLAND TOP (bearish, short) forward edge — EXCESS over each name's base rate")
    print("  (1-day entry lag; placebo = random dates on the SAME tape, same count)")
    print(f"  {'H':>3} {'n':>4} {'excess%':>8} {'raw%':>7} {'win%':>5} "
          f"{'t':>6} {'HAC_t':>6} {'p_plac':>7} {'net%':>7}")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, side="top", n_draws=NDRAWS, gap_pct=GAP)
        print(f"  {h:>3} {r['n_events']:>4} {r['mean']*100:>7.3f} {r['raw_mean']*100:>6.3f} "
              f"{r['win']*100:>4.1f} {r['t']:>6.2f} {r['hac_t']:>6.2f} "
              f"{r['p_placebo']:>7.3f} {r['net']*100:>6.3f}")

    print("\n# Myth check — ISLAND BOTTOM (bullish, long): does the figure reverse symmetrically?")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, side="bottom", n_draws=NDRAWS, gap_pct=GAP)
        print(f"  BOT H={h:>2}: n={r['n_events']:>3}  excess={r['mean']*100:+.3f}%  "
              f"raw={r['raw_mean']*100:+.3f}%  win={r['win']*100:.0f}%  t={r['t']:+.2f}  "
              f"HAC={r['hac_t']:+.2f}  p={r['p_placebo']:.3f}")

    print("\n# Robustness — gap-threshold strictness sweep (island top, short) at H=10")
    for gp in (0.005, 0.01, 0.02):
        r = st.run_experiment(panel, horizon=10, side="top", n_draws=3000, gap_pct=gp)
        print(f"  gap_pct={gp:.3f}: n={r['n_events']:>4}  excess={r['mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}")

    print("\n# SPY-only island top (the README hook) — does the figure beat SPY?")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, names=["SPY"], side="top",
                              n_draws=NDRAWS, gap_pct=GAP)
        print(f"  SPY H={h:>2}: n={r['n_events']:>2}  excess={r['mean']*100:+.3f}%  "
              f"raw={r['raw_mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")
else:
    print("(no _cache/island_close.parquet — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector + inference must recover a PLANTED post-island reversal and the same-tape")
print("  placebo must NOT light up when the true edge is 0.")
for edge in (0.0, 0.25):
    panel, truth = data.synthetic_panel(edge=edge, seed=417, n_planted=8, daily_vol=0.011,
                                        gap_pct=0.04)
    r = st.run_experiment(panel, horizon=20, side="top", n_draws=1500, gap_pct=0.02)
    print(f"  planted edge={edge:+.2f}: planted={truth['n_planted_total']:>3}  "
          f"detected={r['n_islands']:>3}  n={r['n_events']:>3}  "
          f"excess={r['mean']*100:>6.2f}%  t={r['t']:>6.2f}  "
          f"p_placebo={r['p_placebo']:.3f}  win={r['win']*100:.0f}%")
