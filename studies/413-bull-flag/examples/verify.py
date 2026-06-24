"""Reproducible headline run for Study 413 — Bull Flag.

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

from bull_flag import data, strategy as st

ASOF = "2026-05-31"          # pinned as-of; last full bar used is the trading day on/before
NDRAWS = 5000

print("# Bull Flag — objective detector on SPY + 29 large-caps (yfinance daily OHLC)")
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

    bk = st.collect_breakouts(panel, side="up")
    bkd = st.collect_breakouts(panel, side="down")
    tot = sum(len(v) for v in bk.values())
    totd = sum(len(v) for v in bkd.values())
    print(f"breakouts      : {tot} confirmed UP-breaks (SPY alone: {len(bk['SPY'])}); "
          f"{totd} DOWN-breaks (myth check)")

    print("\n# UP-break forward edge — EXCESS over each name's base rate (1-day entry lag)")
    print("  (placebo = random dates on the SAME tape, same count)")
    print(f"  {'H':>3} {'n':>4} {'excess%':>8} {'raw%':>7} {'win%':>5} "
          f"{'t':>6} {'HAC_t':>6} {'p_plac':>7} {'net%':>7}")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, n_draws=NDRAWS)
        print(f"  {h:>3} {r['n_events']:>4} {r['mean']*100:>7.3f} {r['raw_mean']*100:>6.3f} "
              f"{r['win']*100:>4.1f} {r['t']:>6.2f} {r['hac_t']:>6.2f} "
              f"{r['p_placebo']:>7.3f} {r['net']*100:>6.3f}")

    print("\n# Myth check — DOWN-break of the same flag (does it resolve UP far more than down?)")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, side="down", n_draws=NDRAWS)
        print(f"  DOWN H={h:>2}: n={r['n_events']:>3}  excess={r['mean']*100:+.3f}%  "
              f"raw={r['raw_mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")

    print("\n# Robustness — detector strictness (min_pole, max_retrace) at H=20")
    for mp, mr in [(0.10, 0.6), (0.12, 0.5), (0.16, 0.4)]:
        r = st.run_experiment(panel, horizon=20, n_draws=3000, min_pole=mp, max_retrace=mr)
        print(f"  min_pole={mp:.2f} max_retrace={mr:.2f}: n={r['n_events']:>4}  "
              f"excess={r['mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")

    print("\n# SPY-only (the README hook) — does the figure beat SPY buy-and-hold?")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, names=["SPY"], n_draws=NDRAWS)
        print(f"  SPY H={h:>2}: n={r['n_events']:>2}  excess={r['mean']*100:+.3f}%  "
              f"raw={r['raw_mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")
else:
    print("(no _cache/bullflag_close.parquet — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector + inference must recover a PLANTED post-breakout drift and must NOT")
print("  pass the placebo when the true edge is 0 (the naive t can be fooled by geometry).")
for edge in (0.0, 0.20):
    panel, truth = data.synthetic_panel(edge=edge, seed=413, n_planted=8, daily_vol=0.011)
    r = st.run_experiment(panel, horizon=20, n_draws=1500)
    print(f"  planted edge={edge:+.2f}: planted={truth['n_planted_total']:>3}  "
          f"detected={r['n_breakouts']:>3}  n={r['n_events']:>3}  "
          f"excess={r['mean']*100:>6.2f}%  t={r['t']:>6.2f}  "
          f"p_placebo={r['p_placebo']:.3f}  win={r['win']*100:.0f}%")
