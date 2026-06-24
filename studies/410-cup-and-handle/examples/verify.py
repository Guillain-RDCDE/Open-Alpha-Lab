"""Reproducible headline run for Study 410 — Cup & Handle.

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

from cup_and_handle import data, strategy as st

ASOF = "2026-05-31"          # pinned as-of; the last full bar used is the trading day on/before
NDRAWS = 5000

print("# Cup & Handle — objective detector on SPY + 29 large-caps (yfinance daily OHLC)")
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

    bk = st.collect_breakouts(panel)
    tot = sum(len(v) for v in bk.values())
    print(f"breakouts      : {tot} confirmed cup-with-handle breakouts "
          f"(SPY alone: {len(bk['SPY'])})")

    print("\n# Forward edge after a confirmed breakout — EXCESS over each name's base rate")
    print("  (1-day entry lag; placebo = random dates on the SAME tape, same count)")
    print(f"  {'H':>3} {'n':>4} {'excess%':>8} {'raw%':>7} {'win%':>5} "
          f"{'t':>6} {'HAC_t':>6} {'p_plac':>7} {'net%':>7}")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, n_draws=NDRAWS)
        print(f"  {h:>3} {r['n_events']:>4} {r['mean']*100:>7.3f} {r['raw_mean']*100:>6.3f} "
              f"{r['win']*100:>4.1f} {r['t']:>6.2f} {r['hac_t']:>6.2f} "
              f"{r['p_placebo']:>7.3f} {r['net']*100:>6.3f}")

    print("\n# Robustness — detector strictness (rim_tol, handle_max_depth) at H=10")
    for rt, hd in [(0.04, 0.10), (0.06, 0.15), (0.10, 0.20)]:
        r = st.run_experiment(panel, horizon=10, n_draws=3000,
                              rim_tol=rt, handle_max_depth=hd)
        print(f"  rim_tol={rt:.2f} handle<={hd:.2f}: n={r['n_events']:>4}  "
              f"excess={r['mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")

    print("\n# SPY-only (the README hook) — does the figure beat SPY buy-and-hold?")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, names=["SPY"], n_draws=NDRAWS)
        print(f"  SPY H={h:>2}: n={r['n_events']:>2}  excess={r['mean']*100:+.3f}%  "
              f"raw={r['raw_mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")
else:
    print("(no _cache/cuph_close.parquet — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector + inference must recover a PLANTED post-breakout drift and must NOT")
print("  manufacture significance when the true edge is 0.")
for edge in (0.0, 0.20):
    panel, truth = data.synthetic_panel(edge=edge, seed=410, n_planted=8, daily_vol=0.011)
    r = st.run_experiment(panel, horizon=20, n_draws=1500)
    print(f"  planted edge={edge:+.2f}: planted={truth['n_planted_total']:>3}  "
          f"detected={r['n_breakouts']:>3}  n={r['n_events']:>3}  "
          f"excess={r['mean']*100:>6.2f}%  t={r['t']:>6.2f}  "
          f"p_placebo={r['p_placebo']:.3f}  win={r['win']*100:.0f}%")
