"""Reproducible headline run for Study 696 — Double-Bottom.

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from double_bottom import data, strategy as st  # noqa: E402

AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07-10)
NDRAWS = 5_000
MAX_DAYS = 126               # ~6 trading months — the "long timer" window

print("# Double-Bottom — does buying the confirmed W-shaped breakout beat the stock's own drift?")
if not data.have_real():
    print("(cache miss — fetching the 30-name basket once)")
    data.fetch_panel()

panel = data.load_real(asof=AS_OF)
closes = panel["close"]
span_years = (closes.index.max() - closes.index.min()).days / 365.25
print(data_stamp("basket daily closes (30 names)", closes, asof=AS_OF))
print(f"panel          : {closes.shape[1]} names, {len(closes):,} daily bars, "
      f"{closes.index.min().date()} -> {closes.index.max().date()} ({span_years:.1f} years)")
print(f"fingerprint    : {data.fingerprint(panel)}  (sha1 of the close panel)")

bk = st.collect_breakouts(panel)
tot = sum(len(v) for v in bk.values())
print(f"breakouts      : {tot} confirmed double-BOTTOMS across the basket (SPY alone: {len(bk['SPY'])})")

print("\n# THE HEADLINE — forward return after a confirmed double-bottom breakout, "
      "EXCESS over each name's own base rate")
print("  (1-day entry lag; placebo = random dates on the SAME tape, same count per name)")
print(f"  {'H':>3} {'n':>4} {'excess%':>8} {'raw%':>7} {'win%':>5} "
      f"{'t':>6} {'HAC_t':>6} {'p_plac':>7} {'net%':>7}")
headline = {}
for h in st.HORIZONS:
    r = st.run_experiment(panel, horizon=h, n_draws=NDRAWS)
    headline[h] = r
    print(f"  {h:>3} {r['n_events']:>4} {r['mean']*100:>7.3f} {r['raw_mean']*100:>6.3f} "
          f"{r['win']*100:>4.1f} {r['t']:>6.2f} {r['hac_t']:>6.2f} "
          f"{r['p_placebo']:>7.3f} {r['net']*100:>6.3f}")

print("\n# Robustness — detector strictness (tolerance, min_bounce) at H=20")
for tol, mb in [(0.03, 0.07), (0.04, 0.05), (0.06, 0.03)]:
    r = st.run_experiment(panel, horizon=20, n_draws=3000, tolerance=tol, min_bounce=mb)
    print(f"  tolerance={tol:.2f} min_bounce={mb:.2f}: n={r['n_events']:>4}  "
          f"excess={r['mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")

print("\n# SPY-only double bottom (the README hook) — beats SPY buy-and-hold?")
for h in st.HORIZONS:
    r = st.run_experiment(panel, horizon=h, names=["SPY"], n_draws=NDRAWS)
    print(f"  SPY H={h:>2}: n={r['n_events']:>2}  excess={r['mean']*100:+.3f}%  "
          f"raw={r['raw_mean']*100:+.3f}%  t={r['t']:+.2f}  p={r['p_placebo']:.3f}")

print("\n# THE MEASURED-MOVE TARGET — does the pattern's own height forecast a real move?")
print(f"  (target = neckline + (neckline - trough level); window = {MAX_DAYS} trading days; "
      "placebo = magnitude-matched random entries on the same tape)")
mm = st.measured_move_hits(panel, max_days=MAX_DAYS)
print(f"  observed hit rate : {mm['n_hit']}/{mm['n_signals']} = {mm['hit_rate']*100:.1f}%  "
      f"(Wilson 95% [{mm['hit_lo']*100:.1f}%, {mm['hit_hi']*100:.1f}%])")
print(f"  placebo hit rate  : {mm['placebo_hit']}/{mm['placebo_n']} = {mm['placebo_rate']*100:.1f}%  "
      f"(same-magnitude random entries)")
print(f"  two-proportion z (observed vs placebo): {mm['z_vs_placebo']:+.2f}")
print(f"  median days to hit: {mm['median_days_to_hit']:.0f}   "
      f"mean target distance: {mm['mean_rel_move']*100:+.2f}%")

print("\n# THE LONG TIMER — hold to target-hit or timeout, net of costs")
print(f"  (entry = breakout close + 1 day; exit = target-hit close or the {MAX_DAYS}-day timeout close; "
      "excess = vs a holding-period-matched base rate)")
for cb in (5.0, 10.0):
    tp = st.timer_pnl(panel, max_days=MAX_DAYS, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: n={tp['n']:>4}  gross={tp['gross']*100:+.3f}%  "
          f"net={tp['net']*100:+.3f}%  t(vs 0)={tp['t']:+.2f}  HAC_t(vs 0)={tp['hac_t']:+.2f}  "
          f"avg_hold={tp['avg_hold']:.1f}d  target-hit share={tp['hit_share']*100:.1f}%")
tp5 = st.timer_pnl(panel, max_days=MAX_DAYS, cost_bps=5.0)
print(f"  excess over a holding-period-matched base rate: {tp5['excess']*100:+.3f}%  "
      f"HAC t = {tp5['t_excess']:+.2f}")
print("  (the gap between t-vs-0 and t-vs-matched-base is the market's own up-drift over an "
      "average multi-week hold — not the pattern; see docs/references.md)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector + inference must recover a PLANTED post-breakout drift, and the same-tape")
print("  placebo must NOT light up when the true edge is 0 (the naive t can be fooled by the")
print("  figure's own geometry — that is exactly why the placebo is the arbiter).")
null_ts, null_ps = [], []
for s_ in range(20):
    px, _ = data.synthetic_panel(edge=0.0, seed=696 + s_, n_planted=8, daily_vol=0.011)
    r = st.run_experiment(px, horizon=20, n_draws=1500, seed=696 + s_)
    null_ts.append(r["t"])
    null_ps.append(r["p_placebo"])
null_ts = np.asarray(null_ts)
null_ps = np.asarray(null_ps)
print(f"  null (edge=0), 20 seeds: mean naive t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"placebo p < 0.05 in {(null_ps < 0.05).sum()}/20 seeds")
for edge in (0.0, 0.20):
    px, truth = data.synthetic_panel(edge=edge, seed=696, n_planted=8, daily_vol=0.011)
    r = st.run_experiment(px, horizon=20, n_draws=3000)
    print(f"  planted edge={edge:+.2f} (seed 696): planted={truth['n_planted_total']:>3}  "
          f"detected={r['n_breakouts']:>3}  n={r['n_events']:>3}  excess={r['mean']*100:>6.2f}%  "
          f"t={r['t']:>6.2f}  p_placebo={r['p_placebo']:.3f}  win={r['win']*100:.0f}%")
