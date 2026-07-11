"""Reproducible headline run for Study 686 — Stick Sandwich.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic positive control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from stick_sandwich import data, strategy as st  # noqa: E402

COST_BPS = 5.0
PLACEBO_H = 20
PLACEBO_DRAWS = 1000

print("# Stick Sandwich — does a failed rally between two matching closes call a bottom?")
print(f"basket: {len(data.BASKET)} names (SPY + {len(data.BASKET) - 1} long-listed US "
      f"large-caps), as-of {data.AS_OF}")

if not data.have_real():
    print("(cache miss — fetching the basket once)")
    data.load_real(fetch=True)

panel = data.load_real()
spy = panel["SPY"]
print(data_stamp("SPY OHLC", spy, asof=data.AS_OF))
print(f"panel fingerprint: {data.panel_fingerprint(panel)}")

years = max((b.index[-1] - b.index[0]).days / 365.25 for b in panel.values())
total_bars = sum(len(b) for b in panel.values())
print(f"window: ~{years:.1f} years per name, {len(panel)} names, {total_bars:,} total bars")

print("\n# THE DETECTOR — bearish / bullish-rally / bearish, outer closes meeting within "
      f"{st.DEFAULT_TOL * 1e4:.0f} bps, after a confirmed {st.DEFAULT_TREND_LOOKBACK}-day down leg")
res = st.run_experiment(panel, cost_bps=COST_BPS, placebo_horizon=PLACEBO_H,
                        n_draws=PLACEBO_DRAWS)
print(f"  pooled stick sandwiches: n = {res['n_sandwiches']} across {res['n_names']} names")
per_t = res["per_ticker_n"]
top = sorted(per_t.items(), key=lambda kv: -kv[1])[:5]
print(f"  per-ticker range: min={min(per_t.values())}, max={max(per_t.values())}, "
      f"busiest names: {top}")
print(f"  Bonferroni-corrected critical |t| for {len(st.HORIZONS)} horizons: "
      f"{res['bonferroni_crit']:.2f}")

print("\n# THE HEADLINE — sandwich forward return vs the unconditional base rate")
print("  (one-sample t is the MISLEADING number — it is pure beta, the tape's upward drift;")
print("   the Welch sandwich-vs-base-rate t is the test that matters)")
print(f"  {'H':>3} {'n':>5} {'sand_bps':>9} {'win%':>5} {'1samp_t':>8} "
      f"{'base_bps':>9} {'delta_bps':>10} {'net_bps':>8} {'welch_t':>8}")
for h in st.HORIZONS:
    row = res["by_h"][h]
    g, b, ns = row["gross"], row["base"], row["net"]
    wt = row["welch_t"]
    wt_s = f"{wt:+.2f}" if wt is not None else "n/a"
    print(f"  {h:>3} {g['n']:>5} {g['mean_bps']:>9.1f} {g['win']*100:>5.0f} "
          f"{g['t']:>8.2f} {b['mean_bps']:>9.1f} {row['delta_bps']:>10.1f} "
          f"{ns['mean_bps']:>8.1f} {wt_s:>8}")

crit = res["bonferroni_crit"]
survivors = [h for h in st.HORIZONS
             if res["by_h"][h]["welch_t"] is not None
             and abs(res["by_h"][h]["welch_t"]) >= crit]
print(f"  horizons clearing the Bonferroni bar (|welch t| >= {crit:.2f}): "
      f"{survivors if survivors else 'NONE'}")
naive_survivors = [h for h in st.HORIZONS
                  if res["by_h"][h]["welch_t"] is not None
                  and abs(res["by_h"][h]["welch_t"]) >= 2.0]
print(f"  horizons clearing the naive |t| >= 2.00 bar: {naive_survivors if naive_survivors else 'NONE'}")

pb = res["placebo"]
print(f"\n# GEOMETRY PLACEBO ({pb['ticker']}, H={pb['horizon']}) — keep the down/up/down "
      "context, scramble the equal-close test")
print(f"  {pb['n_candidates']} context-matched candidates (down leg + bearish/bullish-rally/"
      "bearish, equal-close ignored)")
print(f"  observed sandwich mean {pb['obs']*1e4:+.1f} bps   placebo p = {pb['p_value']:.4f}   "
      f"({pb['n_draws']} draws)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch sandwich-vs-base-rate detector must NOT fire on a null world (edge=0) and")
print("  must recover a planted post-sandwich reversal. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    p_, _truth = data.synthetic_panel(edge=0.0, seed=686 + s_, n_names=20, n_days=3000)
    r = st.synthetic_detect(p_, horizon=20)
    if r["welch_t"] is not None:
        null_ts.append(r["welch_t"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (edge=0), {len(null_ts)} seeds: mean Welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/{len(null_ts)} seeds")
p_planted, truth_planted = data.synthetic_panel(edge=1.0, seed=686, n_names=20, n_days=3000)
r_planted = st.synthetic_detect(p_planted, horizon=20)
print(f"  planted edge=1.00 (seed 686, {truth_planted['n_planted']} planted sandwiches): "
      f"sandwich {r_planted['mean_bps']:+.1f} bps vs base {r_planted['base_bps']:+.1f} bps  "
      f"delta {r_planted['delta_bps']:+.1f} bps  Welch t = {r_planted['welch_t']:+.2f}")
