"""Reproducible headline run for Study 695 — Inverse Head-and-Shoulders.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached 30-name basket tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from inverse_head_shoulders import data, strategy as st  # noqa: E402

print("# Inverse Head-and-Shoulders — does the bullish bottom pattern predict anything, "
      "and does its measured-move target mean anything?")

if not data.have_real():
    print("(cache miss — fetching the 30-name basket once)")
    data.fetch_panel()

panel = data.load_real()
print(data_stamp(f"basket close ({panel['close'].shape[1]} names)", panel["close"], asof=data.AS_OF))
print(f"basket: {list(panel['close'].columns)}")

sigs = st.collect_signals(panel)
n_total = sum(len(v) for v in sigs.values())
print(f"\nconfirmed inverse-H&S breakouts: {n_total} across {panel['close'].shape[1]} names, "
      f"{data.START} -> {data.AS_OF}")

print("\n# THE HEADLINE — forward return after the confirmed neckline break, "
      "vs the name's own base rate")
print("  (excess = post-breakout forward return minus that name's unconditional forward return "
      "of the same horizon; one-day entry lag; one-sample t and HAC t on the pooled excess; "
      "a same-tape random-date placebo is the honest control for the tape's own up-drift)")
headline = {}
for h in st.HORIZONS:
    r = st.run_experiment(panel, horizon=h, cost_bps=5.0)
    headline[h] = r
    print(f"  {h:>2d}d: n={r['n_events']:>3d}  raw {r['raw_mean']*100:+.3f}%  "
          f"excess {r['mean']*100:+.3f}%  t={r['t']:+.2f}  HAC t={r['hac_t']:+.2f}  "
          f"placebo p={r['p_placebo']:.3f}  net(5bps) {r['net']*100:+.3f}%")

print("\n# Detector-strictness robustness sweep (horizon=20d)")
for tol_s, tol_n, tag in [(0.08, 0.10, "tight"), (0.12, 0.15, "base"), (0.18, 0.20, "loose")]:
    r = st.run_experiment(panel, horizon=20, shoulder_tol=tol_s, neckline_tol=tol_n)
    print(f"  {tag:>5s} (shoulder_tol={tol_s}, neckline_tol={tol_n}): n={r['n_events']:>3d}  "
          f"excess {r['mean']*100:+.3f}%  t={r['t']:+.2f}")

print("\n# THE MEASURED-MOVE TARGET — does 'height = neckline minus head, "
      "projected up' mean anything?")
mm = st.measured_move_hits(panel, max_days=126)
print(f"  {mm['n_signals']} signals, target hit within 126 trading days: "
      f"{mm['n_hit']}/{mm['n_signals']} = {mm['hit_rate']*100:.1f}%  "
      f"(Wilson 95% [{mm['hit_lo']*100:.1f}%, {mm['hit_hi']*100:.1f}%])")
print(f"  magnitude-matched random-entry placebo (same relative target size, {mm['placebo_n']:,} "
      f"draws): {mm['placebo_hit']}/{mm['placebo_n']} = {mm['placebo_rate']*100:.1f}%")
print(f"  median days-to-hit: {mm['median_days_to_hit']:.0f}  |  mean target distance: "
      f"{mm['mean_rel_move']*100:+.1f}% above entry")

print("\n# THE LONG TIMER — hold to target-or-timeout (126d), net of costs")
for cb in (5.0, 10.0):
    tp = st.timer_pnl(panel, max_days=126, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: n={tp['n']}  gross {tp['gross']*100:+.3f}%  "
          f"net {tp['net']*100:+.3f}%  t={tp['t']:+.2f}  HAC t={tp['hac_t']:+.2f}  "
          f"avg hold {tp['avg_hold']:.1f}d  target-hit share {tp['hit_share']*100:.1f}%")
tp5 = st.timer_pnl(panel, max_days=126, cost_bps=5.0)
print(f"  excess over a holding-period-MATCHED base rate: {tp5['excess']*100:+.3f}%  "
      f"HAC t = {tp5['t_excess']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT manufacture significance on pure noise (>=10 seeds, |t|<2) "
      "and must light up on a planted post-breakout continuation edge")
null_ts = []
for s_ in range(695, 715):
    p, _ = data.synthetic_panel(n_names=10, n_days=6000, edge=0.0, n_planted=0, seed=s_)
    r = st.run_experiment(p, horizon=20, seed=s_)
    null_ts.append(r["t"])
null_ts = np.asarray(null_ts)
print(f"  null (pure noise, no planted shape), 20 seeds: mean t = {null_ts.mean():+.3f} "
      f"(sd {null_ts.std(ddof=1):.3f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")

planted, truth = data.synthetic_panel(n_names=10, n_days=6000, edge=0.15, n_planted=8, seed=695)
rp = st.run_experiment(planted, horizon=20, seed=695)
print(f"  planted post-breakout edge=+0.15 (seed 695): n={rp['n_events']} "
      f"(of {truth['n_planted_total']} planted) excess {rp['mean']*100:+.2f}%  "
      f"t = {rp['t']:+.2f}  HAC t = {rp['hac_t']:+.2f}  placebo p = {rp['p_placebo']:.5f}")

print("\n# THE VERDICT")
h20 = headline[20]
print(f"  Signal: forward-return excess over base rate never clears t=2 at any horizon "
      f"(max |t| = {max(abs(headline[h]['t']) for h in st.HORIZONS):.2f}); the measured-move "
      f"target hits {mm['hit_rate']*100:.1f}% of the time vs a magnitude-matched placebo of "
      f"{mm['placebo_rate']*100:.1f}% (the pattern does not beat a random move of the same "
      f"size); the long-timer excess over a holding-matched base rate is "
      f"{tp5['excess']*100:+.3f}% at HAC t = {tp5['t_excess']:+.2f}.")
