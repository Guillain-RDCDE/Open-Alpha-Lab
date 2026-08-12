"""Reproducible headline run for Study 877 — GDPNow Revisions.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached GDPNow + SPY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from gdpnow import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# GDPNow Revisions — does the daily nowcast revision predict forward SPY?")

if not data.have_real():
    print("(cache miss — fetching the GDPNow workbook + SPY once)")
    data.fetch_nowcast()
    data.fetch_spy()

raw = data.load_nowcast()
frame = data.build_real()
fp = fingerprint(frame[["nowcast", "rev", "fwd1"]].reset_index(drop=True)
                 .assign(i=range(len(frame))).set_index("i"))
print(f"[data] GDPNow: {len(raw):,} forecast dates over {raw['qtr'].nunique()} quarters, "
      f"{raw.index.min().date()} -> {raw.index.max().date()}  as-of {data.AS_OF}")
print(f"[data] aligned revision/return pairs: {len(frame):,}  "
      f"{frame.index.min().date()} -> {frame.index.max().date()}  fingerprint={fp}")
print("  SIGNAL is the within-quarter day-over-day change of the top-line GDPNow nowcast; "
      "execution lag = act at the close of the release day (lag 0, headline).")

print("\n# PREDICTIVE REGRESSION — forward SPY return on the revision (Newey-West t)")
for y, lbl in [("fwd1", "1-day"), ("fwd5", "5-day")]:
    r = st.predict_stats(frame, ycol=y)
    print(f"  {lbl:>5} fwd: n={r['n']}  beta={r['beta']*1e4:+.2f} bps per 1pp  "
          f"NW t = {r['t']:+.2f}  R2 = {r['r2']*100:.3f}%")
print("  execution-lag robustness (act next day, lag 1):")
for y, lbl in [("fwd1_l1", "1-day"), ("fwd5_l1", "5-day")]:
    r = st.predict_stats(frame, ycol=y)
    print(f"    {lbl:>5} fwd: beta={r['beta']*1e4:+.2f} bps  NW t = {r['t']:+.2f}  "
          f"(sign flips vs lag 0 -> no stable slope)")

print("\n# DECILE CONDITIONAL — biggest up vs biggest down revisions (fwd 1-day)")
d = st.decile_conditional(frame, ycol="fwd1")
print(f"  base forward 1-day return: {d['base_bps']:+.2f} bps")
print(f"  top-decile UP  (rev >= {d['up_thr']:+.3f} pp): n={d['n_up']}  "
      f"{d['up_bps']:+.2f} bps  NW t = {d['up_t']:+.2f}")
print(f"  bot-decile DOWN(rev <= {d['down_thr']:+.3f} pp): n={d['n_down']}  "
      f"{d['down_bps']:+.2f} bps  NW t = {d['down_t']:+.2f}")
print(f"  up-minus-down Welch t = {d['up_minus_down_welch_t']:+.2f}")
d1 = st.decile_conditional(frame, ycol="fwd1_l1")
print(f"  [lag 1] the top-decile result flips: UP {d1['up_bps']:+.2f} bps (t={d1['up_t']:+.2f})"
      f" — a fragile intraday-timing artefact, not an edge")

print("\n# ERA CUT — the predictive regression on two halves (split 2019-01-01)")
e = st.era_stats(frame)
print(f"  2011-2018 (n={e['early']['n']}): beta={e['early']['beta']*1e4:+.2f} bps  "
      f"NW t = {e['early']['t']:+.2f}")
print(f"  2019-2026 (n={e['late']['n']}): beta={e['late']['beta']*1e4:+.2f} bps  "
      f"NW t = {e['late']['t']:+.2f}   (sign flips across eras)")

print("\n# PLACEBO — shuffle forward returns against revisions (5,000 draws)")
p = st.placebo_pvalue(frame, ycol="fwd1")
print(f"  observed slope {p['obs_beta_bps']:+.2f} bps vs shuffled sd {p['placebo_sd_bps']:.2f} "
      f"-> two-sided p = {p['p_value']:.3f}")

print("\n# TIMER — long SPY for one day after an up-revision, flat otherwise")
for cb in (1.0, 5.0):
    t = st.timer_stats(frame, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: net {t['net']['mean_bps']:+.2f} bps/day "
          f"({t['net']['ann_pct']:+.1f}%/yr, Sharpe {t['net']['sharpe']:+.2f})  vs "
          f"buy-and-hold Sharpe {t['buy_hold']['sharpe']:+.2f}  (exposure {t['exposure']:.2f})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([st.synthetic_detect(data.synthetic(edge=0.0, seed=877 + s, n=2000))["t"]
                   for s in range(20)])
print(f"  null (edge=0), 20 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
sy = st.synthetic_detect(data.synthetic(edge=0.005, seed=877, n=2000))
print(f"  planted (edge=0.005, seed 877): beta={sy['beta_bps']:+.1f} bps  NW t = {sy['t']:+.2f}  "
      f"R2 = {sy['r2']*100:.2f}%")
