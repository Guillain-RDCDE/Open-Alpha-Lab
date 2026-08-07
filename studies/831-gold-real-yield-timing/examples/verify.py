"""Reproducible headline run for Study 831 — Gold Real-Yield Timing.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tape under
``_cache/`` (fetching once, with retries, on a cache miss), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from gold_real_yield import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Gold Real-Yield Timing — does the real-yield TREND predict forward gold?")

if not data.have_real():
    print("(cache miss — fetching GLD/TIP/IEF/^TNX once, with retries)")
    data.fetch()

df = data.load_series()
print(f"[data] {len(df):,} days  {df.index.min().date()} -> {df.index.max().date()}  "
      f"as-of {data.AS_OF}  tape fp {data.fingerprint(df,'GLD_close')}  ry fp {data.fingerprint(df,'ry')}")
print("  SURVIVORSHIP: single fixed ETF/yield tapes, no cross-sectional membership.")

print("\n# THE INVERSE LINK (contemporaneous, descriptive) — the famous fact")
il = st.inverse_link(df)
print(f"  corr(gold ret, same-day real-yield change) = {il['corr']:+.2f}  "
      f"beta = {il['beta']:+.2f}  NW t = {il['t']:+.2f}  (n={il['n']:,})")

print("\n# THE HEADLINE SORT — forward 21d GLD on the (lagged) real-yield-fall rank")
qs = st.quintile_spread(df, horizon=21, lookback=63)
p = st.placebo_pvalue(df, horizon=21, lookback=63, n_perm=1000, seed=831)
print(f"  Q1 (yields rising) {qs['q1']*100:+.2f}%  |  Q5 (yields falling) {qs['q5']*100:+.2f}%  "
      f"-> spread {qs['spread']*100:+.2f}% (HAC t {qs['t']:+.2f}, placebo p {p:.3f}, n_q~{qs['n_q']})")

print("\n# HORIZON SWEEP")
for h, row in st.horizon_sweep(df).iterrows():
    print(f"  {h:>5}: spread {row['spread']*100:+.2f}%  HAC t {row['t']:+.2f}")

print("\n# LOOKBACK SWEEP")
for lb, row in st.lookback_sweep(df).iterrows():
    print(f"  {lb:>5}: spread {row['spread']*100:+.2f}%  HAC t {row['t']:+.2f}")

print("\n# SUB-PERIOD SWEEP (21d)")
edges = [("2004-2009", "2004-11-01", "2009-12-31"), ("2010-2015", "2010-01-01", "2015-12-31"),
         ("2016-2020", "2016-01-01", "2020-12-31"), ("2021-2026", "2021-01-01", "2026-06-30")]
for lab, row in st.subperiod_sweep(df, edges, horizon=21).iterrows():
    print(f"  {lab}: spread {row['spread']*100:+.2f}%  HAC t {row['t']:+.2f}  (n={int(row['n'])})")

print("\n# THE TIMER — own GLD when real yields falling, else cash; one-way cost/switch x NAV")
for cb in (2.0, 5.0):
    ov = st.timing_overlay(df, cost_bps=cb)
    print(f"  cost={cb:>3.1f} bps: timer Sharpe {ov['timer_sharpe']:.3f} vs buy-hold "
          f"{ov['bh_sharpe']:.3f} | mean spread {ov['spread_bps_day']:+.2f} bps/day "
          f"(t {ov['spread_t']:+.2f}) | {ov['switches_per_yr']:.1f} switches/yr | "
          f"invested {ov['days_invested_frac']:.3f}")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network (25 seeds, 21d)")
print("  the same contemporaneous link is on; only the PREDICTIVE edge is planted.")
for e in (0.0, 0.005, 0.010, 0.020, 0.040):
    r = st.synthetic_mean_t(data, edge=e, n_seeds=25, n_days=3000)
    print(f"  edge {e:.3f} -> mean spread {r['mean_spread']*100:+.2f}%  mean HAC t {r['mean_t']:+.2f}")
