"""Reproducible headline run for Study 789 — SUE Earnings-Surprise Drift.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached prices + EDGAR-EPS-derived SUE
events under ``_cache/`` (building them once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import as_of, data_stamp  # noqa: E402

from sue_drift import data, strategy as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

print("# SUE Earnings-Surprise Drift — does the classic PEAD line up with standardized "
      "unexpected earnings?")

if not data.have_real():
    print("(cache miss — building prices + EDGAR-EPS SUE events once)")
    data.fetch_panel(start="2005-01-01")

prices = as_of(data.load_prices(), data.AS_OF)      # pin the sample so the fingerprint is stable
events = data.load_events()
et = data.build_event_table(prices, events)
print(data_stamp("basket adjusted closes", prices, asof=data.AS_OF))
print(f"SUE events (ticker, filing date): {len(et)} over {et['ticker'].nunique()} names, "
      f"filed {et['filed'].min().date()} -> {et['filed'].max().date()} "
      f"(strictly on/before the {data.AS_OF} as-of; one row per quarterly print with >=8 prior "
      "seasonal surprises)")

print("\n# THE HEADLINE — forward drift of a top-minus-bottom SUE tercile long-short")
print("# (enter one day after the filing is public; hold 21 / 42 / 63 trading days ~ 1/2/3 mo)")
for H in st.HORIZONS:
    s = st.summarize(prices, et, H, n_buckets=3, placebo=True, n_draws=20_000)
    print(f"  H={H:>2}d  n={s['n_events']}  top {s['top_mean']*100:+.2f}%  "
          f"bot {s['bot_mean']*100:+.2f}%  LONG-SHORT {s['ls_mean']*100:+.2f}%")
    print(f"        one-sample t = {s['t']:+.2f} (naive, clustered)  |  "
          f"calendar-time Newey-West t = {s['nw_t']:+.2f} (robust, {s['nw_periods']} seasons)")
    print(f"        win-rate {s['ls_win']*100:.1f}% (Wilson [{s['win_lo']*100:.1f}%, "
          f"{s['win_hi']*100:.1f}%])  |  right-tail label-shuffle placebo p = {s['p_placebo']:.4f}")

print("\n# Monotonicity — quintile drift low->high SUE (should rise monotonically if PEAD is real)")
for H in st.HORIZONS:
    q5 = st.bucket_means(st.event_drift_frame(prices, et, H), n_buckets=5)
    print(f"  H={H:>2}d: " + "  ".join(f"{x*100:+.2f}%" for x in q5))

print("\n# Within-quarter block placebo (respects the earnings-season clustering)")
for H in st.HORIZONS:
    bp = st.block_placebo_pvalue(st.event_drift_frame(prices, et, H), n_draws=5_000)
    print(f"  H={H:>2}d: block-placebo p = {bp:.4f}")

print("\n# Sign flips with the bucket count (a hallmark of no real signal) — 63d")
fr63 = st.event_drift_frame(prices, et, 63)
for nb, name in ((3, "tercile"), (10, "decile")):
    ls = st.long_short_drift(fr63, n_buckets=nb)
    print(f"  {name:>7} long-short: {ls['ls_mean']*100:+.2f}%  (one-sample t = "
          f"{st.ttest_vs_zero(ls['ls_sample']):+.2f}, n_top={ls['n_top']}, n_bot={ls['n_bot']})")

print("\n# THE COST TEST — long-short net of one-way costs × per-event turnover (+ borrow), 63d")
for cb in (5.0, 10.0):
    nc = st.net_of_costs(prices, et, 63, cost_bps=cb)
    print(f"  {cb:>4.1f} bps one-way: gross {nc['gross']*100:+.2f}%  ->  net {nc['net']*100:+.2f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the long-short detector must NOT fire on a null world (edge=0) and must recover a")
print("  planted post-earnings drift. Null checked over 20 seeds (never a single stream).")
nulls = []
for sd in range(20):
    p, e = data.synthetic_sue(edge=0.0, seed=789 + sd)
    nulls.append(st.synthetic_detect(p, e)["t"])
nulls = np.asarray(nulls)
print(f"  null (edge=0), 20 seeds: mean t = {nulls.mean():+.2f} (sd {nulls.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(nulls) >= 2).sum()}/20 seeds")
prP, evP = data.synthetic_sue(edge=0.08, seed=789)
dP = st.synthetic_detect(prP, evP)
print(f"  planted edge=0.08 (seed 789): long-short {dP['ls_mean']*100:+.2f}%  t = {dP['t']:+.2f}")
