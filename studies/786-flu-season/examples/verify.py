"""Reproducible headline run for Study 786 — Flu-Season.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached CVS / SPY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

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

from flu_season import data as dt, strategy as st  # noqa: E402

print("# Flu-Season — does the pharmacy tape (CVS) rally into flu season?")

print(f"calendar: {len(dt.EVENTS)} flu seasons {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      f"each anchored on the fixed CDC Oct-1 season start (MMWR week 40 convention)")

if not dt.have_real():
    print("(cache miss — fetching CVS + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Flu-Season panel (CVS + SPY, adjusted/total-return)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} seasons have CVS+SPY coverage "
      f"around the anchor")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")

print("\n# THE RUN-UP (the 'rally into the season') — abnormal return, K sessions BEFORE "
      "the Oct-1 season start, CVS minus SPY (gross)")
for label, col in (("2-week run-up (k=10)", "pre_s"), ("1-month run-up (k=21)", "pre_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<24s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE IN-SEASON LEG (does the anticipation give back?) — abnormal return, K "
      "sessions AFTER the season start")
for label, col in (("2-week in-season (k=10)", "post_s"), ("1-month in-season (k=21)", "post_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<24s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event)")
for label, col, k, tail in (("2wk run-up", "pre_s", 10, "right"),
                            ("2wk in-season", "post_s", 10, "left")):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail=tail)
    print(f"  {label:<14s} ({tail}-tail): observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 2-week run-up t-stat")
x = inc["pre_s"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — the run-up net of costs (calendar-known entry, zero look-ahead)")
for base, label in (("pre_s", "2-week run-up"), ("post_s", "2-week in-season")):
    g = st.one_sample_t(inc[base].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<16s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=813 + s, k=10)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (0.01, 0.02):
    planted = st.synthetic_detect(bump=bump, seed=813, k=10)
    print(f"  planted run-up bump=+{bump*100:.1f}% (seed 813): mean AR {planted['mean']*100:+.3f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
