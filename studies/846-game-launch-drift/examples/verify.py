"""Reproducible headline run for Study 846 — Blockbuster Game-Launch Drift.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached TTWO / EA / NTDOY / UBSFY /
SPY tapes under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network. ATVI is delisted (no tape) — its launches show up in the excluded
funnel.

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

from game_launch import data as dt, strategy as st  # noqa: E402

print("# Game-Launch Drift — do publishers (TTWO/EA/NTDOY/UBSFY) move around a AAA launch?")

by_pub = {p: sum(1 for e in dt.EVENTS if e[1] == p) for p in dt.PUBLISHERS + dt.DELISTED}
print(f"calendar: {len(dt.EVENTS)} marquee AAA launches 2013->2024, hardcoded from "
      f"Wikipedia release boxes / publisher press releases; by publisher {by_pub}")

if not dt.have_real():
    print("(cache miss — fetching TTWO + EA + NTDOY + UBSFY + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Game-Launch panel (TTWO + EA + NTDOY + UBSFY + SPY, adjusted/total-return)",
                 panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} launches have publisher+SPY "
      f"coverage around the anchor")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")
print(f"  included by publisher: {inc['publisher'].value_counts().to_dict()}")

print("\n# THE RUN-UP (the 'buy the hype') — abnormal return, K sessions BEFORE the launch, "
      "publisher minus SPY (gross)")
for label, col in (("1-week run-up (k=5)", "pre_s"), ("2-week run-up (k=10)", "pre_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    nw = st.newey_west_t(inc.sort_values("anchor_date")[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"t_NW={nw:+.3f}  hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE POST-LAUNCH DRIFT (the 'ride/sell the news') — abnormal return, K sessions AFTER")
for label, col in (("1-week drift (k=5)", "post_s"), ("~1-month drift (k=20)", "post_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    nw = st.newey_west_t(inc.sort_values("anchor_date")[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"t_NW={nw:+.3f}  hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Per-publisher split of the 20-day post-launch drift")
for publisher in dt.PUBLISHERS:
    sub = inc[inc["publisher"] == publisher]
    s = st.one_sample_t(sub["post_l"].values)
    print(f"  {publisher:<6s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}")

print("\n# Sub-era robustness — 20-day drift, early vs late half (by launch date)")
srt = inc.sort_values("anchor_date")
half = len(srt) // 2
for label, part in (("early", srt.iloc[:half]), ("late", srt.iloc[half:])):
    s = st.one_sample_t(part["post_l"].values)
    print(f"  {label:<5s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}")

print("\n# Random-window placebo (20 seeds x 200 draws per event)")
for label, col, k, tail in (("2wk run-up", "pre_l", 10, "right"),
                            ("20d drift", "post_l", 20, "right")):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail=tail)
    print(f"  {label:<12s} ({tail}-tail): observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 20-day drift t-stat")
x = inc["post_l"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — net of costs (entry date is calendar-known months ahead; exec lag = 0)")
for base, label in (("pre_l", "2-week run-up"), ("post_l", "20-day drift")):
    g = st.one_sample_t(inc[base].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<14s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(drift=0.0, seed=846 + s, k=20)["t"] for s in range(20)])
print(f"  null (drift=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for drift in (0.02, 0.04):
    planted = st.synthetic_detect(drift=drift, seed=846, k=20)
    print(f"  planted launch drift=+{drift*100:.1f}% (seed 846): mean AR {planted['mean']*100:+.3f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
