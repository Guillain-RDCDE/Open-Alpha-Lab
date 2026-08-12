"""Reproducible headline run for Study 844 — Madden-Cover-Curse.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EA / TTWO / SPY tapes under
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

from madden_curse import data as dt, strategy as st  # noqa: E402

print("# Madden-Cover-Curse — do EA / TTWO get an abnormal return around a game launch?")

n_madden = sum(1 for e in dt.EVENTS if e[1] == "EA")
n_2k = sum(1 for e in dt.EVENTS if e[1] == "TTWO")
print(f"calendar: {len(dt.EVENTS)} annual launches ({n_madden} Madden -> EA, {n_2k} NBA 2K "
      f"-> TTWO), 2015->2024, hardcoded from Wikipedia / publisher press releases")

if not dt.have_real():
    print("(cache miss — fetching EA + TTWO + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Madden-Cover-Curse panel (EA + TTWO + SPY, adjusted/total-return)",
                 panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} launches have publisher+SPY "
      f"coverage around the anchor")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")

print("\n# THE RUN-UP (the 'buy the hype') — abnormal return, K sessions BEFORE the "
      "launch, publisher minus SPY (gross)")
for label, col in (("1-week run-up (k=5)", "pre_s"), ("2-week run-up (k=10)", "pre_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    nw = st.newey_west_t(inc.sort_values("anchor_date")[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"t_NW={nw:+.3f}  hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE LAUNCH DRIFT (the 'sell/ride the news') — abnormal return, K sessions AFTER")
for label, col in (("1-week drift (k=5)", "post_s"), ("2-week drift (k=10)", "post_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    nw = st.newey_west_t(inc.sort_values("anchor_date")[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"t_NW={nw:+.3f}  hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Per-publisher split of the 2-week launch drift (EA vs TTWO)")
for publisher in dt.PUBLISHERS:
    sub = inc[inc["publisher"] == publisher]
    s = st.one_sample_t(sub["post_l"].values)
    print(f"  {publisher:<5s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}")

print("\n# Sub-era robustness — 2-week launch drift, early vs late half (by launch date)")
srt = inc.sort_values("anchor_date")
half = len(srt) // 2
for label, part in (("2015-2019", srt.iloc[:half]), ("2020-2024", srt.iloc[half:])):
    s = st.one_sample_t(part["post_l"].values)
    print(f"  {label} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}")

print("\n# Random-window placebo (20 seeds x 200 draws per event)")
for label, col, k, tail in (("2wk run-up", "pre_l", 10, "right"),
                            ("2wk drift", "post_l", 10, "right")):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail=tail)
    print(f"  {label:<12s} ({tail}-tail): observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 2-week launch drift t-stat")
x = inc["post_l"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — the launch drift net of costs (entry date is calendar-known "
      "years ahead; execution lag = 0)")
for base, label in (("pre_l", "2-week run-up"), ("post_l", "2-week drift")):
    g = st.one_sample_t(inc[base].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<14s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(drift=0.0, seed=844 + s, k=10)["t"] for s in range(20)])
print(f"  null (drift=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for drift in (0.02, 0.04):
    planted = st.synthetic_detect(drift=drift, seed=844, k=10)
    print(f"  planted launch drift=+{drift*100:.1f}% (seed 844): mean AR {planted['mean']*100:+.3f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
