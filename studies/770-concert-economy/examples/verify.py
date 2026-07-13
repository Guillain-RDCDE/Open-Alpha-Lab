"""Reproducible headline run for Study 770 — Concert-Economy.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached LYV / SPY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

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

from concert_economy import data as dt, strategy as st  # noqa: E402

print("# Concert-Economy — does Live Nation (LYV) rally INTO festival season?")

n_held = sum(1 for _, fr, c in dt.EVENTS if c is None)
print(f"calendar: {len(dt.EVENTS)} Coachella editions {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      f"{n_held} held (2020 & 2021 COVID-cancelled), hardcoded from Wikipedia")

if not dt.have_real():
    print("(cache miss — fetching LYV + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Concert panel (LYV + SPY, adjusted/total-return)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} Coachella editions have LYV+SPY "
      f"coverage around the anchor")
print("excluded, by reason:")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  {n:2d}x {reason}")

print("\n# THE RUN-UP (the 'rally into') — abnormal return, K sessions BEFORE Coachella, "
      "LYV minus SPY (gross)")
for label, col in (("1-month run-up (k=21)", "ru_1mo"), ("2-month run-up (k=42)", "ru_2mo")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<24s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event; right-tail p)")
for label, col, k in (("1-month run-up", "ru_1mo", 21), ("2-month run-up", "ru_2mo", 42)):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail="right")
    print(f"  {label:<16s}: observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 1-month run-up t-stat")
x = inc["ru_1mo"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — the run-up net of costs (calendar-known entry, zero look-ahead)")
for base, label in (("ru_1mo", "1-month"), ("ru_2mo", "2-month")):
    g = st.one_sample_t(inc[base].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<8s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# THIRD AXIS -- 'sell the news?' in-season abnormal return (Coachella -> ~Labor Day)")
s_in = st.one_sample_t(inc["during"].values)
hr_in = st.hit_rate(inc["during"].values)
pl_in = st.placebo_pvalue(ev, prices, "during", k=st.SEASON_K, tail="left")
print(f"  in-season (k={st.SEASON_K})  n={s_in['n']}  mean={s_in['mean']*100:+.3f}%  "
      f"t={s_in['t']:+.3f}  hit {hr_in['k']}/{hr_in['n']}={hr_in['rate']*100:.1f}%")
print(f"  in-season placebo (left-tail): observed {pl_in['obs']*100:+.3f}% vs placebo mean "
      f"{pl_in['placebo_mean']*100:+.3f}%  -> p = {pl_in['p_value']:.4f}")

print("\n# Fundamental backdrop (LABELLED PROXY, 10-K/10-Q reconstruction)")
qs = dt.quarterly_share()
print(f"  revenue seasonality shares: " + "  ".join(f"{q} {v*100:.0f}%" for q, v in qs.items())
      + f"  (Q3 summer touring is the peak)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the run-up one-sample-t detector must NOT fire on a null world (bump=0) and must "
      "recover a planted pre-festival bump. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=770 + s, k=21)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (0.01, 0.02):
    planted = st.synthetic_detect(bump=bump, seed=770, k=21)
    print(f"  planted run-up bump=+{bump*100:.1f}% (seed 770): mean AR {planted['mean']*100:+.3f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
