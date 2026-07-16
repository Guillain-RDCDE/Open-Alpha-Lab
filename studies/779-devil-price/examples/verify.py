"""Reproducible headline run for Study 779 — Devil-Price.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached S&P 100 + SPY tapes under
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

from devil_price import data as dt, strategy as st  # noqa: E402

print("# Devil-Price — do stocks sitting on a '666' price level then underperform?")
print(f"universe: {len(dt.UNIVERSE)} S&P 100 names + {dt.BENCHMARK}; devil level = mantissa "
      f"6.66 (log-price {dt.DEVIL_LOG:.4f}); weekly rebalance, quintile sort")

if not dt.have_real():
    print("(cache miss — fetching S&P 100 + SPY once)")
    dt.fetch()

panel, spy = dt.load_real()
stamp_df = panel.join(spy.rename("SPY")).dropna(how="all")
print(data_stamp("Devil-Price panel (S&P 100 + SPY, adjusted/total-return)", stamp_df,
                 asof=dt.AS_OF))

events = st.build_panel_events(panel, spy)
table = st.build_sort_table(events)
ns = [e["n"] for e in events]
print(f"\nrebalances resolved: {len(events)} weekly cross-sections "
      f"{table['date'].iloc[0].date()} -> {table['date'].iloc[-1].date()}; "
      f"cross-section size min/median/max = {min(ns)}/{int(np.median(ns))}/{max(ns)}")

print("\n# THE DEVIL QUINTILE — demeaned forward return (near-666 minus peer mean)")
for label, col in (("1-week (k=5)", "devil_s_dm"), ("1-month (k=21)", "devil_l_dm")):
    s = st.one_sample_t(table[col].values)
    print(f"  {label:<16s} n={s['n']:4d}  mean={s['mean']*100:+.4f}%  t={s['t']:+.3f}")

print("\n# THE LONG/SHORT SPREAD — clean minus devil quintile (folklore: positive)")
for label, col in (("1-week LS (k=5)", "ls_s"), ("1-month LS (k=21)", "ls_l")):
    s = st.one_sample_t(table[col].values)
    hr = st.hit_rate(table[col].values)
    print(f"  {label:<18s} n={s['n']:4d}  mean={s['mean']*100:+.4f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# DEVIL QUINTILE vs MARKET — near-666 minus SPY forward return")
for label, col in (("1-week vs SPY", "devil_s_abn"), ("1-month vs SPY", "devil_l_abn")):
    s = st.one_sample_t(table[col].values)
    print(f"  {label:<16s} n={s['n']:4d}  mean={s['mean']*100:+.4f}%  t={s['t']:+.3f}")

print("\n# Random-bucket placebo (20 seeds x 100 draws) — does the devil sort explain the spread?")
for label, h in (("1-week LS", "s"), ("1-month LS", "l")):
    pl = st.placebo_pvalue(events, horizon=h, tail="right")
    print(f"  {label:<12s} observed {pl['obs']*100:+.4f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.4f}% (sd {pl['placebo_sd']*100:.4f}%) over "
          f"{pl['n_draws']:,} draws -> right-tail p = {pl['p_value']:.4f}")

print("\n# Leave-one-year-out jackknife — 1-week LS spread t")
jk = st.jackknife_years(table, col="ls_s")
print(f"  full-sample t = {jk['full_t']:+.3f}  |  jackknife t range "
      f"[{jk['lo']:+.3f}, {jk['hi']:+.3f}] across {jk['n_years']} leave-one-year-out draws")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=789 + s, side="devil")["t"]
                    for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (0.01, 0.02):
    planted = st.synthetic_detect(bump=bump, seed=789, side="devil")
    print(f"  planted devil drag -{bump*100:.1f}% (seed 789): devil-quintile mean "
          f"{planted['mean']*100:+.4f}%  t = {planted['t']:+.2f}  (n={planted['n']} rebalances)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
