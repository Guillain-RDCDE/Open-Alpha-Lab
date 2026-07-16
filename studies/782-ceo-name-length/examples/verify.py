"""Reproducible headline run for Study 782 — CEO-Name-Length.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached tapes under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

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

from ceo_name_length import data as dt, strategy as st  # noqa: E402

print("# CEO-Name-Length — does the length of the CEO's surname predict returns?")
print(f"universe: {len(dt.UNIVERSE)} large caps + {dt.BENCHMARK}; characteristic = "
      f"len(CEO surname), a STATIC 2026-06 snapshot (CEO turnover NOT tracked — disclosed).")

if not dt.have_real():
    print("(cache miss — fetching universe + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("CEO-name-length panel (40 names + SPY, adjusted/total-return)", panel, asof=dt.AS_OF))

ch = dt.characteristics()
print(f"\nsurname-length characteristic: min={int(ch.min())} (e.g. {ch.idxmin()}), "
      f"max={int(ch.max())} (e.g. {ch.idxmax()}), mean={ch.mean():.2f}, sd={ch.std():.2f}")

rets = dt.monthly_returns(prices)
print(f"monthly cross-section: {rets.shape[0]} months x {rets.shape[1]} names, "
      f"{rets.index.min().date()} -> {rets.index.max().date()}")

tm = st.tercile_means(rets, ch)
print(f"\ntercile legs: short = {tm['n_short']} shortest surnames, long = {tm['n_long']} longest")
print(f"  short-leg mean monthly return = {tm['short_leg_mean']*100:+.3f}%")
print(f"  long-leg  mean monthly return = {tm['long_leg_mean']*100:+.3f}%")

print("\n# THE LONG/SHORT BOOK — long longest-surname tercile, short shortest (dollar-neutral)")
ls = st.long_short_series(rets, ch)
s = st.one_sample_t(ls.values)
hr = st.hit_rate(ls.values)
sr = st.sharpe(ls.values)
print(f"  monthly LS: n={s['n']}  mean={s['mean']*100:+.3f}%  sd={s['sd']*100:.3f}%  "
      f"t={s['t']:+.3f}  ann.Sharpe={sr:+.3f}")
print(f"  hit rate (positive months): {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
      f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")
print(f"  annualised LS mean ~ {s['mean']*12*100:+.2f}%/yr")

print("\n# Label-shuffle placebo (20 seeds x 200 draws; permute surname lengths across names)")
pl = st.placebo_pvalue(rets, ch, tail="two")
print(f"  observed LS mean {pl['obs']*100:+.3f}%  vs placebo mean {pl['placebo_mean']*100:+.3f}% "
      f"(sd {pl['placebo_sd']*100:.3f}%) over {pl['n_draws']:,} draws -> two-tail p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-ticker-out) — LS t-stat range")
jk = st.jackknife_t(rets, ch)
print(f"  full-sample t = {s['t']:+.3f}  |  jackknife t range [{jk['lo']:+.3f}, {jk['hi']:+.3f}] "
      f"across {jk['n']} leave-one-out draws")

print("\n# TRADABILITY — net of costs (monthly rebalance; conservative full-turnover charge)")
for cb in (5.0, 10.0):
    lsn = st.long_short_series(rets, ch, cost_bps=cb)
    sn = st.one_sample_t(lsn.values)
    print(f"  net @ {cb:>4.1f} bps/leg: mean {sn['mean']*100:+.3f}%/mo (t={sn['t']:+.2f}, "
          f"Sharpe {st.sharpe(lsn.values):+.2f})")

print("\n# Robustness — a few alternate specs")
for q, lab in ((0.20, "quintile"), (1/3, "tercile"), (0.50, "median-split")):
    lsq = st.long_short_series(rets, ch, q=q)
    sq = st.one_sample_t(lsq.values)
    print(f"  {lab:<12s} (q={q:.2f}): mean {sq['mean']*100:+.3f}%/mo  t={sq['t']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=798 + i)["t"] for i in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (0.004, 0.008):
    planted = st.synthetic_detect(bump=bump, seed=798)
    print(f"  planted slope bump=+{bump:.3f} (seed 798): LS mean {planted['mean']*100:+.3f}%/mo  "
          f"t = {planted['t']:+.2f}")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
