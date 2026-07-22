"""Reproducible headline run for Study 802 — Stock-Split-Modern.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached split calendar + price
panel under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from stock_split_modern import data, strategy as st  # noqa: E402

print("# Stock-Split-Modern — does post-split drift survive in the mega-cap era?")

if not data.have_real():
    print("(cache miss — fetching split calendar + prices once)")
    data.fetch()

prices = data.load_prices()
splits = data.load_splits()
panel = st.build_event_panel(prices, splits)

fp = data.fingerprint_splits(splits)
print(f"[data] splits: {len(splits)} forward-split events "
      f"{splits['date'].min().date()} -> {splits['date'].max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={fp}")
print(f"  events with usable 3m window: {panel['car_3m'].notna().sum()} | "
      f"post-2020: {(panel['era']=='post').sum()} | "
      f"post-2020 mega-cap: {((panel['era']=='post') & panel['is_mega']).sum()}")

print("\n# Raw vs market-adjusted 3m return, modern (post-2020) cohort")
post = panel[panel["era"] == "post"]
print(f"  raw 3m mean:  {post['raw_3m'].mean()*100:+.2f}%   "
      f"market 3m mean: {post['mkt_3m'].mean()*100:+.2f}%   "
      f"ABNORMAL 3m mean: {post['car_3m'].mean()*100:+.2f}%")

print("\n# Cohort table — abnormal (market-adjusted) 3m CAR")
ct = st.cohort_table(panel, col="car_3m")
for name, row in ct.iterrows():
    print(f"  {name:22s}: n={int(row['n']):2d}  mean {row['mean_pct']:+6.2f}%  "
          f"median {row['median_pct']:+6.2f}%  HAC t={row['t_hac']:+.2f}  "
          f"plain t={row['t_plain']:+.2f}  win {row['win']*100:.0f}%")

print("\n# Horizon sweep — modern (post-2020) cohort, abnormal CAR")
hs = st.horizon_sweep(panel, cohort="post")
for lbl, row in hs.iterrows():
    print(f"  {lbl:10s}: n={int(row['n']):2d}  mean {row['mean_pct']:+7.2f}%  "
          f"std {row['std_pct']:6.2f}%  HAC t={row['t_hac']:+.2f}")

print("\n# Around the ex-date [-1,+1] abnormal, modern cohort")
ex = post["car_ex"].dropna().to_numpy()
print(f"  n={len(ex)}  mean {ex.mean()*100:+.2f}%  one-sample t={st.one_sample_t(ex):+.2f}")

print("\n# Placebo — random NON-split entry dates, same names (post-2020 cohort, 3m)")
pl = st.placebo_pvalue(prices, panel, col="car_3m", cohort="post", n_draws=1500)
print(f"  observed split-cohort mean {pl['obs']*100:+.2f}%  vs placebo mean "
      f"{pl['placebo_mean']*100:+.2f}% (sd {pl['placebo_sd']*100:.2f}%) over "
      f"{pl['n_draws']} baskets -> p = {pl['p_value']:.4f}")

print("\n# The timer — buy ex-date close, hold 3m, long-only, abnormal (market-hedged)")
for cb in (5.0, 20.0):
    tm = st.timer_stats(panel, col="car_3m", cost_bps=cb, cohort="post")
    print(f"  cost={cb:>4.1f} bps: gross {tm['gross_pct']:+.2f}% -> net {tm['net_pct']:+.2f}% "
          f"(t={tm['t_net']:+.2f}, Sharpe {tm['sharpe']:.2f}, worst {tm['worst_pct']:+.1f}%, "
          f"n={tm['n']}, ~{tm['events_per_year']:.1f} events/yr)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the HAC detector must NOT fire on a null world (planted=0) and must recover a")
print("  planted post-split abnormal drift. Null checked over 12 seeds.")
null_ts = []
for s_ in range(12):
    pr, mk, sp = data.synthetic_world(planted_bps=0.0, seed=802 + s_)
    null_ts.append(st.synthetic_detect(pr, sp)["t_hac"])
null_ts = np.asarray(null_ts)
print(f"  null (planted=0), 12 seeds: mean HAC t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts)>=2).sum()}/12 seeds")
pr, mk, sp = data.synthetic_world(planted_bps=8.0, seed=802)
sy = st.synthetic_detect(pr, sp)
print(f"  planted +8 bps/day abnormal (seed 802): n={sy['n']} mean CAR {sy['mean_pct']:+.2f}%  "
      f"HAC t = {sy['t_hac']:+.2f}")
