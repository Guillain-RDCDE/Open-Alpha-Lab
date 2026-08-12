"""Reproducible headline run for Study 847 — Rotten-Tomatoes -> Studio.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + six-studio tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp, fingerprint  # noqa: E402

from rotten_tomatoes import data, strategy as st  # noqa: E402

print("# Rotten-Tomatoes -> Studio — does a film's critic tier move the distributing studio?")

films = data.film_table()
n_fresh = int((films["tier"] == "fresh").sum())
n_rotten = int((films["tier"] == "rotten").sum())
print(f"\nfilm table: {len(films)} major wide releases {films['date'].min().date()} -> "
      f"{films['date'].max().date()}  ({n_fresh} fresh / {n_rotten} rotten; hardcoded, "
      "clearly fresh>=75 or rotten<50 only)")

if not data.have_real():
    print("(cache miss — fetching SPY + DIS/WBD/PARA/CMCSA/NFLX/SONY once)")
    data.fetch()

prices = data.load_real()
combined = pd.DataFrame({t: prices[t] for t in data.all_tickers()})
for t in data.all_tickers():
    print(data_stamp(f"{t} close", prices[t].to_frame("Close"), asof=data.AS_OF))
print(f"[panel fingerprint] {fingerprint(combined)}")

events = st.build_event_table(prices)
inc = events[events["included"]]
print(f"\n# included events: {int(events['included'].sum())}/{len(events)} "
      f"({int((inc['tier']=='fresh').sum())} fresh / {int((inc['tier']=='rotten').sum())} rotten)")
if (~events["included"]).any():
    for r in events[~events["included"]].itertuples():
        print(f"    excluded: {r.title} ({r.studio}) — {r.reason}")

for col, label in [("ow_car", "OPENING-WEEKEND CAR [0..+1]"),
                   ("fw_car", "FOLLOWING-WEEK CAR [+2..+6]"),
                   ("full_car", "COMBINED CAR [0..+6]")]:
    ts = st.tier_stats(events, col=col)
    print(f"\n# {label}  (studio market-adjusted abnormal return)")
    print(f"  fresh (n={ts['n_fresh']:>2}): mean {ts['fresh_bps']:+8.1f} bps   "
          f"one-sample t = {ts['fresh_t']:+.3f}")
    print(f"  rotten(n={ts['n_rotten']:>2}): mean {ts['rotten_bps']:+8.1f} bps   "
          f"one-sample t = {ts['rotten_t']:+.3f}")
    print(f"  fresh - rotten GAP: {ts['gap_bps']:+8.1f} bps   Welch t = {ts['gap_welch_t']:+.3f}")

# hit rates on the following-week window
hf = st.hit_rate(events, "fresh", "fw_car", positive=True)
hr = st.hit_rate(events, "rotten", "fw_car", positive=False)
print(f"\n# following-week hit rates (Wilson 95%)")
print(f"  fresh studio UP:   {hf['k']}/{hf['n']} = {hf['rate']*100:.1f}%  "
      f"[{hf['lo']*100:.1f}%, {hf['hi']*100:.1f}%]")
print(f"  rotten studio DOWN:{hr['k']}/{hr['n']} = {hr['rate']*100:.1f}%  "
      f"[{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%]")

print("\n# TIER-LABEL PERMUTATION PLACEBO — fresh-minus-rotten following-week gap")
perm = st.permutation_placebo(events, col="fw_car", n_seeds=20, n_draws_per_seed=1000)
print(f"  observed gap {perm['obs_bps']:+.1f} bps vs placebo {perm['placebo_mean_bps']:+.1f} "
      f"(sd {perm['placebo_sd_bps']:.1f}) over {perm['n_draws']:,} draws -> right-tail "
      f"p = {perm['p_value']:.3f}")

print("\n# RANDOM-DATE PLACEBO — pooled following-week CAR magnitude")
rd = st.random_date_placebo(events, prices, col="fw_car", n_seeds=20, n_draws_per_seed=200)
print(f"  observed pooled {rd['obs_bps']:+.1f} bps vs placebo {rd['placebo_mean_bps']:+.1f} "
      f"(sd {rd['placebo_sd_bps']:.1f}) over {rd['n_draws']:,} draws -> two-sided "
      f"p = {rd['p_value']:.3f}")

print("\n# THE TIMER — long-fresh / short-rotten studio over the following week")
for cb in (0.0, 5.0):
    tm = st.timer_stats(events, col="fw_car", cost_bps=cb)
    print(f"  cost {cb:>4.1f} bps/leg: gross {tm['gross_bps']:+7.1f} bps  net {tm['net_bps']:+7.1f} bps  "
          f"t(net) = {tm['t_net']:+.2f}  (n legs = {tm['n']})")

print("\n# SYNTHETIC POSITIVE CONTROL — the machinery is unbiased (no network)")
null_t = np.array([st.synthetic_detect(edge=0.0, seed=847 + s)["gap_welch_t"] for s in range(20)])
print(f"  null (edge=0), 20 seeds: mean gap-t = {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_t) >= 2).sum()}/20 seeds")
plant_t = np.array([st.synthetic_detect(edge=0.004, seed=847 + s)["gap_welch_t"] for s in range(20)])
print(f"  planted edge=0.004/day, 20 seeds: mean gap-t = {plant_t.mean():+.2f} "
      f"(sd {plant_t.std(ddof=1):.2f}), |t|>=2 in {(np.abs(plant_t) >= 2).sum()}/20 seeds")
sd0 = st.synthetic_detect(edge=0.004, seed=847)
print(f"  planted edge=0.004/day (seed 847): gap {sd0['gap_bps']:+.1f} bps  Welch t = {sd0['gap_welch_t']:+.2f}")
