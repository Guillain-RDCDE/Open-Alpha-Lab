"""Reproducible headline run for Study 851 — Netflix Password Crackdown.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached NFLX / SPY / QQQ tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
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

from nflx_crackdown import data, strategy as st  # noqa: E402

PRE, POST = 1, 5

print("# Netflix Password Crackdown — did the 'scary policy that worked' show up as an")
print("#  abnormal-return signal around its five public dates? (N=5 -> a case study, not")
print("#  a factor: low power by construction.)")

events = data.event_table()
print(f"\ncrackdown calendar: {len(events)} public market-facing dates "
      f"{events['date'].min().date()} -> {events['date'].max().date()} (hardcoded from "
      "the Netflix shareholder letters / newsroom; earnings react next session)")
for _, r in events.iterrows():
    print(f"  {r['date'].date()}  {r['label']}")

if not data.have_real():
    print("(cache miss — fetching NFLX / SPY / QQQ once)")
    data.fetch()

px = data.load_real()
for t in data.TICKERS:
    print(data_stamp(f"{t} close", px[t].to_frame("Close"), asof=data.AS_OF))

r_n = st.daily_returns(px["NFLX"])
r_s = st.daily_returns(px["SPY"])
r_q = st.daily_returns(px["QQQ"])

print("\n# PER-EVENT — event-session (day 0) abnormal NFLX return, market model vs SPY")
mat, kept, betas = st.event_car(r_n, r_s, events["date"], PRE, POST, model="market")
idx = r_n.index
for d, row, b in zip(kept, mat, betas):
    pos = idx.searchsorted(d)
    print(f"  {d.date()}: raw NFLX {r_n.iloc[pos]*100:+6.2f}%  SPY {r_s.iloc[pos]*100:+5.2f}%  "
          f"-> abnormal {row[PRE]*100:+6.2f}%  (est. beta {b:.2f}, winCAR {row.sum()*100:+.2f}%)")

print("\n# THE HEADLINE — cross-event mean event-session abnormal return (vs SPY)")
d0 = st.day0_stats(r_n, r_s, events["date"], PRE, POST, model="market")
wlo, whi = st.wilson_interval(d0["hit"], d0["n"])
print(f"  mean {d0['mean']*100:+.2f}%  one-sample t = {d0['t']:+.3f}  (n={d0['n']})")
print(f"  up-days: {d0['hit']}/{d0['n']}  (Wilson 95% [{wlo*100:.0f}%, {whi*100:.0f}%])")
blo, bhi = st.block_bootstrap_ci(d0["per_event"], n_boot=5000, seed=851)
print(f"  event-bootstrap 95% CI on the mean: [{blo*100:+.1f}%, {bhi*100:+.1f}%]")
wc = st.window_car_stats(r_n, r_s, events["date"], PRE, POST, model="market")
print(f"  whole-window CAR [-{PRE}..+{POST}]: mean {wc['mean']*100:+.2f}%  t = {wc['t']:+.3f}")
pc = st.post_car_stats(r_n, r_s, events["date"], PRE, POST, model="market")
print(f"  post-event CAR [+1..+{POST}]     : mean {pc['mean']*100:+.2f}%  t = {pc['t']:+.3f}")

print("\n# BENCHMARK CROSS-CHECK — same test vs QQQ (tech-heavy)")
d0q = st.day0_stats(r_n, r_q, events["date"], PRE, POST, model="market")
print(f"  vs QQQ: mean day0 abnormal {d0q['mean']*100:+.2f}%  t = {d0q['t']:+.3f}")

print("\n# RANDOM-CALENDAR PLACEBO — is 5 real dates unlike 5 random ones?")
pl = st.placebo_distribution(r_n, r_s, d0["n"], PRE, POST, model="market",
                             n_draws=4000, seed=851, stat="day0")
p_right = st.placebo_pvalue(d0["mean"], pl, "right")
p_left = st.placebo_pvalue(d0["mean"], pl, "left")
print(f"  day0: observed {d0['mean']*100:+.2f}% vs placebo mean {pl.mean()*100:+.2f}% "
      f"(sd {pl.std()*100:.2f}%) over {len(pl):,} draws")
print(f"        right-tail p = {p_right:.3f}  left-tail p = {p_left:.3f}  "
      "(the observed sits in the LEFT tail — one crash outlier, wrong sign for the claim)")

print("\n# ROBUSTNESS — the whole result is one 2022-04-20 crash")
sub = events.loc[events["date"] != "2022-04-20", "date"]
d0s = st.day0_stats(r_n, r_s, sub, PRE, POST, model="market")
print(f"  drop the Q1'22 announcement crash (n=4): mean {d0s['mean']*100:+.2f}%  t = {d0s['t']:+.3f}")
conf = events.loc[events["date"].isin(pd.to_datetime(["2023-07-20", "2023-10-19"])), "date"]
d0c = st.day0_stats(r_n, r_s, conf, PRE, POST, model="market")
print(f"  confirmation earnings only (Q2+Q3'23, n=2): mean {d0c['mean']*100:+.2f}%  t = {d0c['t']:+.3f}")
print("  -> with the crash it is significantly NEGATIVE; without it, nothing. N=5 has no power.")

print("\n# THE TIMER — long NFLX from the event-session close, held N sessions (long-only)")
print("  one round trip of one-way costs charged twice vs the unconditional NFLX N-day baseline")
for h in (1, 3, 5, 10, 21):
    g = st.summarize_trade(st.buy_the_event(px["NFLX"], events["date"], hold=h, cost_bps=0.0), "ret_gross")
    n10 = st.summarize_trade(st.buy_the_event(px["NFLX"], events["date"], hold=h, cost_bps=10.0), "ret_net")
    base = float((px["NFLX"].shift(-h) / px["NFLX"] - 1.0).mean() * 1e4)
    print(f"  hold {h:>2d}d: gross {g['mean_bps']:+8.1f} bps  net(10bps) {n10['mean_bps']:+8.1f} bps  "
          f"win {n10['win_rate']*100:.0f}%  t(net) = {n10['t']:+.2f}  uncond NFLX {base:+.1f} bps")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  a proper power/unbiasedness proof on 30 scheduled pseudo-events: the null must")
print("  not fire, a planted jump must light up.")
null_t = np.array([st.synthetic_detect(*data.synthetic_world(edge=0.0, seed=851 + s))["t"]
                   for s in range(20)])
print(f"  null (edge=0), 20 seeds: mean t = {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_t) >= 2).sum()}/20 seeds")
a, m, e = data.synthetic_world(edge=0.03, seed=851)
sy = st.synthetic_detect(a, m, e)
print(f"  planted (edge=0.03, seed 851): mean day0 {sy['mean']*100:+.2f}%  t = {sy['t']:+.2f}")
print("  (a faithful-engine / power check only — never cited for the real-tape stamp)")
