"""Reproducible headline run for Study 845 — Stadium Naming-Rights Curse.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / sponsor tapes under
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

from stadium_curse import data, strategy as st  # noqa: E402

print("# Stadium Naming-Rights Curse — does buying a stadium's name hex the sponsor?")
print("# (Managerial-hubris / peak-earnings signaling; folklore anchored on Enron & FTX.)")

table = data.deal_table()
tradable = data.tradable_deals()
print(f"\ndeal table: {len(table)} naming-rights deals "
      f"{table['date'].min().date()} -> {table['date'].max().date()} (hardcoded); "
      f"{len(tradable)} have a listed, tradable sponsor tape; "
      f"{len(table) - len(tradable)} untradable cautionary tales "
      "(Enron, MCI/WorldCom, FTX, Crypto.com, SoFi)")

if not data.have_real():
    print("(cache miss — fetching SPY + sponsor tickers once)")
    data.fetch()

spy, prices = data.load_prices()
print(f"\n[data] SPY: {len(spy):,} rows  {spy.index.min().date()} -> {spy.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint={data.fingerprint(spy)}")
print(f"[data] sponsor tapes cached: {len(prices)} of {len(data.tradable_tickers())} tradable "
      "tickers (Comerica/CMA has no Yahoo tape — a named no-coverage drop)")

for W, label in ((252, "1-year"), (504, "2-year")):
    print(f"\n# ======== {label} forward window (window={W} sessions) ========")
    cs = st.car_stats(spy, prices, tradable, window=W)
    wlo, whi = cs["wilson"]
    print(f"  deals on tape (window fits): n = {cs['n']}")
    print(f"  mean sponsor BHAR vs SPY : {cs['mean']*100:+.2f}%   "
          f"median {cs['median']*100:+.2f}%")
    print(f"  one-sample t             : {cs['t']:+.3f}   (NW t = {cs['t_nw']:+.3f})")
    print(f"  hit rate (sponsor < SPY) : {cs['hit']}/{cs['n']} = {cs['hit_rate']*100:.1f}%  "
          f"(Wilson 95% [{wlo*100:.1f}%, {whi*100:.1f}%])")

    es = st.era_split(spy, prices, tradable, window=W)
    print(f"  era split: pre-2010  n={es['pre']['n']}  mean {es['pre']['mean']*100:+.2f}%  "
          f"t={es['pre']['t']:+.2f}   |   post-2010  n={es['post']['n']}  "
          f"mean {es['post']['mean']*100:+.2f}%  t={es['post']['t']:+.2f}")

    pl = st.placebo_pvalue(spy, prices, tradable, window=W, n_draws=3000, seed=845)
    print(f"  placebo (random entry dates on the same names, {pl['n_draws']} draws): "
          f"observed {pl['obs']*100:+.2f}% vs placebo mean {pl['placebo_mean']*100:+.2f}% "
          f"(sd {pl['placebo_sd']*100:.2f}%) -> left-tail p = {pl['p_left']:.3f}")

    ov = st.curse_overlay(spy, prices, tradable, window=W, cost_bps=5.0, borrow_bps_yr=100.0)
    print(f"  tradable overlay (short sponsor / long SPY): gross {ov['gross_mean']*100:+.2f}%  "
          f"net {ov['net_mean']*100:+.2f}%  t(net)={ov['t_net']:+.2f}  "
          f"win {ov['win_rate']*100:.0f}%  cost drag {ov['cost_drag_bps']:.0f} bps")

print("\n# The worst individual sponsors (1-year BHAR) — is the 'curse' a few names?")
_, kept = st.stack_bhar(spy, prices, tradable, window=252)
kept = kept.sort_values("bhar")
for _, r in kept.head(6).iterrows():
    print(f"    {r['ticker']:6s} {r['date'].date()}  {r['bhar']*100:+7.2f}%  {r['venue']}")
print("  ... best:")
for _, r in kept.tail(3).iterrows():
    print(f"    {r['ticker']:6s} {r['date'].date()}  {r['bhar']*100:+7.2f}%  {r['venue']}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the cross-event BHAR test must NOT fire on a null world (edge=0) and must")
print("  recover a planted post-deal underperformance. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    spy_s, pr_s, ev_s = data.synthetic_world(edge=0.0, seed=845 + s_)
    null_ts.append(st.synthetic_detect(spy_s, pr_s, ev_s, window=252)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean t = {null_ts.mean():+.2f}  (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
spy_s, pr_s, ev_s = data.synthetic_world(edge=-0.25, seed=845)
sy = st.synthetic_detect(spy_s, pr_s, ev_s, window=252)
print(f"  planted curse edge=-25% (seed 845): mean {sy['mean']*100:+.1f}%   t = {sy['t']:+.2f}")

print("\n# Fingerprints (per-sponsor, for the record):")
for t in sorted(prices):
    print(f"    {t:6s} {data.fingerprint(prices[t])}")
