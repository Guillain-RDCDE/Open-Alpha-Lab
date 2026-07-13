"""Reproducible headline run for Study 739 — Wildfire-Season.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached utility/insurer/SPY tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from wildfire_season import data, strategy as st  # noqa: E402

PRE, POST = 1, 5

print("# Wildfire-Season — is California fire season a tradable risk event for")
print("#  the state's utilities (EIX/PCG) and property insurers (ALL/TRV/MCY/CB)?")

fires = data.fire_table()
print(f"\nfire table: {len(fires)} major California wildfire events "
      f"{fires['date'].min().date()} -> {fires['date'].max().date()} (hardcoded; "
      f"{int(fires['utility_linked'].sum())} utility-linked, "
      f"{int((~fires['utility_linked']).sum())} non-utility)")

if not data.have_real():
    print("(cache miss — fetching EIX/PCG/ALL/TRV/MCY/CB/SPY once)")
    data.fetch()

series = data.load_real()
for t in data.ALL_TICKERS:
    print(data_stamp(f"{t} close", series[t].to_frame("Close"), asof=data.AS_OF))

ret_spy = st.daily_returns(series["SPY"])
ar_spy = st.abnormal_returns(ret_spy)
basket = st.basket_returns(series, data.BASKET_TICKERS)
ar_bk = st.abnormal_returns(basket)
util = st.basket_returns(series, data.UTIL_TICKERS)
ar_util = st.abnormal_returns(util)
ins = st.basket_returns(series, data.INS_TICKERS)
ar_ins = st.abnormal_returns(ins)

# ---- THE HEADLINE — ignition-day (day 0) abnormal basket return ----
print(f"\n# THE HEADLINE — combined basket ignition-day (day 0) abnormal return, "
      f"window [-{PRE}..+{POST}]")
d0 = st.day0_stats(ar_bk, fires["date"], pre=PRE, post=POST)
wlo, whi = st.wilson_interval(d0["hit_down"], d0["n"])
print(f"  n = {d0['n']} events on the tape")
print(f"  mean ignition-day abnormal basket return: {d0['mean']*1e4:+.2f} bps   "
      f"one-sample t = {d0['t']:+.3f}")
print(f"  hit rate (basket down on the day)        : {d0['hit_down']}/{d0['n']} = "
      f"{d0['hit_down']/d0['n']*100:.1f}%  (Wilson 95% [{wlo*100:.1f}%, {whi*100:.1f}%])")

print(f"\n# Random-calendar placebo (20 seeds x 1,000 draws of {d0['n']} random non-fire days)")
draws0 = np.concatenate([
    st.placebo_distribution(ar_bk, d0["n"], pre=PRE, post=POST, n_draws=1000,
                             seed=739 + s, stat="day0") for s in range(20)])
p0 = st.placebo_pvalue(d0["mean"], draws0, tail="left")
print(f"  observed {d0['mean']*1e4:+.2f} bps vs placebo mean {draws0.mean()*1e4:+.2f} bps "
      f"(sd {draws0.std()*1e4:.2f}) over {len(draws0):,} draws -> left-tail p = {p0:.3f}")

# ---- Event window ----
print(f"\n# Event window — mean abnormal basket return by offset [-{PRE}..+{POST}] (own t)")
cp = st.car_path_stats(ar_bk, fires["date"], pre=PRE, post=POST)
for k, row in cp.iterrows():
    tag = "  <-- ignition day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_ar']*1e4:+8.2f} bps   CAR {row['car']*1e4:+8.2f} bps  "
          f"(t={row['t']:+.2f}){tag}")

# ---- The liability window — [+1..+5] ----
print(f"\n# THE LIABILITY WINDOW — cumulative abnormal return [+1..+{POST}] (does the")
print("  drop arrive AFTER ignition, as who-caused-it news lands?)")
rev = st.reversal_stats(ar_bk, fires["date"], pre=PRE, post=POST)
plo, phi = st.block_bootstrap_ci(rev["per_event"])
jk = st.jackknife_range(rev["per_event"])
print(f"  combined basket [+1..+5] CAR: {rev['mean']*1e4:+.2f} bps   one-sample t = "
      f"{rev['t']:+.3f}  (n={rev['n']})")
print(f"  bootstrap 95% CI on the mean: [{plo*1e4:+.1f}, {phi*1e4:+.1f}] bps")
print(f"  leave-one-out |t| range: [{jk['t_min']:+.2f}, {jk['t_max']:+.2f}]; "
      f"|t|<2 in {jk['n_below2']}/{jk['n']} drops")

# ---- Utility leg vs insurer leg ----
print("\n# UTILITY LEG vs INSURER LEG — which half carries the drop? ([+1..+5] window)")
lc = st.leg_compare(ar_util, ar_ins, fires["date"], pre=PRE, post=POST, window="post")
print(f"  utility leg (EIX/PCG) [+1..+5] CAR: {lc['util_mean']*1e4:+.2f} bps  (t={lc['util_t']:+.2f})")
print(f"  insurer leg (ALL/TRV/MCY/CB) [+1..+5] CAR: {lc['ins_mean']*1e4:+.2f} bps  (t={lc['ins_t']:+.2f})")
print(f"  paired (util - ins) difference: {lc['diff_mean']*1e4:+.2f} bps  (t={lc['diff_t']:+.2f})")

# ---- Utility-linked subset ----
print("\n# UTILITY-LINKED SUBSET — restrict to fires an IOU's equipment (likely) started")
ul = fires.loc[fires["utility_linked"], "date"]
rev_ul = st.reversal_stats(ar_util, ul, pre=PRE, post=POST)
d0_ul = st.day0_stats(ar_util, ul, pre=PRE, post=POST)
jk_ul = st.jackknife_range(rev_ul["per_event"])
print(f"  utility leg, utility-linked fires only (n={rev_ul['n']}):")
print(f"    ignition-day: {d0_ul['mean']*1e4:+.2f} bps (t={d0_ul['t']:+.2f})")
print(f"    [+1..+5] CAR: {rev_ul['mean']*1e4:+.2f} bps (t={rev_ul['t']:+.2f})   "
      f"leave-one-out |t|<2 in {jk_ul['n_below2']}/{jk_ul['n']} drops")

# ---- Basket vs SPY extra drop ----
print("\n# EXTRA DROP vs MARKET — does the CA basket fall harder than SPY? ([+1..+5])")
ed = st.extra_drop(ar_spy, ar_bk, fires["date"], pre=PRE, post=POST, window="post")
print(f"  basket [+1..+5] {ed['basket_mean']*1e4:+.2f} bps  vs  SPY {ed['spy_mean']*1e4:+.2f} bps")
print(f"  paired (basket - SPY) difference: {ed['mean_diff']*1e4:+.2f} bps  (t={ed['t']:+.2f}, n={ed['n']})")

# ---- The seasonal half ----
print("\n# THE SEASONAL HALF — Jul->Dec fire window vs the rest of the year")
print("  (combined-basket daily abnormal return; sell-in-July for California risk?)")
seas = st.seasonal_test(ar_bk, fire_months=data.FIRE_MONTHS)
sp = st.seasonal_placebo(ar_bk, k_months=len(data.FIRE_MONTHS), n_draws=4000, seed=739)
p_seas = float((np.abs(sp) >= abs(seas["diff"])).mean())
print(f"  fire-window mean {seas['in_mean']*1e4:+.3f} bps/day (n={seas['n_in']}) vs "
      f"rest {seas['out_mean']*1e4:+.3f} bps/day (n={seas['n_out']})")
print(f"  gap (in - out) {seas['diff']*1e4:+.3f} bps/day   Welch t = {seas['t']:+.3f}   "
      f"random-window two-sided p = {p_seas:.3f}")

# ---- The timer ----
print("\n# THE TIMER — SHORT the basket on the ignition headline (fires are bad news for")
print("  the basket, so the folklore trade is a short). Enter at the ignition-session")
print("  close, hold N sessions; shorts pay 5 bps one-way (x2) + 300 bps/yr borrow.")
for hold in (5, 10, 21):
    lg_g = st.fire_timer(st.basket_nav(series, data.BASKET_TICKERS), fires["date"], hold=hold,
                         cost_bps=0.0, borrow_bps_annual=0.0)
    g = st.summarize_timer(lg_g, col="ret_gross")
    lg_n = st.fire_timer(st.basket_nav(series, data.BASKET_TICKERS), fires["date"], hold=hold,
                         cost_bps=5.0, borrow_bps_annual=300.0)
    n_ = st.summarize_timer(lg_n, col="ret_net")
    print(f"  hold {hold:>2d}d: gross {g['mean_bps']:+8.2f} bps (median {g['median_bps']:+7.2f})  "
          f"net {n_['mean_bps']:+8.2f} bps  t(net)={n_['t']:+.2f}  win {n_['win_rate']*100:.0f}%")

# ---- Synthetic control ----
print("\n# Synthetic positive control — deterministic, no network")
print("  the day0 detector must NOT fire on a null world (dip=0) and must recover a")
print("  planted event-day dip. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    close, ev = data.synthetic_world(dip=0.0, seed=739 + s_)
    null_ts.append(st.synthetic_detect(close, ev, stat="day0")["t"])
null_ts = np.asarray(null_ts)
print(f"  null (dip=0), 20 seeds: mean t = {null_ts.mean():+.2f}  (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
close, ev = data.synthetic_world(dip=-0.03, seed=739)
sy = st.synthetic_detect(close, ev, stat="day0")
print(f"  planted dip=-3.0% (seed 739): mean {sy['mean']*1e4:+.1f} bps   t = {sy['t']:+.2f}")
