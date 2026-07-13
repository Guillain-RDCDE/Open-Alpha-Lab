"""Reproducible headline run for Study 769 - prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The equity numbers come from the cached yfinance pulls under ``_cache/`` (month-end Adj
Close for DIS, SPY); the parks numbers come from the hardcoded, cited, **approximate**
attendance + ticket-price series in :mod:`disney_parks.data`, released with the TEA/AECOM
Theme Index's real ~mid-following-year lag. Deterministic, offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from disney_parks import data, strategy as st

att = data.load_attendance()
growth = data.attendance_growth()
eq = data.load_equities()
dis, spy = eq["DIS"], eq["SPY"]
frame = data.build_frame()

print("# Parks attendance - HARDCODED, CITED, APPROXIMATE (a labelled proxy, not a feed)")
print(f"window        : {att.index[0].year} - {att.index[-1].year} (year-end, millions of visits)")
print("levels        : " + ", ".join(f"{y}:{v:.0f}" for y, v in zip(att.index.year, att.values)))
print("YoY growth %  : " + ", ".join(f"{y}:{v:+.0f}" for y, v in zip(growth.index.year, growth.values)))
print(f"release lag   : Theme Index for year Y is public ~{data.THEME_INDEX_RELEASE_MONTH:02d}/Y+1 "
      f"(e.g. 2019 -> {data.release_date(2019).date()})")

print("\n# Context - does DIS even beat SPY over the window? (month-end Adj Close)")
for name, s in (("DIS", dis), ("SPY", spy)):
    m = st.summarize(s)
    print(f"  {name}: CAGR {m['cagr']*100:6.2f}%  vol {m['vol']*100:5.1f}%  "
          f"Sharpe {m['sharpe']:.2f}  maxDD {m['mdd']*100:6.1f}%  (n={m['n']} mo)")
ae = st.annual_excess_t(dis, spy)
print(f"  annual excess (DIS - SPY): {ae['mean_excess']*100:+.2f}%/yr  t = {ae['t']:+.3f}  "
      f"(p = {ae['p']:.3f}, n = {ae['n']} yrs)")

print("\n# Lead-lag - does release-lagged parks momentum predict DIS forward 12m return?")
lla = st.lead_lag(frame, "pg", horizon=12, lag=1, excess=False)
lle = st.lead_lag(frame, "pg", horizon=12, lag=1, excess=True)
llp = st.lead_lag(frame, "ph", horizon=12, lag=1, excess=True)
print(f"  DIS  fwd12  ~ parks growth : slope {lla['slope']:+.5f}  NW t = {lla['t']:+.2f}  (n={lla['n']})")
print(f"  DIS-SPY fwd12 ~ parks growth: slope {lle['slope']:+.5f}  NW t = {lle['t']:+.2f}  (n={lle['n']})  <- the DIS-specific tell")
print(f"  DIS-SPY fwd12 ~ price hike  : slope {llp['slope']:+.5f}  NW t = {llp['t']:+.2f}  (n={llp['n']})")

print("\n# Regime split - DIS forward 12m when parks momentum > 0 vs the base rate (Welch t)")
ra = st.regime_split(frame, horizon=12, lag=1, excess=False)
re = st.regime_split(frame, horizon=12, lag=1, excess=True)
print(f"  absolute : cond {ra['cond_mean']*100:+.2f}% (n={ra['n_cond']}) vs base {ra['base_mean']*100:+.2f}%  "
      f"Welch t = {ra['t']:+.2f}")
print(f"  excess   : cond {re['cond_mean']*100:+.2f}%           vs base {re['base_mean']*100:+.2f}%  "
      f"Welch t = {re['t']:+.2f}  <- vanishes vs SPY")

print("\n# Timing backtest - hold DIS when parks momentum > 0 else SPY (1-mo lag, 10bps/leg)")
bt = st.timing_backtest(frame, lag=1, cost_bps=10.0, hold_bench=True)
print(f"  switches {bt['n_switches']:.0f}  DIS exposure {bt['exposure_dis']*100:.0f}%")
for k, lbl in (("gross", "rule gross"), ("net", "rule net  "),
               ("buy_hold_dis", "hold DIS  "), ("buy_hold_spy", "hold SPY  ")):
    s = bt[k]
    print(f"  {lbl}: ann {s['ann_ret']*100:6.2f}%  vol {s['ann_vol']*100:5.1f}%  Sharpe {s['sharpe']:.3f}")

print("\n# Synthetic positive control - engine lights up iff a forward edge is planted")
c0 = st.control_recovers(data.synthetic(edge=0.0), 0.0)
c1 = st.control_recovers(data.synthetic(edge=0.02), 0.02)
print(f"  edge = 0.00 (null)   : NW t = {c0['t']:+.2f}   (must stay small)")
print(f"  edge = 0.02 (planted): NW t = {c1['t']:+.2f}   (must light up)")
