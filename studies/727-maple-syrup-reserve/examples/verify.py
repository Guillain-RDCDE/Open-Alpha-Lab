"""Reproducible headline run for Study 727 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The equity-proxy numbers come from the cached yfinance pulls under ``_cache/`` (month-end
Adj Close for RSI.TO, SB=F, ^GSPTSE, sliced to the desk as-of); the maple-price numbers
come from the hardcoded, cited, **approximate** PPAQ bulk-price series in
:mod:`maple_syrup_reserve.data`. Offline & deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from maple_syrup_reserve import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402

maple = data.load_maple_price()
prox = {t: repro.as_of(s.to_frame("x")).iloc[:, 0] for t, s in data.load_proxies().items()}
RSI, SB, TSX = prox["RSI.TO"], prox["SB=F"], prox["^GSPTSE"]
bench_r = TSX.pct_change().dropna()

print("# PPAQ bulk maple price — HARDCODED, CITED, APPROXIMATE (a labelled proxy, not a feed)")
print(f"window        : {maple.index[0].year} - {maple.index[-1].year} (annual CAD/lb, administered)")
print("levels        : " + ", ".join(f"{y}:{v:.2f}" for y, v in zip(maple.index.year, maple.values)))
sm = st.summarize(maple, periods_per_year=1.0)
print(f"maple CAGR    : {sm['cagr']*100:6.2f}%   vol {sm['vol']*100:5.1f}%   "
      f"Sharpe {sm['sharpe']:.2f}   maxDD {sm['mdd']*100:6.1f}%")
f = data.RESERVE_FACTS
print(f"reserve/heist : {f['operator']} · QC ~{f['world_share_pct']}% of world output · "
      f"~{f['reserve_capacity_mlb']}M-lb reserve · {f['heist_year']} heist ~C${f['heist_value_cad_m']}M "
      f"/ {f['heist_tonnes']:,} t")

tsx_ye = TSX.resample("YE").last()
tsx_ye = tsx_ye[(tsx_ye.index.year >= 2010) & (tsx_ye.index.year <= 2024)]
ss = st.summarize(tsx_ye, periods_per_year=1.0)
print(f"TSX CAGR (YE) : {ss['cagr']*100:6.2f}%   vol {ss['vol']*100:5.1f}%   maxDD {ss['mdd']*100:6.1f}%")

ae = st.annual_excess_t(maple, TSX)
print(f"\n# Does maple reward a holder better than stocks? (annual excess, maple - TSX)")
print(f"mean annual excess : {ae['mean_excess']*100:+.2f}%   t = {ae['t']:+.3f}   "
      f"p = {ae['p']:.3f}   (n = {ae['n']} years)")

print("\n# The tradable proxies (monthly Adj Close) vs TSX — LABELLED PROXIES")
for t, lvl in (("RSI.TO", RSI), ("SB=F", SB)):
    s = st.summarize(lvl)
    nw = st.newey_west_alpha_t(lvl.pct_change().dropna(), bench_r, lags=6)
    print(f"  {t:7s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:.2f}  "
          f"maxDD {s['mdd']*100:6.1f}%  | alpha {nw['alpha_ann']*100:+6.2f}%/yr  beta {nw['beta']:+.2f}  "
          f"NW t {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")
sp = st.summarize(TSX)
print(f"  {'^GSPTSE':7s} CAGR {sp['cagr']*100:6.2f}%  vol {sp['vol']*100:5.1f}%  Sharpe {sp['sharpe']:.2f}  "
      f"maxDD {sp['mdd']*100:6.1f}%")

print("\n# The sugaring-season seasonal (Feb-Apr) on RSI.TO monthly returns")
rsi_r = RSI.pct_change().dropna()
ms = st.month_stats(rsi_r)
nm = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for m in range(1, 13):
    row = ms.loc[m]
    tag = " <- sugaring" if m in st.SUGARING_MONTHS else ""
    print(f"  {nm[m-1]:4s}: mean {row['mean']*100:+6.2f}%  t {row['tstat']:+.2f}  "
          f"t_HAC {row['tstat_hac']:+.2f}  n={int(row['n'])}{tag}")
sea = st.season_tstat(rsi_r)
print(f"  season(Feb-Apr) mean {sea['season_mean']*100:+.2f}%  rest {sea['rest_mean']*100:+.2f}%  "
      f"spread {sea['spread']*100:+.2f}%/mo  t {sea['tstat']:+.2f}  (n_s={sea['n_season']}, n_r={sea['n_rest']})")
ci = st.season_bootstrap_ci(rsi_r, n_boot=5000)
print(f"  block-bootstrap 95% CI on spread: [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%]  "
      f"(point {ci['point']*100:+.2f}%, n_boot={ci['n_boot']})")

print("\n# The sugaring-season timer (long RSI.TO Feb-Apr, else hold TSX) vs buy-and-hold")
timer = st.seasonal_timer(rsi_r, bench_r)
net = st.apply_costs(timer, n_legs_per_year=2, cost_bps_one_way=15)
for lab, r in (("timer (gross)", timer), ("timer (net 15bp/leg)", net),
               ("buy & hold RSI.TO", rsi_r), ("buy & hold TSX", bench_r)):
    s = st.summary_ret(r, rf=bench_r)
    print(f"  {lab:22s} CAGR {s['cagr']*100:+6.2f}%  Sharpe(exc-TSX) {s['sharpe']:5.2f}  "
          f"vol {s['vol']*100:5.1f}%  maxDD {s['mdd']*100:6.1f}%")

print("\n# Synthetic positive control — the seasonality engine recovers a planted spring premium")
world, truth = data.synthetic_world()
cr = st.control_recovers(world["ret"], planted_sign=1)
seac = st.season_tstat(world["ret"])
print(f"  planted spring premium {truth['spring_premium']*100:.1f}%/yr -> recovered spread "
      f"{cr['spread']*100:+.2f}%/mo  t {seac['tstat']:+.2f}  sign_ok={cr['sign_ok']}")

allm = pd.DataFrame({"RSI.TO": RSI, "SB=F": SB, "GSPTSE": TSX}).dropna(how="all")
print(f"\n[data] proxies : {len(allm):,} rows  {allm.index.min().date()} -> {allm.index.max().date()}"
      f"  as-of {repro.DEFAULT_AS_OF}  fingerprint={repro.fingerprint(allm)}")
print(f"[data] maple   : {len(maple)} rows  {maple.index.min().date()} -> {maple.index.max().date()}"
      f"  fingerprint={repro.fingerprint(maple.to_frame('maple'))}")
