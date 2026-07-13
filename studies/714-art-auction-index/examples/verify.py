"""Reproducible headline run for Study 714 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The equity-proxy numbers come from the cached yfinance pulls under ``_cache/`` (month-end
Adj Close for MCHN.SW, KER.PA, SPY); the art-index numbers come from the hardcoded,
cited, **approximate** contemporary-art auction series in :mod:`art_auction_index.data`.
Deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from art_auction_index import data, strategy as st

idx = data.load_art_index()
prox = data.load_proxies()
spy = prox["SPY"]

print("# Art auction index — HARDCODED, CITED, APPROXIMATE (a labelled proxy, not a feed)")
print(f"window         : {idx.index[0].year} - {idx.index[-1].year} (year-end levels, base 100)")
print("levels         : " + ", ".join(f"{y}:{v:.0f}" for y, v in zip(idx.index.year, idx.values)))
pk_d, pk_l = data.art_peak()
print(f"reported peak  : {pk_d.date()} level {pk_l:.0f} (2022 records-season blow-off top)")
si = st.summarize(idx, periods_per_year=1.0)
print(f"index CAGR     : {si['cagr']*100:6.2f}%   vol {si['vol']*100:5.1f}%   "
      f"Sharpe {si['sharpe']:.2f}   maxDD {si['mdd']*100:6.1f}%")
print(f"2007->2009     : {(idx.loc['2009-12-31']/idx.loc['2007-12-31']-1)*100:6.1f}% (financial-crisis crash)")
print(f"2022->2024     : {(idx.loc['2024-12-31']/idx.loc['2022-12-31']-1)*100:6.1f}% (recent correction)")

spy_ye = spy.resample("YE").last()
spy_ye = spy_ye[(spy_ye.index.year >= 2000) & (spy_ye.index.year <= 2025)]
ss = st.summarize(spy_ye, periods_per_year=1.0)
print(f"SPY  CAGR (YE) : {ss['cagr']*100:6.2f}%   vol {ss['vol']*100:5.1f}%   maxDD {ss['mdd']*100:6.1f}%")

ae = st.annual_excess_t(idx, spy)
print("\n# Did art beat the S&P? (annual excess, index - SPY)")
print(f"mean annual excess : {ae['mean_excess']*100:+.2f}%   t = {ae['t']:+.3f}   "
      f"p = {ae['p']:.3f}   (n = {ae['n']} years)")

print("\n# Tradable equity proxies (monthly Adj Close) vs SPY — LABELLED PROXIES")
print("# (the pure auction houses are all PRIVATE now — no listed way to buy them:)")
for tk, why in data.DELISTED.items():
    print(f"  - {tk}: {why}")
spy_r = spy.pct_change().dropna()
for t in ("MCHN.SW", "KER.PA"):
    lvl = prox[t]
    s = st.summarize(lvl)
    nw = st.newey_west_alpha_t(lvl.pct_change().dropna(), spy_r, lags=6)
    print(f"  {t:8s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:+.2f}  "
          f"maxDD {s['mdd']*100:6.1f}%  | alpha {nw['alpha_ann']*100:+6.2f}%/yr  beta {nw['beta']:.2f}  "
          f"NW t {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")
sp = st.summarize(spy)
print(f"  {'SPY':8s} CAGR {sp['cagr']*100:6.2f}%  vol {sp['vol']*100:5.1f}%  Sharpe {sp['sharpe']:+.2f}  "
      f"maxDD {sp['mdd']*100:6.1f}%")

print("\n# The buyer's-premium / carry haircut — what an auction round-trip really costs")
h = st.net_of_premium_cagr(si["cagr"], buyers_premium=0.25, sellers_commission=0.10,
                           hold_years=7.0, insure_per_year=0.01)
print(f"  gross CAGR {h['gross_cagr']*100:+.2f}%  x round-trip mult {h['round_trip_mult']:.3f} "
      f"=> premium drag {h['spread_drag_annual']*100:+.2f}%/yr "
      f"- insurance {h['insure_per_year']*100:+.2f}%/yr  =>  NET {h['net_cagr']*100:+.2f}%/yr")

print("\n# Synthetic positive control — engine recovers a planted bubble's sign + Sharpe")
syn = data.synthetic_bubble()
cr = st.control_recovers(syn, planted_sign=1)
s = st.summarize(syn)
print(f"  planted bubble: peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  "
      f"recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  "
      f"sign_ok={cr['sign_ok']}")
