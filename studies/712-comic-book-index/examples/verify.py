"""Reproducible headline run for Study 712 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The equity-proxy numbers come from the cached yfinance pulls under ``_cache/`` (month-end
Adj Close for FNKO, SPY); the comic-index numbers come from the hardcoded, cited,
**approximate** graded-key-comic series in :mod:`comic_book_index.data`. Deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comic_book_index import data, strategy as st

idx = data.load_comic_index()
prox = data.load_proxies()
spy = prox["SPY"]

print("# Comic index — HARDCODED, CITED, APPROXIMATE (a labelled proxy, not a feed)")
print(f"window         : {idx.index[0].year} - {idx.index[-1].year} (year-end levels, base 100)")
print(f"levels         : " + ", ".join(f"{y}:{v:.0f}" for y, v in zip(idx.index.year, idx.values)))
pk_d, pk_l = data.comic_peak()
print(f"reported peak  : {pk_d.date()} level {pk_l:.0f} (early-2022 blow-off top)")
si = st.summarize(idx, periods_per_year=1.0)
print(f"index CAGR     : {si['cagr']*100:6.2f}%   vol {si['vol']*100:5.1f}%   "
      f"Sharpe {si['sharpe']:.2f}   maxDD {si['mdd']*100:6.1f}%")
print(f"peak->trough   : {(idx.loc['2024-12-31']/pk_l-1)*100:6.1f}% (early-2022 high to 2024 low)")

spy_ye = spy.resample("YE").last()
spy_ye = spy_ye[(spy_ye.index.year >= 2018) & (spy_ye.index.year <= 2025)]
ss = st.summarize(spy_ye, periods_per_year=1.0)
print(f"SPY  CAGR (YE) : {ss['cagr']*100:6.2f}%   vol {ss['vol']*100:5.1f}%   maxDD {ss['mdd']*100:6.1f}%")

ae = st.annual_excess_t(idx, spy)
print(f"\n# Did comics beat the S&P? (annual excess, index - SPY)")
print(f"mean annual excess : {ae['mean_excess']*100:+.2f}%   t = {ae['t']:+.3f}   "
      f"p = {ae['p']:.3f}   (n = {ae['n']} years)")

print("\n# The only listed proxy (FNKO, monthly Adj Close) vs SPY — a LABELLED, POOR PROXY")
print("#   (CGC/PSA parents + Heritage are private/delisted; there is no pure-play comic stock)")
spy_r = spy.pct_change().dropna()
lvl = prox["FNKO"]
s = st.summarize(lvl)
nw = st.newey_west_alpha_t(lvl.pct_change().dropna(), spy_r, lags=6)
print(f"  {'FNKO':7s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:.2f}  "
      f"maxDD {s['mdd']*100:6.1f}%  | alpha {nw['alpha_ann']*100:+6.2f}%/yr  beta {nw['beta']:.2f}  "
      f"NW t {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")
sp = st.summarize(spy)
print(f"  {'SPY':7s} CAGR {sp['cagr']*100:6.2f}%  vol {sp['vol']*100:5.1f}%  Sharpe {sp['sharpe']:.2f}  "
      f"maxDD {sp['mdd']*100:6.1f}%")

print("\n# The grading-fee + dealer-spread haircut — what flipping really costs (on the index gross CAGR)")
h = st.net_of_costs_cagr(si["cagr"], round_trip_spread=0.25, hold_years=3.0,
                         grading_fee_pct=0.02, carry_per_year=0.01)
print(f"  gross CAGR {h['gross_cagr']*100:+.2f}%  - spread {h['spread_drag_annual']*100:+.2f}%/yr "
      f"- grading {h['grading_drag_annual']*100:+.2f}%/yr - carry {h['carry_per_year']*100:+.2f}%/yr "
      f" =>  NET {h['net_cagr']*100:+.2f}%/yr")

print("\n# Synthetic positive control — engine recovers a planted bubble's sign + Sharpe")
syn = data.synthetic_bubble()
cr = st.control_recovers(syn, planted_sign=1)
s = st.summarize(syn)
print(f"  planted bubble: peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  "
      f"recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  "
      f"sign_ok={cr['sign_ok']}")
