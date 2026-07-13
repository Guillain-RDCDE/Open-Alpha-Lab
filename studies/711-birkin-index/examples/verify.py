"""Reproducible headline run for Study 711 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The equity/gold-proxy numbers come from the cached yfinance pulls under ``_cache/``
(month-end Adj Close for RMS.PA, MC.PA, KER.PA, SPY, GLD); the resale-index numbers come
from the hardcoded, cited, **approximate** Birkin secondary-market series in
:mod:`birkin_index.data`. Deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from birkin_index import data, strategy as st

idx = data.load_resale_index()
prox = data.load_proxies()
spy = prox["SPY"]
gld = prox["GLD"]

myth = data.baghunter_myth()
print("# The claim under test — Baghunter (2016), recycled by luxury-investment media")
print(f"claimed Birkin return : {myth['cagr']*100:.1f}%/yr over {myth['window']} "
      f"(vs the {myth['sp500']*100:.1f}%/yr it quoted for the S&P) — 'never a down year'")

print("\n# Birkin resale index — HARDCODED, CITED, APPROXIMATE (a labelled proxy, not a feed)")
print(f"window         : {idx.index[0].year} - {idx.index[-1].year} (year-end levels, base 100)")
print("levels         : " + ", ".join(f"{y}:{v:.0f}" for y, v in zip(idx.index.year, idx.values)))
si = st.summarize(idx, periods_per_year=1.0)
print(f"index CAGR     : {si['cagr']*100:6.2f}%   vol {si['vol']*100:5.1f}%   "
      f"Sharpe {si['sharpe']:.2f}   maxDD {si['mdd']*100:6.1f}%")

for nm, s in (("SPY", spy), ("GLD", gld)):
    ye = s.resample("YE").last()
    ye = ye[(ye.index.year >= 2015) & (ye.index.year <= 2025)]
    ss = st.summarize(ye, periods_per_year=1.0)
    ae = st.annual_excess_t(idx, s)
    print(f"{nm}  CAGR (YE) : {ss['cagr']*100:6.2f}%   vol {ss['vol']*100:5.1f}%   maxDD {ss['mdd']*100:6.1f}%   "
          f"| excess (idx-{nm}) mean {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  p={ae['p']:.3f}  (n={ae['n']})")

print("\n# Tradable equity proxies (monthly Adj Close) vs SPY — LABELLED PROXIES")
spy_r = spy.pct_change().dropna()
for t in ("RMS.PA", "MC.PA", "KER.PA"):
    lvl = prox[t]
    s = st.summarize(lvl)
    nw = st.newey_west_alpha_t(lvl.pct_change().dropna(), spy_r, lags=6)
    print(f"  {t:7s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:.2f}  "
          f"maxDD {s['mdd']*100:6.1f}%  | alpha {nw['alpha_ann']*100:+6.2f}%/yr  beta {nw['beta']:.2f}  "
          f"NW t {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")
sp = st.summarize(spy)
gl = st.summarize(gld)
print(f"  {'SPY':7s} CAGR {sp['cagr']*100:6.2f}%  vol {sp['vol']*100:5.1f}%  Sharpe {sp['sharpe']:.2f}  maxDD {sp['mdd']*100:6.1f}%")
print(f"  {'GLD':7s} CAGR {gl['cagr']*100:6.2f}%  vol {gl['vol']*100:5.1f}%  Sharpe {gl['sharpe']:.2f}  maxDD {gl['mdd']*100:6.1f}%")

print("\n# The consignment/illiquidity haircut — what a Birkin flip really costs (on index gross CAGR)")
h = st.net_of_carry_cagr(si["cagr"], round_trip_spread=0.30, hold_years=3.0, insure_per_year=0.005)
print(f"  gross CAGR {h['gross_cagr']*100:+.2f}%  - spread drag {h['spread_drag_annual']*100:+.2f}%/yr "
      f"- insurance {h['insure_per_year']*100:+.2f}%/yr  =>  NET {h['net_cagr']*100:+.2f}%/yr")

print("\n# Synthetic positive control — engine recovers a planted steady compounder")
syn = data.synthetic_compounder()
cr = st.control_recovers(syn, planted_sign=1)
s = st.summarize(syn)
print(f"  planted compounder: peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  "
      f"recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  "
      f"sign_ok={cr['sign_ok']}")
