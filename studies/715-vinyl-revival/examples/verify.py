"""Reproducible headline run for Study 715 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The equity-proxy numbers come from the cached yfinance pulls under ``_cache/`` (month-end
Adj Close for WMG, SPOT, UMG.AS, SPY); the vinyl-revenue numbers come from the hardcoded,
cited, **approximate** RIAA year-end series in :mod:`vinyl_revival.data`. Deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vinyl_revival import data, strategy as st

vidx = data.load_vinyl_index()
rev = data.vinyl_revenue_musd()
prox = data.load_proxies()
spy = prox["SPY"]

print("# Vinyl-revenue index — HARDCODED, CITED, APPROXIMATE (RIAA year-end; a proxy, not a feed)")
print(f"window         : {rev.index[0].year} - {rev.index[-1].year} (year-end $M retail value)")
print("revenue ($M)   : " + ", ".join(f"{y}:{v:.0f}" for y, v in zip(rev.index.year, rev.values)))
si = st.summarize(vidx, periods_per_year=1.0)
print(f"index CAGR     : {si['cagr']*100:6.2f}%   vol {si['vol']*100:5.1f}%   "
      f"Sharpe {si['sharpe']:.2f}   maxDD {si['mdd']*100:6.1f}%   (base 100 @ 2010)")
share = data.vinyl_share()
print(f"vinyl share    : {share.iloc[-1]:.0f}% of total U.S. music revenue at the peak "
      f"(streaming ~{data.streaming_share_2024():.0f}%) — the omitted context")

spy_ye = spy.resample("YE").last()
spy_ye = spy_ye[(spy_ye.index.year >= 2018) & (spy_ye.index.year <= 2025)]
ss = st.summarize(spy_ye, periods_per_year=1.0)
print(f"SPY  CAGR (YE) : {ss['cagr']*100:6.2f}%   vol {ss['vol']*100:5.1f}%   maxDD {ss['mdd']*100:6.1f}%")

ae = st.annual_excess_t(vidx, spy)
print(f"\n# Did the vinyl TREND out-grow the S&P? (annual excess, revenue growth - SPY, overlap)")
print(f"mean annual excess : {ae['mean_excess']*100:+.2f}%   t = {ae['t']:+.3f}   "
      f"p = {ae['p']:.3f}   (n = {ae['n']} years)  [revenue growth is NOT an investable return]")

print("\n# Tradable equity proxies (monthly Adj Close) vs SPY — LABELLED PROXIES")
spy_r = spy.pct_change().dropna()
for t in ("WMG", "SPOT", "UMG.AS"):
    lvl = prox[t]
    s = st.summarize(lvl)
    nw = st.newey_west_alpha_t(lvl.pct_change().dropna(), spy_r, lags=6)
    print(f"  {t:7s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:.2f}  "
          f"maxDD {s['mdd']*100:6.1f}%  | alpha {nw['alpha_ann']*100:+6.2f}%/yr  beta {nw['beta']:.2f}  "
          f"NW t {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")
sp = st.summarize(spy)
print(f"  {'SPY':7s} CAGR {sp['cagr']*100:6.2f}%  vol {sp['vol']*100:5.1f}%  Sharpe {sp['sharpe']:.2f}  "
      f"maxDD {sp['mdd']*100:6.1f}%")

print("\n# The collector carry haircut — what physically owning the trend costs")
h = st.net_of_collector_carry(si["cagr"], round_trip_spread=0.30, hold_years=5.0, storage_per_year=0.01)
print(f"  charitable (records appreciate at revenue growth):")
print(f"    gross {h['gross_cagr']*100:+.2f}%  - spread drag {h['spread_drag_annual']*100:+.2f}%/yr "
      f"- storage {h['storage_per_year']*100:+.2f}%/yr  =>  NET {h['net_cagr']*100:+.2f}%/yr")
h0 = st.net_of_collector_carry(0.0, round_trip_spread=0.30, hold_years=5.0, storage_per_year=0.01)
print(f"  realistic (per-record resale ~flat as reissues expand supply):")
print(f"    gross {h0['gross_cagr']*100:+.2f}%  =>  NET {h0['net_cagr']*100:+.2f}%/yr")

print("\n# Synthetic positive control — engine recovers a planted revival's sign + Sharpe")
syn = data.synthetic_revival()
cr = st.control_recovers(syn, planted_sign=1)
s = st.summarize(syn)
print(f"  planted revival: peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  "
      f"recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  "
      f"sign_ok={cr['sign_ok']}")
