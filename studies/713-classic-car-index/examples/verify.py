"""Reproducible headline run for Study 713 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The equity-proxy + benchmark numbers come from the cached yfinance pulls under ``_cache/``
(month-end Adj Close for RACE, AML.L, SPY [total return]; ^GSPC [price only]); the
car-index numbers come from the hardcoded, cited, **approximate** collector-car series in
:mod:`classic_car_index.data`. Deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classic_car_index import data, strategy as st

idx = data.load_car_index()
prox = data.load_proxies()
spy = prox["SPY"]      # dividend-adjusted -> total return
gspc = prox["^GSPC"]   # price only


def _ye(s):
    s2 = s.resample("YE").last()
    return s2[(s2.index.year >= 2005) & (s2.index.year <= 2025)]


print("# Collector-car index — HARDCODED, CITED, APPROXIMATE (a labelled proxy, not a feed)")
print(f"window         : {idx.index[0].year} - {idx.index[-1].year} (year-end levels, base 100)")
print("levels         : " + ", ".join(f"{y}:{v:.0f}" for y, v in zip(idx.index.year, idx.values)))
pk_d, pk_l = data.car_peak()
print(f"reported high  : {pk_d.date()} level {pk_l:.0f} (KFLII crest)")
si = st.summarize(idx, periods_per_year=1.0)
print(f"index CAGR     : {si['cagr']*100:6.2f}%   vol {si['vol']*100:5.1f}%   "
      f"Sharpe {si['sharpe']:.2f}   maxDD {si['mdd']*100:6.1f}%")
print(f"from high      : {(idx.loc['2024-12-31']/pk_l-1)*100:6.1f}% (a plateau, not a crash)")

ss = st.summarize(_ye(spy), periods_per_year=1.0)
sp = st.summarize(_ye(gspc), periods_per_year=1.0)
print(f"S&P total ret  : CAGR {ss['cagr']*100:6.2f}%   vol {ss['vol']*100:5.1f}%   maxDD {ss['mdd']*100:6.1f}%")
print(f"S&P price only : CAGR {sp['cagr']*100:6.2f}%   vol {sp['vol']*100:5.1f}%   maxDD {sp['mdd']*100:6.1f}%")

print("\n# Did cars beat the S&P? (annual excess, index - benchmark)")
ae_tr = st.annual_excess_t(idx, spy)
ae_po = st.annual_excess_t(idx, gspc)
print(f"vs total return : mean {ae_tr['mean_excess']*100:+.2f}%/yr   t = {ae_tr['t']:+.3f}   "
      f"p = {ae_tr['p']:.3f}   (n = {ae_tr['n']} years)")
print(f"vs price only   : mean {ae_po['mean_excess']*100:+.2f}%/yr   t = {ae_po['t']:+.3f}   "
      f"p = {ae_po['p']:.3f}   (n = {ae_po['n']} years)")

print("\n# The smoothness trap — appraisal-smoothing de-bias (Geltner AR(1) un-smoothing)")
ds = st.desmooth_returns(idx)
print(f"AR(1) rho      : {ds['rho']:+.3f}  (a smooth, laggy index)")
print(f"vol            : reported {ds['vol_obs']*100:5.1f}%  ->  de-smoothed {ds['vol_desmoothed']*100:5.1f}%")
print(f"Sharpe         : reported {ds['sharpe_obs']:+.2f}  ->  de-smoothed {ds['sharpe_desmoothed']:+.2f}")

print("\n# Tradable equity proxies (monthly Adj Close) vs SPY total return — LABELLED PROXIES")
spy_r = spy.pct_change().dropna()
for t in ("RACE", "AML.L"):
    lvl = prox[t]
    s = st.summarize(lvl)
    nw = st.newey_west_alpha_t(lvl.pct_change().dropna(), spy_r, lags=6)
    print(f"  {t:7s} CAGR {s['cagr']*100:7.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:+.2f}  "
          f"maxDD {s['mdd']*100:6.1f}%  | alpha {nw['alpha_ann']*100:+7.2f}%/yr  beta {nw['beta']:.2f}  "
          f"NW t {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")
spm = st.summarize(spy)
print(f"  {'SPY':7s} CAGR {spm['cagr']*100:7.2f}%  vol {spm['vol']*100:5.1f}%  Sharpe {spm['sharpe']:+.2f}  "
      f"maxDD {spm['mdd']*100:6.1f}%")

print("\n# The carry/illiquidity haircut — what owning the metal really costs (on the index gross CAGR)")
h = st.net_of_carry_cagr(si["cagr"], round_trip_spread=0.22, hold_years=7.0, carry_per_year=0.025)
print(f"  gross CAGR {h['gross_cagr']*100:+.2f}%  - spread drag {h['spread_drag_annual']*100:+.2f}%/yr "
      f"- carry {h['carry_per_year']*100:+.2f}%/yr  =>  NET {h['net_cagr']*100:+.2f}%/yr")

print("\n# Synthetic positive control — engine recovers a planted boom's sign + Sharpe")
syn = data.synthetic_boom()
cr = st.control_recovers(syn, planted_sign=1)
s = st.summarize(syn, periods_per_year=1.0)
print(f"  planted boom: peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  "
      f"recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  "
      f"sign_ok={cr['sign_ok']}")
