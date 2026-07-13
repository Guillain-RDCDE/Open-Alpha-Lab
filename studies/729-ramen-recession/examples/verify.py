"""Reproducible headline run for Study 729 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The noodle-stock numbers come from the cached yfinance pulls under ``_cache/`` (month-end
Adj Close for 2897.T, 2875.T, ^N225); the "ramen index" (WINA world demand) and the NBER
recession windows come from the hardcoded, cited series in :mod:`ramen_recession.data`.
Offline & deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ramen_recession import data, strategy as st

px = data.load_prices()
n225 = px["^N225"]
br = n225.pct_change().dropna()

print("# THE HEADLINE — does the ramen index LEAD the market? (WINA demand vs Nikkei annual)")
mkt = n225.resample("YE").last().pct_change().dropna()
ll = st.lead_lag_corr(data.ramen_growth() / 100.0, mkt, leads=range(-2, 3))
for k, v in ll["per_lead"].items():
    tag = "  <- the 'tell' lead" if k == ll["best_lead"] else ""
    print(f"  lead {k:+d}: r {v['r']:+.3f}  t {v['t']:+.2f}  p {v['p']:.3f}  n {v['n']}{tag}")
dv = st.demand_in_vs_out_recession(data.ramen_growth(), data.recession_years())
print(f"  demand growth: recession yrs {dv['in_mean']:+.2f}%/yr vs other {dv['out_mean']:+.2f}%/yr  "
      f"diff {dv['diff']:+.2f}  t {dv['t']:+.2f}  p {dv['p']:.3f}  (n_in={dv['n_in']}, n_out={dv['n_out']})")

ri = data.load_ramen_index()
g = data.ramen_growth()
print(f"  ramen index {ri.index[0].year} {ri.iloc[0]:.1f}bn -> {ri.index[-1].year} {ri.iloc[-1]:.1f}bn  "
      f"(CAGR {(ri.iloc[-1]/ri.iloc[0])**(1/(ri.index[-1].year-ri.index[0].year))-1:+.2%}/yr)")
print(f"  demand growth: 2008 {g[g.index.year==2008].iloc[0]:+.1f}%  2009 {g[g.index.year==2009].iloc[0]:+.1f}%  "
      f"2014 {g[g.index.year==2014].iloc[0]:+.1f}%  2020 {g[g.index.year==2020].iloc[0]:+.1f}%")

print("\n# Full-sample risk/return — noodle makers vs the Nikkei (month-end Adj Close, yfinance)")
print(f"window (^N225) : {n225.index[0].date()} -> {n225.index[-1].date()}")
for t in ("2897.T", "2875.T", "^N225"):
    s = st.summarize(px[t])
    print(f"  {t:7s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:.2f}  "
          f"maxDD {s['mdd']*100:6.1f}%  $1->${st.terminal_wealth(px[t]):5.2f}  (n={s['n']})")

print("\n# CAPM alpha vs Nikkei (Newey-West, 6-lag) — the honest complication (survivorship!)")
for t in ("2897.T", "2875.T"):
    nw = st.newey_west_alpha_t(px[t].pct_change().dropna(), br, lags=6)
    print(f"  {t:7s} alpha {nw['alpha_ann']*100:+6.2f}%/yr  beta {nw['beta']:.2f}  "
          f"NW t(alpha) {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")

print("\n# Downside defensiveness — bull vs bear beta (split at Nikkei=0)")
for t in ("2897.T", "2875.T"):
    bb = st.bull_bear_beta(px[t].pct_change().dropna(), br, split=0.0)
    print(f"  {t:7s} down-beta {bb['down_beta']:+.2f}  up-beta {bb['up_beta']:+.2f}  "
          f"full {bb['full_beta']:.2f}  asymmetry {bb['asymmetry']:+.2f}  defensive={bb['defensive']}")

print("\n# Recession-window excess vs the Nikkei (paired t)")
for t in ("2897.T", "2875.T"):
    re = st.recession_excess_t(px[t].pct_change().dropna(), br, data.recession_mask)
    print(f"  {t:7s} n={re['n']:2d}  mean {re['mean_stock']*100:+.2f}%/mo vs Nikkei {re['mean_bench']*100:+.2f}%/mo  "
          f"excess {re['mean_excess']*100:+.2f}%/mo  t={re['t']:+.2f} p={re['p']:.3f}  | cum {re['cum_stock']*100:+.1f}% vs {re['cum_bench']*100:+.1f}%")

print("\n# Per-recession provenance (compounded %)")
for t in ("2897.T", "2875.T"):
    bd = st.recession_breakdown(px[t].pct_change().dropna(), br, data.NBER_RECESSIONS)
    for name, d in bd.items():
        print(f"  {t:7s} {name:14s} n={d['n']:2d}  {d['stock']*100:+7.1f}%  vs Nikkei {d['bench']*100:+7.1f}%")

print(f"\n# Tradability — the double look-ahead: WINA lag ~{data.WINA_RELEASE_LAG_MONTHS} mo + "
      f"NBER lag ~{data.NBER_ANNOUNCE_LAG_MONTHS} mo -> the tell is un-actable in real time.")

print("\n# Synthetic positive controls — the engines are faithful (machinery proofs)")
ig, mk = data.synthetic_leading_index(lead=1, seed=729)
cl = st.control_recovers_lead(ig, mk, planted_lead=1)
print(f"  lead control:      planted lead 1 -> recovered best_lead {cl['best_lead']}  "
      f"r={cl['best_r']:.2f}  t={cl['best_t']:.1f}  ok={cl['recovered_lead_ok']}")
mkt2, stock2 = data.synthetic_defensive(beta_down=0.5, beta_up=1.0, seed=7290)
cd = st.control_recovers_defensive(stock2, mkt2, 0.5, 1.0)
print(f"  defensive control: planted 0.50/1.00 -> recovered down {cd['down_beta']:.2f}  up {cd['up_beta']:.2f}  "
      f"defensive={cd['recovered_defensive']}")
