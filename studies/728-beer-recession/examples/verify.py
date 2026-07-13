"""Reproducible headline run for Study 728 — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The beer-name numbers come from the cached yfinance pulls under ``_cache/`` (month-end
Adj Close for TAP, SAM, SPY); the NBER recession windows come from the hardcoded, cited
dates in :mod:`beer_recession.data`. Offline & deterministic.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from beer_recession import data, strategy as st

px = data.load_prices()
spy = px["SPY"]
spy_r = spy.pct_change().dropna()

print("# Beer names vs SPY — month-end Adj Close via yfinance (as-of 2026-06-30)")
print(f"window (SPY)   : {spy.index[0].date()} -> {spy.index[-1].date()}  (n={len(spy)-1} monthly returns)")
for t in ("TAP", "SAM", "SPY"):
    s = st.summarize(px[t])
    print(f"  {t:5s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  Sharpe {s['sharpe']:.2f}  "
          f"maxDD {s['mdd']*100:6.1f}%  (n={s['n']})")

print("\n# CAPM alpha vs SPY (Newey-West, 6-lag) — after paying for market beta")
for t in ("TAP", "SAM"):
    nw = st.newey_west_alpha_t(px[t].pct_change().dropna(), spy_r, lags=6)
    print(f"  {t:5s} alpha {nw['alpha_ann']*100:+6.2f}%/yr  beta {nw['beta']:.2f}  "
          f"NW t(alpha) {nw['t_alpha']:+.2f}  p {nw['p_alpha']:.3f}  (n={nw['n']})")

print("\n# Downside defensiveness — bull vs bear beta (split at SPY=0)")
for t in ("TAP", "SAM"):
    bb = st.bull_bear_beta(px[t].pct_change().dropna(), spy_r, split=0.0)
    print(f"  {t:5s} down-beta {bb['down_beta']:.2f}  up-beta {bb['up_beta']:.2f}  "
          f"full-beta {bb['full_beta']:.2f}  asymmetry {bb['asymmetry']:+.2f}  "
          f"defensive={bb['defensive']}  (n_down={bb['n_down']}, n_up={bb['n_up']})")

print("\n# The direct counter-cyclical test — recession-window excess vs SPY (paired t)")
for t in ("TAP", "SAM"):
    re = st.recession_excess_t(px[t].pct_change().dropna(), spy_r, data.recession_mask)
    print(f"  {t:5s} n={re['n']:2d} rec-months  mean {re['mean_stock']*100:+.2f}%/mo vs SPY "
          f"{re['mean_bench']*100:+.2f}%/mo  excess {re['mean_excess']*100:+.2f}%/mo  "
          f"t={re['t']:+.2f} p={re['p']:.3f}  | cum {t} {re['cum_stock']*100:+.1f}% vs SPY {re['cum_bench']*100:+.1f}%")

print("\n# Per-recession provenance — where any aggregate 'edge' comes from (compounded %)")
for t in ("TAP", "SAM"):
    bd = st.recession_breakdown(px[t].pct_change().dropna(), spy_r, data.NBER_RECESSIONS)
    for name, d in bd.items():
        print(f"  {t:5s} {name:14s} n={d['n']:2d}  {t} {d['stock']*100:+7.1f}%  vs SPY {d['bench']*100:+7.1f}%")

print("\n# Tradability — the opportunity cost of holding 'beer for defense' ($1 buy-and-hold)")
for t in ("TAP", "SAM", "SPY"):
    yrs = (px[t].index[-1] - px[t].index[0]).days / 365.25
    print(f"  {t:5s} $1 -> ${st.terminal_wealth(px[t]):6.2f}  over {yrs:.1f}y")
print(f"  NBER announce lag: ~{data.NBER_ANNOUNCE_LAG_MONTHS} months -> you cannot rotate into "
      "beer at a recession's start (look-ahead).")

print("\n# Synthetic positive control — engine recovers a PLANTED asymmetric (defensive) beta")
mkt, stock = data.synthetic_defensive(beta_down=0.5, beta_up=1.0)
cr = st.control_recovers(stock, mkt, planted_down=0.5, planted_up=1.0)
print(f"  planted down/up beta 0.50/1.00  ->  recovered down {cr['down_beta']:.2f}  up {cr['up_beta']:.2f}  "
      f"recovered_defensive={cr['recovered_defensive']}")
