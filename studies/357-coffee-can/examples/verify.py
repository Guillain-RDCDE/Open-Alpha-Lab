"""Reproducible headline run for Study 357 (Coffee-Can) — prints every number quoted in
docs/results.md and frozen into notebooks/build_notebooks.py (the ``R`` dict).

The real-tape section needs the cached yfinance panel (``_cache/coffee_can_monthly.csv``);
if it is absent it is fetched once (network). The synthetic survivorship control is offline
and fully deterministic (fixed seed).

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coffee_can import data, strategy as st

# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
prices = data.load_real() if data.have_real() else data.fetch_real()
basket = data.BASKET
rets = data.monthly_returns(prices)
bret = rets[basket]
spy = rets["SPY"]

can = st.buy_and_hold(bret)                     # zero-turnover coffee can (gross)
can_net = can.copy()
can_net.iloc[0] -= st.entry_cost(len(basket), 10.0)   # one-off 10bps entry
reb = st.rebalanced(bret, rebal=12, cost_bps=10.0)    # annually rebalanced, net of cost

print("# Real tape — yfinance monthly TOTAL-RETURN, basket vs SPY")
print(f"window     : {prices.index.min().date()} -> {prices.index.max().date()}")
print(f"months     : {len(spy)}   years: {len(spy)/12:.1f}")
print(f"basket     : {', '.join(basket)}")
print("\n# Summary stats (monthly TR)")
for name, s in [("coffee_can (gross)", can), ("coffee_can (net entry)", can_net),
                ("rebal_12m (net)", reb), ("SPY (index)", spy)]:
    d = st.summary(s)
    print(f"  {name:24s} CAGR={d['cagr']*100:6.2f}%  vol={d['vol']*100:5.2f}%  "
          f"Sharpe={d['sharpe']:.3f}  MDD={d['mdd']*100:6.1f}%")

print("\n# Signal test — paired t of the can's monthly EXCESS over SPY")
for lag in (0, 6):
    r = st.paired_t(can, spy, hac_lag=lag)
    print(f"  hac_lag={lag}:  mean_excess_ann={r['mean_ann']*100:+6.2f}%  "
          f"t={r['t']:+5.2f}  p={r['p']:.4f}  n={r['n']}")
rv = st.paired_t(can, reb, hac_lag=6)
print(f"  can vs rebalanced (hac6): mean_ann={rv['mean_ann']*100:+.2f}%  t={rv['t']:+.2f}  p={rv['p']:.4f}")

print("\n# Cost / turnover tailwind (the legitimate part)")
for cb in (0, 5, 10, 20):
    print(f"  rebal cost {cb:>2}bps/reset -> CAGR {st.cagr(st.rebalanced(bret,12,cb))*100:.3f}%")
print(f"  coffee-can pays cost ONCE at entry: {st.entry_cost(len(basket),10.0)*100:.2f}% of NAV, then never again")

# --------------------------------------------------------------------------- #
# Synthetic survivorship positive control (offline, deterministic)
# --------------------------------------------------------------------------- #
print("\n# Synthetic survivorship control — every firm has the SAME true 8%/yr drift")
prices_s, alive_s = data.synthetic_panel(n_firms=400, n_months=240, mu_ann=0.08,
                                         sigma_ann=0.22, death_hazard_ann=0.03, seed=357)
n_dead = int(prices_s.iloc[-1].isna().sum())
g = st.survivorship_gap(prices_s, k=10, lookback=24)
print(f"  firms={prices_s.shape[1]}  delisted_by_end={n_dead} ({n_dead/prices_s.shape[1]*100:.0f}%)")
print(f"  famous (look-ahead) basket CAGR = {g['cagr_famous']*100:6.2f}%")
print(f"  honest (ex-ante)    basket CAGR = {g['cagr_honest']*100:6.2f}%")
print(f"  whole universe (EW, deaths in)  = {g['cagr_all']*100:6.2f}%   <- the honest truth (~8%)")
print(f"  SURVIVORSHIP GAP famous-vs-honest = {g['gap_vs_honest']*100:+.2f} pp/yr  (entirely fake)")
print(f"  SURVIVORSHIP GAP famous-vs-all    = {g['gap_vs_all']*100:+.2f} pp/yr")

print("\n# No-death control — gap shrinks toward the look-ahead-only floor")
prices_nd, _ = data.synthetic_panel(n_firms=400, n_months=240, death_hazard_ann=0.0, seed=357)
gnd = st.survivorship_gap(prices_nd, k=10, lookback=24)
print(f"  famous={gnd['cagr_famous']*100:.2f}%  honest={gnd['cagr_honest']*100:.2f}%  "
      f"all={gnd['cagr_all']*100:.2f}%  gap_vs_all={gnd['gap_vs_all']*100:+.2f}pp")
print(f"  (all-firm CAGR {gnd['cagr_all']*100:.2f}% ~ planted 8% confirms the engine is unbiased "
      f"when nothing dies)")
