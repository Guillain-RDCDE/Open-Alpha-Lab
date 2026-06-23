"""Reproducible headline run for Study 392 — Glassdoor-Sentiment.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices under ``_cache/``
if present (the real-tape numbers) together with the CONSTRUCTED sentiment proxy, and always
runs the synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from glassdoor_sentiment import data, strategy as st

print("# Glassdoor-Sentiment — happiest-minus-grumpiest long-short on a 40-name basket")
print("#   sentiment input = CONSTRUCTED proxy (NOT real Glassdoor data), independent of returns")

if data.have_real():
    d = data.load_real()
    prices, stars = d["prices"], d["stars"]
    span_years = (prices.index.max() - prices.index.min()).days / 365.25
    monthly = st.to_monthly_returns(prices)
    print(f"\nprice days     : {len(prices)}  ({prices.index.min().date()} -> "
          f"{prices.index.max().date()}, {span_years:.1f} years)")
    print(f"monthly spreads: {len(st.long_short_returns(monthly, stars))}")
    hi, lo = st.quintile_masks(stars)
    print(f"happiest (Q5)  : {sorted(stars[hi].index.tolist())}")
    print(f"grumpiest (Q1) : {sorted(stars[lo].index.tolist())}")

    print("\n# Headline long-short (q=0.20, 1-period lag)")
    s = st.summarize(prices, stars, n_placebo=5000)
    print(f"  months          : {s['n_months']}")
    print(f"  mean monthly    : {s['mean_m']*100:+.2f}%   annualised : {s['ann_mean']*100:+.1f}%")
    print(f"  one-sample t    : {s['t']:+.2f}")
    print(f"  Sharpe (ann)    : {s['sharpe']:.2f}   95% CI : [{s['sharpe_lo']:+.2f}, {s['sharpe_hi']:+.2f}]")
    print(f"  win-rate        : {s['win_rate']*100:.0f}%")
    print(f"  placebo p       : {s['p_placebo']:.3f}")

    print("\n# Net of costs (10 bps one-way x turnover + 50 bps/yr borrow)")
    spread = st.long_short_returns(monthly, stars)
    c = st.net_of_costs(spread, cost_bps=10.0, rebalances_per_year=1.0, borrow_bps=50.0)
    print(f"  gross monthly={c['gross_mean_m']*100:+.3f}%  net monthly={c['net_mean_m']*100:+.3f}%  "
          f"gross Sharpe={c['gross_sharpe']:.2f}  net Sharpe={c['net_sharpe']:.2f}  "
          f"(all-in {c['annual_cost_bps']:.0f} bps/yr)")

    print("\n# Robustness — quintile fraction q")
    for q in (0.10, 0.20, 0.30):
        sp = st.long_short_returns(monthly, stars, q=q)
        pl = st.placebo_pvalue(monthly, stars, q=q, n_draws=2000, seed=392)
        print(f"  q={q:.2f}: ann={sp.values.mean()*12*100:+.1f}%  t={st.t_stat(sp.values):+.2f}  "
              f"sharpe={st.sharpe(sp.values):.2f}  placebo p={pl['p_value']:.3f}")

    print("\n# Robustness — 10 independent constructed panels (does ANY clear t=2?)")
    ts = []
    for sd in range(392, 402):
        star2 = data.constructed_sentiment(list(prices.columns), seed=sd)
        ts.append(st.t_stat(st.long_short_returns(monthly, star2).values))
    print(f"  t by seed: {[round(t,2) for t in ts]}")
    print(f"  max |t| across seeds: {max(abs(t) for t in ts):.2f}  (none reaches 2)")
else:
    print("(no _cache/basket_prices.csv — run data.fetch_basket() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must NOT manufacture significance at edge=0, and MUST light up at a large edge")
for edge in (0.0, 0.010):
    syn = data.synthetic_panel(edge=edge, seed=392)
    s = st.summarize(syn["prices"], syn["stars"], n_placebo=3000)
    print(f"  planted edge={edge:+.3f}: ann={s['ann_mean']*100:+5.1f}%  t={s['t']:+.2f}  "
          f"sharpe={s['sharpe']:.2f}  win={s['win_rate']*100:.0f}%  placebo p={s['p_placebo']:.3f}")
