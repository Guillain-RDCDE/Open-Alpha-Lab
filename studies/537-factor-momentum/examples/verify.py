"""Reproducible headline run for Study 537 — Factor-Momentum (Ehsani-Linnainmaa 2022).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached basket prices under ``_cache/``
(the real-tape numbers) and always runs the synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from factor_momentum import data, strategy as st

NDRAWS = 3000

print("# Factor-Momentum — time-series momentum ON the factors (yfinance survivor basket)")
if data.have_real():
    prices = data.load_prices()
    panel = data.build_factor_panel(prices)
    span = (panel.index.max() - panel.index.min()).days / 365.25
    print(f"basket         : {len([c for c in prices.columns if c != 'SPY'])} names + SPY")
    print(f"factor panel   : {panel.shape[0]} months x {panel.shape[1]} factors  "
          f"({panel.index.min().date()} -> {panel.index.max().date()}, {span:.1f} yr)")
    print(f"factors        : {', '.join(panel.columns)}")
    print(f"prices fp      : {data.fingerprint(prices)}   panel fp: {data.fingerprint(panel)}")

    print("\n# Are the factors themselves persistent? (lag-1 autocorrelation)")
    for c in panel.columns:
        print(f"  {c:>8}: static mean {panel[c].mean()*1200:>6.2f}%/yr   "
              f"autocorr(1) = {panel[c].autocorr(1):>+.3f}")

    print("\n# Static (untimed) average factor vs TIMED factor-momentum")
    stat = st.summary(st.static_factor_series(panel))
    print(f"  static avg factor : mean {stat['mean']*100:>6.2f}%/yr  t = {stat['t']:>5.2f}")

    print("\n# Factor-momentum meta-premium by timing lookback (mean of sign(trail)*factor)")
    print(f"  {'lb':>3} {'mean%':>7} {'vol%':>6} {'SR':>5} {'t':>6} {'hac_t':>6} "
          f"{'hit%':>5} {'mdd%':>6} {'p_plac':>7}")
    for lb in st.LOOKBACKS:
        s = st.summary(st.factor_momentum_series(panel, lookback=lb))
        p = st.placebo_pvalue(panel, lookback=lb, n_draws=NDRAWS)
        print(f"  {lb:>3} {s['mean']*100:>6.2f} {s['vol']*100:>6.1f} {s['sharpe']:>5.2f} "
              f"{s['t']:>6.2f} {s['hac_t']:>6.2f} {s['hit_rate']*100:>4.0f} "
              f"{s['max_drawdown']*100:>6.1f} {p['p_value']:>7.3f}")

    print("\n# Net of one-way costs (10 bps x turnover) + 50 bps/yr borrow on short sleeves")
    print(f"  {'lb':>3} {'gross%':>8} {'net%':>8} {'turn':>5} {'short':>6} {'net_t':>6}")
    for lb in st.LOOKBACKS:
        c = st.net_of_costs(panel, lookback=lb)
        print(f"  {lb:>3} {c['gross']*100:>7.2f} {c['net']*100:>7.2f} "
              f"{c['avg_turnover']:>5.2f} {c['avg_short_frac']:>6.2f} {c['net_t']:>6.2f}")

    print("\n# Per-factor timed contribution (lb=3) — which sleeves carry any premium")
    timed = st.timed_factor_returns(panel, lookback=3)
    for cc in timed.columns:
        print(f"  {cc:>8}: timed mean {timed[cc].mean()*1200:>6.2f}%/yr  "
              f"t = {st.ttest_vs_zero(timed[cc]):>5.2f}")

    print("\n# Seed-robustness of the lb=3 placebo p (20 seeds)")
    ps = [st.placebo_pvalue(panel, lookback=3, n_draws=1500, seed=s)["p_value"]
          for s in range(20)]
    print(f"  placebo p: mean {np.mean(ps):.3f}  min {min(ps):.3f}  max {max(ps):.3f}")

    print("\n# Robustness — long-only-on-negative (hold flat instead of shorting a factor)")
    for lb in (3, 12):
        s = st.summary(st.factor_momentum_series(panel, lookback=lb,
                                                 long_only_on_negative=True))
        print(f"  lb={lb:>2}: mean {s['mean']*100:>6.2f}%/yr  t = {s['t']:>5.2f}")
else:
    print("(no _cache/fm_prices.parquet — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  white-noise factors (ar=0) must NOT manufacture a premium; planted AR(1) must.")
for ar in (0.0, 0.30):
    panel_s, _ = data.synthetic_factors(ar=ar, n_months=240, mu=0.0, sigma=0.04, seed=537)
    s = st.summary(st.factor_momentum_series(panel_s, lookback=3))
    p = st.placebo_pvalue(panel_s, lookback=3, n_draws=2000)
    print(f"  planted ar={ar:+.2f}: mean {s['mean']*100:>6.2f}%/yr  t = {s['t']:>5.2f}  "
          f"placebo p = {p['p_value']:.3f}")
