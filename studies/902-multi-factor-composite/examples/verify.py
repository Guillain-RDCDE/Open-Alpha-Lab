"""Reproducible headline run for Study 902 — Multi-Factor Composite.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the study's cached yfinance tape under
``_cache/`` for the real numbers, and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from multi_factor import data, strategy as st

COST = 2.0  # one-way bps charged per unit of rebalancing turnover (realistic ETF spread)

print("# Multi-Factor Composite — equal-weight VLUE/QUAL/MTUM/USMV/SIZE sleeve vs SPY")
print("# (Dedup: 601 tested EACH ETF's exposure vs SPY; this tests the COMBINED sleeve as a")
print("#  portfolio. 638 value-momentum, 401 signal-stacking, 242 QMJ = academic long-shorts.)")

if not data.have_real():
    print("(missing cache: _cache/mfc_prices.parquet — run data.fetch() once)")
else:
    try:
        from quantlab import repro, stats as qstats
    except Exception:
        repro = qstats = None

    prices = data.load_prices()
    mret = data.monthly_total_returns(prices)
    cash = mret[data.CASH]
    spy = mret[data.BENCH]

    comp = st.equal_weight_composite(mret, data.SLEEVE, cost_bps=COST)
    comp_iv = st.inverse_vol_composite(mret, data.SLEEVE, cost_bps=COST)
    common = comp.index

    print("\n# Data stamp")
    print(f"tape          : {prices.shape[0]} days x {prices.shape[1]} tickers, "
          f"{prices.index.min().date()} -> {prices.index.max().date()} "
          f"(yfinance auto-adjusted = TOTAL RETURN, net of each fund fee)")
    print(f"as-of         : monthly stats sliced to {data.AS_OF} (last complete month)")
    print(f"blend window  : {common.min().date()} -> {common.max().date()}  "
          f"({len(common)} complete months, common to all 5 sleeves)")
    if repro is not None:
        panel = pd.concat([mret[data.SLEEVE + [data.BENCH, data.CASH]]], axis=1).loc[common]
        print(f"fingerprint   : monthly_panel={repro.fingerprint(panel)}")

    print("\n# The composite sleeve (equal-weight, monthly rebalanced)")
    cstat_g = st.ann_stats(comp["gross"], cash)
    cstat_n = st.ann_stats(comp["net"], cash)
    sstat = st.ann_stats(spy.loc[common], cash)
    print(f"  turnover/rebalance : mean {comp['turnover'].mean()*100:.2f}% of NAV/mo  "
          f"(cost {comp['cost'].mean()*1e4:.2f} bps/mo, {comp['cost'].mean()*1e4*12:.1f} bps/yr)")
    print(f"  composite GROSS: CAGR {cstat_g['cagr']*100:.2f}%  vol {cstat_g['vol']*100:.1f}%  "
          f"exSharpe {cstat_g['sharpe']:.3f}  maxDD {cstat_g['maxdd']*100:.1f}%")
    print(f"  composite NET  : CAGR {cstat_n['cagr']*100:.2f}%  vol {cstat_n['vol']*100:.1f}%  "
          f"exSharpe {cstat_n['sharpe']:.3f}  maxDD {cstat_n['maxdd']*100:.1f}%")
    print(f"  SPY            : CAGR {sstat['cagr']*100:.2f}%  vol {sstat['vol']*100:.1f}%  "
          f"exSharpe {sstat['sharpe']:.3f}  maxDD {sstat['maxdd']*100:.1f}%")

    print("\n# Axis 1 — the excess-of-cash Sharpe race (both legs minus BIL)")
    race_g = st.sharpe_race(comp["gross"], spy, cash)
    race_n = st.sharpe_race(comp["net"], spy, cash)
    print(f"  GROSS: comp exSharpe {race_g['sharpe_comp']:.3f} vs SPY {race_g['sharpe_spy']:.3f} "
          f"(adv {race_g['sharpe_adv']:+.3f})  active {race_g['active_bps']:+.1f} bps/mo "
          f"NW t {race_g['t_active_nw']:+.2f}")
    print(f"  NET  : comp exSharpe {race_n['sharpe_comp']:.3f} vs SPY {race_n['sharpe_spy']:.3f} "
          f"(adv {race_n['sharpe_adv']:+.3f})  active {race_n['active_bps']:+.1f} bps/mo "
          f"NW t {race_n['t_active_nw']:+.2f}  win-rate {race_n['win_rate']*100:.0f}%")
    print(f"  vols : comp {race_n['vol_comp']*100:.1f}% vs SPY {race_n['vol_spy']*100:.1f}%")

    print("\n# Bootstrap CI on the Sharpe ADVANTAGE (paired moving-block, net)")
    boot = st.adv_bootstrap_ci(comp["net"], spy, cash, seed=902)
    print(f"  adv {boot['obs']:+.3f}  95% CI [{boot['lo']:+.3f}, {boot['hi']:+.3f}]  "
          f"P(adv<0) = {boot['frac_negative']:.3f}  (n_boot {boot['n_boot']})")
    if qstats is not None:
        ci = qstats.sharpe_ci_bootstrap((comp["net"] - cash).dropna(), periods_per_year=12,
                                        seed=902)
        print(f"  composite excess Sharpe {ci['sharpe']:.3f}  95% CI "
              f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] (quantlab CBB)")

    print("\n# Axis 2 — two-era robustness (does the advantage survive a sample split?)")
    eras = st.era_split(comp["net"], spy, cash)
    for name in ("early", "late"):
        e = eras[name]
        print(f"  {name:5s} {e['start']} -> {e['end']} ({e['n_months']}m): "
              f"adv {e['sharpe_adv']:+.3f}  active {e['active_bps']:+.1f} bps/mo  "
              f"NW t {e['t_active_nw']:+.2f}")

    print("\n# Axis 3 — the diversification pitch (factor-timing risk)")
    ftr = st.factor_timing_risk(mret, data.SLEEVE, comp["gross"], cash, spy)
    print(f"  composite vol {ftr['comp_vol_pct']:.1f}% vs mean single {ftr['mean_single_vol_pct']:.1f}% "
          f"(min single {ftr['min_single_vol_pct']:.1f}%, SPY {race_n['vol_spy']*100:.1f}%)")
    print(f"  composite exSharpe {ftr['comp_sharpe']:.3f} vs mean single {ftr['mean_single_sharpe']:.3f} "
          f"(best single {ftr['best_single_sharpe']:.3f}, SPY {ftr['spy_sharpe']:.3f})")
    print(f"  avg cross-sleeve annual dispersion {ftr['annual_cross_dispersion_pct']:.1f} pp "
          f"-> blend's own year-to-year sd {ftr['comp_annual_sd_pct']:.1f} pp")

    print("\n# Per single-factor sleeve (common blend window)")
    ss = st.single_sleeve_stats(mret, data.SLEEVE, cash, spy)
    for tk in data.SLEEVE:
        r = ss.loc[tk]
        print(f"  {tk}: CAGR {r['cagr_pct']:.2f}%  vol {r['vol_pct']:.1f}%  "
              f"exSharpe {r['sharpe']:.3f}  active NW t {r['active_t_nw']:+.2f}  "
              f"maxDD {r['maxdd_pct']:.1f}%")

    print("\n# Inverse-vol weighting (robustness alt)")
    race_iv = st.sharpe_race(comp_iv["net"], spy, cash)
    print(f"  inv-vol NET exSharpe {race_iv['sharpe_comp']:.3f} vs SPY {race_iv['sharpe_spy']:.3f} "
          f"(adv {race_iv['sharpe_adv']:+.3f})  active NW t {race_iv['t_active_nw']:+.2f}")

    print("\n# Calendar-year table (composite net vs SPY, %)")
    cy = st.calendar_year_table(comp["net"], spy)
    for yr, row in cy.iterrows():
        print(f"  {yr}: comp {row['composite']:+6.2f}  SPY {row['SPY']:+6.2f}  "
              f"diff {row['diff']:+6.2f}")

print("\n# Synthetic control — machinery proof only, never market evidence")
print("  the Sharpe-advantage active NW t must stay ~0 on the null (edge=0) and light up")
print("  positive when a per-annum blend edge is planted.")
for label, kw in [("null    (edge=+0%/yr)", dict(edge_ann=0.0)),
                  ("planted (edge=+3%/yr)", dict(edge_ann=0.03))]:
    w = data.synthetic_world(n_months=168, edge_ann=kw["edge_ann"], seed=902)
    d = st.synthetic_detect(w)
    print(f"  {label}: Sharpe adv {d['sharpe_adv']:+.3f}  active {d['active_bps']:+.1f} bps/mo  "
          f"NW t {d['t_active_nw']:+.2f}  (n={d['n_months']})")
