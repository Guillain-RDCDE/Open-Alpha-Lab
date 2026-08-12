"""Reproducible headline run for Study 896 — Risk-Parity + Trend.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached six-ETF tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd

from rp_trend import data, strategy as st

try:
    from quantlab.repro import data_stamp
except Exception:                                    # repo root not importable — degrade
    data_stamp = None

CRISES = {
    "2008 GFC": ("2008-04-01", "2009-06-30"),
    "2020 COVID": ("2020-02-01", "2020-04-30"),
    "2022 bond bear": ("2022-01-01", "2022-12-31"),
}

print("# Risk-Parity + Trend — inverse-vol SPY/TLT/GLD/DBC, 200d trend gate per sleeve, monthly, one lag")
if data.have_real():
    px = data.load_prices()
    ret = st.daily_returns(px)
    cash = ret[data.CASH]
    if data_stamp is not None:
        print(data_stamp("RP+Trend six-ETF panel", px, asof=data.AS_OF))
    print(f"cache          : {len(px)} daily closes  {px.index.min().date()} -> "
          f"{px.index.max().date()}  (sleeves {'/'.join(data.SLEEVES)}, cash {data.CASH})")

    r = st.race(px, ret, cash, data.SLEEVES)
    print(f"\n# Headline race — post-burn-in {r['n_days']} days "
          f"({r['plain_excess'].index.min().date()} -> {r['plain_excess'].index.max().date()}, "
          f"{r['years']:.1f} yrs), gross, excess-of-cash both legs")
    for leg, p in (("plain RP", r["plain"]), ("RP+trend", r["trend"])):
        print(f"  {leg:<9}: CAGR {p['cagr_pct']:+6.2f}%  vol {p['vol_ann_pct']:5.2f}%  "
              f"Sharpe {p['sharpe']:+.3f}  maxDD {p['maxdd_pct']:+6.2f}%  wealth x{p['wealth_mult']:.2f}")
    print(f"  Sharpe advantage (trend - plain): {r['sharpe_adv']:+.3f}")
    print(f"  drawdown relief (trend - plain) : {r['dd_relief_pp']:+.2f} pp "
          f"({r['plain']['maxdd_pct']:.2f}% -> {r['trend']['maxdd_pct']:.2f}%)")
    print(f"  excess-return diff (trend - plain): {r['ret_diff_ann_pct']:+.2f}%/yr  "
          f"HAC t = {r['t_ret_diff']:+.2f}")
    print(f"  trend gate on {r['avg_gate_trend']*100:.1f}% of sleeve-months; avg risky exposure "
          f"{r['avg_risky_trend']:.3f}; turnover plain {r['turnover_plain']:.2f}x / "
          f"trend {r['turnover_trend']:.2f}x NAV/yr")

    print("\n# Paired block-bootstrap of the Sharpe DIFFERENCE (2000 resamples, 21d blocks)")
    bs = st.sharpe_diff_bootstrap(r["trend_excess"], r["plain_excess"], n_boot=2000)
    print(f"  observed {bs['obs']:+.3f}  95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]  "
          f"P(diff > 0) = {bs['p_gt0']:.3f}  ->  "
          f"{'clears zero' if bs['lo'] > 0 else 'STRADDLES zero'}")

    print("\n# Two-era robustness cut")
    for e in st.era_cut(px, ret, cash, data.SLEEVES):
        print(f"  {e['era']} {e['start']}..{e['end']}: Sharpe plain {e['sharpe_plain']:+.3f} "
              f"trend {e['sharpe_trend']:+.3f} adv {e['sharpe_adv']:+.3f}  "
              f"maxDD {e['maxdd_plain_pct']:+.2f}% -> {e['maxdd_trend_pct']:+.2f}%  "
              f"t_ret {e['t_ret_diff']:+.2f}  (n={e['n_days']})")

    print("\n# Crisis-window max drawdown (true cash-inclusive NAV)")
    for nm, (a, b) in CRISES.items():
        mp = (r["plain_total"].index >= pd.Timestamp(a)) & (r["plain_total"].index <= pd.Timestamp(b))
        print(f"  {nm:<14}: plain {st.max_drawdown(r['plain_total'][mp])*100:+.2f}%  "
              f"trend {st.max_drawdown(r['trend_total'][mp])*100:+.2f}%")

    print("\n# Costs — one-way bps x turnover x NAV per monthly rebalance (long-or-cash, no borrow)")
    for c in st.cost_sweep(px, ret, cash, data.SLEEVES):
        print(f"  {c['cost_bps']:>4.0f} bps: Sharpe plain {c['sharpe_plain']:+.3f} "
              f"trend {c['sharpe_trend']:+.3f} adv {c['sharpe_adv']:+.3f}  "
              f"maxDD {c['maxdd_plain_pct']:+.2f}%/{c['maxdd_trend_pct']:+.2f}%  "
              f"CAGR {c['cagr_plain_pct']:+.2f}%/{c['cagr_trend_pct']:+.2f}%  t_ret {c['t_ret_diff']:+.2f}")

    print("\n# Placebo — shuffle each sleeve's monthly gate in time (200 seeds; same on/off rate, no timing)")
    pl = st.placebo_shuffle(px, ret, cash, data.SLEEVES, n_seeds=200)
    print(f"  Sharpe adv: observed {pl['obs_sharpe_adv']:+.3f}  placebo mean "
          f"{pl['placebo_mean_adv']:+.3f}  p = {pl['p_sharpe']:.3f}")
    print(f"  max DD    : observed {pl['obs_maxdd_pct']:+.2f}%  placebo mean "
          f"{pl['placebo_mean_maxdd_pct']:+.2f}%  p = {pl['p_dd']:.3f}  "
          f"(share of shuffles as shallow as observed)")

    print("\n# Calendar-year total return (%): plain RP vs RP+trend")
    cy = st.calendar_years(r)
    print(cy.to_string())
else:
    print("(no _cache/rp_trend_prices.parquet — run data.fetch() once to build the cache)")

print("\n# Synthetic control — 20 seeds/world, deterministic, no network")
print("  NULL (edge=0, no persistent downtrend): the trend gate must NOT improve risk-adjusted return;")
print("  PLANTED (edge=1, sustained bear regimes): a 200d gate MUST cut drawdown and lift the Sharpe.")
for name, e in (("null (edge=0)", 0.0), ("planted (edge=1)", 1.0)):
    sc = st.synthetic_check(edge=e, n_seeds=20)
    print(f"  {name:<18}: mean Sharpe adv {sc['mean_sharpe_adv']:+.3f} +/- {sc['sd_sharpe_adv']:.3f}  "
          f"mean DD relief {sc['mean_dd_relief_pp']:+.2f} pp  mean t_ret {sc['mean_t_ret_diff']:+.2f}  "
          f"share adv>0 {sc['share_adv_pos']*100:.0f}%")
