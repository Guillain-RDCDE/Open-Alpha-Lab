"""Reproducible headline run for Study 791 — Advertising-Brand-Capital.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EDGAR + yfinance tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from advertising_brand import data, strategy as st  # noqa: E402

FRAC, COST_BPS, BORROW_BPS = 1 / 3, 10.0, 100.0

print("# Advertising-Brand-Capital — do heavy advertisers earn a forward return premium?")

if not data.have_real():
    print("(cache miss — fetching EDGAR fundamentals + yfinance prices once)")
    data.fetch_fundamentals()
    data.fetch_prices()

real = data.load_real(allow_survivorship_bias=True)
sig = data.build_signal(real)
cov = data.coverage_stats(real)

print(f"\n# COVERAGE (the honest scope) — the advertising line is thin even here")
print(f"  basket names: {cov['n_basket']} | with an EDGAR advertising series: {cov['n_adv_names']}")
print(f"  advertising fill: {cov['adv_cells']}/{cov['adv_possible']} name-years "
      f"({cov['adv_fill_pct']:.0f}%), FY {cov['adv_fy_min']}-{cov['adv_fy_max']}")
print(f"  names filing the broader Marketing&Advertising tag (over-states pure ads): "
      f"{cov['n_marketing_tag']}")

months = sig.dropna(how="all").index
print(f"  ranked months: {len(months)} ({months.min().date()} -> {months.max().date()})")

last = months.max()
lo, sh = st.tertile_members(sig.loc[last])
print(f"\n# Final-month ({last.date()}) tertiles by advertising intensity")
print(f"  LONG  (heavy advertisers): {', '.join(sorted(lo))}")
print(f"  SHORT (light advertisers): {', '.join(sorted(sh))}")

r = st.race(real, sig, frac=FRAC, cost_bps=COST_BPS, borrow_bps=BORROW_BPS,
            n_shuffles=300, seed=791)

print("\n# THE RACE — long heavy-advertisers / short light-advertisers, vs the market")
for label, key in [("Long (heavy adv, top tertile)", "long"),
                   ("Short (light adv, bottom tertile)", "short"),
                   ("Long - short (advertising spread)", "long_short"),
                   ("S&P 500 (SPY)", "spy")]:
    s = st.summarize(r[key])
    print(f"  {label:36s}: CAGR {s['cagr']*100:6.2f}%  Sharpe {s['sharpe']:5.2f}  "
          f"maxDD {s['max_dd']*100:6.1f}%  mean {s['mean_ann']*100:+.2f}%/yr")

print("\n# Signal-axis test — Newey-West HAC t (REAL needs t>=2 AND placebo survival)")
for label, key in [("Long - short spread", "test_ls"),
                   ("Long - SPY", "test_long_vs_spy"),
                   ("Short - SPY", "test_short_vs_spy")]:
    d = r[key]
    print(f"  {label:22s}: {d['mean_ann']*100:+.2f}%/yr  HAC t = {d['tstat']:+.2f}  (n={d['n']})")
print(f"  positive-spread months: {r['hit']}/{r['n_ls']} = {r['hit_rate']*100:.1f}% "
      f"(Wilson 95% [{r['hit_lo']*100:.1f}%, {r['hit_hi']*100:.1f}%])")
print(f"  Welch t, pooled leg name-month returns (heavy vs light): {r['welch_legs']:+.2f}")

print("\n# Placebo — does the SIGNAL do the work, or any split? (label-shuffle null, 400 draws)")
print(f"  advertising long-short: {r['test_ls']['mean_ann']*100:+.2f}%/yr  |  "
      f"shuffled null mean {r['placebo_null'].mean()*100:+.2f}%/yr "
      f"(sd {r['placebo_null'].std()*100:.2f})")
print(f"  percentile in null = {r['placebo_pctile']:.1f}  ->  two-sided p = {r['placebo_p']:.4f}")

print("\n# Costs + short-borrow — the thin edge net of frictions")
print(f"  avg one-way turnover: {r['avg_turnover']*100:.2f}%/month (a slow, annual characteristic)")
print(f"  gross long-short: {r['test_ls']['mean_ann']*100:+.2f}%/yr (t={r['test_ls']['tstat']:+.2f})  ->  "
      f"net of {COST_BPS:.0f}bps turnover + {BORROW_BPS:.0f}bps/yr borrow: "
      f"{r['test_ls_net']['mean_ann']*100:+.2f}%/yr (t={r['test_ls_net']['tstat']:+.2f})")

print("\n# Robustness — long/short fraction")
for frac, name in [(1/2, "half"), (1/3, "tertile"), (1/4, "quartile"), (1/5, "quintile")]:
    b = st.signal_books(sig, real["returns"], frac=frac)
    d = st.hac_tstat(b["long_short"])
    print(f"  top/bottom {name:8s}: {d['mean_ann']*100:+.2f}%/yr  HAC t = {d['tstat']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the harness must NOT manufacture a spread at edge=0 and MUST recover a planted one.")
for edge in (0.0, 0.06):
    sg, rr, bb, truth = data.synthetic_panel(edge=edge, seed=791)
    b = st.signal_books(sg, rr, frac=1 / 3)
    d = st.hac_tstat(b["long_short"])
    print(f"  edge={edge:.2f}: recovered long-short {d['mean_ann']*100:+.2f}%/yr  HAC t = {d['tstat']:+.2f}")

print(f"\n# Fingerprint (adv + rev + returns panels): "
      f"{data.fingerprint(real['adv'], real['rev'], real['returns'])}")
