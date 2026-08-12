"""Reproducible headline run for Study 894 — Trend Overlay on 60/40.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF closes under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from trend6040 import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

COST = 3.0        # one-way switching cost, bps (generous for liquid ETFs)
TAX = 0.25        # short-term-gains tax rate on realised overlay exits

print("# Trend Overlay on 60/40 — does a 200-day filter cut drawdown and keep the return?")

if not data.have_real():
    print("(cache miss — fetching SPY/IEF/AGG/BIL once)")
    data.fetch()

px = data.load_prices()
print(f"[data] {px.shape[1]} tickers, {len(px)} rows  "
      f"{px.index.min().date()} -> {px.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(px)}")
print("  SHORT HISTORY: BIL (cash leg) launches 2007-05 -> the balanced book here spans "
      "~2007-2026. Named on the Signal axis.")

# ---- the two arms, gross and net ----
bt_g = st.trend_overlay(px, cost_bps=0.0, tax_rate=0.0)     # pure gross
bt_c = st.trend_overlay(px, cost_bps=COST, tax_rate=0.0)    # + switching costs
bt_t = st.trend_overlay(px, cost_bps=COST, tax_rate=TAX)    # + switching costs + tax

g = st.excess_race(bt_g, "r_gross")
c = st.excess_race(bt_c, "r_net")
t = st.excess_race(bt_t, "r_net")

print("\n# THE HEADLINE — trend-overlaid 60/40 vs static 60/40 (excess of BIL cash)")
print(f"  n = {g['n_days']} days")
print(f"  GROSS   : overlay Sharpe {g['sharpe_strat']:.3f} vs static {g['sharpe_bench']:.3f}"
      f"  (adv {g['sharpe_adv']:+.3f})")
print(f"            max DD {g['maxdd_strat']*100:+.1f}% vs {g['maxdd_bench']*100:+.1f}%"
      f"  (cut {g['dd_cut']*100:+.1f}pp)")
print(f"            CAGR {g['cagr_strat']*100:.2f}% vs {g['cagr_bench']*100:.2f}%"
      f"   vol {g['vol_strat']*100:.1f}% vs {g['vol_bench']*100:.1f}%")
print(f"            return diff {g['diff_bps']:+.3f} bps/day  NW(10) t = {g['t_nw_diff']:+.2f}"
      f"  (one-sample t = {g['t_1s_diff']:+.2f})")
print(f"  NET(cost): adv {c['sharpe_adv']:+.3f}  DD cut {c['dd_cut']*100:+.1f}pp  "
      f"CAGR {c['cagr_strat']*100:.2f}%")
print(f"  NET(+tax): adv {t['sharpe_adv']:+.3f}  DD cut {t['dd_cut']*100:+.1f}pp  "
      f"CAGR {t['cagr_strat']*100:.2f}%  <- a 25% short-term tax flips the Sharpe edge negative")

print("\n# IS THE SHARPE ADVANTAGE REAL? paired block-bootstrap (gross)")
bs = st.sharpe_adv_bootstrap(bt_g, which="r_gross", n_boot=3000, block=20, seed=894)
print(f"  Sharpe advantage {bs['adv']:+.3f}  95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]  "
      f"P(adv>0) = {bs['p_pos']:.3f}")
dm = st.diff_mean_ci_bootstrap((bt_g['r_gross'] - bt_g['r_bench']), n_boot=3000, block=20, seed=894)
print(f"  return diff {dm['mean_bps']:+.3f} bps/day  95% CI [{dm['lo_bps']:+.3f}, {dm['hi_bps']:+.3f}]")

print("\n# ROBUSTNESS — two eras (split 2017-01-01), gross")
ec = st.era_cut(bt_g, "2017-01-01", "r_gross")
for lbl, r in [("2007-2016", ec["early"]), ("2017-2026", ec["late"])]:
    print(f"  {lbl}: n={r['n_days']}  adv {r['sharpe_adv']:+.3f}  DD cut {r['dd_cut']*100:+.1f}pp  "
          f"(overlay Sharpe {r['sharpe_strat']:.2f} vs static {r['sharpe_bench']:.2f})")

print("\n# CALENDAR YEARS — overlay vs static total return (net of 3bps switching)")
cy = st.calendar_year_table(bt_c, "r_net")
for yr in (2008, 2018, 2022):
    if yr in cy.index:
        row = cy.loc[yr]
        print(f"  {yr}: overlay {row['overlay_%']:+.1f}%  static {row['static_%']:+.1f}%  "
              f"(diff {row['diff_pp']:+.1f}pp)")

print("\n# THE TIMER — Sharpe advantage & DD cut across a cost grid (net, no tax)")
for row in st.timer(px, cost_grid=(0.0, 1.0, 3.0, 5.0), tax_rate=0.0):
    print(f"  cost {row['cost_bps']:>4.1f} bps: adv {row['sharpe_adv']:+.3f}  "
          f"DD cut {row['dd_cut']*100:+.1f}pp  CAGR {row['cagr_strat']*100:.2f}% vs "
          f"{row['cagr_bench']*100:.2f}%")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
adv_p, dd_p, adv_n = [], [], []
for s_ in range(12):
    dp = st.synthetic_detect(data.synthetic_prices(edge=1.0, seed=894 + s_, n_days=5000))
    dn = st.synthetic_detect(data.synthetic_prices(edge=0.0, seed=894 + s_, n_days=5000))
    adv_p.append(dp["sharpe_adv"]); dd_p.append(dp["dd_cut"]); adv_n.append(dn["sharpe_adv"])
adv_p, dd_p, adv_n = np.array(adv_p), np.array(dd_p), np.array(adv_n)
print(f"  planted (edge=1), 12 seeds: Sharpe adv mean {adv_p.mean():+.3f}, DD cut "
      f"{dd_p.mean()*100:+.1f}pp, adv>0 in {(adv_p > 0).sum()}/12")
print(f"  null    (edge=0), 12 seeds: Sharpe adv mean {adv_n.mean():+.3f} (no bear to duck)")
print("  -> the machinery recovers a planted trend benefit and stays flat on the null; "
      "faithful-engine check only, never cites the real stamp.")
