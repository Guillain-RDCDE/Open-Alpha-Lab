"""Reproducible headline run for Study 910 — Managed-Distribution CEF.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/mdc_prices.csv, built once from yfinance) and always-offline on the synthetic control.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from md_cef import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def show(tag: str, s: dict) -> None:
    print(f"  {tag:<8s} n={s['n']:>3d} {s['start']}->{s['end']}  "
          f"annFund {s['fund_ann_pct']:+5.1f}% vs SPY {s['spy_ann_pct']:+5.1f}%  "
          f"| exSharpe {s['fund_sharpe']:.2f} vs SPY {s['spy_sharpe']:.2f} "
          f"(adv {s['sharpe_adv']:+.2f})  | exret {s['fund_exret_bps']:+6.1f} bps/mo "
          f"(HAC t {s['t_exret']:+.2f})  | vsSPY {s['vs_spy_bps']:+6.1f} bps (t {s['t_vs_spy']:+.2f})  "
          f"| alpha {s['alpha_ann_pct']:+5.1f}%/yr (t {s['t_alpha']:+.2f}) beta {s['beta']:.2f} "
          f"R2 {s['r2']:.2f}  | maxDD {s['max_dd_pct']:.0f}% vol {s['fund_vol_pct']:.0f}%")


print("# Managed-Distribution CEF — do persistent-discount, big-payout CEFs beat their asset class?")

if not data.have_real():
    print("(cache miss — fetching the yfinance total-return tape once)")
    data.fetch()

prices = data.load_prices()
panel = data.monthly_panel(prices, asof=data.AS_OF)
cash, spy = panel["BIL"], panel["SPY"]
cols = ["PCEF", "PDI", "UTF", "BST", "RQI", "SPY", "BIL"]
print(repro.data_stamp("monthly panel", panel[cols].dropna(how="all"), asof=data.AS_OF))
print(f"tape: {prices.index.min().date()} -> {prices.index.max().date()} daily; monthly stats "
      f"sliced to as-of {data.AS_OF} (last complete month)")
print("  SURVIVORSHIP/short-history: flagship survivor CEFs only (blow-ups absent -> upper "
      "bound); the 4-name basket has ~11.5y of tape (BST-limited). Named on the Signal axis.")

print("\n# Per-fund, excess-of-cash vs SPY (each fund's own full window)")
for t in ["PCEF", "PDI", "UTF", "BST", "RQI"]:
    show(t, st.fund_stats(panel[t], spy, cash))

print("\n# THE HEADLINE — equal-weight basket (PDI/UTF/BST/RQI), monthly rebalanced")
basket = st.equal_weight_basket(panel, data.BASKET)
b = st.fund_stats(basket, spy, cash)
show("BASKET", b)
bc = st.bootstrap_sharpe_ci(st.excess(basket, cash).to_numpy())
print(f"  basket excess-of-cash Sharpe bootstrap CI: {bc['sharpe']:.2f} "
      f"[{bc['ci_low']:.2f}, {bc['ci_high']:.2f}]  frac_neg={bc['frac_neg']:.3f}  (n_boot={bc['n_boot']})")
pc = st.bootstrap_sharpe_ci(st.excess(panel["PCEF"], cash).to_numpy())
print(f"  PCEF   excess-of-cash Sharpe bootstrap CI: {pc['sharpe']:.2f} "
      f"[{pc['ci_low']:.2f}, {pc['ci_high']:.2f}]  frac_neg={pc['frac_neg']:.3f}")

print("\n# READ: the excess-of-cash return is REAL (HAC t>2, CI clear of 0) — but the excess-vs-")
print("#       excess Sharpe TRAILS SPY and the CAPM alpha is ~0-to-negative: a levered-beta clone.")

print("\n# ERA cut at 2022-01 (the rate-hike regime — leverage got expensive, discounts blew out)")
print("  -- BASKET")
for k, s in st.era_stats(basket, spy, cash, split="2022-01-01").items():
    if s:
        show(k, s)
print("  -- PCEF")
for k, s in st.era_stats(panel["PCEF"], spy, cash, split="2022-01-01").items():
    if s:
        show(k, s)

print("\n# Calendar-year total returns — basket (%)")
cy = st.calendar_year_table(basket)
print("  " + "  ".join(f"{int(y)}:{v:+.1f}" for y, v in zip(cy.index, cy["ret_pct"])))
print(f"  basket max drawdown: {b['max_dd_pct']:.0f}%   (RQI alone: "
      f"{st.fund_stats(panel['RQI'], spy, cash)['max_dd_pct']:.0f}% — the mREIT trap in miniature)")

print("\n# TRADABILITY — costs are NOT the killer (buy-and-hold, ~15 bps CEF spread)")
c = st.costed_net(panel, data.BASKET, cash, spy)
print(f"  gross {c['gross_exret_bps']:+.1f} bps/mo (t {c['t_gross']:+.2f}, Sharpe {c['gross_sharpe']:.2f}) "
      f"-> net {c['net_exret_bps']:+.1f} bps/mo (t {c['t_net']:+.2f}, Sharpe {c['net_sharpe']:.2f})  "
      f"charge {c['charge_bps_per_mo']:.2f} bps/mo  | SPY Sharpe {c['spy_sharpe']:.2f}")
print("  The hidden LEVERED EQUITY BETA (~1.0, R2 0.73) — not costs — erases the edge: you get a")
print("  better Sharpe just holding SPY. Post-2022 the excess return collapsed to ~+32 bps (t<1).")

print("\n# SYNTHETIC CONTROL — deterministic, no network (machinery proof, never market evidence)")
for carry, leak, tag in [(0.0, 0.0, "null (pure levered beta)"),
                         (0.05, 0.0, "planted +5%/yr net carry"),
                         (0.05, 0.05, "return-of-capital trap (carry==leak)")]:
    d = st.synthetic_detect(data.synthetic_world(carry_annual=carry, roc_leak_annual=leak, seed=910))
    print(f"  {tag:<38s}: alpha {d['alpha_ann_pct']:+.2f}%/yr (t {d['t_alpha']:+.2f})  "
          f"beta {d['beta']:.2f} (t {d['t_beta']:.1f})  R2 {d['r2']:.2f}")
null_t = np.array([st.synthetic_detect(data.synthetic_world(carry_annual=0.0, seed=910 + s))["t_alpha"]
                   for s in range(20)])
print(f"  null over 20 seeds: |t_alpha|>=2 in {(np.abs(null_t) >= 2).sum()}/20 "
      f"(~5% false-positive rate — the estimator is unbiased)")
