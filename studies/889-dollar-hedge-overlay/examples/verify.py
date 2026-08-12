"""Reproducible headline run for Study 889 — Broad Dollar-Hedge Overlay.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached tape under ``_cache/`` (fetching
once on a cache miss), and always runs the synthetic control with no network.

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

from dollar_hedge import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Broad Dollar-Hedge Overlay — does the EAFE hedged-minus-unhedged gap = the rate diff?")

if not data.have_real():
    print("(cache miss — fetching the tape once)")
    data.fetch()

prices = data.load_prices()
panel = data.monthly_panel(prices)
print(f"[data] {len(panel)} months  {panel.index.min().date()} -> {panel.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint={fingerprint(panel)}")
print("  Foreign short rate = EAFE-weighted policy blend (ECB .40/BOJ .25/BOE .20/SNB .15);")
print("  fx_foreign = SPOT USD return of the same 4-currency basket (UUP embeds collateral")
print("  yield and is NOT used for the carry identity). Young ETFs: HEFA starts 2014-03.")

for name, (h, u, lbl) in data.PAIRS.items():
    pf = st.pair_frame(panel, h, u)
    s = st.pair_stats(pf)
    print(f"\n==== {name}  ({lbl})")
    print(f"  n={s['n']}  {s['start']} -> {s['end']}")
    print("  # THE IDENTITY — carry_hat = (hedged - unhedged) + fx_foreign")
    print(f"  carry_hat: {s['carry_ann_pct']:+.2f}%/yr  HAC t = {s['t_carry']:+.2f}   "
          f"observed rate differential {s['rate_diff_ann_pct']:+.2f}%/yr")
    print(f"  hedge reg (diff = a + b*(-fx_foreign)): beta = {s['beta']:.2f} (t {s['t_beta']:+.1f}), "
          f"alpha = {s['alpha_ann_pct']:+.2f}%/yr (t {s['t_alpha']:+.2f}), R2 = {s['r2']:.2f}")
    mc = st.mean_boot_ci(pf["carry_hat"].values)
    print(f"  carry bootstrap: {mc['mean_ann_pct']:+.2f}%/yr  95% CI [{mc['ci_low']:+.2f}, "
          f"{mc['ci_high']:+.2f}]  frac<=0 {mc['frac_le_zero']:.3f}")
    era = st.era_split(pf, cut="2022-01-01")
    print(f"  era pre-2022 : carry {era['pre']['carry_ann_pct']:+.2f}%/yr t = {era['pre']['t_carry']:+.2f} "
          f"(n={era['pre']['n']}, obs diff {era['pre']['rate_diff_ann_pct']:+.2f})")
    print(f"  era 2022+    : carry {era['post']['carry_ann_pct']:+.2f}%/yr t = {era['post']['t_carry']:+.2f} "
          f"(n={era['post']['n']}, obs diff {era['post']['rate_diff_ann_pct']:+.2f})")
    print("  # THE EXCESS-OF-CASH SHARPE RACE (both legs minus BIL)")
    ch = st.sharpe_boot_ci(pf["h_ex"].values)
    cu = st.sharpe_boot_ci(pf["u_ex"].values)
    print(f"  hedged   ex-cash Sharpe {ch['sharpe']:.2f}  95% CI [{ch['ci_low']:.2f}, {ch['ci_high']:.2f}]")
    print(f"  unhedged ex-cash Sharpe {cu['sharpe']:.2f}  95% CI [{cu['ci_low']:.2f}, {cu['ci_high']:.2f}]  "
          f"(advantage {s['sharpe_adv']:+.2f}; CIs overlap)")
    print(f"  max drawdown: hedged {st.max_drawdown(pf['hedged'].values)*100:.1f}%  "
          f"unhedged {st.max_drawdown(pf['unhedged'].values)*100:.1f}%")

# Overlay + spread (on the clean HEFA/EFA pair)
pf = st.pair_frame(panel, "HEFA", "EFA")
print("\n# THE 'HEDGE WHEN THE US OUT-YIELDS' OVERLAY (HEFA/EFA, signal = prior-month diff_rate > 0)")
ov = st.overlay_switch(pf, thresh_ann_pct=0.0, cost_bps_oneway=3.0)
print(f"  switches over {ov['n']} months: {ov['switches']} (share hedged {ov['share_hedged']:.2f}) "
      f"-> the US out-yields EAFE almost the whole sample, so the switch ~ 'always hedge'")
print(f"  Sharpe ex-cash: overlay {ov['sharpe_overlay_ex']:.2f} | always-hedged {ov['sharpe_hedged_ex']:.2f} "
      f"| always-unhedged {ov['sharpe_unhedged_ex']:.2f}")
print(f"  overlay advantage vs unhedged {ov['adv_vs_unhedged']:+.2f}, vs hedged {ov['adv_vs_hedged']:+.2f} "
      f"(the switch adds NOTHING over just hedging; cost drag {ov['cost_drag_ann_pct']:.3f}%/yr)")
print("\n# ISOLATING THE CARRY — long hedged / short unhedged (= long the dollar), costed")
sp = st.spread_trade(pf, borrow_annual_bps=50.0, cost_bps_oneway=3.0, turnover_per_year=2.0)
print(f"  spread net {sp['net_diff_ann_pct']:+.2f}%/yr  HAC t = {sp['t_net_diff']:+.2f} "
      f"(charge {sp['charge_ann_pct']:.2f}%/yr) -> the fx vol it carries kills the t; NOT a clean carry")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([st.synthetic_detect(
    data.synthetic_world(n_months=180, carry_annual=0.0, seed=889 + s))["t_carry"] for s in range(20)])
print(f"  null (carry=0), 20 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_t) >= 2).sum()}/20")
planted = st.synthetic_detect(data.synthetic_world(n_months=180, carry_annual=0.03, seed=889))
print(f"  planted (carry=+3%/yr): recovered {planted['carry_ann_pct']:+.2f}%/yr "
      f"(HAC t {planted['t_carry']:+.2f}), hedge beta {planted['beta']:.2f}")
