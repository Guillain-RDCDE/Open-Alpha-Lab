"""Reproducible headline run for Study 619 — BITO Roll Drag.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached yfinance tape under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bito_roll_drag import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402

print("# BITO Roll Drag — futures ETF vs spot bitcoin vs spot ETF (yfinance, total-return)")
if data.have_real():
    f = data.load_real()
    print(repro.data_stamp("BITO/IBIT/BTC-USD/BTC=F daily TR closes", f, asof=data.ASOF))

    # ------------------------------------------------------------------ headline vs spot
    s = st.summarize_drag(f, col="bito", ref="spot")
    print("\n# BITO vs spot bitcoin — since inception (total-return legs)")
    print(f"  window        : {s['start']} -> {s['end']}  ({s['years']:.2f} years, "
          f"{s['n_days']} daily gaps)")
    print(f"  daily gap     : {s['gap_bps_day']:+.2f} bps/day  ->  annualized drag "
          f"{s['drag_ann_pct']:+.2f}%/yr   HAC(21) t = {s['hac_t']:+.2f}  "
          f"(daily gap is noisy: BTC-USD closes ~00:00 UTC vs BITO 16:00 ET)")
    print(f"  cumulative TR : BITO {s['tr_col_pct']:+.2f}%  vs spot {s['tr_ref_pct']:+.2f}%  "
          f"->  shortfall {s['shortfall_pp']:+.2f} pp   wealth ratio {s['wealth_ratio']:.3f}")

    m = st.monthly_gap(f, col="bito", ref="spot")
    print(f"  monthly gap   : {m['gap_bps_month']:+.2f} bps/month over {m['n_months']} complete "
          f"months   HAC(3) t = {m['hac_t']:+.2f}   ({m['neg_share']*100:.1f}% of months negative)")

    # beta sanity: the gap is a fair race
    r_si = f[["bito", "ibit"]].dropna().pct_change().dropna()
    r_ss = f[["bito", "spot"]].dropna().pct_change().dropna()
    beta_ibit = float(np.polyfit(r_si["ibit"], r_si["bito"], 1)[0])
    beta_spot = float(np.polyfit(r_ss["spot"], r_ss["bito"], 1)[0])
    print(f"  beta check    : BITO~IBIT beta = {beta_ibit:.3f} (matched closes) ; "
          f"BITO~spot beta = {beta_spot:.3f} (timestamp attenuation only)")

    # ------------------------------------------------------------------ vs IBIT (clean)
    si = st.summarize_drag(f, col="bito", ref="ibit")
    sc = st.summarize_drag(f, col="ibit", ref="spot")
    print("\n# BITO vs IBIT — the clean matched-close race (spot-ETF era, 2024-01-11+)")
    print(f"  window        : {si['start']} -> {si['end']}  ({si['years']:.2f} years, "
          f"{si['n_days']} daily gaps)")
    print(f"  daily gap     : {si['gap_bps_day']:+.2f} bps/day  ->  annualized drag "
          f"{si['drag_ann_pct']:+.2f}%/yr   HAC(21) t = {si['hac_t']:+.2f}")
    for lags in (5, 63):
        h = st.hac_t(st.daily_gap(f, "bito", "ibit").values, lags=lags)
        print(f"                  HAC({lags}) t = {h['t']:+.2f}  (lag robustness)")
    print(f"  cumulative TR : BITO {si['tr_col_pct']:+.2f}%  vs IBIT {si['tr_ref_pct']:+.2f}%  "
          f"->  shortfall {si['shortfall_pp']:+.2f} pp   wealth ratio {si['wealth_ratio']:.3f}")
    print(f"  IBIT vs spot  : {sc['drag_ann_pct']:+.2f}%/yr   HAC(21) t = {sc['hac_t']:+.2f}  "
          f"(control: the spot ETF tracks within noise, so the timestamp offset does not bias means)")
    print(f"  fee arithmetic: BITO ER 0.95% - IBIT 0.25% = 0.70%/yr; observed gap "
          f"{si['drag_ann_pct']:+.2f}%/yr  ->  ~{abs(si['drag_ann_pct']) - 0.70:.2f}%/yr beyond fees "
          f"= the roll/carry toll")

    # ------------------------------------------------------------------ roll-window attribution
    flags = data.roll_window_flags(f.index, width=5)
    ra_s = st.roll_attribution(f, flags, col="bito", ref="spot")
    ra_i = st.roll_attribution(f, flags, col="bito", ref="ibit")
    print("\n# Roll-window attribution — is the toll paid AT the roll? (5 trading days into expiry Friday)")
    print(f"  vs spot : in-window {ra_s['in_bps_day']:+.2f} bps/day (n={ra_s['n_in']}) vs "
          f"outside {ra_s['out_bps_day']:+.2f} (n={ra_s['n_out']})   Welch t = {ra_s['welch_t']:+.2f}")
    print(f"            window holds {ra_s['share_days']*100:.1f}% of days and "
          f"{ra_s['share_shortfall']*100:.1f}% of the log shortfall")
    print(f"  vs IBIT : in-window {ra_i['in_bps_day']:+.2f} bps/day (n={ra_i['n_in']}) vs "
          f"outside {ra_i['out_bps_day']:+.2f} (n={ra_i['n_out']})   Welch t = {ra_i['welch_t']:+.2f}")
    print(f"            window holds {ra_i['share_days']*100:.1f}% of days and "
          f"{ra_i['share_shortfall']*100:.1f}% of the log shortfall")
    print("  -> no significant concentration: the toll accrues DAILY as the futures premium "
          "converges, it is not an expiry-week execution event")

    # ------------------------------------------------------------------ contango / backwardation
    b = st.basis_series(f)
    idx = b.index
    tau = []
    for d in idx:
        lf = data.last_friday(d.year, d.month)
        if d > lf:
            nm = d + pd.offsets.MonthBegin(1)
            lf = data.last_friday(nm.year, nm.month)
        tau.append((lf - d).days)
    tau = np.array(tau, dtype=float)
    ann_basis = pd.Series(b.values / np.maximum(tau, 1) * 365.0, index=idx)
    print("\n# Contango / backwardation — the carry the roll locks in")
    print(f"  contango days : {float((b > 0).mean())*100:.1f}% of days (front basis > 0; "
          f"median basis {float(b.median())*1e4:+.1f} bps at mean {tau.mean():.1f} days to expiry)")
    print(f"  annualized front basis: median {float(ann_basis.median())*100:+.2f}%/yr "
          f"(2022: {float(ann_basis[ann_basis.index.year == 2022].median())*100:+.2f}%/yr ; "
          f"2024-26: {float(ann_basis[ann_basis.index.year >= 2024].median())*100:+.2f}%/yr)")

    gap = m["series"]
    print("\n  calendar-year monthly gap vs spot (complete months):")
    for y in (2021, 2022, 2023, 2024, 2025, 2026):
        g = gap[gap.index.year == y]
        if len(g) == 0:
            continue
        cum = ((1.0 + g).prod() - 1.0) * 100.0
        print(f"    {y}: {g.mean()*1e4:+7.1f} bps/mo over {len(g):2d} months  "
              f"(cum {cum:+.2f} pp)")
    back = gap[gap.index.year == 2022]
    cont = gap[gap.index.year >= 2023]
    w = st.welch_t(back.values, cont.values)
    print(f"  2022 (backwardated, median ann basis -4.7%): {back.mean()*1e4:+.1f} bps/mo (n={len(back)})")
    print(f"  2023+ (contango era)                       : {cont.mean()*1e4:+.1f} bps/mo (n={len(cont)})")
    print(f"  Welch t of the difference = {w['t']:+.2f}  -> directionally the carry story "
          f"(the drag FLIPPED SIGN in the backwardated year) but NOT certified at t >= 2")
    rs = st.regime_split(f, col="bito", ref="ibit")
    print(f"  daily lag-1 basis-sign split (vs IBIT): contango {rs['contango_ann_pct']:+.2f}%/yr "
          f"vs backwardation {rs['backward_ann_pct']:+.2f}%/yr   Welch t = {rs['welch_t']:+.2f} "
          f"(underpowered; see synthetic-control caveat)")

    # ------------------------------------------------------------------ tradability
    print("\n# Tradability — long IBIT / short BITO, dollar-neutral, daily rebalance")
    print("  (short pays borrow on full notional; 2 bps one-way on rebalancing turnover; "
          "positions from the prior close)")
    for borrow in (0.0, 0.02, 0.05):
        sp = st.spread_trade(f, borrow_ann=borrow, cost_bps=2.0)
        print(f"  borrow {borrow*100:>4.1f}%/yr: gross {sp['gross_ann_pct']:+.2f}%/yr -> net "
              f"{sp['net_ann_pct']:+.2f}%/yr   HAC t = {sp['net_hac_t']:+.2f}   "
              f"Sharpe {sp['sharpe']:+.2f}")
    sp2 = st.spread_trade(f, borrow_ann=0.02, cost_bps=2.0)
    print(f"  spread vol {sp2['vol_ann_pct']:.2f}%/yr, worst day {sp2['worst_day_pct']:+.2f}%  "
          f"(n={sp2['n_days']} days)")
else:
    print("(no _cache/brd_prices.csv — run data.fetch_tape() once to build the cache)")

# ---------------------------------------------------------------------- synthetic control
print("\n# Synthetic control — deterministic, no network (machinery proof, never market evidence)")
print("  a GBM spot + front-month curve with a PLANTED annualized basis + fee; the detector")
print("  must recover the planted drag and must NOT invent one from basis noise alone.")
for basis, fee in ((0.0, 0.0), (0.10, 0.0095)):
    w = data.synthetic_world(basis_ann=basis, fee_ann=fee, seed=619)
    s = st.summarize_drag(w)
    flags = data.roll_window_flags(w.index)
    ra = st.roll_attribution(w, flags)
    planted = -(basis + fee) * 100.0
    print(f"  planted {planted:+6.2f}%/yr (basis {basis*100:.0f}% + fee {fee*100:.2f}%): "
          f"measured {s['drag_ann_pct']:+6.2f}%/yr   HAC t = {s['hac_t']:+7.2f}   "
          f"roll-window Welch t = {ra['welch_t']:+.2f} (diffuse by construction)")
print("  caveat the control exposed: conditioning the DAILY gap on the lagged basis sign is")
print("  artifact-prone (AR(1) basis noise mean-reverts into the gap), so regime attribution")
print("  leans on calendar-year basis levels, not the daily sign flip.")
