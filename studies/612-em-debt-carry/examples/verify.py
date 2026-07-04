"""Reproducible headline run for Study 612 — EM Debt Carry.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached total-return + price-only
tapes under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd

from em_debt_carry import data, strategy as st

print("# EM Debt Carry — EMB (hard-currency) vs IEF, with the EMLC local-currency contrast (yfinance)")
if data.have_real():
    from quantlab import repro

    tr, px = data.load_real()

    # data stamp on the joint monthly TR+PX panel over the EMB window (the headline window)
    mtr, mpx = st.aligned_monthly(tr, px, ["EMB", "IEF", "SPY", "BIL"])
    joint = pd.concat([mtr.add_suffix("_tr"), mpx.add_suffix("_px")], axis=1)
    print(repro.data_stamp("em-debt-carry monthly panel (EMB window)", joint, asof=data.AS_OF))

    print("\n# The PROMISED carry — coupon pickup (TR minus price-only, EMB coupon minus IEF coupon)")
    yp = st.yield_pickup(mtr, mpx, "EMB", "IEF")
    print(f"  EMB coupon stream : {yp['coupon_name_ann_pct']:+.2f}%/yr  (HAC t = {yp['coupon_name_t']:+.1f})")
    print(f"  IEF coupon stream : {yp['coupon_bench_ann_pct']:+.2f}%/yr")
    print(f"  yield PICKUP      : {yp['pickup_bps_mo']:+.2f} bps/mo = {yp['pickup_ann_pct']:+.2f}%/yr  "
          f"HAC t = {yp['hac_t']:+.2f}  (n={yp['n_months']})")
    roll = yp["rolling"]
    print(f"  rolling-12m pickup: min {roll.min():+.2f}%  median {roll.median():+.2f}%  "
          f"max {roll.max():+.2f}%  (always positive: {bool((roll > 0).all())})")

    print("\n# The COLLECTED carry — total-return spread EMB minus IEF (the decisive Signal stat)")
    ss = st.spread_stats(mtr, "EMB", "IEF")
    print(f"  window {ss['start'].date()} -> {ss['end'].date()}  (n={ss['n_months']} months, "
          f"{ss['n_months']/12:.1f} years)")
    print(f"  TR spread: {ss['spread_bps_mo']:+.2f} bps/mo = {ss['spread_ann_pct']:+.2f}%/yr  "
          f"HAC t = {ss['hac_t']:+.2f}  skew = {ss['skew']:+.2f}")
    print(f"  $100 -> ${ss['wealth_name']:.1f} (EMB TR)  vs  ${ss['wealth_bench']:.1f} (IEF TR)")

    print("\n# Risk & return — EMB window, monthly TR, Sharpe excess-vs-excess over BIL")
    pt = st.perf_table(mtr)
    for c in ["EMB", "IEF", "SPY", "BIL"]:
        r = pt.loc[c]
        sh = f"{r['sharpe_excess']:+.2f}" if pd.notna(r["sharpe_excess"]) else "  n/a"
        print(f"  {c:>4}: CAGR {r['cagr_pct']:+6.2f}%/yr  vol {r['vol_pct']:5.2f}%  "
              f"Sharpe(excess) {sh}  maxDD {r['max_dd_pct']:+.2f}%")

    print("\n# The carry-crash profile — spread conditional on the equity tape (worst-decile SPY months)")
    ro = st.riskoff_profile(mtr, "EMB", "IEF")
    print(f"  worst {ro['q']*100:.0f}% SPY months (n={ro['n_off']}, cut {ro['cut_pct']:+.1f}%): "
          f"spread {ro['off_bps_mo']:+.1f} bps/mo")
    print(f"  all other months  (n={ro['n_rest']}): spread {ro['rest_bps_mo']:+.1f} bps/mo")
    print(f"  risk-off gap: {ro['gap_bps_mo']:+.1f} bps/mo  Welch t = {ro['welch_t']:+.2f}  "
          f"| corr(spread, SPY) = {ro['corr_spy']:+.2f}")

    print("\n# What's left after duration AND the hidden equity beta (excess-on-excess, NW HAC 6 lags)")
    print("  EMB_excess = alpha + b_IEF * IEF_excess + b_SPY * SPY_excess + e")
    cp = st.carry_premium(mtr, "EMB")
    print(f"  alpha {cp['alpha_bps_mo']:+.2f} bps/mo ({cp['alpha_ann_pct']:+.2f}%/yr)  "
          f"HAC t = {cp['t_alpha']:+.2f} | b_IEF {cp['beta_IEF']:+.2f} (t {cp['t_IEF']:+.2f})  "
          f"b_SPY {cp['beta_SPY']:+.2f} (t {cp['t_SPY']:+.2f})  R2 {cp['r2']:.2f}")

    print("\n# The local-currency contrast — EMLC (2010-08+): same borrowers, plus their currencies")
    mtr2, mpx2 = st.aligned_monthly(tr, px, ["EMLC", "EMB", "IEF", "SPY", "BIL"])
    print(f"  common window {mtr2.index.min().date()} -> {mtr2.index.max().date()}  (n={len(mtr2)})")
    s_lc = st.spread_stats(mtr2, "EMLC", "IEF")
    s_hc = st.spread_stats(mtr2, "EMB", "IEF")
    s_x = st.spread_stats(mtr2, "EMB", "EMLC")
    y_lc = st.yield_pickup(mtr2, mpx2, "EMLC", "IEF")
    print(f"  EMLC coupon stream: {y_lc['coupon_name_ann_pct']:+.2f}%/yr  "
          f"(pickup vs IEF {y_lc['pickup_ann_pct']:+.2f}%/yr, HAC t = {y_lc['hac_t']:+.2f})")
    print(f"  EMLC - IEF  TR spread: {s_lc['spread_bps_mo']:+.2f} bps/mo = {s_lc['spread_ann_pct']:+.2f}%/yr  "
          f"HAC t = {s_lc['hac_t']:+.2f} | $100 -> ${s_lc['wealth_name']:.1f} vs ${s_lc['wealth_bench']:.1f}")
    print(f"  EMB  - IEF  TR spread (same window): {s_hc['spread_bps_mo']:+.2f} bps/mo = "
          f"{s_hc['spread_ann_pct']:+.2f}%/yr  HAC t = {s_hc['hac_t']:+.2f}")
    print(f"  EMB  - EMLC TR spread: {s_x['spread_bps_mo']:+.2f} bps/mo = {s_x['spread_ann_pct']:+.2f}%/yr  "
          f"HAC t = {s_x['hac_t']:+.2f}  (the FX leg's bill)")

    print("\n# Crisis autopsies — peak-to-trough TOTAL-return drawdown inside each window (daily tape)")
    ct = st.crisis_table(tr, ["EMB", "EMLC", "IEF", "SPY"])
    for label, row in ct.iterrows():
        cells = "  ".join((f"{tk} {row[tk]:+.1f}%" if pd.notna(row[tk]) else f"{tk}   n/a")
                          for tk in ["EMB", "EMLC", "IEF", "SPY"])
        print(f"  {label:<20}: {cells}")
    print("  (EMLC listed 2010-07, so its GFC cell is empty)")

    print("\n# Third axis — does the spread survive the crises it insures? (fixed windows, monthly spread)")
    cl = st.crisis_ledger(mtr, "EMB", "IEF")
    for lab, cum, n in cl["per_window"]:
        print(f"  {lab:<20}: cumulative spread {cum:+.2f}%  over {n} months")
    print(f"  CRISIS months (n={cl['n_crisis']}): {cl['crisis_bps_mo']:+.1f} bps/mo, "
          f"cumulative {cl['crisis_cum_pct']:+.1f}%")
    print(f"  CALM   months (n={cl['n_calm']}): {cl['calm_bps_mo']:+.1f} bps/mo "
          f"(HAC t = {cl['calm_hac_t']:+.2f}), cumulative {cl['calm_cum_pct']:+.1f}%")
    print(f"  crisis-vs-calm gap: Welch t = {cl['welch_t']:+.2f}")
    print(f"  whole-sample cumulative spread: {cl['total_cum_pct']:+.1f}%  "
          f"(the five windows hand back {abs(cl['crisis_cum_pct'])/cl['calm_cum_pct']*100:.0f}% "
          f"of the calm-month carry)")

    print("\n# Robustness — HAC lag choice and subperiods (EMB - IEF TR spread)")
    sp = mtr["EMB"] - mtr["IEF"]
    for lags in (3, 6, 12):
        mu, t = st.nw_tstat(sp, lags=lags)
        print(f"  NW lags={lags:>2}: spread {mu*1e4:+.2f} bps/mo  t = {t:+.2f}")
    for label, lo in [("ex-GFC (2010-01+)", "2010-01-01"), ("last decade (2016-07+)", "2016-07-01")]:
        mu, t = st.nw_tstat(sp.loc[lo:])
        print(f"  {label:<22}: spread {mu*1e4:+.2f} bps/mo  t = {t:+.2f}  (n={len(sp.loc[lo:])})")

    print("\n# Costs — buy-and-hold, stated for the record")
    print("  Two one-way ETF trades over 18.5 years at 5 bps one-way = ~10 bps total, ~0.5 bps/yr")
    print("  amortised; EMB's 0.39%/yr (EMLC's 0.30%/yr) expense ratio is already inside the tape.")
    print("  The verdict is not about costs — it is about what the collected spread actually is.")
else:
    print("(no _cache/edc_tr.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the spread test (HAC t), the coupon detector and the risk-off split must recover")
print("  PLANTED values and must NOT manufacture significance under the null.")
for spread, beta_spy in [(0.0, 0.0), (0.0030, 0.45)]:
    ttr, tpx = data.synthetic_world(spread=spread, beta_spy=beta_spy, seed=612)
    smtr, smpx = st.aligned_monthly(ttr, tpx, ["EMD", "IEF", "SPY", "BIL"])
    sss = st.spread_stats(smtr, "EMD", "IEF")
    syp = st.yield_pickup(smtr, smpx, "EMD", "IEF")
    sro = st.riskoff_profile(smtr, "EMD", "IEF")
    print(f"  planted spread={spread*1e4:+6.1f} bps/mo, beta_spy={beta_spy:.2f}: "
          f"spread {sss['spread_bps_mo']:+.2f} bps/mo (HAC t {sss['hac_t']:+.2f}) | "
          f"pickup {syp['pickup_bps_mo']:+.2f} bps/mo (t {syp['hac_t']:+.1f}, planted "
          f"{(0.0045-0.0020)*1e4:.0f}) | risk-off gap {sro['gap_bps_mo']:+.1f} bps/mo "
          f"(Welch t {sro['welch_t']:+.2f})")
