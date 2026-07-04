"""Reproducible headline run for Study 614 — CLO Equity Yield ("the 15% machine").

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

from clo_equity_yield import data, strategy as st

NAMES = ["ECC", "OXLC"]

print("# CLO Equity Yield — ECC / OXLC vs plain HYG and a credit/equity-matched benchmark (yfinance)")
if data.have_real():
    from quantlab import repro

    tr, px = data.load_real()

    # data stamp on the joint monthly TR+PX panel over the ECC window (the headline window)
    mtr_all, mpx_all = st.aligned_monthly(tr, px, ["ECC", "OXLC", "HYG", "SPY", "BIL"])
    joint = pd.concat([mtr_all.add_suffix("_tr"), mpx_all.add_suffix("_px")], axis=1)
    print(repro.data_stamp("clo-equity monthly panel (ECC window)", joint, asof=data.AS_OF))

    print("\n# Distribution decomposition — total return vs price-only vs the payout stream")
    print("  (monthly distribution component = TR return - price-only return; HAC/Newey-West t, 6 lags)")
    for nm in NAMES:
        mtr, mpx = st.aligned_monthly(tr, px, [nm, "HYG", "SPY", "BIL"])
        d = st.decompose(mtr, mpx, nm)
        w_tr = float((1.0 + mtr[nm]).prod()) * 100
        w_px = float((1.0 + mpx[nm]).prod()) * 100
        print(f"  {nm:>4} [{mtr.index.min().date()} -> {mtr.index.max().date()}, n={d['n_months']}]: "
              f"distribution {d['dist_bps_mo']:+.2f} bps/mo = {d['dist_ann_pct']:+.2f}%/yr  "
              f"HAC t = {d['dist_hac_t']:+.2f}")
        print(f"       TR CAGR {d['tr_cagr_pct']:+.2f}%/yr | price-only CAGR {d['px_cagr_pct']:+.2f}%/yr"
              f" | $100 -> ${w_tr:.1f} (TR) vs ${w_px:.1f} (price-only)")

    print("\n# Risk & return — ECC window (2014-11 -> 2026-06), monthly TR, Sharpe excess-vs-excess over BIL")
    pt = st.perf_table(mtr_all)
    for c in ["ECC", "OXLC", "HYG", "SPY", "BIL"]:
        r = pt.loc[c]
        sh = f"{r['sharpe_excess']:+.2f}" if pd.notna(r["sharpe_excess"]) else "  n/a"
        print(f"  {c:>4}: CAGR {r['cagr_pct']:+6.2f}%/yr  vol {r['vol_pct']:5.2f}%  "
              f"Sharpe(excess) {sh}  maxDD {r['max_dd_pct']:+.2f}%")
    for nm in NAMES:
        m_own, _ = st.aligned_monthly(tr, px, [nm, "HYG", "SPY", "BIL"])
        mu_ex, t_ex = st.nw_tstat(m_own[nm] - m_own["BIL"])
        print(f"  {nm} excess over bills (own window): {mu_ex*1e4:+.2f} bps/mo  HAC t = {t_ex:+.2f} "
              f"(statistically cash)")

    print("\n# HEADLINE — total-return spread vs plain HYG (the one-click alternative; HAC 6 lags)")
    for nm in NAMES:
        mtr, _ = st.aligned_monthly(tr, px, [nm, "HYG", "SPY", "BIL"])
        sp = st.tr_spread_vs(mtr, nm)
        print(f"  {nm:>4} [n={sp['n']}]: spread {sp['spread_bps_mo']:+.2f} bps/mo "
              f"({sp['spread_ann_pct']:+.2f}%/yr)  HAC t = {sp['spread_t']:+.2f} | "
              f"fund TR CAGR {sp['name_cagr_pct']:+.2f}%/yr vs HYG {sp['bench_cagr_pct']:+.2f}%/yr")

    print("\n# Alpha vs the credit/equity-matched benchmark (excess-on-excess, NW HAC 6 lags)")
    print("  fund_excess = alpha + b_HYG * HYG_excess + b_SPY * SPY_excess + e")
    for nm in NAMES:
        mtr, _ = st.aligned_monthly(tr, px, [nm, "HYG", "SPY", "BIL"])
        br = st.benchmark_race(mtr, nm)
        print(f"  {nm:>4}: alpha {br['alpha_bps_mo']:+7.2f} bps/mo ({br['alpha_ann_pct']:+.2f}%/yr)  "
              f"HAC t = {br['t_alpha']:+.2f} | b_HYG {br['beta_HYG']:+.2f} (t {br['t_HYG']:+.2f})  "
              f"b_SPY {br['beta_SPY']:+.2f} (t {br['t_SPY']:+.2f})  R2 {br['r2']:.2f}")
        print(f"        vs own beta-matched benchmark: {nm} CAGR {br['name_cagr_pct']:+.2f}%/yr vs "
              f"bench {br['bench_cagr_pct']:+.2f}%/yr | maxDD {br['name_dd_pct']:+.1f}% vs "
              f"{br['bench_dd_pct']:+.1f}% | spread HAC t = {br['t_spread']:+.2f}")

    print("\n# Robustness — the zero spread / zero alpha is not a lag choice or a subperiod artefact")
    for nm in NAMES:
        mtr, _ = st.aligned_monthly(tr, px, [nm, "HYG", "SPY", "BIL"])
        for lags in (3, 6, 12):
            sp = st.tr_spread_vs(mtr, nm, lags=lags)
            cp = st.carry_premium(mtr, nm, lags=lags)
            print(f"  {nm:>4} NW lags={lags:>2}: spread vs HYG {sp['spread_bps_mo']:+.2f} bps/mo "
                  f"t = {sp['spread_t']:+.2f} | alpha {cp['alpha_bps_mo']:+.2f} bps/mo t = {cp['t_alpha']:+.2f}")
    for label, a in [("2016-01+ (ex the 2015 crunch entry)", "2016-01-01"),
                     ("2020-07+ (post-COVID / high-rates)", "2020-07-01")]:
        sub_tr, sub_px = mtr_all.loc[a:], mpx_all.loc[a:]
        for nm in NAMES:
            d = st.decompose(sub_tr, sub_px, nm)
            sp = st.tr_spread_vs(sub_tr, nm)
            print(f"  {label} {nm:>4}: dist {d['dist_ann_pct']:+.2f}%/yr t = {d['dist_hac_t']:+.1f} | "
                  f"TR {d['tr_cagr_pct']:+.2f}%/yr PX {d['px_cagr_pct']:+.2f}%/yr | "
                  f"spread vs HYG {sp['spread_bps_mo']:+.2f} bps/mo t = {sp['spread_t']:+.2f} (n={sp['n']})")

    print("\n# Third axis — return OF capital or ON capital? (returns arithmetic)")
    for nm in NAMES:
        mtr, mpx = st.aligned_monthly(tr, px, [nm, "HYG", "SPY", "BIL"])
        cs = st.capital_return_split(mtr, mpx, nm)
        print(f"  {nm:>4}: distribution {cs['dist_ann_pct']:+.2f}%/yr, price leg {cs['px_cagr_pct']:+.2f}%/yr"
              f" -> {cs['financed_by_price_pct']:.1f}% of the payout stream offset by price erosion; "
              f"kept (TR) {cs['kept_ann_pct']:+.2f}%/yr")

    print("\n# Crisis autopsies — peak-to-trough TOTAL-return drawdown inside each window (daily tape)")
    ct = st.crisis_table(tr, NAMES + ["HYG", "SPY"])
    for label, row in ct.iterrows():
        cells = "  ".join(f"{tk} {row[tk]:+.1f}%" for tk in NAMES + ["HYG", "SPY"])
        print(f"  {label:<22}: {cells}")

    print("\n# Costs — buy-and-hold, stated for the record")
    print("  Two one-way trades over 11.7 years at ~10 bps one-way (CEF spreads are wider than ETF")
    print("  spreads) = ~20 bps total, ~1.7 bps/yr amortised. The funds' own drag — management +")
    print("  incentive fees + leverage interest, roughly 9-13%/yr of NAV per the funds' reports —")
    print("  is already inside the tape. The mirage is not the commission; the wrapper is.")
else:
    print("(no _cache/ceq_tr.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the payout detector (distribution-component HAC t) and the alpha test (NW regression) must")
print("  recover PLANTED values and must NOT manufacture significance under the null.")
for carry, alpha in [(0.0, 0.0), (0.012, -0.015)]:
    ttr, tpx = data.synthetic_world(carry=carry, alpha=alpha, seed=614)
    smtr, smpx = st.aligned_monthly(ttr, tpx, ["CLOE", "HYG", "SPY", "BIL"])
    sd = st.decompose(smtr, smpx, "CLOE")
    scp = st.carry_premium(smtr, "CLOE")
    print(f"  planted carry={carry*1e4:+7.1f} bps/mo, alpha={alpha*1e4:+7.1f} bps/mo: "
          f"dist {sd['dist_bps_mo']:+.2f} bps/mo (t {sd['dist_hac_t']:+.2f}) | "
          f"alpha {scp['alpha_bps_mo']:+.2f} bps/mo (t {scp['t_alpha']:+.2f}) | "
          f"b_HYG {scp['beta_HYG']:.2f} b_SPY {scp['beta_SPY']:.2f} (planted 1.60 / 0.45)")
