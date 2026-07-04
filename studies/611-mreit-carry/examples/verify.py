"""Reproducible headline run for Study 611 — mREIT Carry.

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

from mreit_carry import data, strategy as st

NAMES = ["REM", "NLY", "AGNC"]

print("# mREIT Carry — REM / NLY / AGNC vs a duration-matched levered-IEF benchmark (yfinance)")
if data.have_real():
    from quantlab import repro

    tr, px = data.load_real()

    # data stamp on the joint monthly TR+PX panel over the REM window (the headline window)
    mtr_all, mpx_all = st.aligned_monthly(tr, px, ["REM", "IEF", "SPY", "BIL"])
    joint = pd.concat([mtr_all.add_suffix("_tr"), mpx_all.add_suffix("_px")], axis=1)
    print(repro.data_stamp("mreit-carry monthly panel (REM window)", joint, asof=data.AS_OF))

    print("\n# Carry decomposition — total return vs price-only vs the dividend stream")
    print("  (monthly dividend component = TR return - price-only return; HAC/Newey-West t, 6 lags)")
    for nm in NAMES:
        mtr, mpx = st.aligned_monthly(tr, px, [nm, "IEF", "SPY", "BIL"])
        d = st.decompose(mtr, mpx, nm)
        w_tr = float((1.0 + mtr[nm]).prod()) * 100
        w_px = float((1.0 + mpx[nm]).prod()) * 100
        print(f"  {nm:>4} [{mtr.index.min().date()} -> {mtr.index.max().date()}, n={d['n_months']}]: "
              f"dividend {d['div_bps_mo']:+.2f} bps/mo = {d['div_ann_pct']:+.2f}%/yr  "
              f"HAC t = {d['div_hac_t']:+.2f}")
        print(f"       TR CAGR {d['tr_cagr_pct']:+.2f}%/yr | price-only CAGR {d['px_cagr_pct']:+.2f}%/yr"
              f" | $100 -> ${w_tr:.1f} (TR) vs ${w_px:.1f} (price-only)")

    print("\n# Risk & return — REM window (2007-06 -> 2026-06), monthly TR, Sharpe excess-vs-excess over BIL")
    pt = st.perf_table(mtr_all)
    for c in ["REM", "IEF", "SPY", "BIL"]:
        r = pt.loc[c]
        sh = f"{r['sharpe_excess']:+.2f}" if pd.notna(r["sharpe_excess"]) else "  n/a"
        print(f"  {c:>4}: CAGR {r['cagr_pct']:+6.2f}%/yr  vol {r['vol_pct']:5.2f}%  "
              f"Sharpe(excess) {sh}  maxDD {r['max_dd_pct']:+.2f}%")
    mu_ex, t_ex = st.nw_tstat(mtr_all["REM"] - mtr_all["BIL"])
    print(f"  REM excess over bills: {mu_ex*1e4:+.2f} bps/mo  HAC t = {t_ex:+.2f} "
          f"(indistinguishable from cash)")

    print("\n# Carry premium vs the duration-matched benchmark (excess-on-excess, NW HAC 6 lags)")
    print("  mREIT_excess = alpha + b_IEF * IEF_excess + b_SPY * SPY_excess + e")
    for nm in NAMES:
        mtr, _ = st.aligned_monthly(tr, px, [nm, "IEF", "SPY", "BIL"])
        br = st.benchmark_race(mtr, nm)
        print(f"  {nm:>4}: alpha {br['alpha_bps_mo']:+7.2f} bps/mo ({br['alpha_ann_pct']:+.2f}%/yr)  "
              f"HAC t = {br['t_alpha']:+.2f} | b_IEF {br['beta_IEF']:+.2f} (t {br['t_IEF']:+.2f})  "
              f"b_SPY {br['beta_SPY']:+.2f} (t {br['t_SPY']:+.2f})  R2 {br['r2']:.2f}")
        print(f"        vs own beta-matched benchmark: {nm} CAGR {br['name_cagr_pct']:+.2f}%/yr vs "
              f"bench {br['bench_cagr_pct']:+.2f}%/yr | maxDD {br['name_dd_pct']:+.1f}% vs "
              f"{br['bench_dd_pct']:+.1f}% | spread HAC t = {br['t_spread']:+.2f}")

    print("\n# Robustness — REM alpha across HAC lag choices and ex-GFC subperiod")
    for lags in (3, 6, 12):
        cp = st.carry_premium(mtr_all, "REM", lags=lags)
        print(f"  NW lags={lags:>2}: alpha {cp['alpha_bps_mo']:+.2f} bps/mo  t = {cp['t_alpha']:+.2f}")
    sub_tr = mtr_all.loc["2010-01-01":]
    sub_px = mpx_all.loc["2010-01-01":]
    cp = st.carry_premium(sub_tr, "REM")
    d = st.decompose(sub_tr, sub_px, "REM")
    print(f"  ex-GFC (2010-01+): alpha {cp['alpha_bps_mo']:+.2f} bps/mo t = {cp['t_alpha']:+.2f} | "
          f"dividend {d['div_ann_pct']:+.2f}%/yr t = {d['div_hac_t']:+.1f} | "
          f"TR CAGR {d['tr_cagr_pct']:+.2f}%/yr, price-only {d['px_cagr_pct']:+.2f}%/yr")

    print("\n# Crisis autopsies — peak-to-trough TOTAL-return drawdown inside each window (daily tape)")
    ct = st.crisis_table(tr, NAMES + ["SPY", "IEF"])
    for label, row in ct.iterrows():
        cells = "  ".join(f"{tk} {row[tk]:+.1f}%" for tk in NAMES + ["SPY", "IEF"])
        print(f"  {label:<20}: {cells}")
    print("  (AGNC listed 2008-05, so its GFC cell covers only the tail of the crisis)")

    print("\n# Costs — buy-and-hold, stated for the record")
    print("  Two one-way ETF trades over 19.1 years at 5 bps one-way = ~10 bps total, ~0.5 bps/yr")
    print("  amortised; REM's 0.48%/yr expense ratio is already inside the tape. The mirage is not")
    print("  costs — the vehicle's own total return sits below T-bills.")
else:
    print("(no _cache/mrc_tr.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the carry detector (dividend-component HAC t) and the alpha test (NW regression) must")
print("  recover PLANTED values and must NOT manufacture significance under the null.")
for carry, alpha in [(0.0, 0.0), (0.010, -0.0070)]:
    ttr, tpx = data.synthetic_world(carry=carry, alpha=alpha, seed=611)
    smtr, smpx = st.aligned_monthly(ttr, tpx, ["MREIT", "IEF", "SPY", "BIL"])
    sd = st.decompose(smtr, smpx, "MREIT")
    scp = st.carry_premium(smtr, "MREIT")
    print(f"  planted carry={carry*1e4:+7.1f} bps/mo, alpha={alpha*1e4:+7.1f} bps/mo: "
          f"div {sd['div_bps_mo']:+.2f} bps/mo (t {sd['div_hac_t']:+.2f}) | "
          f"alpha {scp['alpha_bps_mo']:+.2f} bps/mo (t {scp['t_alpha']:+.2f}) | "
          f"b_IEF {scp['beta_IEF']:.2f} b_SPY {scp['beta_SPY']:.2f} (planted 2.50 / 0.55)")
