"""Reproducible headline run for Study 593 — HFEA (UPRO/TMF 55/45).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached yfinance tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from hfea_leveraged_6040 import data as d, strategy as st

print("# HFEA (UPRO/TMF 55/45) — the leveraged-portfolio allocation claim (yfinance)")
if d.have_real():
    from quantlab import repro

    px = d.load_real()
    tape = d.build_tape(px, asof=d.AS_OF)
    print(repro.data_stamp("hfea daily tape", tape, cols=["spy", "tlt", "rf"],
                           asof=d.AS_OF))

    print("\n# Synthesis validation — the study-100 recipe, per leg")
    print("  daily identity r_3x = 3*r_index - 2*r_bill - fee/252; all-in fee calibrated on")
    print("  the real-fund overlap (terminal-NAV match); corr must clear 0.99.")
    cal_s = d.calibrate_fee(tape["spy"], tape["rf"], tape["upro"])
    cal_b = d.calibrate_fee(tape["tlt"], tape["rf"], tape["tmf"])
    for nm, c in [("UPRO (3x SPY)", cal_s), ("TMF  (3x TLT)", cal_b)]:
        print(f"  {nm}: fee {c['fee_ann']*100:.2f}%/yr  daily-return corr {c['corr']:.4f}  "
              f"ann tracking err {c['te_ann']*100:.2f}%  NAV ratio {c['nav_ratio']:.4f}  "
              f"({c['start']} -> {c['end']}, {c['n_days']:,} days)")

    legs = d.spliced_legs(tape, cal_s["fee_ann"], cal_b["fee_ann"])
    n_syn = int((legs["src"] == "synth").sum())
    n_real = int((legs["src"] == "real").sum())
    print(f"  spliced legs: {legs.index.min().date()} -> {legs.index.max().date()}  "
          f"({n_syn:,} synthetic days pre-launch, {n_real:,} real-fund days)")

    hfea = st.rebalanced(legs["s3x"], legs["b3x"], st.HFEA_W)
    sixty = st.rebalanced(legs["spy"], legs["tlt"], st.SIXTY_FORTY_W)
    rf = legs["rf"]
    print(f"  quarterly resets: {hfea['n_rebalances']}  avg NAV traded per reset "
          f"{hfea['avg_turnover']*100:.1f}%")

    print("\n# Full-period race (2002-07 -> 2026-06, gross; excess-vs-excess Sharpe)")
    rows = [("HFEA 55/45", hfea["ret"]), ("SPY", legs["spy"]),
            ("60/40 SPY/TLT", sixty["ret"]), ("UPRO leg (3x SPY)", legs["s3x"]),
            ("TMF leg (3x TLT)", legs["b3x"])]
    for nm, r in rows:
        p = st.perf(r, rf)
        print(f"  {nm:18s} CAGR {p['cagr_pct']:+7.2f}%  vol {p['vol_ann_pct']:5.1f}%  "
              f"maxDD {p['max_dd_pct']:6.1f}%  Sharpe {p['sharpe']:5.2f}  "
              f"({p['years']:.1f}y, {p['n_months']} months)")
    wm = {nm: float((1.0 + r).prod()) for nm, r in rows[:3]}
    print("  terminal wealth multiple of $1: "
          + "  ".join(f"{nm} x{v:.1f}" for nm, v in wm.items()))

    print("\n# 'Compounds faster' — HAC t on the mean monthly LOG-return gap")
    r_spy = st.race(hfea["ret"], legs["spy"], rf)
    r_6040 = st.race(hfea["ret"], sixty["ret"], rf)
    print(f"  HFEA vs SPY  : gap {r_spy['dlog_ann_pct']:+.2f}%/yr log-CAGR   "
          f"HAC t = {r_spy['t_dlog_hac']:+.2f}  (n={r_spy['n_months']}, lags={r_spy['lags']})")
    print(f"  HFEA vs 60/40: gap {r_6040['dlog_ann_pct']:+.2f}%/yr log-CAGR   "
          f"HAC t = {r_6040['t_dlog_hac']:+.2f}  (n={r_6040['n_months']}, lags={r_6040['lags']})")

    print("\n# Regime split — the corr flip is the claim's own mechanism, so the split is")
    print("# ex-ante justified (2022-01 = the Fed-hiking corr-flip year), not return-snooped")
    mp, mb = st.monthly(hfea["ret"]), st.monthly(legs["spy"])
    dlog = np.log1p(mp) - np.log1p(mb)
    pre = dlog[dlog.index <= pd.Period("2021-12", "M")].to_numpy()
    post = dlog[dlog.index >= pd.Period("2022-01", "M")].to_numpy()
    h_pre, h_post = st.hac_mean_t(pre), st.hac_mean_t(post)
    wt = ((pre.mean() - post.mean())
          / np.sqrt(pre.var(ddof=1) / len(pre) + post.var(ddof=1) / len(post)))
    print(f"  2002-08..2021-12: gap {h_pre['mean']*12*100:+.2f}%/yr  HAC t = {h_pre['t']:+.2f}  (n={h_pre['n']})")
    print(f"  2022-01..2026-06: gap {h_post['mean']*12*100:+.2f}%/yr  HAC t = {h_post['t']:+.2f}  (n={h_post['n']})")
    print(f"  Welch t of the regime DIFFERENCE = {wt:+.2f}")
    d64 = np.log1p(mp) - np.log1p(st.monthly(sixty["ret"]))
    h64 = st.hac_mean_t(d64[d64.index <= pd.Period("2021-12", "M")].to_numpy())
    print(f"  (vs 60/40, pre-2022: gap {h64['mean']*12*100:+.2f}%/yr  HAC t = {h64['t']:+.2f})")
    for nm, sub in [("2002-08..2021-12", slice(None, "2021-12-31")),
                    ("2022-01..2026-06", slice("2022-01-01", None))]:
        ph = st.perf(hfea["ret"].loc[sub], rf.loc[sub])
        ps = st.perf(legs["spy"].loc[sub], rf.loc[sub])
        print(f"  {nm}: HFEA CAGR {ph['cagr_pct']:+.2f}% (Sharpe {ph['sharpe']:.2f})  "
              f"vs SPY {ps['cagr_pct']:+.2f}% (Sharpe {ps['sharpe']:.2f})")

    print("\n# The 2022 autopsy — the stock-bond correlation flip hits BOTH legs at once")
    for nm, r in [("UPRO leg", legs["s3x"]), ("TMF leg", legs["b3x"]),
                  ("HFEA 55/45", hfea["ret"]), ("SPY", legs["spy"]),
                  ("60/40", sixty["ret"]), ("TLT (1x)", legs["tlt"])]:
        print(f"  2022 calendar year {nm:10s}: {st.year_return(r, 2022):+7.1f}%")
    print(f"  monthly SPY/TLT correlation: 2002-2021 {st.stock_bond_corr(legs, end='2021-12'):+.2f}"
          f"   2022 {st.stock_bond_corr(legs, start='2022-01', end='2022-12'):+.2f}"
          f"   2024-2026 {st.stock_bond_corr(legs, start='2024-01'):+.2f}"
          f"   full {st.stock_bond_corr(legs):+.2f}")
    dd = st.drawdown_state(hfea["ret"])
    print(f"  HFEA drawdown: peak {dd['peak']} -> trough {dd['trough']}  depth {dd['depth_pct']:.1f}%"
          f"   still {dd['now_vs_peak_pct']:+.1f}% vs peak at the as-of")
    print(f"  contrast years: 2008 HFEA {st.year_return(hfea['ret'], 2008):+.1f}% vs SPY "
          f"{st.year_return(legs['spy'], 2008):+.1f}%   |   2020 HFEA "
          f"{st.year_return(hfea['ret'], 2020):+.1f}% vs SPY {st.year_return(legs['spy'], 2020):+.1f}%")

    print("\n# Costs & robustness")
    for cb in (2.0, 5.0, 10.0):
        h = st.rebalanced(legs["s3x"], legs["b3x"], st.HFEA_W, cost_bps=cb)
        p = st.perf(h["ret"], rf)
        rr = st.race(h["ret"], legs["spy"], rf)
        print(f"  one-way cost {cb:4.1f} bps: net CAGR {p['cagr_pct']:+.2f}%  "
              f"gap vs SPY {rr['dlog_ann_pct']:+.2f}%/yr  t = {rr['t_dlog_hac']:+.2f}  "
              f"(annualised drag {h['ann_cost_bps']:.1f} bps/yr)")
    legs_hi = d.spliced_legs(tape, 0.03, 0.03)
    h_hi = st.rebalanced(legs_hi["s3x"], legs_hi["b3x"], st.HFEA_W)
    p_hi = st.perf(h_hi["ret"], rf)
    r_hi = st.race(h_hi["ret"], legs_hi["spy"], rf)
    print(f"  pessimistic 3.0%/yr all-in synthesis fee pre-launch: CAGR {p_hi['cagr_pct']:+.2f}%  "
          f"gap {r_hi['dlog_ann_pct']:+.2f}%/yr  t = {r_hi['t_dlog_hac']:+.2f}")
    real = legs[legs["src"] == "real"]
    h_real = st.rebalanced(real["s3x"], real["b3x"], st.HFEA_W)
    p_r = st.perf(h_real["ret"], real["rf"])
    p_rs = st.perf(real["spy"], real["rf"])
    rr = st.race(h_real["ret"], real["spy"], real["rf"])
    print(f"  real-funds-only window ({real.index.min().date()} ->): HFEA CAGR {p_r['cagr_pct']:+.2f}% "
          f"(Sharpe {p_r['sharpe']:.2f}) vs SPY {p_rs['cagr_pct']:+.2f}% (Sharpe {p_rs['sharpe']:.2f})  "
          f"gap {rr['dlog_ann_pct']:+.2f}%/yr  t = {rr['t_dlog_hac']:+.2f}")
else:
    print("(no _cache/hfea_prices.csv — run data.fetch_prices() once to build the cache)")

print("\n# Synthetic control — plant / remove the diversification engine (20 seeds, 40y)")
print("  PLANTED (rho=-0.6, bond carry +6%): the levered 55/45 must out-compound 1x stock;")
print("  NULL (rho=+0.6, zero carry — the 2022 world made permanent): it must NOT.")
for nm, rho, carry in [("PLANTED", -0.6, 0.06), ("NULL   ", +0.6, 0.0)]:
    c = st.synthetic_check(rho=rho, bond_mu_exc=carry, n_seeds=20, n_years=40)
    print(f"  {nm} rho={rho:+.1f} carry={carry*100:.0f}%: mean gap {c['mean_gap_ann_pct']:+.2f}%/yr  "
          f"mean HAC t = {c['mean_t']:+.2f} (sd {c['sd_t']:.2f})  "
          f"share t>=2: {c['share_t_ge_2']*100:.0f}%  ({c['n_seeds']} seeds)")
