"""Reproducible headline run for Study 599 — Tax-Loss Harvesting.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY/IVV total-return tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from tax_loss_harvesting import data, strategy as st

print("# Tax-Loss Harvesting — SPY daily total-return, lot-level TLH engine (yfinance)")
if data.have_real():
    from quantlab import repro

    spy, ivv = data.load_real()
    print(repro.data_stamp("SPY total-return close", spy.to_frame("SPY"), asof=data.AS_OF))
    tc = data.twin_check(spy, ivv)
    print(f"twin check (SPY vs IVV daily returns, {tc['n_days']:,} common days): "
          f"corr = {tc['corr']:.4f}, mean return gap = {tc['gap_ann_bps']:+.2f} bps/yr, "
          f"tracking-diff vol = {tc['te_ann_bps']:.0f} bps/yr")

    print("\n# Full-period lump-sum — the basis lock-up punchline")
    r = st.after_tax_alpha(spy, st_rate=0.35, lt_rate=0.15)
    print(f"  $100k lump sum 1993-02 -> 2026-06 ({r['years']:.1f}y) at 35/15: "
          f"harvests = {r['h']['n_harvests']}, after-tax alpha = {r['alpha_bps']:+.1f} bps/yr "
          f"(the 1993 lot never trades below basis again after month one)")

    print("\n# Full-period contributor ($10k + $1k/month, 1993-02 -> 2026-06) — the rate grid")
    grid = [(0.35, 0.15, False, "35/15 (top bracket, ST offsets)"),
            (0.24, 0.15, False, "24/15 (middle bracket)"),
            (0.15, 0.15, False, "15/15 (deferral only, no rate arb)"),
            (0.35, 0.15, True,  "35/15 + step-up at death"),
            (0.00, 0.00, False, "0/0   (0% bracket - MYTH CHECK)"),
            (0.00, 0.00, True,  "0/0   + step-up (low-bracket donor)")]
    for stt, ltt, su, lab in grid:
        rr = st.after_tax_alpha(spy, initial=10_000, monthly_contrib=1_000,
                                st_rate=stt, lt_rate=ltt, step_up=su)
        print(f"  {lab:<36s} after-tax alpha {rr['alpha_bps']:+6.1f} bps/yr | "
              f"terminal after-tax uplift {rr['uplift_pct']:+6.2f}% | "
              f"harvests {rr['h']['n_harvests']}")

    rc = st.after_tax_alpha(spy, initial=10_000, monthly_contrib=1_000,
                            st_rate=0.35, lt_rate=0.15)
    h = rc["h"]
    st_share = 100.0 * h["losses_st"] / (h["losses_st"] + h["losses_lt"])
    print(f"  35/15 detail: harvested losses ${h['losses_st'] + h['losses_lt']:,.0f} "
          f"({st_share:.1f}% short-term), tax savings reinvested ${h['tax_saved']:,.0f}, "
          f"terminal tax TLH ${h['terminal_tax']:,.0f} vs B&H ${rc['b']['terminal_tax']:,.0f}")

    print("\n# Harvest yield by calendar year (contributor, % of avg NAV) — path dependence")
    hy = st.harvest_yield_by_year(h)
    top = hy.sort_values(ascending=False).head(6)
    print("  richest years: " + ", ".join(f"{int(y)}: {v:.2f}%" for y, v in top.items()))
    bear = hy[hy.index.isin(st.BEAR_YEARS)].to_numpy()
    calm = hy[~hy.index.isin(st.BEAR_YEARS)].to_numpy()
    tw, dof = st.welch_t(bear, calm)
    print(f"  bear years {st.BEAR_YEARS}: mean {bear.mean():.2f}%/yr (n={len(bear)}) vs "
          f"calm {calm.mean():.2f}%/yr (n={len(calm)})  Welch t = {tw:+.2f} (dof {dof:.1f})")

    print("\n# 10-year lump-sum cohorts (quarterly starts) — the Signal-axis test at 35/15")
    co = st.cohort_alphas(spy, horizon_years=10, st_rate=0.35, lt_rate=0.15)
    a = co["alpha_bps"].to_numpy()
    lag = 40  # 10y horizon / quarterly starts = 39 overlapping quarters
    print(f"  cohorts = {len(co)} ({co.index[0].date()} -> {co.index[-1].date()} starts)")
    print(f"  after-tax alpha: mean {a.mean():+.1f} bps/yr | median {np.median(a):+.1f} | "
          f"sd {a.std(ddof=1):.1f} | min {a.min():+.1f} | max {a.max():+.1f}")
    print(f"  HAC/NW t (lag {lag}) = {st.nw_tstat(a, lag):+.2f}   share > 0: "
          f"{100 * (a > 0).mean():.1f}%   zero-harvest cohorts: {(co['n_harvests'] == 0).sum()} "
          f"of {len(co)}")
    best, worst = co["alpha_bps"].idxmax(), co["alpha_bps"].idxmin()
    print(f"  best cohort start {best.date()} ({co.loc[best, 'alpha_bps']:+.1f} bps/yr); "
          f"worst {worst.date()} ({co.loc[worst, 'alpha_bps']:+.1f} bps/yr)")
    dec_means = co["alpha_bps"].groupby(co.index.year // 10 * 10).mean()
    print("  by start decade: " + ", ".join(f"{int(d)}s: {v:+.1f}" for d, v in dec_means.items()))

    print("\n# Cohort rate grid — the alpha is your tax bracket, not the market")
    for stt, ltt, lab in [(0.35, 0.15, "35/15"), (0.24, 0.15, "24/15"), (0.15, 0.15, "15/15")]:
        cg = st.cohort_alphas(spy, st_rate=stt, lt_rate=ltt)
        ag = cg["alpha_bps"].to_numpy()
        print(f"  {lab}: mean {ag.mean():+6.1f} bps/yr | median {np.median(ag):+6.1f} | "
              f"HAC t (lag 40) = {st.nw_tstat(ag, 40):+.2f}")

    print("\n# Decay — average harvested loss (% of initial stake) by year since inception")
    dec = st.decay_profile(spy, horizon_years=10, st_rate=0.35, lt_rate=0.15)
    print("  " + ", ".join(f"y{int(k) + 1}: {v:.2f}%" for k, v in dec.items()))
    print(f"  year-1 harvest = {dec.iloc[0]:.2f}% of stake; years 7-10 average = "
          f"{dec.iloc[6:].mean():.2f}% — the basis locks up")

    print("\n# Threshold robustness (35/15, 10y cohorts)")
    for th in (0.005, 0.01, 0.05):
        ct = (co if abs(th - 0.01) < 1e-12
              else st.cohort_alphas(spy, threshold=th, st_rate=0.35, lt_rate=0.15))
        at = ct["alpha_bps"].to_numpy()
        print(f"  harvest when loss > {100 * th:.1f}%: mean {at.mean():+6.1f} bps/yr | "
              f"HAC t = {st.nw_tstat(at, 40):+.2f} | harvests/cohort {ct['n_harvests'].mean():.1f}")
else:
    print("(no _cache/tlh_spy.csv — run data.fetch() once to build the cache)")

print("\n# Synthetic positive control — deterministic GBM, 20 seeds per cell, no network")
print("  TLH alpha must RISE with the planted volatility knob and sit at ~zero (-costs)")
print("  when tax rates are 0/0, whatever the path:")
ctl = st.synthetic_control(sigmas=(0.02, 0.18, 0.35), n_seeds=20)
for sig, row in ctl.iterrows():
    print(f"  sigma = {sig:.2f}: alpha(35/15) = {row['alpha_bps_35_15']:+7.2f} bps/yr "
          f"(sd {row['alpha_sd']:.1f} across seeds) | alpha(rates 0/0) = "
          f"{row['alpha_bps_rates0']:+.2f} bps/yr")
print("  (machinery proof only — never cited in support of the Signal stamp)")
