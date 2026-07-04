"""Reproducible headline run for Study 626 — Unemployment-Trend-Timing (GTT).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached tape under
``_cache/`` (BLS unemployment + ^GSPC + ^IRX + Shiller dividend yield) and always
runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd

from unemployment_trend_timing import data, strategy as st

COST = 5.0   # headline one-way cost, bps x NAV per switch

print("# Growth-Trend Timing — 200d SMA gated by rising unemployment (LNS14000000)")
if data.have_real():
    from quantlab import repro

    df = data.load_monthly()
    print(repro.data_stamp("gtt_monthly", df, cols=["eq_ret", "cash_ret", "unrate"],
                           asof=data.AS_OF))
    yrs = len(df) / 12.0
    print(f"tape            : {df.index.min().date()} -> {df.index.max().date()} "
          f"({len(df)} months, {yrs:.1f} years), S&P 500 total return "
          f"(^GSPC price + Shiller dividend yield), cash = 3m T-bill "
          f"(^IRX; 1948-59 ERP-2011 Table B-73 annual averages)")
    print(f"signals         : month-end close vs 200d SMA; unemployment (1-month "
          f"reporting lag) vs its 12m SMA; ONE execution lag (signal month m -> "
          f"position month m+1); cost {COST:.0f} bps one-way x NAV per switch")

    r = st.run_race(df, cost_bps=COST)
    print("\n# The race (net of costs, total-return, excess-vs-excess Sharpe)")
    for k, lab in (("bh", "Buy & hold"), ("faber", "Faber 200d SMA"), ("gtt", "GTT (filtered)")):
        p = r[k]
        lev = (1.0 + r["series"][k]).cumprod().iloc[-1]
        print(f"  {lab:15s}: CAGR {p['cagr_pct']:6.2f}%  vol {p['vol_pct']:5.2f}%  "
              f"Sharpe {p['sharpe']:5.3f}  maxDD {p['maxdd_pct']:7.2f}%  "
              f"$1 -> ${lev:,.0f}")

    print("\n# Whipsaws — the claim's mechanical half")
    for k, lab in (("faber_spells", "Faber"), ("gtt_spells", "GTT")):
        s = r[k]
        print(f"  {lab:5s}: {s['switches']:3d} switches, {s['n_spells']:2d} risk-off spells "
              f"({s['whipsaws']:2d} whipsaws / {s['saves']:2d} saves), "
              f"{s['pct_out']:.1f}% of months out of the market")
    fs, gs = r["faber_spells"], r["gtt_spells"]
    print(f"  switches cut {100*(1-gs['switches']/fs['switches']):.0f}% | whipsaw spells cut "
          f"{100*(1-gs['whipsaws']/fs['whipsaws']):.0f}% (whipsaw = a risk-off spell where "
          f"being out LOST money vs staying in)")

    print("\n# Active-return inference (Newey-West HAC t, monthly)")
    for k, lab in (("t_gtt_vs_faber", "GTT - Faber"), ("t_gtt_vs_bh", "GTT - B&H"),
                   ("t_faber_vs_bh", "Faber - B&H")):
        t = r[k]
        print(f"  {lab:12s}: {t['mean_bps']:+7.2f} bps/mo  HAC t = {t['tstat']:+5.2f} "
              f"(lags {t['lags']}, n {t['n']})")

    # HAC lag sensitivity: the whipsaw reversals make active returns NEGATIVELY
    # autocorrelated, so here the HAC correction legitimately SHRINKS the SE.
    af = r["series"]["gtt"] - r["series"]["faber"]
    acf1 = float(af.autocorr(1))
    tt = {L: st.hac_tstat(af, lags=L)["tstat"] for L in (0, 1, 6, 12)}
    print(f"  lag sensitivity : plain (lag-0) t = {tt[0]:+.2f}; HAC t = "
          f"{tt[1]:+.2f} / {tt[6]:+.2f} / {tt[12]:+.2f} at 1/6/12 lags "
          f"(acf1 = {acf1:+.3f} — negative, so HAC shrinks the SE)")

    print("\n# Cost sweep — GTT-minus-Faber active return (GTT trades LESS, so t rises)")
    for cb in (0.0, 5.0, 10.0, 25.0):
        t = st.run_race(df, cost_bps=cb)["t_gtt_vs_faber"]
        print(f"  {cb:4.0f} bps one-way: {t['mean_bps']:+6.2f} bps/mo  HAC t = {t['tstat']:+5.2f}")

    print("\n# Exposure vs skill — the honest decomposition + rotation placebo")
    dec = st.exposure_decomposition(df, cost_bps=COST)
    print(f"  GTT is long in {dec['share_extra_pct']:.1f}% of months where Faber is flat;")
    print(f"  edge {dec['edge_bps']:+.2f} bps/mo = mechanical exposure {dec['mechanical_bps']:+.2f} "
          f"+ pure timing {dec['timing_bps']:+.2f}")
    pl = st.rotation_placebo(df, cost_bps=COST)
    print(f"  placebo (all {pl['n_rot']} circular rotations of the filter, deterministic): "
          f"obs {pl['obs_bps']:+.2f} vs rotation mean {pl['placebo_mean_bps']:+.2f} bps/mo, "
          f"p = {pl['p_value']:.4f}")

    print("\n# Robustness")
    for w in (6, 12, 24):
        rw = st.run_race(data.load_monthly(u_window=w), cost_bps=COST)
        t = rw["t_gtt_vs_faber"]
        print(f"  unemployment SMA window {w:2d}m: GTT-Faber {t['mean_bps']:+6.2f} bps/mo  "
              f"HAC t = {t['tstat']:+5.2f}  (GTT Sharpe {rw['gtt']['sharpe']:.3f})")
    rp = st.run_race(df, cost_bps=COST, ret_col="eq_ret_px")
    t = rp["t_gtt_vs_faber"]
    print(f"  price-only tape (no dividends): GTT-Faber {t['mean_bps']:+6.2f} bps/mo  "
          f"HAC t = {t['tstat']:+5.2f}")

    print("\n# Sub-period — before/after the GTT publication (Feb 2016)")
    sp = st.subperiod(df, cost_bps=COST)
    for lab, rr in (("pre-2016 ", sp["pre"]), ("post-2016", sp["post"])):
        t = rr["t_gtt_vs_faber"]
        print(f"  {lab}: GTT-Faber {t['mean_bps']:+6.2f} bps/mo  HAC t = {t['tstat']:+5.2f}  "
              f"(GTT Sharpe {rr['gtt']['sharpe']:.3f} vs Faber {rr['faber']['sharpe']:.3f}, "
              f"n {t['n']})")

    print("\n# The six bears — cumulative net return through each episode")
    episodes = {"1974 bear": ("1973-01-31", "1975-06-30"),
                "1987 crash": ("1987-08-31", "1988-03-31"),
                "2000-02 bust": ("2000-03-31", "2003-06-30"),
                "2008 GFC": ("2007-10-31", "2009-12-31"),
                "2020 COVID": ("2020-01-31", "2020-12-31"),
                "2022 bear": ("2022-01-31", "2023-06-30")}
    for name, (a, b) in episodes.items():
        sl = slice(pd.Timestamp(a), pd.Timestamp(b))
        gw = r["series"]["gtt_w"].loc[sl]
        fw = r["series"]["faber_w"].loc[sl]
        cum = {k: (1.0 + r["series"][k].loc[sl]).prod() - 1.0 for k in ("bh", "faber", "gtt")}
        print(f"  {name:13s}: B&H {cum['bh']*100:+7.1f}%  Faber {cum['faber']*100:+7.1f}% "
              f"(out {int((1-fw).sum()):2d}m)  GTT {cum['gtt']*100:+7.1f}% "
              f"(out {int((1-gw).sum()):2d}m)")
else:
    print("(_cache missing — run `python -c \"from unemployment_trend_timing import data; "
          "data.fetch_all()\"` once from the study root to build it)")

print("\n# Synthetic control — deterministic, no network")
print("  the exposure-matched rotation placebo must (a) light up on a PLANTED recession")
print("  drag the filter can see, (b) stay quiet when the filter is noise, and (c) refuse")
print("  the raw-t bait when the filter only adds exposure (no drag planted).")
for gap, linked, tag in ((0.02, True, "planted drag, filter informative"),
                         (0.02, False, "null: filter is pure noise     "),
                         (0.0, True, "exposure-only bait (no drag)   ")):
    sw = data.synthetic_world(gap=gap, linked=linked, seed=626)
    rs = st.run_race(sw, cost_bps=COST)
    t = rs["t_gtt_vs_faber"]
    pl = st.rotation_placebo(sw, cost_bps=COST)
    print(f"  {tag}: raw t = {t['tstat']:+5.2f}  placebo p = {pl['p_value']:.3f}  "
          f"(obs {pl['obs_bps']:+6.2f} vs rot mean {pl['placebo_mean_bps']:+6.2f} bps/mo)")
