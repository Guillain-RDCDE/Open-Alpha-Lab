"""Reproducible headline run for Study 601 — Factor ETFs, Live Test.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the study's cached yfinance tape under
``_cache/`` (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from factor_etf_live_test import data, strategy as st

print("# Factor ETFs — Live Test: USMV / MTUM / VLUE / QUAL vs SPY since inception")
print("# (Dedup guard: 330-low-volatility-anomaly & 242-quality-minus-junk grade the ACADEMIC")
print("#  cross-sections; the unit under test here is the LIVE PRODUCT, net of its own fee.)")

if not data.have_real():
    print("(missing cache: _cache/fel_prices.csv — run data.fetch_tape() once)")
else:
    try:
        from quantlab import repro
    except Exception:
        repro = None

    tape = data.load_tape()
    mret = data.monthly_total_returns(tape[[c for c in tape.columns if c != "^IRX"]])
    rf = data.monthly_rf(tape["^IRX"])
    spy = mret["SPY"].dropna()
    spy_ex = (spy - rf).dropna()
    wml = data.sector_momentum_spread(mret[data.SECTORS].dropna())
    vmg = data.value_spread(mret)
    qmb = data.quality_spread(mret)
    SPREADS = {"USMV": None, "MTUM": wml, "VLUE": vmg, "QUAL": qmb}
    SPREAD_NAME = {"MTUM": "sector 12-1 WML", "VLUE": "IWD-IWF value spread",
                   "QUAL": "SPHQ-SPY quality spread"}

    print("\n# Data stamp")
    print(f"tape          : {tape.shape[0]} days x {tape.shape[1]} tickers, "
          f"{tape.index.min().date()} -> {tape.index.max().date()} "
          f"(yfinance auto-adjusted closes = TOTAL RETURN, net of each fund's 0.15%/yr fee)")
    print(f"as-of         : monthly stats sliced to {data.AS_OF} (last complete month; "
          f"the partial current month is dropped)")
    if repro is not None:
        panel = pd.concat([mret[["SPY"] + data.FUNDS], rf.rename("rf")], axis=1)
        print(f"fingerprint   : monthly_panel={repro.fingerprint(panel)}")
    for tk in data.FUNDS:
        s = mret[tk].dropna()
        print(f"  {tk}: {s.index.min().date()} -> {s.index.max().date()}  "
              f"({len(s)} complete months)")

    print("\n# Per-fund audit (monthly total returns; excess = minus ^IRX rf, lagged 1m)")
    for tk in data.FUNDS:
        r = mret[tk].dropna()
        idx = r.index.intersection(spy.index).intersection(rf.dropna().index)
        r, sp, f = r.loc[idx], spy.loc[idx], rf.loc[idx]
        cp = st.capm(r - f, sp - f, lags=6)
        vr = st.vol_ratio_ci(r, sp, seed=601)
        cap = st.up_down_capture(r, sp)
        sf, ss = st.ann_stats(r, f), st.ann_stats(sp, f)
        act = r - sp
        t_act = st.nw_tstat(act, lags=6)
        print(f"\n  == {tk}  ({idx.min().date()} -> {idx.max().date()}, {len(idx)} months)")
        print(f"  beta {cp['beta']:.3f} (NW se {cp['se_beta']:.3f}; t vs 1 = "
              f"{cp['t_beta_vs1']:+.2f})   R^2 {cp['r2']*100:.1f}%")
        print(f"  CAPM alpha {cp['alpha_ann']*100:+.2f}%/yr   NW t = {cp['t_alpha']:+.2f}")
        print(f"  realized vol ratio {vr['obs']:.3f}  (95% CI [{vr['lo']:.3f}, {vr['hi']:.3f}])"
              f"   fund vol {sf['vol']*100:.1f}% vs SPY {ss['vol']*100:.1f}%")
        print(f"  capture: up {cap['up']*100:.1f}%  down {cap['down']*100:.1f}%  "
              f"(n_up={cap['n_up']}, n_down={cap['n_down']})")
        print(f"  CAGR {sf['cagr']*100:.2f}% vs SPY {ss['cagr']*100:.2f}%   "
              f"Sharpe {sf['sharpe']:.3f} vs {ss['sharpe']:.3f}   "
              f"maxDD {sf['maxdd']*100:.1f}% vs {ss['maxdd']*100:.1f}%")
        print(f"  growth of $1: {sf['wealth']:.2f} vs SPY {ss['wealth']:.2f}   "
              f"active mean {act.mean()*1e4:+.1f} bps/mo  NW t = {t_act:+.2f}")

    print("\n# Axis 1 — EXPOSURE delivery (the stated-style test)")
    print("  USMV's promise is mechanical (a lower-risk profile); the other three must load on")
    print("  their style's realized long-short spread (two-factor NW regression + sign split).")
    r = mret["USMV"].dropna()
    idx = r.index.intersection(spy.index).intersection(rf.dropna().index)
    cp = st.capm(r.loc[idx] - rf.loc[idx], spy.loc[idx] - rf.loc[idx], lags=6)
    vr = st.vol_ratio_ci(r.loc[idx], spy.loc[idx], seed=601)
    print(f"  USMV: beta {cp['beta']:.3f}, t(beta<1) = {cp['t_beta_vs1']:+.2f}   "
          f"vol ratio {vr['obs']:.3f} CI [{vr['lo']:.3f}, {vr['hi']:.3f}] "
          f"-> {100*(1-vr['obs']):.0f}% less vol than SPY")
    for tk in ("MTUM", "VLUE", "QUAL"):
        r = mret[tk].dropna()
        idx = r.index.intersection(spy.index).intersection(rf.dropna().index)
        sl = st.style_loading(r.loc[idx] - rf.loc[idx], spy.loc[idx] - rf.loc[idx],
                              SPREADS[tk], lags=6)
        sp_split = st.spread_sign_split((r - spy).loc[idx], SPREADS[tk], seed=601)
        print(f"  {tk}: loading on {SPREAD_NAME[tk]} = {sl['loading']:+.3f}  "
              f"NW t = {sl['t_loading']:+.2f}  (2-factor R^2 {sl['r2']*100:.1f}%, n={sl['n']})")
        print(f"        spread-sign split: active {sp_split['mean_pos_bps']:+.1f} bps/mo in "
              f"factor-won months vs {sp_split['mean_neg_bps']:+.1f} in factor-lost  "
              f"(diff {sp_split['diff_bps']:+.1f} bps, Welch t {sp_split['welch_t']:+.2f}, "
              f"placebo p {sp_split['p_placebo']:.4f})")

    print("\n# Axis 2 — ALPHA delivery (CAPM alpha NW t; lags 3/6/12 robustness)")
    for tk in data.FUNDS:
        r = mret[tk].dropna()
        idx = r.index.intersection(spy.index).intersection(rf.dropna().index)
        ts = [st.capm(r.loc[idx] - rf.loc[idx], spy.loc[idx] - rf.loc[idx], lags=lg)
              for lg in (3, 6, 12)]
        print(f"  {tk}: alpha {ts[1]['alpha_ann']*100:+.2f}%/yr   NW t = "
              + "  ".join(f"{t['t_alpha']:+.2f} (lags {lg})" for t, lg in zip(ts, (3, 6, 12))))

    print("\n# Axis 3 — did ANY of them beat SPY outright? (same-window races, net of fees)")
    for tk in data.FUNDS:
        r = mret[tk].dropna()
        idx = r.index.intersection(spy.index).intersection(rf.dropna().index)
        sf = st.ann_stats(r.loc[idx], rf.loc[idx])
        ss = st.ann_stats(spy.loc[idx], rf.loc[idx])
        act = (r - spy).loc[idx]
        print(f"  {tk}: CAGR {sf['cagr']*100:.2f}% vs SPY {ss['cagr']*100:.2f}%  "
          f"(gap {100*(sf['cagr']-ss['cagr']):+.2f} pp/yr)   Sharpe {sf['sharpe']:.3f} vs "
          f"{ss['sharpe']:.3f}   active NW t = {st.nw_tstat(act, lags=6):+.2f}")

    print("\n# Style-proxy sanity (the regressors themselves, full common window)")
    print(f"  sector WML : {wml.index.min().date()} -> {wml.index.max().date()}  "
          f"mean {wml.mean()*1e4:+.1f} bps/mo  vol {wml.std()*np.sqrt(12)*100:.1f}%/yr  "
          f"NW t(mean) = {st.nw_tstat(wml):+.2f}")
    print(f"  IWD-IWF    : {vmg.index.min().date()} -> {vmg.index.max().date()}  "
          f"mean {vmg.mean()*1e4:+.1f} bps/mo  NW t = {st.nw_tstat(vmg):+.2f}")
    print(f"  SPHQ-SPY   : {qmb.index.min().date()} -> {qmb.index.max().date()}  "
          f"mean {qmb.mean()*1e4:+.1f} bps/mo  NW t = {st.nw_tstat(qmb):+.2f}  "
          f"(SPHQ switched to the S&P 500 Quality index in 2016 — proxy caveat)")

print("\n# Synthetic control — machinery proof only, never market evidence")
print("  the CAPM-alpha NW t, style-loading NW t and beta-below-1 t must stay quiet on the")
print("  null (beta=1, loading=0, alpha=0) and recover the planted parameters.")
for label, kw in [("null  (b=1.0, s=0.0, a=+0%)",
                   dict(beta=1.0, loading=0.0, alpha_ann=0.0)),
                  ("planted (b=0.7, s=0.5, a=+3%)",
                   dict(beta=0.7, loading=0.5, alpha_ann=0.03))]:
    w = data.synthetic_world(n_months=168, seed=601, **kw)
    sl = st.style_loading(w["fund"], w["mkt"], w["spread"], lags=6)
    cp = st.capm(w["fund"], w["mkt"], lags=6)
    print(f"  {label}: beta {cp['beta']:.3f} (t vs 1 = {cp['t_beta_vs1']:+.2f})  "
          f"loading {sl['loading']:+.3f} (t {sl['t_loading']:+.2f})  "
          f"alpha {sl['alpha_ann']*100:+.2f}%/yr (t {sl['t_alpha']:+.2f})")
