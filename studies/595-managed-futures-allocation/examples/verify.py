"""Reproducible headline run for Study 595 — Managed-Futures Sleeve.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the shared futures cache at the repo root
plus the study's cached ETF tape under ``_cache/`` (the real-tape numbers), and always runs
the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from managed_futures_allocation import data, strategy as st

NDRAWS = 5_000
SLEEVE = 0.15
FEE = 0.0085          # the live ETF wrapper (DBMF 0.85%/yr) charged against the replication
COST_BPS = 5.0        # one-way bps x traded futures notional (headline)

print("# Managed-Futures Sleeve — does a 15% trend sleeve improve a 60/40?")
print("# (Siblings 31-trade-winds / 518-tsmom graded the STANDALONE premium Weak;")
print("#  the unit under test here is the PORTFOLIO improvement.)")

if not data.have_real():
    print("(missing cache: shared trade_winds_futures.parquet and/or _cache/mf_etfs.csv —")
    print(" run data.fetch_etfs() once; the futures cache is built by study 31)")
else:
    try:
        from quantlab import repro
    except Exception:
        repro = None

    fut = data.load_futures()
    fm = data.drop_partial_last_month(data.monthly_from_daily_returns(fut), fut.index.max())
    etfs = data.load_etfs()
    tr = data.monthly_total_returns(etfs[["SPY", "VBMFX", "DBMF", "KMLM"]])
    rf = data.monthly_rf(etfs["^IRX"])
    p6040_full = st.portfolio_6040(tr["SPY"], tr["VBMFX"])

    book = st.tsmom_book(fm, lookback=12, cost_bps=COST_BPS)
    idx = p6040_full.index.intersection(book["net"].index).intersection(rf.dropna().index)
    p6040, rf_i = p6040_full.loc[idx], rf.loc[idx]
    e6040 = p6040 - rf_i
    mf_net = book["net"].loc[idx]                 # futures excess (net of turnover costs)
    mf_funded = rf_i + mf_net                     # cash-collateralised sleeve, gross of fee
    blend = st.blend_returns(p6040, mf_net, rf_i, SLEEVE, FEE)

    print("\n# Data stamp")
    print(f"futures cache : {fut.shape[0]} days x {fut.shape[1]} contracts, "
          f"{fut.index.min().date()} -> {fut.index.max().date()} "
          f"(partial {fut.index.max().strftime('%Y-%m')} dropped from monthly stats)")
    print(f"ETF tape      : SPY/VBMFX {etfs['SPY'].dropna().index.min().date()} -> "
          f"{etfs['SPY'].dropna().index.max().date()}; DBMF from "
          f"{etfs['DBMF'].dropna().index.min().date()}, KMLM from "
          f"{etfs['KMLM'].dropna().index.min().date()} (total-return closes; ^IRX = rf)")
    print(f"joint window  : {idx.min().date()} -> {idx.max().date()}  ({len(idx)} months)")
    if repro is not None:
        joint = pd.concat([p6040, mf_net, rf_i], axis=1)
        joint.columns = ["p6040", "mf_net", "rf"]
        print(f"fingerprint   : joint={repro.fingerprint(joint)}  "
              f"futures_monthly={repro.fingerprint(fm)}")

    print("\n# The sleeve itself (12-mo TSMOM, 18 futures, inverse-vol, 1-mo lag, "
          f"{COST_BPS:.0f} bps one-way; futures shorts post margin, no borrow fee)")
    s_mf = st.ann_stats(mf_funded, rf_i)
    print(f"  net ann excess Sharpe {st.sharpe_excess(mf_net):.3f}   NW t(mean) = "
          f"{st.nw_tstat(mf_net):+.2f}   vol {s_mf['vol']*100:.1f}%   maxDD {s_mf['maxdd']*100:.1f}%")
    print(f"  CAGR (funded, gross of ETF fee) {s_mf['cagr']*100:.2f}%/yr   "
          f"avg gross leverage {book['leverage']:.2f}x   avg turnover {book['turnover'].mean():.2f}x NAV/mo")
    print("  (standalone premium already graded Weak by siblings 31 & 518 — not re-litigated)")

    print("\n# Axis 1a — the 'near-zero correlation' half of the pitch")
    c = st.corr_ci(mf_net, e6040)
    print(f"  corr(MF, 60/40)  = {c['r']:+.3f}  95% CI [{c['lo']:+.3f}, {c['hi']:+.3f}]  (n={c['n']})")
    c2 = st.corr_ci(mf_net, tr["SPY"].loc[idx] - rf_i)
    print(f"  corr(MF, stocks) = {c2['r']:+.3f}  95% CI [{c2['lo']:+.3f}, {c2['hi']:+.3f}]")
    c3 = st.corr_ci(mf_net, tr["VBMFX"].loc[idx] - rf_i)
    print(f"  corr(MF, bonds)  = {c3['r']:+.3f}  95% CI [{c3['lo']:+.3f}, {c3['hi']:+.3f}]")

    print("\n# Axis 1b — the improvement (excess-vs-excess; sleeve pays the 0.85%/yr ETF fee)")
    a = st.ann_stats(p6040, rf_i)
    b = st.ann_stats(blend, rf_i)
    print(f"  60/40        : CAGR {a['cagr']*100:.2f}%  vol {a['vol']*100:.2f}%  "
          f"Sharpe {a['sharpe']:.3f}  maxDD {a['maxdd']*100:.1f}%  MAR {a['mar']:.3f}")
    print(f"  +15% sleeve  : CAGR {b['cagr']*100:.2f}%  vol {b['vol']*100:.2f}%  "
          f"Sharpe {b['sharpe']:.3f}  maxDD {b['maxdd']*100:.1f}%  MAR {b['mar']:.3f}")
    al = st.nw_alpha(mf_net, e6040, lags=6)
    print(f"  MF alpha vs 60/40: {al['alpha_ann']*100:+.2f}%/yr  NW t = {al['t_alpha']:+.2f}  "
          f"beta {al['beta']:+.3f}  R^2 {al['r2']*100:.1f}%   (mean-variance criterion)")
    bs = st.bootstrap_dsharpe(e6040, mf_net, SLEEVE, FEE, n_draws=NDRAWS, seed=595)
    print(f"  dSharpe (blend - 60/40) = {bs['obs']:+.3f}   block-bootstrap 95% CI "
          f"[{bs['lo']:+.3f}, {bs['hi']:+.3f}]   one-sided p = {bs['p_onesided']:.3f}  "
          f"({bs['n_draws']} draws, block {bs['block']})")

    print("\n# Sleeve-size robustness (same bootstrap)")
    for sl in (0.10, 0.15, 0.20):
        bl2 = st.blend_returns(p6040, mf_net, rf_i, sl, FEE)
        b2 = st.ann_stats(bl2, rf_i)
        bs2 = st.bootstrap_dsharpe(e6040, mf_net, sl, FEE, n_draws=NDRAWS, seed=595)
        print(f"  sleeve {sl*100:.0f}%: Sharpe {b2['sharpe']:.3f}  maxDD {b2['maxdd']*100:.1f}%  "
          f"dSharpe {bs2['obs']:+.3f}  CI [{bs2['lo']:+.3f}, {bs2['hi']:+.3f}]  p {bs2['p_onesided']:.3f}")

    print("\n# Cost & lookback robustness (alpha t is the fragile joint)")
    for cb in (2.0, 5.0, 10.0):
        bk = st.tsmom_book(fm, lookback=12, cost_bps=cb)
        em = bk["net"].loc[idx]
        al2 = st.nw_alpha(em, e6040, lags=6)
        print(f"  cost {cb:>4.1f} bps: alpha {al2['alpha_ann']*100:+.2f}%/yr  "
              f"NW t = {al2['t_alpha']:+.2f}")
    for lb in (9, 12, 15):
        bk = st.tsmom_book(fm, lookback=lb, cost_bps=COST_BPS)
        em = bk["net"].reindex(idx).dropna()
        e6 = e6040.reindex(em.index)
        al2 = st.nw_alpha(em, e6, lags=6)
        bs2 = st.bootstrap_dsharpe(e6, em, SLEEVE, FEE, n_draws=NDRAWS, seed=595)
        print(f"  lookback {lb:>2d}m: book Sharpe {st.sharpe_excess(em):.3f}  "
              f"alpha NW t = {al2['t_alpha']:+.2f}  dSharpe {bs2['obs']:+.3f}  "
              f"CI [{bs2['lo']:+.3f}, {bs2['hi']:+.3f}]")
    print("  NW lag sensitivity (12m, 5bp): "
          + "  ".join(f"lags={lg}: t={st.nw_alpha(mf_net, e6040, lags=lg)['t_alpha']:+.2f}"
                      for lg in (3, 6, 12)))

    print("\n# Sub-periods — where the benefit lives (regime dependence)")
    for lo, hi in [("2001-09", "2008-12"), ("2009-01", "2019-12"), ("2020-01", "2026-05")]:
        sl_ = slice(pd.Timestamp(lo), pd.Timestamp(hi) + pd.offsets.MonthEnd(0))
        s60 = st.sharpe_excess(e6040.loc[sl_])
        sbl = st.sharpe_excess((blend - rf_i).loc[sl_])
        smf = st.sharpe_excess(mf_net.loc[sl_])
        print(f"  {lo} -> {hi}: 60/40 Sharpe {s60:+.2f}   blend {sbl:+.2f}   "
              f"dSharpe {sbl - s60:+.3f}   (sleeve standalone {smf:+.2f})")

    print("\n# 2022 — the crisis-alpha year (calendar total returns)")
    rows = [("SPY", tr["SPY"]), ("VBMFX (bonds)", tr["VBMFX"]), ("60/40", p6040),
            ("MF sleeve (funded, net)", mf_funded), ("60/40 + 15% sleeve", blend),
            ("DBMF (live)", tr["DBMF"]), ("KMLM (live)", tr["KMLM"])]
    for name, ser in rows:
        print(f"  {name:<26s}: {st.calendar_year_return(ser, 2022)*100:+7.2f}%")
    print(f"  sleeve contribution to the blend in 2022 ~ +{0.15*st.calendar_year_return(mf_funded, 2022)*100:.1f} pp")

    print("\n# Third axis — is the benefit just BONDS IN DISGUISE?")
    alb = st.nw_alpha(mf_net, tr["VBMFX"].loc[idx] - rf_i, lags=6)
    print(f"  corr(MF, bonds) = {c3['r']:+.3f}  CI [{c3['lo']:+.3f}, {c3['hi']:+.3f}]   "
          f"beta on bonds {alb['beta']:+.3f}   R^2 {alb['r2']*100:.1f}%")
    p4555 = 0.45 * tr["SPY"].loc[idx] + 0.55 * tr["VBMFX"].loc[idx]
    d = st.ann_stats(p4555, rf_i)
    print(f"  45/55 (shift the same 15 pts into bonds instead): Sharpe {d['sharpe']:.3f}  "
          f"maxDD {d['maxdd']*100:.1f}%  2022 {st.calendar_year_return(p4555, 2022)*100:+.2f}%")
    print(f"  vs blend: Sharpe {b['sharpe']:.3f}  maxDD {b['maxdd']*100:.1f}%  "
          f"2022 {st.calendar_year_return(blend, 2022)*100:+.2f}%")
    print(f"  2022: bonds {st.calendar_year_return(tr['VBMFX'], 2022)*100:+.2f}% vs "
          f"sleeve {st.calendar_year_return(mf_funded, 2022)*100:+.2f}% — opposite signs")

    print("\n# Live ETF check — DBMF/KMLM, the one-ticket wrappers (short window, descriptive)")
    dbmf = tr["DBMF"].dropna().iloc[1:]           # drop the partial inception month
    lidx = dbmf.index.intersection(p6040_full.index).intersection(rf.dropna().index)
    p60l, rfl = p6040_full.loc[lidx], rf.loc[lidx]
    e60l = p60l - rfl
    print(f"  live window: {lidx.min().date()} -> {lidx.max().date()}  ({len(lidx)} months)")
    cd = st.corr_ci(dbmf.loc[lidx] - rfl, e60l)
    print(f"  corr(DBMF, 60/40) = {cd['r']:+.3f}  CI [{cd['lo']:+.3f}, {cd['hi']:+.3f}]")
    cr = st.corr_ci(dbmf.loc[lidx] - rfl, mf_net.reindex(lidx))
    print(f"  corr(DBMF, replication book) = {cr['r']:+.3f}  CI [{cr['lo']:+.3f}, {cr['hi']:+.3f}]")
    kmlm = tr["KMLM"].dropna().iloc[1:]
    kidx = kmlm.index.intersection(p6040_full.index).intersection(rf.dropna().index)
    ck = st.corr_ci(kmlm.loc[kidx] - rf.loc[kidx], (p6040_full.loc[kidx] - rf.loc[kidx]))
    print(f"  corr(KMLM, 60/40) = {ck['r']:+.3f}  CI [{ck['lo']:+.3f}, {ck['hi']:+.3f}]  (n={ck['n']})")
    sd = st.ann_stats(dbmf.loc[lidx], rfl)
    print(f"  DBMF net-of-fee: CAGR {sd['cagr']*100:.2f}%  vol {sd['vol']*100:.1f}%  "
          f"Sharpe {sd['sharpe']:.3f}  maxDD {sd['maxdd']*100:.1f}%")
    bl_l = 0.85 * p60l + 0.15 * dbmf.loc[lidx]
    a_l, b_l = st.ann_stats(p60l, rfl), st.ann_stats(bl_l, rfl)
    bs_l = st.bootstrap_dsharpe(e60l, dbmf.loc[lidx] - rfl, SLEEVE, 0.0, n_draws=NDRAWS, seed=595)
    print(f"  60/40 Sharpe {a_l['sharpe']:.3f} maxDD {a_l['maxdd']*100:.1f}%  ->  +15% DBMF "
          f"Sharpe {b_l['sharpe']:.3f} maxDD {b_l['maxdd']*100:.1f}%   "
          f"dSharpe {bs_l['obs']:+.3f} CI [{bs_l['lo']:+.3f}, {bs_l['hi']:+.3f}] p {bs_l['p_onesided']:.3f}")

    print("\n# Random-sign placebo — same sizing/lag/costs/fee, trend sign replaced by a coin")
    rows_p = st.random_sign_book(fm, range(1, 25), cost_bps=COST_BPS)
    shs, dss = [], []
    for r in rows_p:
        em = r["net"].reindex(idx).dropna()
        e6 = e6040.reindex(em.index)
        shs.append(st.sharpe_excess(em))
        blx = (1 - SLEEVE) * e6 + SLEEVE * (em - FEE / 12.0)
        dss.append(st.sharpe_excess(blx) - st.sharpe_excess(e6))
    print(f"  {len(rows_p)} seeds: mean placebo book Sharpe {np.mean(shs):+.3f} "
          f"(sd {np.std(shs):.3f}); mean placebo dSharpe {np.mean(dss):+.4f} (sd {np.std(dss):.4f})")
    print("  -> a zero-edge uncorrelated sleeve HURTS the blend (cost+fee drag); the observed")
    print("     +dSharpe is trend structure, not automatic diversification arithmetic")

print("\n# Synthetic control — machinery proof only, never market evidence")
print("  the dSharpe bootstrap + alpha t must stay quiet on a planted ZERO-edge sleeve and")
print("  light up on a planted Sharpe-0.8 sleeve (312 months, corr 0, fee charged).")
for ms in (0.0, 0.8):
    w = data.synthetic_world(n_months=312, mf_sharpe=ms, rho_bond=0.0, seed=595)
    e60s = 0.6 * w["stock"] + 0.4 * w["bond"]
    bs_s = st.bootstrap_dsharpe(e60s, w["mf"], SLEEVE, FEE, n_draws=3_000, seed=595)
    al_s = st.nw_alpha(w["mf"], e60s, lags=6)
    print(f"  planted MF Sharpe {ms:.1f}: dSharpe {bs_s['obs']:+.3f}  "
          f"CI [{bs_s['lo']:+.3f}, {bs_s['hi']:+.3f}]  p {bs_s['p_onesided']:.3f}  "
          f"alpha NW t = {al_s['t_alpha']:+.2f}")
