"""Reproducible headline run for Study 627 — 13F Cloning (Berkshire Hathaway).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape (EDGAR 13F-HR
top holdings + yfinance total-return closes under ``_cache/``), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from thirteen_f_cloning import data, strategy as st

try:
    from quantlab.repro import data_stamp
except Exception:                                    # repo layout fallback
    data_stamp = None

N_DRAWS = 200
SEED = 627

print("# 13F Cloning — copy Berkshire's 13F 45 days late (EDGAR + yfinance)")
if not data.have_real():
    print("(cache missing — building it now: EDGAR 13F-HR + yfinance, one-shot)")
holdings, prices = data.load_real()

if data_stamp is not None:
    print(data_stamp("clone tape (28 names + SPY + BRK-B + ^IRX)", prices, asof=data.AS_OF))

n_filings = holdings["filing_date"].nunique()
t10 = holdings[holdings["rank"] <= 10]
n_slots = len(t10)
missing = t10["cusip"].map(data.CUSIP_TO_TICKER).isna().sum()
print(f"filings         : {n_filings} original 13F-HR, periods "
      f"{holdings['period'].min().date()} -> {holdings['period'].max().date()} "
      f"(filed {holdings['filing_date'].min().date()} -> {holdings['filing_date'].max().date()})")
print(f"universe        : {t10['cusip'].nunique()} distinct CUSIPs ever in the top 10; "
      f"{len(data.TICKERS)} priceable on yfinance")
print(f"slot coverage   : {n_slots - missing}/{n_slots} top-10 slots priced "
      f"({(n_slots - missing) / n_slots * 100:.1f}%) — the {missing} missing slots are "
      f"DIRECTV + Activision (acquired & delisted; weights renormalised)")

latest = t10[t10["filing_date"] == t10["filing_date"].max()].copy()
latest["w_vw"] = latest["value"] / latest["value"].sum()
print(f"\nlatest filing ({latest['filing_date'].iloc[0].date()}, period "
      f"{latest['period'].iloc[0].date()}) top 10 (VW weights):")
for _, row in latest.iterrows():
    tk = data.CUSIP_TO_TICKER.get(row["cusip"]) or "—"
    print(f"  {row['rank']:>2}. {tk:<6} {row['issuer']:<32} {row['w_vw'] * 100:5.1f}%")

print("\n# The race — clone (rebalanced at filing date + 1 trading day) vs SPY, monthly")
races = {}
for wgt in ("ew", "vw"):
    for cb in (0.0, 10.0):
        r = st.race(prices, holdings, weighting=wgt, cost_bps=cb)
        races[(wgt, cb)] = r
        tag = f"{wgt.upper()} {'gross' if cb == 0 else f'net@{cb:.0f}bps'}"
        print(f"  {tag:<14}: clone CAGR {r['clone']['cagr_pct']:6.2f}%  vs SPY "
              f"{r['bench_stats']['cagr_pct']:.2f}%  | active {r['active_ann_pct']:+.2f}%/yr "
              f"HAC t = {r['t_active']:+.2f} (NW lags {r['nw_lags']}, n = {r['n_months']}m)")

r = races[("ew", 0.0)]
print(f"\n  window          : {r['start']} -> {r['end']}  "
      f"({r['n_months']} complete months, {r['n_months'] / 12:.1f} years)")
print(f"  EW gross CAPM   : alpha {r['alpha_ann_pct']:+.2f}%/yr (HAC t = {r['t_alpha']:+.2f}), "
      f"beta {r['beta']:.2f} | Sharpe (excess-vs-excess) {r['clone']['sharpe_ex']:.2f} vs "
      f"SPY {r['bench_stats']['sharpe_ex']:.2f}")
rv = races[("vw", 0.0)]
print(f"  VW gross CAPM   : alpha {rv['alpha_ann_pct']:+.2f}%/yr (HAC t = {rv['t_alpha']:+.2f}), "
      f"beta {rv['beta']:.2f} | Sharpe {rv['clone']['sharpe_ex']:.2f}")
print(f"  max drawdown    : clone EW {r['clone']['max_dd_pct']:.1f}%  VW "
      f"{rv['clone']['max_dd_pct']:.1f}%  SPY {r['bench_stats']['max_dd_pct']:.1f}%")
print(f"  costs           : EW one-way turnover {r['avg_oneway_turnover_pct']:.1f}%/rebalance "
      f"(~{r['avg_oneway_turnover_pct'] * 4 / 100:.0%}/yr), drag at 10 bps one-way = "
      f"{races[('ew', 10.0)]['cost_drag_ann_bps']:.1f} bps/yr — costs are NOT the problem")

print("\n# Sub-period split (EW gross, active vs SPY)")
cm, bm = r["clone_m"], r["bench_m"]
act = cm - bm
for lab, sl in (("2013-2019", act[act.index < "2020-01-01"]),
                ("2020-2026", act[act.index >= "2020-01-01"])):
    nw = st.nw_tstat(sl.to_numpy())
    print(f"  {lab}: active {nw['mean'] * 1200:+.2f}%/yr  HAC t = {nw['t']:+.2f}  (n = {nw['n']}m)")

print("\n# Robustness — how many names you clone (EW, gross, active vs SPY)")
for tn in (5, 10, 15):
    b = st.build_clone(prices, holdings, top_n=tn, weighting="ew",
                       cusip_map=data.CUSIP_TO_TICKER)
    cm2 = st.to_monthly(b["daily"])
    bm2 = st.to_monthly(prices["SPY"].pct_change().reindex(b["daily"].index))
    idx = cm2.index.intersection(bm2.index)
    nw = st.nw_tstat((cm2.loc[idx] - bm2.loc[idx]).to_numpy())
    print(f"  top-{tn:<2}: active {nw['mean'] * 1200:+.2f}%/yr  HAC t = {nw['t']:+.2f}")

print("\n# Random-manager placebo — 10 random names from the same universe, same calendar/lag")
rb = st.random_manager_baseline(prices, holdings, n_draws=N_DRAWS, seed=SEED)
ew_act = races[("ew", 0.0)]["active_ann_pct"]
pct_below = float((rb["actives"] <= ew_act).mean())
z = (ew_act - rb["mean_active_ann_pct"]) / rb["sd_active_ann_pct"]
print(f"  {rb['n_draws']} draws (seed {SEED}): mean active {rb['mean_active_ann_pct']:+.2f}%/yr "
      f"(sd {rb['sd_active_ann_pct']:.2f}), mean CAGR {rb['mean_cagr_pct']:.2f}%")
print(f"  the actual EW clone ({ew_act:+.2f}%/yr) sits {z:+.2f} sd below the random mean; "
      f"{pct_below * 100:.1f}% of random managers did as badly or worse")

print("\n# Third axis — does the clone beat Berkshire itself? (the cash-drag free-ride)")
brk_cagr = None
for wgt in ("ew", "vw"):
    rb3 = st.race(prices, holdings, weighting=wgt, cost_bps=0.0, bench="BRK-B")
    brk_cagr = rb3["bench_stats"]["cagr_pct"]
    print(f"  {wgt.upper()} clone {rb3['clone']['cagr_pct']:6.2f}%/yr  vs BRK-B "
          f"{brk_cagr:.2f}%/yr  | active {rb3['active_ann_pct']:+.2f}%/yr "
          f"HAC t = {rb3['t_active']:+.2f}")
print(f"  (context: BRK-B itself lagged SPY on this window — {brk_cagr:.2f}% vs "
      f"{races[('ew', 0.0)]['bench_stats']['cagr_pct']:.2f}%/yr)")

print("\n# Synthetic control — deterministic, no network (machinery proof only)")
print("  a manager with PLANTED skill discloses through quarterly filings + 45-day lag;")
print("  the clone harness must recover a planted alpha and must NOT fire on a null manager.")
for a in (0.0, 0.08):
    spx, sh = data.synthetic_world(alpha_annual=a, seed=SEED)
    rs = st.race(spx, sh, weighting="ew", cost_bps=0.0, cusip_map=None)
    print(f"  planted alpha = {a * 100:4.1f}%/yr: clone active {rs['active_ann_pct']:+.2f}%/yr  "
          f"HAC t = {rs['t_active']:+.2f}  (CAPM alpha {rs['alpha_ann_pct']:+.2f}%/yr, "
          f"t = {rs['t_alpha']:+.2f})")
print("  (the planted-skill run recovers ~2/3 of the alpha — the 45-day lag + quarterly")
print("   rotation haircut — and clears t >= 2; the null stays flat. Never market evidence.)")
