"""Reproducible headline run for Study 906 — EM Local Bonds, FX-Hedged (a proxy).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached total-return parquet under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402
from quantlab.stats import sharpe_ci_bootstrap  # noqa: E402

from em_hedged import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# EM Local Bonds FX-Hedged — does stripping the FX leave a real local-rate carry?")

if not data.have_real():
    print("(cache miss — fetching the six-ETF total-return panel once)")
    data.fetch()

px = data.load_prices()
m = st.monthly_returns(px)
print(f"[data] {px.shape[1]} tickers, monthly {m.index.min().date()} -> {m.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint(Close)={fingerprint(px)}")
print("  PROXY: the 'hedge' is a long-UUP (DXY-basket) overlay, NOT the EMLC currency basket "
      "-> strips only PART of the EM-FX. Named on the Signal axis.")

print("\n# THE RACE — excess-vs-excess (minus BIL), three local-EM ETFs vs USD-EM (EMB)")
for local in data.LOCAL:
    r = st.race(m, local=local)
    print(f"\n== {local}  (n={r['n']} {r['start']}->{r['end']}) ==")
    print(f"  EMLC~UUP: beta {r['emlc_uup_beta']:+.2f} (t={r['t_emlc_uup']:+.2f}) R2 {r['emlc_uup_r2']:.2f}"
          f"  ->  hedge ratio b = {r['hedge_b']:+.3f} (long-USD overlay)")
    print(f"  UNHEDGED {local}: excess {r['local_exc_ann_pct']:+.2f}%/yr  Sharpe {r['local_sharpe']:+.2f}  (HAC t {r['t_local']:+.2f})")
    print(f"  HEDGED   {local}: excess {r['hedged_exc_ann_pct']:+.2f}%/yr  Sharpe {r['hedged_sharpe']:+.2f}  (HAC t {r['t_hedged']:+.2f})")
    print(f"  BENCH    EMB : excess {r['bench_exc_ann_pct']:+.2f}%/yr  Sharpe {r['bench_sharpe']:+.2f}  (HAC t {r['t_bench']:+.2f})")
    print(f"  hedged - EMB premium: {r['prem_diff_ann_pct']:+.2f}%/yr (HAC t {r['t_prem_diff']:+.2f}, Welch {r['welch_hedged_vs_bench']:+.2f})")

# Headline = EMLC
r = st.race(m, local="EMLC")

print("\n# BOOTSTRAP Sharpe CI (circular block bootstrap, excess-of-cash)")
for tag, s in [("unhedged EMLC", st.excess(m, "EMLC")),
               ("hedged EMLC", st.hedged_series(m, "EMLC")),
               ("EMB (USD-EM)", st.excess(m, "EMB"))]:
    ci = sharpe_ci_bootstrap(s, n_boot=3000, periods_per_year=12, seed=906, method="cbb")
    print(f"  {tag:16s}: Sharpe {ci['sharpe']:+.2f}  95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  frac<0 {ci['frac_negative']:.2f}")

print("\n# WALK-FORWARD hedge (36m rolling b, lag 1 — the implementable, no-look-ahead version)")
wf = st.rolling_hedge_series(m, "EMLC", window=36, min_periods=24)
oe = st.excess(m, "UUP"); a = wf.align(oe, join="inner")
rr = st.hac_ols(a[0].to_numpy(), a[1].to_numpy())
print(f"  n={len(wf)} {wf.index.min().date()}->{wf.index.max().date()}  excess {wf.mean()*12*100:+.2f}%/yr  "
      f"Sharpe {st.sharpe_ann(wf):+.2f}  (HAC t {st.newey_west_t(wf.to_numpy()):+.2f})")
print(f"  residual EM-FX beta left by the DXY proxy = {rr['beta']:+.3f} (R2 {rr['r2']:.2f})")

print("\n# ERA SPLIT (2021-01-01, single full-sample hedge ratio)")
es = st.era_split(m, split="2021-01-01", local="EMLC")
for k, v in es.items():
    print(f"  {k:5s} n={v['n']} {v['start']}->{v['end']}: hedged {v['hedged_ann_pct']:+.2f}%/yr (t {v['t_hedged']:+.2f})  "
          f"prem-vs-EMB {v['prem_diff_ann_pct']:+.2f}% (t {v['t_prem_diff']:+.2f})")

print("\n# DRAWDOWN (total-return NAV)")
dd_emlc = st.max_drawdown(px["EMLC"]); dd_emb = st.max_drawdown(px["EMB"])
he = st.hedged_series(m, "EMLC"); dd_hed = st.max_drawdown((1 + he).cumprod())
print(f"  EMLC        : {dd_emlc['depth_pct']:+.1f}%  ({dd_emlc['peak']} -> {dd_emlc['trough']})")
print(f"  EMB         : {dd_emb['depth_pct']:+.1f}%  ({dd_emb['peak']} -> {dd_emb['trough']})")
print(f"  hedged-EMLC : {dd_hed['depth_pct']:+.1f}%  ({dd_hed['peak']} -> {dd_hed['trough']})  <- FX-strip cuts the drawdown")

print("\n# THE TIMER — cost the UUP overlay (re-struck monthly to |b| of NAV)")
c = st.costed(m, local="EMLC")
print(f"  b={c['hedge_b']:+.3f}  charge {c['charge_ann_pct']:.2f}%/yr  gross {c['gross_hedged_ann_pct']:+.2f} -> "
      f"net {c['net_hedged_ann_pct']:+.2f}%/yr (t {c['t_net_hedged']:+.2f}, Sharpe {c['net_hedged_sharpe']:+.2f})")
print(f"  net premium vs EMB: {c['net_prem_diff_ann_pct']:+.2f}%/yr (t {c['t_net_prem_diff']:+.2f})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
planted = st.synthetic_detect(data.synthetic_world(carry_annual=0.04, seed=906))
null_t = np.array([st.synthetic_detect(data.synthetic_world(carry_annual=0.0, seed=906 + s))["t_hedged"]
                   for s in range(20)])
print(f"  planted (carry=4%/yr, seed 906): hedged {planted['hedged_exc_ann_pct']:+.2f}%/yr (HAC t {planted['t_hedged']:+.2f}), "
      f"hedge b {planted['hedge_b']:+.2f}")
print(f"  null (carry=0), 20 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
