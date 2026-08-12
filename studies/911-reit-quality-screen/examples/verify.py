"""Reproducible headline run for Study 911 — REIT Quality Screen.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached total-return tape under
``_cache/`` (fetching once on a cache miss) and always runs the synthetic control offline.

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
from quantlab.analytics import mean_tstat_hac  # noqa: E402
from quantlab.stats import annualized_sharpe  # noqa: E402

from reit_quality import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# REIT Quality Screen — does a durable-income equity sleeve beat the broad REIT "
      "index, and is the mortgage-REIT carry a trap?")

if not data.have_real():
    print("(cache miss — fetching the ETF total-return tape once)")
    data.fetch()

px = data.load_prices()
m = st.monthly_returns(px)
common = st.common_sample(m, ["VNQ", "REZ", "RWR", "REM", "SPY", "BIL"])
print(f"[data] daily panel {len(px):,} rows  {px.index.min().date()} -> {px.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint(Close)={fingerprint(px)}")
print(f"[data] common monthly sample {common.index.min().date()} -> {common.index.max().date()}  "
      f"({len(common)} months)")
print("  SURVIVORSHIP / SHORT-HISTORY: XLRE only lists 2015-10; the sector funds are young; "
      "all magnitudes are named on the Signal axis.")

# --- excess Sharpe race ---------------------------------------------------- #
print("\n# THE RACE — excess-vs-excess Sharpe (all minus BIL T-bill)")
tab = st.sharpe_table(m, ["REZ", "VNQ", "RWR", "REM", "SPY"], rf="BIL")
for c in ["REZ", "VNQ", "RWR", "REM", "SPY"]:
    row = tab.loc[c]
    # cross-check the excess Sharpe against quantlab.stats.annualized_sharpe
    e = st.excess(common, c, "BIL")
    ql = annualized_sharpe(e, periods_per_year=12)
    print(f"  {c:4s}: excess Sharpe {row['excess_sharpe']:+.3f} (ql {ql:+.3f})  "
          f"ann {row['ann_ret_pct']:+.2f}%  vol {row['ann_vol_pct']:.1f}%")

# --- the durable-income tilt ----------------------------------------------- #
print("\n# LEG 1 — the durable-income tilt: quality (REZ) vs broad (VNQ)")
sp = st.spread_stats(m, "REZ", "VNQ")
qc = mean_tstat_hac((m["REZ"] - m["VNQ"]).dropna())   # quantlab cross-check
print(f"  REZ - VNQ spread: {sp['mean_bps']:+.1f} bps/mo  HAC t = {sp['t_nw']:+.2f}  "
      f"(ql HAC t {qc['tstat']:+.2f}, n={sp['n']})")
book = st.book_spread_stats(m, ["VNQ", "REZ", "RWR"], "VNQ")
print(f"  quality book (VNQ,REZ,RWR eq-wt) - VNQ: {book['mean_bps']:+.1f} bps/mo  "
      f"HAC t = {book['t_nw']:+.2f}  (n={book['n']})")
adv = st.sharpe_advantage(m, "REZ", "VNQ", rf="BIL")
print(f"  Sharpe advantage REZ over VNQ: {adv['advantage']:+.3f}  "
      f"95% CI [{adv['ci_low']:+.3f}, {adv['ci_high']:+.3f}]  frac<0 = {adv['frac_neg']:.2f}")

print("\n# LEG 1 — robustness, two eras (split 2017-01)")
for e in st.era_spreads(m, "REZ", "VNQ", cut="2017-01-01"):
    print(f"  {e['era']}: n={e['n']}  {e['mean_bps']:+.1f} bps/mo  HAC t = {e['t_nw']:+.2f}")

# --- the leveraged-carry trap ---------------------------------------------- #
print("\n# LEG 2 — the leveraged-carry trap: mortgage REITs (REM)")
tg = st.trap_gap(m, "REM", "REZ", "VNQ", rf="BIL")
print(f"  REM   : ann {tg['trap_ann_pct']:+.2f}%/yr  excess Sharpe {tg['trap_sharpe']:+.3f}")
print(f"  REZ   : ann {tg['quality_ann_pct']:+.2f}%/yr  excess Sharpe {tg['quality_sharpe']:+.3f}")
print(f"  VNQ   : ann {tg['broad_ann_pct']:+.2f}%/yr  excess Sharpe {tg['broad_sharpe']:+.3f}")
print(f"  REZ - REM spread {tg['quality_minus_trap_bps']:+.1f} bps/mo  "
      f"HAC t = {tg['t_quality_minus_trap']:+.2f}")
for e in st.era_spreads(m, "REZ", "REM", cut="2017-01-01"):
    print(f"    {e['era']}: {e['mean_bps']:+.1f} bps/mo  HAC t = {e['t_nw']:+.2f}")

# --- drawdowns ------------------------------------------------------------- #
print("\n# RISK — daily total-return max drawdowns")
for c in ["REZ", "VNQ", "REM", "SPY"]:
    dd = st.max_drawdown(px, c)
    print(f"  {c:4s}: {dd['depth_pct']:+.1f}%  ({dd['peak']} -> {dd['trough']})")

# --- costs ----------------------------------------------------------------- #
print("\n# TRADABILITY — cost the quality book (monthly rebalanced) vs buy-and-hold VNQ")
for cb in (2.0, 5.0, 10.0):
    cst = st.costed_book(m, ["VNQ", "REZ", "RWR"], "VNQ", cost_bps_oneway=cb)
    print(f"  cost {cb:>4.1f} bps/side: gross {cst['gross_bps_mo']:+.1f} -> net "
          f"{cst['net_bps_mo']:+.1f} bps/mo (drag {cst['drag_bps_yr']:.1f} bps/yr, "
          f"t_net = {cst['t_net']:+.2f}, ~{cst['net_ann_pct']:+.2f}%/yr)")

# --- synthetic control ----------------------------------------------------- #
print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
straddle, trap_flag, null_adv = 0, 0, []
for s_ in range(20):
    w0 = data.synthetic_world(edge_ann=0.0, seed=911 + s_, n_months=228)
    a0 = st.sharpe_advantage(w0, "QUAL", "BROAD", rf="CASH", n_boot=800)
    null_adv.append(a0["advantage"])
    if a0["ci_low"] < 0 < a0["ci_high"]:
        straddle += 1
    if st.excess_sharpe(w0, "TRAP", "CASH") < st.excess_sharpe(w0, "BROAD", "CASH"):
        trap_flag += 1
null_adv = np.asarray(null_adv)
print(f"  null (edge=0), 20 seeds: Sharpe advantage mean {null_adv.mean():+.3f} "
      f"(sd {null_adv.std(ddof=1):.3f}); bootstrap CI straddles zero in {straddle}/20 seeds; "
      f"trap correctly flagged in {trap_flag}/20")
w1 = data.synthetic_world(edge_ann=0.03, seed=911, n_months=228)
d1 = st.synth_detect(w1)
print(f"  planted (+3%/yr edge, seed 911): QUAL-BROAD HAC t = {d1['spread_t']:+.2f}, "
      f"Sharpe adv = {d1['adv']:+.3f} (CI low {d1['adv_ci_low']:+.3f} clear of zero), "
      f"trap flagged = {d1['trap_flagged']}")
