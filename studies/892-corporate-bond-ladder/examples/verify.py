"""Reproducible headline run for Study 892 — Corporate-Bond Ladder.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached total-return prices under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control offline.

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

from bond_ladder import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Corporate-Bond Ladder — does a held-to-maturity ladder beat a constant-maturity fund?")

if not data.have_real():
    print("(cache miss — fetching total-return closes once via yfinance)")
    data.fetch()

prices = data.load_prices()
ret = data.monthly_returns(prices)
print(f"[data] {prices.shape[1]} tickers, {len(ret)} joint months  "
      f"{ret.index.min().date()} -> {ret.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(prices)={fingerprint(prices)}")
print("  SELECTION: SHY/IEI/IEF/TLT/AGG/BND/LQD/BIL are all live survivor ETFs; the joint "
      "window starts 2007-06 (BND/BIL inception). Named on the Signal axis.")
print(f"  ladder durations: EW={data.ladder_duration(data.EW_LADDER):.1f}y  "
      f"DurMatched={data.ladder_duration(data.DUR_LADDER):.1f}y  vs AGG "
      f"{data.DURATION['AGG']:.1f}y")

print("\n# HEADLINE — duration-matched Treasury ladder vs AGG (excess-of-BIL Sharpe race)")
r = st.race(ret, data.DUR_LADDER, fund="AGG", cash="BIL")
print(f"  ladder : ann {r['ladder_ann_pct']:+.2f}%/yr  exSharpe {r['ladder_ex_sharpe']:.3f} "
      f"CI [{r['ladder_sharpe_ci'][0]:+.2f}, {r['ladder_sharpe_ci'][1]:+.2f}]  "
      f"maxDD {r['ladder_maxdd_pct']:.1f}%")
print(f"  AGG    : ann {r['fund_ann_pct']:+.2f}%/yr  exSharpe {r['fund_ex_sharpe']:.3f} "
      f"CI [{r['fund_sharpe_ci'][0]:+.2f}, {r['fund_sharpe_ci'][1]:+.2f}]  "
      f"maxDD {r['fund_maxdd_pct']:.1f}%")
print(f"  ladder - fund: {r['diff_ann_pct']:+.2f}%/yr ({r['diff_bps_mo']:+.2f} bps/mo)  "
      f"HAC t = {r['t_hac']:+.2f}  (1-sample t = {r['t_1s']:+.2f})")
print(f"  diff-Sharpe {r['diff_sharpe']:+.3f}  CI [{r['diff_sharpe_ci'][0]:+.2f}, "
      f"{r['diff_sharpe_ci'][1]:+.2f}]  (straddles 0 => not distinguishable)")

print("\n# THE NAIVE LADDER — equal-weight SHY/IEI/IEF/TLT (what retail actually buys)")
rew = st.race(ret, data.EW_LADDER, fund="AGG", cash="BIL")
print(f"  EW ladder (dur {rew['ladder_dur']:.1f}y): ann {rew['ladder_ann_pct']:+.2f}%/yr  "
      f"exSharpe {rew['ladder_ex_sharpe']:.3f}  maxDD {rew['ladder_maxdd_pct']:.1f}%  "
      f"-> LOSES to AGG (Sharpe {rew['fund_ex_sharpe']:.3f}) on extra duration")

print("\n# ERA CUT — a real structural premium is stable; a duration/credit artefact flips sign")
et = st.era_table(ret, data.DUR_LADDER, fund="AGG")
for era, row in et.iterrows():
    print(f"  {era}: n={int(row['n'])}  ladder {row['ladder_ann_pct']:+.2f}%  "
          f"AGG {row['fund_ann_pct']:+.2f}%  diff {row['diff_ann_pct']:+.2f}%/yr "
          f"(HAC t {row['t_hac']:+.2f})")

print("\n# 2022 RATE SHOCK — the stress test the claim leans on")
cy = st.calendar_year_table(ret, data.DUR_LADDER, fund="AGG", cash="BIL")
for yr in (2021, 2022, 2023):
    if yr in cy.index:
        row = cy.loc[yr]
        print(f"  {yr}: ladder {row['ladder']:+.2f}%  AGG {row['fund']:+.2f}%  "
              f"gap {row['ladder_minus_fund']:+.2f} pp")

print("\n# TRADABILITY — cost the annually-rolled ETF ladder; the one-ticker fund is free")
for tb, tt in ((3.0, 0.15), (5.0, 0.30)):
    c = st.costed_race(ret, data.DUR_LADDER, fund="AGG", cash="BIL",
                       spread_bps_oneway=tb, annual_turnover=tt)
    print(f"  spread {tb:.0f}bp x turnover {tt:.0%}: cost {c['ladder_cost_bps_yr']:.1f} bps/yr "
          f"-> gross {c['gross_diff_ann_pct']:+.2f}% => net {c['net_diff_ann_pct']:+.2f}%/yr "
          f"(HAC t {c['t_hac_net']:+.2f})")

print("\n# SYNTHETIC CONTROL — deterministic, no network (machinery proof only)")
null_t = np.array([st.synthetic_detect(data.synthetic_world(edge_annual=0.0, seed=892 + s))["t_hac"]
                   for s in range(20)])
print(f"  null (edge=0), 20 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_t) >= 2).sum()}/20")
planted = st.synthetic_detect(data.synthetic_world(edge_annual=0.015, seed=892))
print(f"  planted (+1.5%/yr): recovered {planted['diff_ann_pct']:+.2f}%/yr  HAC t = {planted['t_hac']:+.2f}")

print(f"\nFingerprint {fingerprint(prices)}")
