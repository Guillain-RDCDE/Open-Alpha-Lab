"""Reproducible headline run for Study 907 — Senior Loans vs High-Yield.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic once the real tapes are cached under
``_cache/`` (fetched once on a miss via yfinance); always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from loans_vs_hy import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Senior Loans vs High-Yield — is the 'seniority premium' a real risk-adjusted edge?")

if not data.have_real():
    print("(cache miss — fetching the six total-return tapes once via yfinance)")
    data.fetch()

px_full = data.load_prices()
# Race everything on the COMMON window bounded by BKLN's 2011-03-03 inception, so the HY
# arms don't get to bank the 2008 GFC the loan arms never saw. (SRLN still joins the loan
# composite at its own 2013 listing via the NaN-skipping mean — a short-history caveat.)
px = px_full.loc[data.BKLN_INCEPTION:]
print(f"[data] {px.shape[1]} tapes, {len(px)} rows  "
      f"{px.index.min().date()} -> {px.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(px)}")
print("  SURVIVORSHIP / short history: SRLN lists 2013-04-04 (joins the composite loan "
      "sleeve mid-sample); all four credit ETFs are live survivors. Named on the Signal axis.")

ret = st.to_returns(px)
cash = ret["BIL"]
loans_c = st.composite(ret, data.LOAN_LEGS)      # mean(BKLN, SRLN)
hy_c = st.composite(ret, data.HY_LEGS)           # mean(HYG, JNK)

print("\n# THE ARMS — total return, excess-of-cash Sharpe (2011-03-03 -> 2026-06-30)")
for name, r in [("BKLN (loan)", ret["BKLN"]), ("SRLN (loan)", ret["SRLN"]),
                ("HYG (HY)", ret["HYG"]), ("JNK (HY)", ret["JNK"]),
                ("LOANS composite", loans_c), ("HY composite", hy_c),
                ("IEF (dur ref)", ret["IEF"])]:
    s = st.arm_stats(r, cash)
    print(f"  {name:16s} CAGR {s['cagr']*100:5.2f}%  vol {s['vol']*100:4.1f}%  "
          f"exSharpe {s['sharpe']:+.3f}  maxDD {s['max_dd']*100:6.1f}%")

print("\n# THE FLAGSHIP PAIR — BKLN (loans) vs HYG (HY)")
spf = st.spread_stats(ret["BKLN"], ret["HYG"])
advf = st.sharpe_advantage(st.excess(ret["BKLN"], cash), st.excess(ret["HYG"], cash))
print(f"  excess-Sharpe: loans {advf['sharpe_loans']:+.3f} vs HY {advf['sharpe_hy']:+.3f}  "
      f"-> advantage {advf['advantage']:+.3f}")
print(f"  return spread (loans-HY): {spf['mean_bps']:+.2f} bps/day "
      f"({spf['ann_pct']:+.2f}%/yr)  NW t = {spf['t_nw']:+.2f}")

print("\n# THE COMPOSITE SLEEVE — loans(BKLN,SRLN) vs HY(HYG,JNK)")
adv = st.sharpe_advantage(st.excess(loans_c, cash), st.excess(hy_c, cash))
sp = st.spread_stats(loans_c, hy_c)
print(f"  excess-Sharpe: loans {adv['sharpe_loans']:+.3f} vs HY {adv['sharpe_hy']:+.3f}  "
      f"-> advantage {adv['advantage']:+.3f}")
print(f"  return spread (loans-HY): {sp['mean_bps']:+.2f} bps/day "
      f"({sp['ann_pct']:+.2f}%/yr)  NW t = {sp['t_nw']:+.2f}")
boot = st.bootstrap_sharpe_adv(st.excess(loans_c, cash), st.excess(hy_c, cash),
                               n_boot=5000, seed=907)
print(f"  BOOTSTRAP Sharpe advantage: {boot['advantage']:+.3f}  95% CI "
      f"[{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]  loans-wins {boot['frac_loans_wins']*100:.0f}%")

print("\n# ERA ROBUSTNESS — excess-Sharpe of each leg + loans-HY spread")
cuts = [("2011-03-03", "2016-01-01", "2011-15 (energy build-up)"),
        ("2016-01-01", "2020-01-01", "2016-19"),
        ("2020-01-01", "2023-01-01", "2020-22 (COVID + hike)"),
        ("2023-01-01", "2026-07-01", "2023-26")]
for e in st.era_table(loans_c, hy_c, cash, cuts):
    print(f"  {e['era']:26s} n={e['n_days']:4d}  ShL {e['sharpe_loans']:+.2f}  "
          f"ShHY {e['sharpe_hy']:+.2f}  adv {e['advantage']:+.2f}  "
          f"spread {e['spread_bps']:+.2f}bps (t {e['spread_t']:+.2f})")

print("\n# STRESS TABLE — total return (%) through the credit episodes")
windows = [("2015-06-01", "2016-02-15", "energy default wave 2015-16"),
           ("2020-02-19", "2020-03-23", "COVID liquidity crash"),
           ("2022-01-01", "2022-10-15", "2022 rate shock")]
for row in st.stress_table(px[["BKLN", "SRLN", "HYG", "JNK"]], windows):
    print(f"  {row['episode']:28s} BKLN {row['BKLN']:+6.1f}  SRLN {row['SRLN']:+6.1f}  "
          f"HYG {row['HYG']:+6.1f}  JNK {row['JNK']:+6.1f}")

print("\n# CALENDAR YEAR — total return (%), loans vs HY composite")
cy = st.calendar_year_table(loans_c, hy_c)
print(cy.to_string())

print("\n# TRADABILITY — long loans / short HY, costed")
print("  gross spread + monthly-rebal round-trip cost + borrow on the short HY leg")
for cb, bo in [(5.0, 60.0), (3.0, 40.0)]:
    tm = st.costed_long_short(loans_c, hy_c, cost_bps=cb, borrow_bps_yr=bo)
    print(f"  cost {cb:.0f}bps/side + {bo:.0f}bps borrow: gross {tm['gross_ann_pct']:+.2f}%/yr "
          f"-> net {tm['net_ann_pct']:+.2f}%/yr (NW t {tm['t_net_nw']:+.2f}, "
          f"Sharpe {tm['net_sharpe']:+.2f})")

print("\n# SYNTHETIC CONTROL — deterministic, no network")
null_adv = []
for s_ in range(12):
    f0, _ = data.synthetic_pair(sharpe_edge=0.0, seed=907 + s_, n_days=4000)
    r0 = st.to_returns(f0)
    null_adv.append(st.sharpe_advantage(st.excess(r0["LOANS"], r0["CASH"]),
                                        st.excess(r0["HY"], r0["CASH"]))["advantage"])
null_adv = np.asarray(null_adv)
print(f"  null (edge=0), 12 seeds: mean Sharpe advantage {null_adv.mean():+.3f} "
      f"(sd {null_adv.std(ddof=1):.3f}) — no systematic edge")
fp, _ = data.synthetic_pair(sharpe_edge=0.6, seed=907, n_days=4000)
det = st.synthetic_detect(fp, n_boot=2000, seed=907)
print(f"  planted (edge=0.6, seed 907): advantage {det['advantage']:+.3f}, "
      f"loans-wins {det['frac_loans_wins']*100:.0f}%, CI "
      f"[{det['ci95'][0]:+.3f}, {det['ci95'][1]:+.3f}]")
