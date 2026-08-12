"""Reproducible headline run for Study 888 — CLO AAA Carry.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/clo_prices.csv, built once from yfinance) and always-offline on the synthetic
control.

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

from clo_aaa import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

RACE_LEGS = ["JAAA", "ICLO", "LQD", "IEF", "BKLN"]
HIGH_START = "2023-01-01"          # 5% cash plateau: fat AAA-CLO carry in dollars
ZIRP_END = "2022-06-30"            # rates ~0 before the 2022 hiking cycle

print("# CLO AAA Carry — is the senior-tranche pickup a real, costed, risk-adjusted edge?")

if not data.have_real():
    print("(cache miss — fetching the CLO tape once from yfinance)")
    data.fetch_tape()

prices = data.load_prices()
ret = data.daily_returns(prices)
fp = fingerprint(prices, cols=data.TICKERS)
print(f"[data] {prices.shape[1]} tickers, {len(prices)} rows  "
      f"{prices.index.min().date()} -> {prices.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(prices)={fp}")
print("  SHORT HISTORY: JAAA from 2020-10-19, ICLO from 2022-12; ONE rate cycle, NO CLO "
      "credit-stress event in-sample -> realized Sharpes are an upper bound (Signal axis).")

print("\n# THE RACE — excess-of-cash (minus BIL) Sharpe, full available history per leg")
print("  (annualised excess, vol, Sharpe [block-bootstrap 95% CI], HAC t on daily excess, maxDD)")
tbl = st.race(ret, RACE_LEGS, cash=data.CASH)
for leg, row in tbl.iterrows():
    print(f"  {leg:<5s} {row['start']}->{row['end']} n={int(row['n']):>4d}  "
          f"exc {row['excess_ann_pct']:+6.2f}%/yr  vol {row['vol_ann_pct']:5.2f}%  "
          f"Sharpe {row['sharpe']:+5.2f} [{row['sharpe_lo']:+.2f},{row['sharpe_hi']:+.2f}]  "
          f"HAC t {row['t_hac']:+6.2f}  maxDD {row['maxdd_pct']:+6.2f}%  "
          f"fracNeg {row['frac_neg']:.3f}")

print("\n# JAAA carry stats (the headline leg)")
j = st.carry_stats(ret, "JAAA", cash=data.CASH)
print(f"  JAAA excess-of-cash: {j['excess_ann_pct']:+.2f}%/yr on {j['vol_ann_pct']:.2f}% vol "
      f"-> Sharpe {j['sharpe']:+.2f} (95% CI [{j['sharpe_lo']:+.2f},{j['sharpe_hi']:+.2f}], "
      f"bootstrap frac<0 = {j['frac_neg']:.3f})")
print(f"  daily excess {j['excess_bps_day']:+.2f} bps  HAC t = {j['t_hac']:+.2f}  "
      f"(one-sample t = {j['t_1s']:+.2f})  maxDD {j['maxdd_pct']:+.2f}%  n={j['n']}")

print("\n# HEAD-TO-HEAD — JAAA excess minus each benchmark excess (== JAAA - bench, common window)")
for b in ("LQD", "IEF", "BKLN"):
    h = st.head_to_head(ret, "JAAA", b, cash=data.CASH)
    print(f"  JAAA - {b:<4s} {h['start']}->{h['end']} n={h['n']:>4d}: "
          f"{h['diff_ann_pct']:+6.2f}%/yr  HAC t {h['t_hac']:+5.2f}  spreadSharpe {h['sharpe']:+.2f}")

print("\n# ERA CUT — ZIRP (<=2022-06, rates ~0) vs high-rate plateau (2023+)")
for tag, s, e in [("ZIRP  ", None, ZIRP_END), ("HighRt", HIGH_START, None)]:
    r = st.carry_stats(ret[(ret.index <= pd.Timestamp(e)) if e else (ret.index >= pd.Timestamp(s))],
                       "JAAA", cash=data.CASH)
    print(f"  {tag} {r.get('start','-')}->{r.get('end','-')} n={r.get('n',0):>4d}: "
          f"JAAA excess {r.get('excess_ann_pct',float('nan')):+.2f}%/yr  Sharpe {r.get('sharpe',float('nan')):+.2f}  "
          f"HAC t {r.get('t_hac',float('nan')):+.2f}")

print("\n# TRADABILITY (a) — buy-and-hold JAAA funded by cash, charge ETF bid-ask on rebalances")
print("  (JAAA ER ~0.20%/yr is ALREADY inside the total-return NAV; extra = spread x turnover)")
for reb in (1.0, 12.0):
    c = st.costed_carry(ret, "JAAA", cash=data.CASH, spread_bps_oneway=3.0, rebalances_per_year=reb)
    print(f"  {reb:>4.0f} rebal/yr: gross {c['gross_ann_pct']:+.2f}%/yr - charge "
          f"{c['charge_ann_pct']:.2f}% -> net {c['net_ann_pct']:+.2f}%/yr  "
          f"Sharpe {c['net_sharpe']:+.2f}  HAC t {c['t_net_hac']:+.2f}")

print("\n# TRADABILITY (b) — relative isolation: long JAAA / short LQD, borrow + spread both legs")
rel = st.relative_trade(ret, "JAAA", "LQD", borrow_annual_bps=40.0)
print(f"  gross {rel['gross_ann_pct']:+.2f}%/yr - charge {rel['charge_ann_pct']:.2f}% -> "
      f"net {rel['net_ann_pct']:+.2f}%/yr  Sharpe {rel['net_sharpe']:+.2f}  HAC t {rel['t_net_hac']:+.2f}  "
      f"(also SHORT ~8y duration — a rate bet, not pure carry)")

print("\n# SYNTHETIC CONTROL — deterministic, no network (machinery proof, never market evidence)")
null_s, null_t, null_neg = [], [], []
for sd in range(12):
    w = data.synthetic_world(carry_annual=0.0, seed=888 + sd)
    d = st.synthetic_detect(w)
    null_s.append(d["sharpe"]); null_t.append(d["t_hac"]); null_neg.append(d["frac_neg"])
null_s, null_t = np.asarray(null_s), np.asarray(null_t)
print(f"  null (carry=0), 12 seeds: excess Sharpe mean {null_s.mean():+.2f} (sd {null_s.std(ddof=1):.2f}), "
      f"HAC |t|>=2 in {(np.abs(null_t)>=2).sum()}/12")
wp = data.synthetic_world(carry_annual=0.012, seed=888)
dp = st.synthetic_detect(wp)
print(f"  planted (+1.2%/yr carry): excess {dp['excess_ann_pct']:+.2f}%/yr  Sharpe {dp['sharpe']:+.2f}  "
      f"HAC t {dp['t_hac']:+.2f}  (95% CI [{dp['sharpe_lo']:+.2f},{dp['sharpe_hi']:+.2f}])")
