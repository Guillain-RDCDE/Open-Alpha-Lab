"""Reproducible headline run for Study 908 — Optimized-Roll Commodities.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/roll_prices.csv, built once from yfinance) and always-offline on the synthetic
control.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from opt_roll import data, strategy as st
from quantlab import repro

BENCH = ["DBC", "GSG", "DJP"]
ERAS = [
    ("deep-contango 2010-2015", "2010-09", "2015-12"),
    ("recovery 2016-2020", "2016-01", "2020-12"),
    ("backwardation 2021-2026", "2021-01", "2026-06"),
]

print("# Optimized-Roll Commodities — USCI vs DBC / GSG / DJP (yfinance, total-return, excess of BIL)")
prices = data.load_prices()
rets = data.monthly_returns(prices, asof=data.AS_OF)
ex = st.excess_frame(rets, cash=data.CASH)

common = st.common_sample(ex, ["USCI"] + BENCH)
print(repro.data_stamp("excess-of-cash monthly panel", common,
                       cols=["USCI"] + BENCH, asof=data.AS_OF))
print(f"tape: {prices.index.min().date()} -> {prices.index.max().date()} daily; "
      f"common excess window {common.index.min().date()} -> {common.index.max().date()} "
      f"({len(common)} months), USCI inception 2010-08-10 gates the start")

print("\n# Full sample — excess-of-cash Sharpe race (optimized USCI vs each benchmark)")
print("  (Sharpe adv = SR(USCI) - SR(bench); paired block-bootstrap 95% CI on the advantage;")
print("   HAC/Newey-West t on the monthly return difference USCI - bench, 6 lags)")
for b in BENCH:
    r = st.sharpe_race(ex, "USCI", b)
    print(f"  USCI vs {b:<4s} n={r['n']}  ex-ret USCI {r['ann_ex_opt']:+.2f}% (vol {r['vol_opt']:.1f}%) "
          f"Sharpe {r['sharpe_opt']:+.3f}  |  {b} {r['ann_ex_bench']:+.2f}% (vol {r['vol_bench']:.1f}%) "
          f"Sharpe {r['sharpe_bench']:+.3f}")
    print(f"           Sharpe adv {r['sharpe_adv']:+.3f}  CI[{r['adv_ci_lo']:+.3f}, {r['adv_ci_hi']:+.3f}]  "
          f"frac<=0 {r['adv_frac_le0']:.3f}  |  diff {r['diff_ann_pct']:+.2f}%/yr  HAC t {r['t_diff']:+.2f}")

print("\n# Era cut — the edge is NOT era-robust (it inverts in 2016-2020)")
for b in BENCH:
    print(f"  -- USCI vs {b}")
    for e in st.era_race(ex, "USCI", b, ERAS):
        print(f"     {e['era']:<26s} n={e['n']:>3d}  diff {e['diff_ann_pct']:+6.2f}%/yr  "
              f"HAC t {e['t_diff']:+.2f}  Sharpe adv {e['sharpe_adv']:+.3f}")

print("\n# Costed — total returns already net each fund's expense ratio (USCI 1.03% vs GSG 0.48%);")
print("#  this adds an incremental bid-ask charge on reconstitution turnover (USCI the wider spread)")
for b in BENCH:
    c = st.costed_race(ex, "USCI", b)
    print(f"  USCI vs {b:<4s} net Sharpe adv {c['sharpe_adv_net']:+.3f}  diff_net {c['diff_ann_pct_net']:+.2f}%/yr "
          f"(HAC t {c['t_diff_net']:+.2f})  charges USCI {c['charge_opt_ann_pct']:.2f}% vs {b} {c['charge_bench_ann_pct']:.2f}%/yr")

print("\n# Corroborating optimized cousin — PDBC (2014-11+) vs front-month GSG")
r = st.sharpe_race(ex, "PDBC", "GSG")
print(f"  PDBC vs GSG n={r['n']} {r['start'][:10]}->{r['end'][:10]}  Sharpe adv {r['sharpe_adv']:+.3f} "
      f"CI[{r['adv_ci_lo']:+.3f}, {r['adv_ci_hi']:+.3f}]  diff {r['diff_ann_pct']:+.2f}%/yr  HAC t {r['t_diff']:+.2f}")

print("\n# Drawdown & total-return context (2010-09 -> 2026-06)")
for c in ["USCI"] + BENCH:
    s = rets[c].dropna()
    s = s[s.index >= common.index.min()]
    ann = (np.prod(1.0 + s.to_numpy()) ** (12.0 / len(s)) - 1.0) * 100
    print(f"  {c:<4s} total-return {ann:+.2f}%/yr  max drawdown {st.max_drawdown(s.to_numpy()):.1f}%  "
          f"(expense ratio {data.EXPENSE_RATIOS[c]:.2f}%)")

print("\n# Synthetic control — deterministic, no network (machinery proof, never market evidence)")
print("  planted roll-edge world: the race must recover a positive Sharpe advantage when")
print("  roll_edge_annual > 0 and NOTHING when it is 0 (the null).")
for edge in (0.0, 0.03):
    w = data.synthetic_world(roll_edge_annual=edge, seed=908)
    d = st.synthetic_detect(w)
    print(f"  planted {edge*100:+4.1f}%/yr: Sharpe adv {d['sharpe_adv']:+.3f}  "
          f"CI[{d['adv_ci_lo']:+.3f}, {d['adv_ci_hi']:+.3f}]  diff {d['diff_ann_pct']:+.2f}%/yr  HAC t {d['t_diff']:+.2f}")

print(f"\nFingerprint {repro.fingerprint(common, cols=['USCI'] + BENCH)}")
