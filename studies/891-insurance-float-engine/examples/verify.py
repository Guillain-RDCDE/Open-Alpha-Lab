"""Reproducible headline run for Study 891 — Insurance Float Engine.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/insfloat_prices.parquet, built once from yfinance) and always-offline on the
synthetic control.

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

from quantlab.repro import data_stamp, fingerprint  # noqa: E402

from insurance_float import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

ERAS = [
    ("2007-06-01", "2009-12-31", "GFC 2007-09"),
    ("2010-01-01", "2015-12-31", "2010-15"),
    ("2016-01-01", "2020-12-31", "2016-20"),
    ("2021-01-01", "2026-06-30", "2021-26"),
    ("2010-01-01", "2026-06-30", "post-GFC 2010+"),
]

print("# Insurance Float Engine — does a P&C insurer basket beat the market on the float?")
if not data.have_real():
    print("no real cache; run insurance_float.data.fetch() once (needs network). Aborting.")
    sys.exit(0)

prices = data.load_prices()
ret = data.monthly_returns(prices, asof=data.AS_OF)
print(data_stamp("monthly total returns", ret, cols=data.TICKERS, asof=data.AS_OF))
print(f"tape: {prices.index.min().date()} -> {prices.index.max().date()} daily; "
      f"{len(ret)} complete months {ret.index.min().date()} -> {ret.index.max().date()}")
print(f"Fingerprint {fingerprint(ret, cols=data.TICKERS)}")

print("\n# Descriptives (monthly total returns, excess Sharpe vs BIL cash)")
for c in ["KIE", "IAK", "SPY", "KBE"]:
    s = st.ann_stats(ret, c)
    print(f"  {c}: CAGR {s['cagr_pct']:6.2f}%  vol {s['vol_pct']:6.2f}%  "
          f"exSharpe {s['sharpe']:+.3f}  maxDD {s['maxdd_pct']:6.1f}%")

print("\n# Excess-vs-excess Sharpe race (both legs minus BIL) + HAC t on the return diff")
for ins in data.INSURERS:
    r = st.sharpe_race(ret, ins)
    print(f"  {ins} vs SPY [{r['start']}->{r['end']}, n={r['n']}]: "
          f"exSharpe {r['sharpe_ins']:+.3f} vs SPY {r['sharpe_mkt']:+.3f}  "
          f"advantage {r['advantage']:+.3f}  |  {ins}-SPY {r['diff_ann_pct']:+.2f}%/yr "
          f"(HAC t={r['t_diff']:+.2f})")

print("\n# Bootstrap 95% CI on the excess Sharpe (circular block bootstrap)")
for c in ["KIE", "IAK", "SPY", "KBE"]:
    b = st.bootstrap_sharpe(ret, c)
    print(f"  {c}: exSharpe {b['sharpe']:+.3f}  CI[{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]  "
          f"P(Sharpe<0)={b['frac_negative']:.2f}")

print("\n# CAPM decomposition — insurer excess on market excess (does alpha survive?)")
for ins in data.INSURERS:
    c = st.capm(ret, ins)
    print(f"  {ins}: alpha {c['alpha_ann_pct']:+.2f}%/yr (HAC t={c['t_alpha']:+.2f})  "
          f"beta {c['beta']:.3f} (t vs 1 = {c['t_beta_vs1']:+.2f})  R2={c['r2']:.3f}")

print("\n# Two-factor decomposition — insurer excess on [market, (bank - market) spread]")
print("#   the decisive test: is the 'float premium' just financial-sector beta?")
for ins in data.INSURERS:
    t = st.decompose_financials(ret, ins)
    print(f"  {ins}: alpha {t['alpha_ann_pct']:+.2f}%/yr (HAC t={t['t_alpha']:+.2f})  "
          f"beta_mkt {t['beta_mkt']:.3f}  load_bank {t['load_bank']:+.3f} "
          f"(t={t['t_load_bank']:+.2f})  R2={t['r2']:.3f}")

print("\n# Within-financials — insurer minus bank (a DIFFERENT claim; not vs the market)")
for ins in data.INSURERS:
    v = st.insurer_vs_bank(ret, ins)
    print(f"  {ins}-KBE: {v['diff_ann_pct']:+.2f}%/yr (HAC t={v['t_diff']:+.2f})")

print("\n# Era cut — the structural story should hold across eras, not just in the crash")
et = st.era_table(ret, ERAS, "KIE")
for tag, row in et.iterrows():
    print(f"  {tag:16s} n={int(row['n']):3d}  KIE {row['sharpe_ins']:+.3f}  "
          f"SPY {row['sharpe_mkt']:+.3f}  KBE {row['sharpe_bank']:+.3f}  "
          f"| KIE-SPY adv {row['advantage']:+.3f}")

print("\n# Tradability (a) one-month-lag rotation: own KIE when it's beaten SPY over 12m")
rot = st.rotation_strategy(ret, "KIE")
print(f"  net {rot['net_ann_pct']:+.2f}%/yr (exSharpe {rot['sharpe_net']:+.3f})  vs "
      f"always-SPY {rot['always_mkt_ann_pct']:+.2f}%  always-KIE {rot['always_ins_ann_pct']:+.2f}%  "
      f"({rot['switches']} switches, KIE {rot['share_ins']*100:.0f}% of the time)")

print("# Tradability (b) isolation trade: long KIE / short SPY, borrow 40 bps + 5 bps x 1/yr")
iso = st.isolation_trade(ret, "KIE")
print(f"  gross {iso['gross_ann_pct']:+.2f}%/yr (t={iso['t_gross']:+.2f}) -> "
      f"net {iso['net_ann_pct']:+.2f}%/yr (t={iso['t_net']:+.2f})  charges {iso['charge_ann_pct']:.2f}%/yr")

print("\n# Calendar-year total returns (%)")
cy = st.calendar_year_table(ret, ["KIE", "IAK", "SPY", "KBE"])
print(cy.to_string())

print("\n# Synthetic control — deterministic, no network (machinery proof, never evidence)")
print("  planted float edge must be recovered; the zero-edge null must NOT fire.")
for edge in (0.0, 0.04):
    d = st.synthetic_detect(data.synthetic_world(edge_ann=edge, seed=891, n_months=240))
    print(f"  planted {edge*100:+4.1f}%/yr: Sharpe adv {d['advantage']:+.3f}  "
          f"CAPM alpha {d['capm_alpha_ann_pct']:+.2f}%/yr (t={d['capm_t_alpha']:+.2f})  "
          f"two-factor alpha {d['two_alpha_ann_pct']:+.2f}%/yr (t={d['two_t_alpha']:+.2f})  "
          f"bank load {d['load_bank']:+.3f} (t={d['t_load_bank']:+.2f})")
