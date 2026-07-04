"""Reproducible headline run for Study 618 — GBTC Premium Cycle.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached GBTC + BTC-USD closes under
``_cache/`` (fetched once on a cache miss), and always runs the synthetic control offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from gbtc_premium_cycle import data, strategy as st
from quantlab import repro

df = data.load_real(as_of=data.AS_OF)
print(repro.data_stamp("GBTC + BTC-USD (yfinance daily closes, GBTC trading days)",
                       df, cols=["gbtc", "btc", "prem"], asof=data.AS_OF))

print("\n# The wrapper's mechanics (modeled bitcoin-per-share, sources in data.py)")
import pandas as pd
print(f"  BPS at 91:1 split (2018-01-26): "
      f"{data.btc_per_share(pd.DatetimeIndex(['2018-01-26']))[0] / data.SPINOFF_FACTOR:.7f} "
      f"BTC/share (Grayscale disclosed ~0.00101)")
print(f"  BPS at ETF conversion (2024-01-11): "
      f"{data.btc_per_share(pd.DatetimeIndex(['2024-01-11']))[0] / data.SPINOFF_FACTOR:.8f} "
      f"BTC/share (Grayscale disclosed ~0.00089)")

print("\n# Three regimes — one wrapper (premium level, HAC lag = 63 days on the level)")
for r in st.regime_table(df):
    print(f"  {r['regime']:<26s} {r['start']} -> {r['end']}  n={r['n']:>5d}  "
          f"mean {r['mean_pct']:+7.2f}%  HAC t={r['t_hac']:+6.2f}  "
          f"min {r['min_pct']:+7.2f}% ({r['min_date']})  max {r['max_pct']:+7.2f}% ({r['max_date']})")
print(f"  first sustained discount (5 straight closes < NAV): {st.first_sustained_discount(df)}")
regs = data.regimes(df)
w = st.welch_t(regs["premium era (2015-2021)"]["prem"].to_numpy(),
               regs["discount era (2021-2024)"]["prem"].to_numpy())
print(f"  premium era vs discount era daily premium: Welch t = {w:+.2f}")
etf = regs["ETF era (2024-)"]["prem"]
print(f"  ETF-era calibration residual: mean {etf.mean()*100:+.3f}%  sd {etf.std(ddof=1)*100:.3f}%  "
      f"mean |prem| {etf.abs().mean()*100:.3f}%")

print("\n# The dated trade — the 2023 convergence (long GBTC / short BTC = d log(1+prem))")
full = st.convergence_test(df, "2023-01-03", "2024-01-11", lags=10)
print(f"  full-2023 window   : {full['start']} -> {full['end']}  ({full['n_days']} days)  "
      f"prem {full['prem_entry_pct']:+.2f}% -> {full['prem_exit_pct']:+.2f}%")
print(f"    hedged drift: {full['total_log_pct']:+.2f} log-pts total, "
      f"{full['mean_daily_bps']:+.2f} bps/day, HAC(10) t = {full['t_hac']:+.2f}")
trig = st.convergence_test(df, "2023-06-16", "2024-01-11", lags=10)
print(f"  ex-ante trigger    : enter close 2023-06-16 (ONE trading day after BlackRock's "
      f"2023-06-15 filing), exit conversion close")
print(f"    {trig['start']} -> {trig['end']}  ({trig['n_days']} days)  "
      f"prem {trig['prem_entry_pct']:+.2f}% -> {trig['prem_exit_pct']:+.2f}%")
print(f"    hedged drift: {trig['total_log_pct']:+.2f} log-pts total, "
      f"{trig['mean_daily_bps']:+.2f} bps/day, HAC(10) t = {trig['t_hac']:+.2f}")

print("\n# Costs on the triggered trade (10 bps one-way x 4 legs, short-BTC borrow 5 %/yr)")
cc = st.convergence_costs(df, "2023-06-16", "2024-01-11", cost_bps=10.0, borrow_pa=5.0)
print(f"  gross {cc['gross_log_pct']:+.2f} log-pts - legs {cc['legs_cost_pct']:.2f}pp - "
      f"borrow {cc['borrow_cost_pct']:.2f}pp ({cc['years']:.2f} yrs) = "
      f"net {cc['net_log_pct']:+.2f} log-pts = {cc['net_simple_pct']:+.2f}% simple")
for cb in (25.0, 50.0):
    c2 = st.convergence_costs(df, "2023-06-16", "2024-01-11", cost_bps=cb, borrow_pa=5.0)
    print(f"  at {cb:.0f} bps/leg: net {c2['net_simple_pct']:+.2f}% simple")

print("\n# Event days (one-day hedged move, z vs discount-era daily sd)")
for e in st.event_days(df, sd_window=("2021-02-23", "2024-01-10")):
    print(f"  {e['date']}  {e['move_pct']:+6.2f}%  z={e['z']:+5.1f}  {e['label']}")

print("\n# Third axis — who could harvest each regime? The accredited create-and-dump arb")
coh = st.lockup_arb_cohorts(df)
s = st.lockup_arb_summary(coh)
print(f"  monthly creation cohorts 2015-06 .. 2021-06, 126-td lockup, hedged, net of fee "
      f"drag + 3 legs x 25 bps + 5 %/yr borrow")
print(f"  early cohorts (created < 2020-09): n={s['n_early']}  mean net "
      f"{s['early_mean_pct']:+.2f} log-pts/cohort  worst {s['early_worst_pct']:+.2f}  "
      f"share>0 {s['early_share_pos']*100:.0f}%")
print(f"  blow-up cohorts (created >= 2020-09, exit into the flip): n={s['n_late']}  "
      f"mean net {s['late_mean_pct']:+.2f}  worst {s['late_worst_pct']:+.2f}")
print(f"  early vs late Welch t = {s['welch_t']:+.2f}")

print("\n# Synthetic control — planted three-regime world (machinery proof, never evidence)")
for label, kw in [("planted premium+drift", dict(drift_on=True, premium_on=True)),
                  ("planted premium, NO convergence drift", dict(drift_on=False, premium_on=True)),
                  ("null (premium = noise around 0)", dict(drift_on=False, premium_on=False))]:
    sw = data.synthetic_world(seed=618, **kw)
    c0, c1 = sw.attrs["conv_window"]
    ct = st.convergence_test(sw, str(c0.date()), str(c1.date()), lags=10)
    pr = sw.attrs["regime_slices"]["premium"]
    lvl = sw.loc[pr[0]:pr[1], "prem"].mean() * 100
    print(f"  {label:<42s} plateau mean {lvl:+7.2f}%  conv-window HAC t = {ct['t_hac']:+6.2f}  "
        f"({ct['total_log_pct']:+7.2f} log-pts)")
