"""Reproducible headline run for Study 893 — Vol-Target 60/40.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached total-return closes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from vt6040 import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Vol-Target 60/40 — does a volatility thermostat improve the balanced book?")

if not data.have_real():
    print("(cache miss — fetching SPY/IEF/AGG/BIL total-return closes once)")
    data.fetch()

px = data.load_prices()
ret = st.to_returns(px)
cash = ret[data.CASH_TICKER]
print(f"[data] {px.shape[1]} tickers, {len(ret)} return rows  "
      f"{ret.index.min().date()} -> {ret.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(prices)={fingerprint(px)}")
print("  SHORT HISTORY: BIL (cash proxy) lists 2007-05-30 and bounds the joint window — a "
      "young-ETF caveat named on the Signal axis.")

r = st.race(ret)   # matched-risk gross overlay, 21-day window
print(f"\n# THE RACE — static 60/40 vs vol-targeted 60/40 (matched avg risk, target "
      f"{r['target_vol']*100:.2f}%/yr, 21-day window, gross)")
for tag, s in [("static  ", r["static"]), ("vol-tgt ", r["vol_target"])]:
    print(f"  {tag}: excess-Sharpe {s['sharpe']:.3f}  CAGR {s['cagr']*100:5.2f}%  "
          f"vol {s['vol']*100:5.2f}%  maxDD {s['max_dd']*100:6.1f}%")
print(f"  Sharpe gain (excess-vs-excess): {r['sharpe_gain']:+.3f}")
print(f"  spanning alpha (leverage-clean): {r['alpha_ann']*100:+.2f}%/yr  "
      f"HAC t = {r['t_alpha']:+.2f}  (beta {r['alpha_beta']:.3f})")
print(f"  raw return-diff HAC t = {r['diff_t_nw']:+.2f}  (inflated by the level tilt — see notes)")
print(f"  leverage: avg {r['avg_leverage']:.2f}x, levered {r['frac_levered']*100:.0f}% of days, "
      f"capped {r['frac_capped']*100:.0f}%, overlay turnover {r['turnover_ann']:.1f}x/yr")

bs = st.bootstrap_sharpe_diff(r["vt_net"], r["static_net"], cash, n_boot=2000)
print(f"\n# BOOTSTRAP — circular block CI on the excess Sharpe difference (2,000 resamples)")
print(f"  gain {bs['point']:+.3f}  95% CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]  "
      f"P(vt wins) = {bs['frac_a_wins']*100:.1f}%")

print("\n# ROBUSTNESS — two eras (split 2015-01-01)")
for lo, hi, lbl in [("2007-05-31", "2015-01-01", "2007-2014"),
                    ("2015-01-01", "2026-07-01", "2015-2026")]:
    sub = ret[(ret.index >= lo) & (ret.index < hi)]
    rr = st.race(sub)
    print(f"  {lbl}: n={rr['n_days']}  gain {rr['sharpe_gain']:+.3f}  alpha-t {rr['t_alpha']:+.2f}  "
          f"maxDD {rr['dd_static']*100:.1f}% -> {rr['dd_vt']*100:.1f}%")

print("\n# WINDOW ROBUSTNESS — no magic point (gross, matched risk)")
for w in (21, 42, 63):
    rr = st.race(ret, window=w)
    print(f"  {w:2d}-day: gain {rr['sharpe_gain']:+.3f}  alpha-t {rr['t_alpha']:+.2f}  "
          f"turnover {rr['turnover_ann']:.1f}x/yr  maxDD -> {rr['dd_vt']*100:.1f}%")

print("\n# COSTED — one-way bps on overlay turnover + 50 bps/yr borrow on the levered fraction")
sweep = st.cost_sweep(ret, window=21, borrow_bps_yr=50.0)
for c, row in sweep.iterrows():
    print(f"  {c:>4.0f} bp: static {row['static_sharpe']:.3f}  vt {row['vt_sharpe']:.3f}  "
          f"gain {row['sharpe_gain']:+.3f}  maxDD(vt) {row['dd_vt']*100:.1f}%")

print("\n# CRASH-YEAR DRAWDOWN — the robust half (calendar-year total return)")
cy_s = st.calendar_year_returns(r["static_net"])
cy_v = st.calendar_year_returns(r["vt_net"])
for yr in (2008, 2022):
    print(f"  {yr}: static {cy_s.loc[yr]*100:+.1f}%  ->  vol-target {cy_v.loc[yr]*100:+.1f}%")

print("\n# SYNTHETIC CONTROL — the machinery is unbiased (offline, no network)")
null_t = np.array([st.synthetic_detect(data.synthetic_prices(seed=893 + s, sigma_hi=0.006)[0])["t_alpha"]
                   for s in range(30)])
plan_t = np.array([st.synthetic_detect(data.synthetic_prices(seed=893 + s)[0])["t_alpha"]
                   for s in range(30)])
print(f"  null  (flat vol),  30 seeds: alpha-t mean {null_t.mean():+.2f}  |t|>=2 in "
      f"{(abs(null_t) >= 2).sum()}/30")
print(f"  planted (clustered), 30 seeds: alpha-t mean {plan_t.mean():+.2f}  t>=2 in "
      f"{(plan_t >= 2).sum()}/30")
print("  (A faithful-engine / power check only — never cited in support of the real-tape stamp.)")
