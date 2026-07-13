"""Reproducible headline run for Study 748 — CEO-Age-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached monthly total-return tape
under ``_cache/`` if present (the real long/short), and always runs the synthetic
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ceo_age_effect import data, strategy as st

print("# CEO-Age-Effect — long-young / short-old, HAC t and CAPM alpha-vs-beta")

ages = data.curated_ages()
ny = int((ages["bucket"] == "young").sum())
no = int((ages["bucket"] == "old").sum())
print(f"curated table: {len(ages)} CEOs ({ny} young < {data.YOUNG_MAX_AGE}, {no} old), "
      f"scored {data.SCORE_DATE}, table fp {data.fingerprint(ages)}")

if data.have_real():
    px = data.fetch_prices()
    ret = data.build_returns(px)
    print(f"tape: {len(ret)} months {ret.index.min().date()} -> {ret.index.max().date()} "
          f"| price fp {data.fingerprint(px)} | L/S panel fp {data.fingerprint(ret)}")

    bs = st.bucket_stats(ret)
    print("\n# Baskets (annualised, monthly total return)")
    for k in ("young", "old", "mkt"):
        b = bs[k]
        print(f"  {k:>6}: ret {b['ann_ret']*100:+7.2f}%/yr  vol {b['ann_vol']*100:6.2f}%  "
              f"Sharpe {b['sharpe']:+.2f}")

    h = bs["ls"]["hac"]
    print("\n# H1 - the long/short premium (Newey-West HAC)")
    print(f"  L/S mean {h['mean']*100:+.3f}%/mo (ann {bs['ls']['ann_ret']*100:+.2f}%/yr)  "
          f"HAC t = {h['t']:+.2f} ({h['lags']} lags, n {h['n']})")
    pl = st.placebo_pvalue(px, data, n_perm=2000)
    print(f"  label-shuffle placebo: obs |t| {pl['obs_t']:.2f}, p = {pl['p']:.3f}")

    ca = st.capm_alpha(ret["ls"].to_numpy(), ret["mkt"].to_numpy())
    print("\n# H2 - alpha vs beta (CAPM, HAC covariance) -- the decisive control")
    print(f"  alpha {ca['alpha']*12*100:+.2f}%/yr (t {ca['alpha_t']:+.2f})  "
          f"market beta {ca['beta']:+.2f} (t {ca['beta_t']:+.1f})  R2 {ca['r2']:.2f}")

    print("\n# H3 - robustness: cutoff sweep")
    print(st.cutoff_sweep(data, px, [50, 55, 60]).round(3).to_string())
    print("\n# H3 - robustness: sub-period sweep (sign flips by regime)")
    splits = [("2018-2020", "2018-01-01", "2020-12-31"),
              ("2021-2022", "2021-01-01", "2022-12-31"),
              ("2023-2026", "2023-01-01", "2026-06-30")]
    print(st.subperiod_sweep(ret, splits).round(3).to_string())

    print("\n# Execution lag (membership is calendar-known -> lag is immaterial)")
    for lag in (0, 1):
        ls = st.lag_returns(ret["ls"], lag)
        hh = st.hac_mean_t(ls.to_numpy())
        print(f"  lag {lag}: L/S {ls.mean()*12*100:+.2f}%/yr  HAC t {hh['t']:+.2f}  n {hh['n']}")

    print("\n# Costs + short borrow")
    nc = st.net_of_costs(ret, cost_bps=5.0, borrow_ann_bps=75.0, annual_turnover=0.30)
    print(f"  gross {nc['gross_ann']*100:+.2f}%/yr (Sharpe {nc['gross_sharpe']:.2f})  "
          f"-> net {nc['net_ann']*100:+.2f}%/yr (Sharpe {nc['net_sharpe']:.2f})")
else:
    print("(no _cache - run data.fetch_prices(fetch=True) once to build it)")

print("\n# Synthetic positive control - deterministic, no network (25 seeds)")
print("  young firms carry a higher beta AND a plantable age_alpha; the CAPM alpha t must")
print("  stay ~0 at the null (beta tilt only) and rise past 2 once a real premium is planted.")
for a in (0.0, 0.004, 0.008, 0.012):
    r = st.synthetic_mean_alpha_t(data, age_alpha=a, n_seeds=25)
    print(f"  age_alpha {a*12*100:+5.1f}%/yr: mean CAPM alpha t {r['mean_alpha_t']:+.2f}  "
          f"mean raw HAC t {r['mean_raw_t']:+.2f}")
