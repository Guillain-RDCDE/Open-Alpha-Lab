"""Reproducible headline run for Study 664 — ESG Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ESGU/SUSA/SPY/IVV/IVW/IVE/QUAL/
^IRX tapes under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp, fingerprint  # noqa: E402

from esg_premium import data, strategy as st  # noqa: E402

print("# ESG Premium — does 'doing well by doing good' pay, or is it a relabelled growth tilt?")

if not data.have_real():
    print("(cache miss - fetching ESGU/SUSA/SPY/IVV/IVW/IVE/QUAL/^IRX once)")
    data.fetch()

tapes = data.load_real()
for t in data.TICKERS:
    print(data_stamp(t, tapes[t], asof=data.AS_OF))

closes = {t: tapes[t]["Close"] for t in data.TICKERS}
rets = {t: st.daily_returns(closes[t]) for t in data.TICKERS if t != "^IRX"}
rf = st.rf_daily(closes["^IRX"].reindex(rets["SPY"].index).ffill())

print("\n# Fund facts (hardcoded, no network)")
for t in ["ESGU", "SUSA", "SPY", "IVV", "IVW", "IVE", "QUAL"]:
    f = data.FUND_FACTS[t]
    print(f"  {t:>5}: {f['name']:<38} inception {f['inception']}  "
          f"expense ratio {f['expense_ratio']*100:.3f}%  ({f['role']})")

print("\n# THE HEADLINE — tracking difference & Sharpe, each ESG fund vs its plain-vanilla peer")
for fund, bench in data.PAIRS.items():
    ts = st.tracking_stats(rets[fund], rets[bench], rf)
    print(f"\n  {fund} vs {bench}  [{ts['start']} -> {ts['end']}, n={ts['n']:,} days, "
          f"{ts['years']:.1f}y]")
    print(f"    CAGR           : {fund} {ts['fund_cagr']*100:+.2f}%  vs  "
          f"{bench} {ts['bench_cagr']*100:+.2f}%")
    print(f"    Ann. vol       : {fund} {ts['fund_vol']*100:.2f}%  vs  "
          f"{bench} {ts['bench_vol']*100:.2f}%")
    print(f"    Excess-of-cash Sharpe : {fund} {ts['fund_sharpe_xs']:+.3f}  vs  "
          f"{bench} {ts['bench_sharpe_xs']:+.3f}")
    print(f"    Tracking error (ann.) : {ts['tracking_error']*100:.2f}%   "
          f"Information ratio: {ts['information_ratio']:+.3f}")
    print(f"    Active return (ann.)  : {ts['active_mean_ann']*100:+.3f}%")

print("\n# ACTIVE-RETURN SPREAD TEST — long fund / short benchmark, daily")
spread = {}
for fund, bench in data.PAIRS.items():
    sp = st.spread_test(rets[fund], rets[bench])
    spread[fund] = sp
    print(f"\n  {fund} - {bench}  (n={sp['n']:,})")
    print(f"    mean active return: {sp['mean_daily_bps']:+.3f} bps/day  "
          f"({sp['gross_ann_pct']:+.3f}%/yr gross)")
    print(f"    Newey-West t (5 lags, PRIMARY): {sp['nw_t']:+.2f}   "
          f"Welch t (cross-check): {sp['welch_t']:+.2f}")
    print(f"    net of costs (2 legs x 5bps one-way + 30bps/yr borrow): "
          f"{sp['net_ann_pct']:+.3f}%/yr")
    print(f"    hit rate (days fund > bench): {sp['hit_rate']*100:.1f}%")

print("\n# FACTOR DECOMPOSITION — is the gap just a growth/value or quality tilt?")
gv_spread = rets["IVW"] - rets["IVE"]
q_spread = rets["QUAL"] - rets["SPY"]
factor = {}
for fund, bench in data.PAIRS.items():
    fac = st.raw_vs_factor_alpha(rets[fund], rets[bench], rets[bench], gv_spread, q_spread)
    factor[fund] = fac
    print(f"\n  {fund} (benchmark {bench} as the market factor):")
    print(f"    raw active return          : {fac['raw_ann_pct']:+.3f}%/yr   "
          f"NW t = {fac['raw_t']:+.2f}")
    print(f"    factor-model alpha         : {fac['factor_alpha_ann_pct']:+.3f}%/yr   "
          f"NW t = {fac['factor_alpha_t']:+.2f}")
    print(f"    growth-value beta (IVW-IVE): {fac['beta_growth_value']:+.3f}   "
          f"t = {fac['beta_growth_value_t']:+.2f}")
    print(f"    quality beta (QUAL-{bench})   : {fac['beta_quality']:+.3f}   "
          f"t = {fac['beta_quality_t']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the NW detector must NOT fire on a null world (premium=0) and must recover a")
print("  planted premium. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    w = data.synthetic_world(premium_bps=0.0, seed=664 + s_)
    null_ts.append(st.synthetic_detect(w)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (premium=0), 20 seeds: mean NW t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
w = data.synthetic_world(premium_bps=5.0, seed=664)
sy = st.synthetic_detect(w)
print(f"  planted premium=+5.0 bps/day (seed 664): mean active {sy['mean']*1e4:+.3f} bps/day  "
      f"NW t = {sy['t']:+.2f}")

fp_esgu = fingerprint((rets["ESGU"] - rets["SPY"]).to_frame("active").dropna())
fp_susa = fingerprint((rets["SUSA"] - rets["IVV"]).to_frame("active").dropna())
print(f"\n[data] active-return fingerprints: ESGU-SPY={fp_esgu}  SUSA-IVV={fp_susa}")
