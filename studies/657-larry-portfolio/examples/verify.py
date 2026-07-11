"""Reproducible headline run for Study 657 — Larry Portfolio.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached IJS/IEF/SPY/SHY tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp, fingerprint  # noqa: E402

from larry_portfolio import data, strategy as st  # noqa: E402

COST_BPS = 2.0

print("# Larry Portfolio — does 30% small-value / 70% bonds match a 60/40 at much less risk?")

if not data.have_real():
    print("(cache miss — fetching IJS / IEF / SPY / SHY once)")
    data.fetch()

px = data.load_real()
print(data_stamp("IJS/IEF/SPY/SHY total-return panel (joint)", px, asof=data.AS_OF))
for tk in data.TICKERS:
    print(f"       {tk:>4}  fingerprint={fingerprint(px[[tk]])}")
print(f"joint window: {px.index[0].date()} -> {px.index[-1].date()}  ({len(px):,} rows)")

rets = st.to_returns(px)
rf = rets[data.CASH]

larry = st.rebalanced_blend(rets, {"IJS": 0.30, "IEF": 0.70},
                            rebalance="annual", cost_bps=COST_BPS)
sixty = st.rebalanced_blend(rets, {"SPY": 0.60, "IEF": 0.40},
                            rebalance="annual", cost_bps=COST_BPS)
spy100 = st.single_asset(rets, "SPY")
ijs100 = st.single_asset(rets, "IJS")
ief100 = st.single_asset(rets, "IEF")

s_larry = st.stats(larry, rf=rf)
s_sixty = st.stats(sixty, rf=rf)
s_spy = st.stats(spy100, rf=rf)
s_ijs = st.stats(ijs100, rf=rf)
s_ief = st.stats(ief100, rf=rf)


def _row(name, d):
    print(f"  {name:<28} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
          f"Sharpe(xs) {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%")


print(f"\n# THE HEADLINE — annual rebalance, {COST_BPS:.0f} bps/rebalance, "
      f"{data.START} -> {data.AS_OF}")
_row("Larry (30% IJS / 70% IEF)", s_larry)
_row("60/40 (60% SPY / 40% IEF)", s_sixty)
_row("100% stocks (SPY)", s_spy)
_row("100% small-value (IJS)", s_ijs)
_row("100% bonds (IEF)", s_ief)

print("\n# Does Larry MATCH 60/40's return? (bootstrap CI, circular block, 21d, 2000 reps)")
boot_ret = st.bootstrap_diff(larry, sixty, rf=None, metric="mean", n_boot=2000, seed=657)
print(f"  CAGR gap (Larry - 60/40): {(s_larry['cagr']-s_sixty['cagr'])*100:+.2f} pts/yr")
print(f"  mean-return diff point={boot_ret['point']*100:+.2f}%/yr  "
      f"95% CI [{boot_ret['ci95'][0]*100:+.2f}%, {boot_ret['ci95'][1]*100:+.2f}%]  "
      f"(CI includes 0: {boot_ret['ci95'][0] <= 0 <= boot_ret['ci95'][1]})")
diff_daily = (larry - sixty).dropna().to_numpy()
t_ret = st.hac_tstat(diff_daily)
print(f"  HAC t on daily (Larry - 60/40) return difference: {t_ret:+.2f}")

print("\n# Is the RISK-ADJUSTED (Sharpe, excess-of-cash) edge real?")
print("  (a distinct question from the raw return gap above — a ratio-level bootstrap, not")
print("   the same HAC t on the mean, since excess-of-cash cancels out of a same-two-arm diff)")
boot_sh = st.bootstrap_diff(larry, sixty, rf=rf, metric="sharpe", n_boot=2000, seed=657)
print(f"  Sharpe(xs) gap (Larry - 60/40): {s_larry['sharpe']-s_sixty['sharpe']:+.3f}")
print(f"  Bootstrap Sharpe-diff point={boot_sh['point']:+.3f}  "
      f"95% CI [{boot_sh['ci95'][0]:+.3f}, {boot_sh['ci95'][1]:+.3f}]  "
      f"(CI includes 0: {boot_sh['ci95'][0] <= 0 <= boot_sh['ci95'][1]})  "
      f"Larry wins {boot_sh['frac_a_wins']*100:.0f}% of resamples")

print("\n# Equity-risk claim — vol / drawdown / correlation to a pure-equity crash")
corr_larry_spy = larry.corr(spy100)
corr_sixty_spy = sixty.corr(spy100)
print(f"  vol:  Larry {s_larry['vol']*100:.1f}%  vs 60/40 {s_sixty['vol']*100:.1f}%  "
      f"vs 100% SPY {s_spy['vol']*100:.1f}%")
print(f"  maxDD: Larry {s_larry['max_dd']*100:.1f}%  vs 60/40 {s_sixty['max_dd']*100:.1f}%  "
      f"vs 100% SPY {s_spy['max_dd']*100:.1f}%")
print(f"  daily corr to SPY: Larry {corr_larry_spy:.2f}  vs 60/40 {corr_sixty_spy:.2f}")

print("\n# THIRD AXIS — has the small-value premium (IJS - SPY) itself decayed?")
spread = st.premium_series(rets)
ps = st.premium_stats(spread)
print(f"  whole sample {data.START[:4]}-{data.AS_OF[:4]}: {ps['ann_pct']:+.2f}%/yr, "
      f"HAC t = {ps['hac_t']:+.2f}  (n={ps['n']:,})")
ec = st.era_contrast(spread, data.DECAY_SPLIT)
print(f"  {data.START[:4]} -> {data.DECAY_SPLIT[:4]}: {ec['early_ann_pct']:+.2f}%/yr "
      f"(n={ec['n_early']:,}, HAC t = {ec['hac_t_early']:+.2f})")
print(f"  {data.DECAY_SPLIT[:4]} -> {data.AS_OF[:4]}: {ec['late_ann_pct']:+.2f}%/yr "
      f"(n={ec['n_late']:,}, HAC t = {ec['hac_t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['welch_t_diff']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the premium-detection statistic (HAC t on the SV-MKT spread) must NOT fire on a")
print("  null (premium=0, CAPM-neutral beta=1) world, and must recover a planted premium.")
print("  Null checked over 20 seeds (never a single stream).")
null_ts = []
null_gaps = []
for s_ in range(20):
    panel = data.synthetic_world(premium=0.0, seed=657 + s_)
    d = st.synthetic_detect(panel)
    null_ts.append(d["premium_hac_t"])
    null_gaps.append(d["cagr_gap"])
null_ts = np.asarray(null_ts)
null_gaps = np.asarray(null_gaps)
print(f"  null (premium=0), 20 seeds: mean premium HAC t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
print(f"  (at premium=0, Larry still trails 60/40 by {null_gaps.mean()*100:+.2f} pts/yr CAGR on "
      "average — the structural cost of running less equity beta, not a detector failure)")
panel = data.synthetic_world(premium=0.05, seed=657)
sy = st.synthetic_detect(panel)
print(f"  planted premium=+5.0%/yr (seed 657): premium HAC t = "
      f"{sy['premium_hac_t']:+.2f}  ({sy['premium_ann_pct']:+.2f}%/yr recovered)  "
      f"CAGR gap Larry-60/40 = {sy['cagr_gap']*100:+.2f} pts/yr  "
      f"(Larry Sharpe {sy['larry_sharpe']:.3f} vs 60/40 {sy['sixty_sharpe']:.3f})")
