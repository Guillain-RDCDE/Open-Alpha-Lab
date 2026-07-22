"""Reproducible headline run for Study 792 — Cross-sectional commodity momentum.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF-basket tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from commodity_momentum import data, strategy as st  # noqa: E402

print("# Cross-sectional commodity momentum — does a 12-1 basket sort pay?")

if not data.have_real():
    print("(cache miss — fetching the ETF basket once)")
    data.fetch()

prices = data.load_prices()
print(data_stamp("commodity ETF basket (total-return closes)", prices, asof=data.AS_OF))

mret = data.monthly_returns(prices)
print(f"monthly return panel: {mret.shape[0]} months x {mret.shape[1]} ETFs "
      f"({mret.index.min().date()} -> {mret.index.max().date()}); "
      f"tickers = {', '.join(mret.columns)}")

print("\n# THE HEADLINE — 12-1 cross-sectional momentum L/S (top third vs bottom third)")
rep = st.momentum_report(mret)
print(f"  L/S months traded : {rep['n_months']} ({rep['start']} -> {rep['end']})")
print(f"  mean monthly      : {rep['mean_bps']:+.2f} bps  (~{rep['ann_pct']:+.2f}%/yr)")
print(f"  HAC/Newey-West t  : {rep['hac_t']:+.3f}  (Bartlett, {rep['hac_lags']} lags)  <-- decisive")
print(f"  one-sample t      : {rep['one_sample_t']:+.3f}")
print(f"  Sharpe (annualised): {rep['sharpe']:+.3f}")
print(f"  hit rate (months>0): {rep['hit_rate']*100:.1f}%   avg one-way turnover: {rep['avg_turnover']:.2f}x NAV")

print("\n# Leg decomposition (excess-of-basket) — where is the P&L?")
legs = st.long_short_legs(mret)
print(f"  winner (long) leg  : {legs['long_excess_bps']:+.2f} bps/mo excess (HAC t = {legs['long_t']:+.2f})")
print(f"  loser (short) leg  : {legs['short_excess_bps']:+.2f} bps/mo excess (HAC t = {legs['short_t']:+.2f})")
print(f"  equal-weight basket: {legs['basket_ann_pct']:+.2f}%/yr")

print(f"\n# Random-rank placebo ({st.PLACEBO_SEEDS} seeds — momentum signal replaced by noise ranks)")
pl = st.random_placebo(mret)
print(f"  real HAC t = {pl['real_t']:+.3f}  vs  placebo mean t {pl['mean_t']:+.3f} (sd {pl['sd_t']:.3f}); "
      f"placebo mean Sharpe {pl['mean_sharpe']:+.3f}")
print(f"  frac of placebos with |t|>=2: {pl['frac_t_ge_2']*100:.0f}%   p(placebo t >= real t) = {pl['p_value']:.3f}")

print(f"\n# Sub-period contrast (split {data.ERA_SPLIT})")
sp = st.subperiod_contrast(mret, data.ERA_SPLIT)
print(f"  early ({sp['n_early']} mo): {sp['early_bps']:+.2f} bps/mo, HAC t = {sp['early_t']:+.2f}, Sharpe {sp['early_sharpe']:+.2f}")
print(f"  late  ({sp['n_late']} mo): {sp['late_bps']:+.2f} bps/mo, HAC t = {sp['late_t']:+.2f}, Sharpe {sp['late_sharpe']:+.2f}")
print(f"  Welch t of the difference (late - early): {sp['welch_t_diff']:+.2f}")

print("\n# THE COSTED TIMER — one-way cost x traded notional; short leg pays 50 bps/yr borrow")
for cb in (5.0, 10.0):
    tm = st.costed_timer(mret, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_ann_pct']:+.2f}%/yr -> net {tm['net_ann_pct']:+.2f}%/yr  "
          f"(net HAC t = {tm['net_t']:+.2f}, net Sharpe {tm['net_sharpe']:.2f}, turnover {tm['avg_turnover']:.2f}x)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the 12-1 sort must NOT fire on a null world (mom_edge=0) and must recover a planted")
print("  momentum edge. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    w = data.synthetic_world(mom_edge=0.0, seed=792 + s_)
    null_ts.append(st.synthetic_detect(w)["hac_t"])
null_ts = np.asarray(null_ts)
print(f"  null (mom_edge=0), 20 seeds: mean HAC t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
planted = st.synthetic_detect(data.synthetic_world(mom_edge=1.0, seed=792))
print(f"  planted mom_edge=1.0 (seed 792): L/S {planted['mean_bps']:+.2f} bps/mo, "
      f"HAC t = {planted['hac_t']:+.2f}, Sharpe {planted['sharpe']:.2f}")
