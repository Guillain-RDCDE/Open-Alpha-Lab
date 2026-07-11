"""Reproducible headline run for Study 658 - Put-Write-Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached PUTW/SPY/BIL tape under
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

from quantlab.repro import data_stamp  # noqa: E402

from put_write_premium import data, strategy as st  # noqa: E402

print("# Put-Write-Premium - does systematically writing S&P 500 puts (PUTW) harvest the VRP "
      "and beat buy&hold risk-adjusted?")

if not data.have_real():
    print("(cache miss - fetching PUTW / SPY / BIL once)")
    data.fetch()

px = data.load_real()
print(data_stamp("PUTW/SPY/BIL closes", px, asof=data.AS_OF))
print(f"window: {px.index.min().date()} -> {px.index.max().date()}  "
      f"({len(px)} joint trading days; PUTW inception {data.START} is the binding constraint)")

ret = st.daily_returns(px)
ex = st.excess(ret, cash_col=data.CASH)
ex_putw = ex["PUTW"].to_numpy()
ex_spy = ex["SPY"].to_numpy()
spy_ret = ret["SPY"].to_numpy()

print("\n# THE HEADLINE - excess-of-cash return (HAC t, 10 lags)")
r_putw = st.excess_return_stats(ex_putw)
r_spy = st.excess_return_stats(ex_spy)
print(f"  PUTW - BIL: {r_putw['ann_pct']:+.2f}%/yr   HAC t = {r_putw['t']:+.2f}   (n={r_putw['n']})")
print(f"  SPY  - BIL: {r_spy['ann_pct']:+.2f}%/yr   HAC t = {r_spy['t']:+.2f}   (n={r_spy['n']})")
diff = st.hac_mean_t(ex_putw - ex_spy, lags=10)
print(f"  PUTW - SPY (paired excess gap): {diff['mean']*252*100:+.2f}%/yr   "
      f"HAC t = {diff['t']:+.2f}")

print("\n# CAPM alpha/beta - is the premium just truncated equity beta?")
cb = st.capm_alpha_beta(ex_putw, ex_spy)
print(f"  alpha: {cb['alpha_ann_pct']:+.2f}%/yr   HAC t = {cb['t_alpha']:+.2f}")
print(f"  beta:  {cb['beta']:.3f}   HAC t = {cb['t_beta']:+.2f}   (n={cb['n']})")

print("\n# Sharpe (excess-of-cash) - did it beat SPY risk-adjusted, over its own live tape?")
sh_putw = st.sharpe_ann(ex_putw)
sh_spy = st.sharpe_ann(ex_spy)
print(f"  Sharpe PUTW: {sh_putw:.2f}   Sharpe SPY: {sh_spy:.2f}   gap: {sh_putw - sh_spy:+.2f}")
bs = st.bootstrap_sharpe_diff(ex_putw, ex_spy)
print(f"  circular block bootstrap (block=21d, n={bs['n_boot']}): point {bs['point']:+.2f}, "
      f"95% CI [{bs['ci_lo']:+.2f}, {bs['ci_hi']:+.2f}], "
      f"PUTW wins {bs['frac_putw_wins']*100:.1f}% of draws")

print("\n# Crash-conditional beta - does the 'lower beta' widen exactly when it's needed most?")
cx = st.crash_beta_interaction(ex_putw, ex_spy, spy_ret, threshold=data.CRASH_DAY_THRESHOLD)
print(f"  normal-day beta: {cx['beta_normal']:.3f} (t={cx['t_beta_normal']:+.2f})   "
      f"extra crash-day beta: +{cx['crash_beta_extra']:.3f} (t={cx['t_crash_beta_extra']:+.2f})   "
      f"-> crash-day beta ~= {cx['crash_beta_total']:.3f}   "
      f"(n_crash_days={cx['n_crash_days']}, threshold={data.CRASH_DAY_THRESHOLD:+.0%})")

print("\n# Named crash windows - max drawdown, PUTW vs SPY")
for name, (start, end) in data.CRASH_WINDOWS.items():
    dd_p = st.window_drawdown(ret["PUTW"], start, end)
    dd_s = st.window_drawdown(ret["SPY"], start, end)
    print(f"  {name} [{start} -> {end}]: PUTW {dd_p*100:+.1f}%   SPY {dd_s*100:+.1f}%")

print("\n# Full-sample max drawdown, monthly capture, tails")
mdd_p = st.max_drawdown(ret["PUTW"])
mdd_s = st.max_drawdown(ret["SPY"])
print(f"  max drawdown: PUTW {mdd_p*100:.1f}%   SPY {mdd_s*100:.1f}%")
cap = st.monthly_capture(px)
print(f"  monthly capture (n={cap['n_months']}, {cap['n_up']} up / {cap['n_dn']} down): "
      f"up {cap['up_capture']*100:.1f}%   down {cap['dn_capture']*100:.1f}%")
print(f"  worst month: PUTW {cap['worst_month_putw']*100:+.1f}% ({cap['worst_month_date']})   "
      f"SPY {cap['worst_month_spy']*100:+.1f}%")
t_putw = st.tail_stats(ret["PUTW"])
t_spy = st.tail_stats(ret["SPY"])
print(f"  ann. vol: PUTW {t_putw['vol_ann_pct']:.1f}%   SPY {t_spy['vol_ann_pct']:.1f}%")
print(f"  skew (daily): PUTW {t_putw['skew']:+.2f}   SPY {t_spy['skew']:+.2f}")
print(f"  worst day: PUTW {t_putw['worst_day_pct']:+.2f}%   SPY {t_spy['worst_day_pct']:+.2f}%")

print("\n# Synthetic positive control - deterministic, no network")
print("  the CAPM-alpha detector must NOT fire on a null world (harvest=0, fairly priced")
print("  options) and must recover a planted variance risk premium. Null over 20 seeds.")
null_ts = []
for s_ in range(20):
    w = data.synthetic_world(harvest=0.0, seed=658 + s_)
    null_ts.append(st.synthetic_detect(w)["t_alpha"])
null_ts = np.asarray(null_ts)
print(f"  null (harvest=0), 20 seeds: mean t_alpha = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
w = data.synthetic_world(harvest=0.25, seed=658)
sy = st.synthetic_detect(w)
print(f"  planted harvest=+0.25 (seed 658): alpha {sy['alpha_ann_pct']:+.2f}%/yr   "
      f"t_alpha = {sy['t_alpha']:+.2f}   beta = {sy['beta']:.3f}")
