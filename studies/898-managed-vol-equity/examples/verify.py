"""Reproducible headline run for Study 898 — Managed-Vol Equity.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY/BIL parquet under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

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

from managed_vol import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Managed-Vol Equity — does a constant-vol thermostat on SPY raise the Sharpe?")

if not data.have_real():
    print("(cache miss — fetching SPY + BIL once)")
    data.fetch()

px = data.load_prices()
ex = data.excess_returns()
print(f"[data] SPY+BIL {len(px)} rows  {px.index.min().date()} -> {px.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint={fingerprint(px)}")
print("  Cash leg = BIL (real total-return T-bill ETF); every race is excess-of-cash on "
      "both legs. BIL history starts 2007-05-30, so the sample begins there.")

r = st.race(ex)
print("\n# THE HEADLINE — 12% vol target, 21d window, cap 2.0x, gross, excess-of-cash")
print(f"  managed : Sharpe {r['strat']['sharpe']:.3f}  vol {r['strat']['vol_ann_pct']:.1f}%  "
      f"maxDD {r['strat']['maxdd_pct']:.1f}%  excess-CAGR {r['strat']['cagr_pct']:.2f}%  "
      f"wealth x{r['strat']['wealth_mult']:.2f}")
print(f"  buy&hold: Sharpe {r['bh']['sharpe']:.3f}  vol {r['bh']['vol_ann_pct']:.1f}%  "
      f"maxDD {r['bh']['maxdd_pct']:.1f}%  excess-CAGR {r['bh']['cagr_pct']:.2f}%  "
      f"wealth x{r['bh']['wealth_mult']:.2f}")
print(f"  Sharpe gap {r['sharpe_gap']:+.3f}  |  HAC alpha {r['alpha_ann_pct']:+.2f}%/yr "
      f"(t = {r['t_alpha']:+.2f}, beta = {r['beta']:.2f}, appraisal = {r['appraisal']:.2f})")
print(f"  avg weight {r['avg_w']:.2f}  levered {r['share_levered']*100:.1f}% of days  "
      f"(at cap {r['share_capped']*100:.1f}%)  turnover {r['turnover_ann']:.2f}x NAV/yr, n={r['n_days']}")

print("\n# LEVERAGE-TIMING DECOMPOSITION — is it 'just' leverage-timing?")
print("  mean(managed excess) = beta*mean(B&H excess) [exposure] + alpha [timing]")
print(f"  exposure {r['exposure_bps']:+.3f} bps/day  +  timing (alpha) {r['timing_bps']:+.3f} bps/day")
print(f"  net managed-minus-B&H daily mean {r['mean_diff_bps']:+.3f} bps (HAC t = {r['t_diff']:+.2f})")
print("  A constant scale leaves Sharpe unchanged, so the whole Sharpe gap IS the timing "
      "term; with avg weight 0.96 the drawdown cut is genuine de-risking-in-storms, not "
      "just holding less.")

print("\n# THERMOSTAT — does it deliver the promised constant vol?")
vt = st.vol_tracking(ex)
print(f"  managed rolling-21d vol: median {vt['median_roll_vol_pct']:.1f}% "
      f"(p10-p90 {vt['p10_pct']:.1f}-{vt['p90_pct']:.1f}%), in +/-5pt band {vt['share_in_band']*100:.1f}% of days")
print(f"  buy&hold: median {vt['bh_median_roll_vol_pct']:.1f}%, p90 {vt['bh_p90_pct']:.1f}%")

print("\n# HEART-ATTACK LEDGER — max drawdown (excess-of-cash NAV) per crash window")
for k, v in st.crash_table(ex).items():
    print(f"  {k:<14} managed {v['strat']:+.1f}%   buy&hold {v['bh']:+.1f}%")

print("\n# PARAMETER GRID — targets 10/12/15% x windows 21/42/63d")
for g in st.grid(ex):
    print(f"  t={g['target']*100:>2.0f}% w={g['window']:>2d}d: Sharpe {g['sharpe']:.3f} "
          f"(bh {g['sharpe_bh']:.3f}, gap {g['sharpe_gap']:+.3f})  maxDD {g['maxdd_pct']:+.1f}%  "
          f"alpha {g['alpha_ann_pct']:+.2f}% (t={g['t_alpha']:+.2f})  avgw {g['avg_w']:.2f}")

print("\n# BOOTSTRAP — Sharpe advantage (managed - B&H), circular block, paired")
ov = st.run_overlay(ex)
bs = st.sharpe_gap_bootstrap(ov["strat"].values, ov["bh"].values)
print(f"  gap {bs['gap']:+.3f}  95% CI [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}]  "
      f"Pr[gap<0] = {bs['frac_negative']:.3f}  (block {bs['block']})")

print("\n# ERA CUT (split 2016-01-01)")
e = st.era_cut(ex)
for k in ("early", "late"):
    d = e[k]
    print(f"  {k:<5} n={d['n']:>4d}: alpha {d['alpha_ann_pct']:+.2f}% (t={d['t_alpha']:+.2f})  "
          f"Sharpe {d['sharpe']:.3f} (bh {d['sharpe_bh']:.3f})  maxDD {d['maxdd_pct']:+.1f}% "
          f"(bh {d['maxdd_bh_pct']:+.1f}%)")

print("\n# COST SWEEP — one-way bps x |dw| x NAV per day + borrow on the levered fraction")
for cb, bs_ in [(1.0, 0.0), (2.0, 0.005), (5.0, 0.01)]:
    rr = st.race(ex, cost_bps=cb, borrow_spread_ann=bs_)
    print(f"  {cb:>3.0f} bp + {bs_*100:.1f}% borrow: Sharpe {rr['strat']['sharpe']:.3f} "
          f"(gap {rr['sharpe_gap']:+.3f})  alpha {rr['alpha_ann_pct']:+.2f}% (t={rr['t_alpha']:+.2f})  "
          f"maxDD {rr['strat']['maxdd_pct']:+.1f}%")

print("\n# PLACEBO — shuffled vol signal, 200 seeds (same weight distribution, no timing)")
pl = st.placebo_shuffle(ex, n_seeds=200)
print(f"  HAC alpha: obs {pl['obs_alpha_ann_pct']:+.2f}% (t={pl['obs_t']:+.2f})  vs placebo "
      f"mean {pl['placebo_mean_alpha']:+.3f}% (sd {pl['placebo_sd_alpha']:.3f})  -> p_alpha = {pl['p_alpha']:.3f}")
print(f"  Sharpe gap: obs {pl['obs_sharpe_gap']:+.3f}  vs placebo mean {pl['placebo_mean_sh_gap']:+.3f}  "
      f"-> p = {pl['p_sh_gap']:.3f}")
print(f"  max drawdown: obs {pl['obs_maxdd_pct']:+.1f}%  vs placebo mean {pl['placebo_mean_maxdd_pct']:+.1f}%  "
      f"-> p_dd = {pl['p_dd']:.3f}")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network (30 seeds, 6000 days)")
null = st.synthetic_check(0.0, n_seeds=30, n_days=6000)
planted = st.synthetic_check(2.0, n_seeds=30, n_days=6000)
print(f"  NULL   (risk priced, mu ~ sig^2): mean alpha t {null['mean_t']:+.2f} "
      f"(sd {null['sd_t']:.2f}), t>=2 in {null['share_t_ge_2']*100:.0f}% of seeds")
print(f"  PLANTED (leverage effect)       : mean alpha t {planted['mean_t']:+.2f}, "
      f"alpha {planted['mean_alpha_ann_pct']:+.1f}%/yr, t>=2 in {planted['share_t_ge_2']*100:.0f}% of seeds")
