"""Reproducible headline run for Study 642 — Turnaround Tuesday.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from turnaround_tuesday import data, strategy as st  # noqa: E402

print("# Turnaround Tuesday — does the market bounce back on Tuesday after a down Monday?")

if not data.have_real():
    print("(cache miss — fetching SPY once)")
    data.fetch()

spy = data.load_real()
print(data_stamp("SPY OHLC + AdjClose", spy, asof=data.AS_OF))

df = st.day_frame(spy["AdjClose"])
pairs = st.monday_tuesday_pairs(df)
print(f"Monday->Tuesday consecutive pairs: {len(pairs)} "
      f"(no market-holiday break between the two sessions)")

years = (df.index.max() - df.index.min()).days / 365.25

print("\n# THE HEADLINE — Tuesday return conditional on the prior Monday closing down")
h = st.headline_stats(df, pairs)
print(f"  down-Monday Tuesday : {h['cond_bps']:+.2f} bps  (n={h['n_down']})")
print(f"  up-Monday Tuesday   : {h['up_bps']:+.2f} bps  (n={h['n_up']})")
print(f"  unconditional Tuesday: {h['uncond_tue_bps']:+.2f} bps  (n={h['n_uncond_tue']})")
print(f"  all trading days    : {h['allday_bps']:+.2f} bps  (n={h['n_allday']})")
print(f"  Welch t  vs unconditional Tuesday = {h['welch_t_vs_uncond']:+.2f}  |  "
      f"vs up-Monday Tuesday = {h['welch_t_vs_up']:+.2f}  |  vs all days = {h['welch_t_vs_allday']:+.2f}")
print(f"  Newey-West t (dummy regression, 5 lags) vs all days   = {h['nw_t_vs_allday']:+.2f} "
      f"(coef {h['nw_coef_vs_allday_bps']:+.2f} bps)")
print(f"  Newey-West t (dummy regression, 5 lags) vs Tuesdays   = {h['nw_t_vs_uncond']:+.2f} "
      f"(coef {h['nw_coef_vs_uncond_bps']:+.2f} bps)")
print(f"  hit rate: Tuesday positive on {h['hit']}/{h['n_down']} down-Monday events = "
      f"{h['hit_rate']*100:.1f}%  (Wilson 95% [{h['hit_lo']*100:.1f}%, {h['hit_hi']*100:.1f}%])")

print("\n# Random-pair placebo (20 seeds x 1,000 draws of "
      f"{h['n_down']} random Tuesdays out of {len(pairs)})")
pl = st.placebo_pvalue(pairs)
print(f"  observed down-Monday-Tuesday mean {pl['obs']*1e4:+.2f} bps vs placebo mean "
      f"{pl['placebo_mean']*1e4:+.2f} bps (sd {pl['placebo_sd']*1e4:.2f}) over "
      f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# Monday-specificity check — is this Monday, or generic reversal-after-down-day?")
wp = st.weekday_pair_stats(df)
for pair, row in wp.iterrows():
    tag = "  <-- turnaround Tuesday" if pair == "Mon->Tue" else ""
    print(f"  {pair}: n={int(row['n'])}, down-day mean {row['down_bps']:+.2f} bps "
          f"vs up-day mean {row['up_bps']:+.2f} bps, Welch t = {row['welch_t']:+.2f}{tag}")
oc = st.other_weekday_check(df)
print(f"  pooled OTHER four weekday pairs (excl. Mon->Tue): n_down={oc['n_down']}, "
      f"down-day mean {oc['down_bps']:+.2f} bps vs up-day mean {oc['up_bps']:+.2f} bps, "
      f"Welch t = {oc['welch_t']:+.2f}")

print(f"\n# Robustness — drop the 2008-09 GFC and 2020-03 COVID crash windows "
      f"({st.CRISIS_WINDOWS})")
ex = st.excl_crisis_check(pairs)
print(f"  down-Monday Tuesday ex-crisis: {ex['mean_bps']:+.2f} bps (n={ex['n']}), "
      f"Welch t vs up-Monday Tuesday = {ex['welch_t']:+.2f}")

print(f"\n# Era contrast — pre vs post 2000 (split {data.ERA_SPLIT}, named in the brief)")
ec = st.era_contrast(df, pairs, data.ERA_SPLIT)
print(f"  {data.START} -> {data.ERA_SPLIT}: down-Monday Tuesday {ec['early_bps']:+.2f} bps "
      f"(n={ec['n_early']}, within-era Welch t = {ec['welch_t_early']:+.2f})")
print(f"  {data.ERA_SPLIT} -> {data.AS_OF}: down-Monday Tuesday {ec['late_bps']:+.2f} bps "
      f"(n={ec['n_late']}, within-era Welch t = {ec['welch_t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['welch_t_diff']:+.2f}")

print("\n# THE TIMER — buy SPY at the Monday close (after that Monday's own close prints")
print("  down), exit at the Tuesday close; one round trip = 2 x one-way cost x NAV.")
for cb in (5.0, 10.0):
    tm = st.timer_stats(pairs, cost_bps=cb, years=years)
    print(f"  cost={cb:>4.1f} bps: gross {tm['gross_bps']:+.1f} bps/event -> net "
          f"{tm['net_bps']:+.1f} bps/event  (~{tm['ann_net_pct']:+.2f}%/yr at "
          f"{tm['events_per_year']:.1f} events/yr, one-sample t = {tm['t_net']:+.2f}, "
          f"Sharpe = {tm['sharpe_net']:.2f}, worst single event {tm['worst_pct']:+.2f}%)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (bounce=0) and must recover a")
print("  planted turnaround bounce. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    close = data.synthetic_world(bounce=0.0, seed=642 + s_)
    null_ts.append(st.synthetic_detect(close)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (bounce=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
close = data.synthetic_world(bounce=0.0020, seed=642)
sy = st.synthetic_detect(close)
print(f"  planted bounce=+20 bps (seed 642): down-Monday mean {sy['cond_bps']:+.2f} bps vs "
      f"up-Monday mean {sy['up_bps']:+.2f} bps  Welch t = {sy['welch_t']:+.2f}")
