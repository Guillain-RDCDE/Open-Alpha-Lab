"""Reproducible headline run for Study 810 — Price Delay.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached cross-section panel under
``_cache/`` (fetching once on a cache miss through the quantlab.universe survivorship
guard), and always runs the synthetic control with no network.

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

from quantlab.repro import fingerprint  # noqa: E402

from price_delay import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Price Delay — do slow-to-price (high-delay) names earn a premium? (Hou-Moskowitz 2005)")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the "
          "survivorship guard)")
    data.fetch()

panel = data.load_panel()
closes = pd.DataFrame({s: panel[s]["Close"] for s in data.UNIVERSE if s in panel})
print(f"[data] {len(panel)} names, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(closes)}")
print("  SURVIVORSHIP: current-membership mega-cap panel — magnitudes are an upper "
      "bound (delisted names absent). Named on the Signal axis.")

weekly = st.weekly_returns(panel)
print(f"[weekly] {len(weekly)} weeks (W-FRI), market = equal-weight cross-sectional mean")

sp = st.delay_spreads(weekly, window=52, n_lags=4, frac=0.3)
h = st.delay_stats(sp)
print(f"\nsort: trailing-52w delay (contemp market + 4 weekly lags), long top30% (HIGH "
      f"delay) / short bottom30% (LOW delay), {h['n_weeks']} weeks "
      f"(median {int(sp['n'].median())} names/week)")
print("# THE HEADLINE — long-high-delay / short-low-delay spread")
print(f"  spread: {h['spread_bps']:+.2f} bps/week  NW(6) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}")
print(f"  books : high-delay {h['long_bps']:+.2f} vs low-delay {h['short_bps']:+.2f} bps  "
      f"(Welch t = {h['welch_t']:+.2f})")
sps = sp["spread"].to_numpy()
print(f"  gross spread Sharpe (no cost): "
      f"{sps.mean() / sps.std(ddof=1) * np.sqrt(52):.2f}")

print("\n# PLACEBO — column-permute the forward returns (1,000 permutations)")
pl = st.placebo_pvalue(weekly, n_seeds=20, n_draws_per_seed=50)
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.delay_stats(sub)
    print(f"  {lbl}: n={ts['n_weeks']}  spread {ts['spread_bps']:+.2f} bps "
          f"(NW t={ts['t_nw']:+.2f})")

print("\n# THE TIMER — long-high-delay / short-low-delay, costed (weekly rebalance)")
print("  2 sides x one-way cost x NAV per week; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/week (cost {tm['cost_bps_per_week']:.2f}/week, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(knob=0.0, seed=810 + s_, n_assets=40, n_days=2000)
    null_t.append(st.synthetic_detect(p0)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (knob=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(knob=0.0018, seed=810, n_assets=40, n_days=2000)
sy = st.synthetic_detect(p1)
print(f"  planted (knob=0.0018, seed 810): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
