"""Reproducible headline run for Study 869 — 52-Week-High Breakout Drift.

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

from breakout_high import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# 52-Week-High Breakout Drift — after a fresh 52w-high, do names drift up or fade?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the "
          "survivorship guard)")
    data.fetch()

panel = data.load_panel()
closes = st.closes_frame(panel)
print(f"[data] {len(panel)} names, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(closes)}")
print("  SURVIVORSHIP: current-membership mega-cap panel — fresh 52w highs are MORE "
      "frequent (survivors trend up); magnitudes are an upper bound. Named on Signal axis.")

for h, lags in ((5, 10), (20, 40)):
    sp = st.breakout_spreads(closes, lookback=252, horizon=h, lag=1)
    hh = st.breakout_stats(sp, nw_lags=lags)
    print(f"\nsort: fresh 52w-high breakout, long just-broke-out / short the rest, "
          f"forward {h}d, {hh['n_days']} event-days ({hh['n_breakouts']:,} breakout obs)")
    print("# THE HEADLINE — long-breakout / short-rest forward return spread")
    print(f"  spread: {hh['spread_bps']:+.2f} bps  NW({lags}) t = {hh['t_nw']:+.2f}  "
          f"one-sample t = {hh['t_1s']:+.2f}")
    print(f"  books : breakout {hh['brk_bps']:+.2f} vs rest {hh['rest_bps']:+.2f} bps  "
          f"(Welch t = {hh['welch_t']:+.2f})")
    print(f"  hit rate (days spread>0): {hh['hit_rate']:.3f}  "
          f"[Wilson95 {hh['hit_lo']:.3f}, {hh['hit_hi']:.3f}]")

    print("# PLACEBO — column-permute the forward returns (1,000 permutations)")
    pl = st.placebo_pvalue(closes, horizon=h, n_seeds=20, n_draws_per_seed=50)
    print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
          f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

    print("# ROBUSTNESS — two eras (split 2018-01-01)")
    for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                        ("2018-01-01", "2026-07-01", "2018-2026")]:
        sub = sp[(sp.index >= lo) & (sp.index < hi)]
        ts = st.breakout_stats(sub, nw_lags=lags)
        print(f"  {lbl}: n={ts['n_days']}  spread {ts['spread_bps']:+.2f} bps "
              f"(NW t={ts['t_nw']:+.2f})")

    print("# THE TIMER — long-breakout / short-rest, costed")
    print("  2 sides x (in+out) one-way cost per event; short book pays 50 bps/yr borrow")
    for cb in (1.0, 5.0):
        tm = st.timer_stats(sp, horizon=h, cost_bps=cb, borrow_bps_yr=50.0)
        print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
              f"{tm['net_bps']:+.2f} bps/event (cost {tm['cost_bps_per_event']:.2f}, "
              f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=869 + s_, n_assets=40, n_days=1200)
    null_t.append(st.synthetic_detect(p0)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.0015, seed=869, n_assets=40, n_days=1500)
sy = st.synthetic_detect(p1)
print(f"  planted (edge=0.0015, seed 869): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
