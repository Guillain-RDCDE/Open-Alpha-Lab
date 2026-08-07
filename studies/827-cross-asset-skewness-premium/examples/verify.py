"""Reproducible headline run for Study 827 — Cross-Asset Skewness Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached nine-ETF closes panel under
``_cache/`` (fetching once on a cache miss via yfinance), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from cross_asset_skew import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Cross-Asset Skewness Premium — do low-skew asset classes go on to earn more?")

if not data.have_real():
    print("(cache miss — fetching the nine-ETF closes panel once via yfinance)")
    data.fetch()

closes = data.load_panel()
print(f"[data] {closes.shape[1]} classes, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={data.fingerprint(closes)}")
print("  SURVIVORSHIP: fixed current-membership class-proxy ETFs — milder than a single-name "
      "universe, still named on the Signal axis.")

W = 126
sp = st.skew_spreads(closes, window=W, frac=0.34)
h = st.skew_stats(sp)
print(f"\nsort: trailing-{W}d realized skew, monthly, long bottom-1/3 / short top-1/3, "
      f"{h['n_months']} months (median {int(sp['n'].median())} classes/month)")
print("# THE HEADLINE — long-low-skew / short-high-skew spread")
print(f"  spread: {h['spread_bps']:+.2f} bps/month  NW(6) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}")
print(f"  books : low-skew {h['lo_bps']:+.2f} vs high-skew {h['hi_bps']:+.2f} bps  "
      f"(Welch t = {h['welch_t']:+.2f})")
print(f"  gross spread Sharpe (no cost, annualised): {h['sharpe']:.2f}")

print("\n# PLACEBO — asset-label-permute the forward returns (1,000 permutations)")
pl = st.placebo_pvalue(closes, window=W, n_seeds=20, n_draws_per_seed=50)
sig = (pl["obs_bps"] - pl["placebo_mean_bps"]) / pl["placebo_sd_bps"]
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.4f} "
      f"(observed ~{sig:+.2f}sigma)")

print("\n# ROBUSTNESS — two eras (split 2016-07-01) and window sweep")
for lo, hi, lbl in [("2007-01-01", "2016-07-01", "2007-2016"),
                    ("2016-07-01", "2026-07-01", "2016-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.skew_stats(sub)
    print(f"  {lbl}: n={ts['n_months']}  spread {ts['spread_bps']:+.2f} bps "
          f"(NW t={ts['t_nw']:+.2f})")
for w in (63, 252):
    hw = st.skew_stats(st.skew_spreads(closes, window=w, frac=0.34))
    print(f"  window {w:>3}d: spread {hw['spread_bps']:+.2f} bps (NW t={hw['t_nw']:+.2f})")

print("\n# THE TIMER — long-low-skew / short-high-skew, costed")
print("  2 sides x one-way cost x NAV per monthly rebalance; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/mo (cost {tm['cost_bps_per_month']:.2f}/mo, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([
    st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=827 + s, n_assets=9, n_days=3000))["t_nw"]
    for s in range(20)
])
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
sy = st.synthetic_detect(data.synthetic_panel(edge=0.004, seed=827, n_assets=9, n_days=4000))
print(f"  planted (edge=0.004, seed 827): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
