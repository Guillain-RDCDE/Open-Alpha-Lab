"""Reproducible headline run for Study 818 — Trend Factor.

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

from quantlab.repro import fingerprint  # noqa: E402

from trend_factor import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Trend Factor — does a blend of MA horizons beat single-MA timing and momentum?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the "
          "survivorship guard)")
    data.fetch()

panel = data.load_panel()
prices = st.close_prices(panel)
ret = st.close_returns(panel)
print(f"[data] {len(panel)} names, {len(prices)} rows  "
      f"{prices.index.min().date()} -> {prices.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(prices)}")
print("  SURVIVORSHIP: current-membership mega-cap panel — magnitudes are an upper "
      "bound (delisted names absent). Named on the Signal axis.")

sp = st.trend_spreads(prices, ret, beta_window=250, frac=0.3)
h = st.trend_stats(sp)
print(f"\nsort: trend factor (MA {data.MA_LAGS}, 250d rolling FM slopes), "
      f"long top30% / short bottom30%, {h['n_days']} days "
      f"(median {int(sp['n'].median())} names/day)")
print("# THE HEADLINE — long-high-trend / short-low-trend spread")
print(f"  spread: {h['spread_bps']:+.2f} bps/day  NW(10) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}")
print(f"  books : high-trend {h['hi_bps']:+.2f} vs low-trend {h['lo_bps']:+.2f} bps  "
      f"(Welch t = {h['welch_t']:+.2f})")
sps = sp["spread"].to_numpy()
print(f"  gross spread Sharpe (no cost): "
      f"{sps.mean() / sps.std(ddof=1) * np.sqrt(252):+.2f}")

print("\n# CONTRAST — the two things the trend factor is claimed to beat")
ma = st.trend_stats(st.single_ma_spreads(prices, ret, lag=200, frac=0.3))
mom = st.trend_stats(st.momentum_spreads(prices, ret, frac=0.3))
print(f"  single-MA(200) timing sort : {ma['spread_bps']:+.2f} bps/day (NW t = {ma['t_nw']:+.2f})")
print(f"  12-1 momentum sort         : {mom['spread_bps']:+.2f} bps/day (NW t = {mom['t_nw']:+.2f})")
print("  (paper claims the trend factor beats BOTH; here it is the WEAKEST of the three)")

print("\n# PLACEBO — column-permute the forward returns (1,000 permutations)")
pl = st.placebo_pvalue(prices, ret, beta_window=250, n_seeds=20, n_draws_per_seed=50)
sigma = (pl["obs_bps"] - pl["placebo_mean_bps"]) / pl["placebo_sd_bps"]
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.4f} "
      f"({sigma:+.2f} sd from mean)")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.trend_stats(sub)
    print(f"  {lbl}: n={ts['n_days']}  spread {ts['spread_bps']:+.2f} bps "
          f"(NW t={ts['t_nw']:+.2f})")

print("\n# THE TIMER — long-high-trend / short-low-trend, costed")
print("  2 sides x one-way cost x NAV per day; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/day (cost {tm['cost_bps_per_day']:.2f}/day, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([
    st.synthetic_detect(
        data.synthetic_panel(edge=0.0, seed=818 + s, n_assets=40, n_days=1500),
        beta_window=120)["t_nw"]
    for s in range(20)
])
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
sy = st.synthetic_detect(
    data.synthetic_panel(edge=0.0015, seed=818, n_assets=40, n_days=1500),
    beta_window=120)
print(f"  planted (edge=0.0015, seed 818): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
