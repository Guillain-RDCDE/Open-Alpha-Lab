"""Reproducible headline run for Study 806 — Prospect-Theory Value.

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

from prospect_theory import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

WIN_DAYS = 1260   # trailing ~5 years of daily returns = the empirical "gamble"

print("# Prospect-Theory Value — do high-TK-value (lottery-like) names go on to earn less?")

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

ret = st.close_returns(panel)
sp = st.tk_spreads(ret, win_days=WIN_DAYS, frac=0.3)
h = st.tk_stats(sp)
print(f"\nsort: trailing-{WIN_DAYS}d TK value, monthly rebalance, long bottom30% (low TK) "
      f"/ short top30% (high TK), {h['n_months']} months "
      f"({sp.index.min().date()} -> {sp.index.max().date()}, median {int(sp['n'].median())} names/mo)")
print("# THE HEADLINE — long-low-TK / short-high-TK spread")
print(f"  spread: {h['spread_bps']:+.2f} bps/month  NW(6) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}")
print(f"  books : low-TK {h['lo_bps']:+.2f} vs high-TK {h['hi_bps']:+.2f} bps  "
      f"(Welch t = {h['welch_t']:+.2f})")
spv = sp["spread"].to_numpy()
print(f"  gross spread Sharpe (no cost, ann.): "
      f"{spv.mean() / spv.std(ddof=1) * np.sqrt(12):.2f}")

print("\n# PLACEBO — column-permute the forward returns (1,000 permutations)")
pl = st.placebo_pvalue(ret, win_days=WIN_DAYS, n_seeds=20, n_draws_per_seed=50)
sigma = (pl["obs_bps"] - pl["placebo_mean_bps"]) / pl["placebo_sd_bps"]
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f} "
      f"({sigma:+.2f}sigma into the right tail)")

print("\n# ROBUSTNESS — two eras (split 2020-01-01)")
for lo, hi, lbl in [("2010-01-01", "2020-01-01", "2015-2019"),
                    ("2020-01-01", "2026-07-01", "2020-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.tk_stats(sub)
    print(f"  {lbl}: n={ts['n_months']}  spread {ts['spread_bps']:+.2f} bps "
          f"(NW t={ts['t_nw']:+.2f})")

print("\n# THE TIMER — long-low-TK / short-high-TK, costed, monthly rebalance")
print("  2 sides x one-way cost x NAV per monthly rebalance; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/mo (cost {tm['cost_bps_per_reb']:.2f}/reb, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=806 + s_, n_assets=40, n_days=1200)
    null_t.append(st.synthetic_detect(p0)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.0020, seed=806, n_assets=40, n_days=1500)
sy = st.synthetic_detect(p1)
print(f"  planted (edge=0.0020, seed 806): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
