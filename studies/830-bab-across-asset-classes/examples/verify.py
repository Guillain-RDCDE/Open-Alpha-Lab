"""Reproducible headline run for Study 830 — BAB Across Asset Classes.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached cross-asset panel under
``_cache/`` (fetching once via yfinance on a cache miss), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from bab_multiasset import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# BAB Across Asset Classes — does betting against beta work at the multi-asset level?")

if not data.have_real():
    print("(cache miss — fetching the nine-ETF cross-asset panel once via yfinance)")
    data.fetch()

df = data.load_series()
print(f"[data] {df.shape[1]} asset classes, {len(df)} rows  "
      f"{df.index.min().date()} -> {df.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(df)}")
print("  Assets: " + ", ".join(f"{t} ({data.ASSET_LABELS[t]})" for t in df.columns))
print("  SURVIVORSHIP: a fixed current-membership set of liquid asset-class ETFs; the "
      "multi-asset 'market' is their equal-weight average. Named on the Signal axis.")

ret = st.close_returns(df)
mkt = st.market_return(ret)
book = st.bab_series(ret)
h = st.bab_stats(book, mkt)

print("\n# THE HEADLINE — beta-neutral BAB factor (long low-beta levered / short high-beta de-levered)")
print(f"  BAB return : {h['bab_bps']:+.2f} bps/day  NW(10) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}")
print(f"  gross Sharpe (ann): {h['sharpe']:+.2f}")
print(f"  CAPM alpha : {h['alpha_bps']:+.2f} bps/day  (HAC t = {h['alpha_t']:+.2f}); "
      f"realized market beta = {h['realized_beta']:+.2f}")
print(f"  legs       : low-beta leg {h['lo_bps']:+.2f} bps (beta_L {h['beta_L']:.2f}) vs "
      f"high-beta leg {h['hi_bps']:+.2f} bps (beta_H {h['beta_H']:.2f})")

print("\n# PLACEBO — column-permute the asset returns (1,000 permutations)")
pl = st.placebo_pvalue(ret, n_seeds=20, n_draws_per_seed=50)
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> "
      f"two-sided p = {pl['p_value']:.4f}  ({pl['sigma']:.2f}sigma)")

print("\n# ROBUSTNESS — two eras (split 2016-07-01)")
for lo, hi, lbl in [("2007-01-01", "2016-07-01", "2007-2016"),
                    ("2016-07-01", "2026-07-01", "2016-2026")]:
    sub = book[(book.index >= lo) & (book.index < hi)]
    ts = st.bab_stats(sub, mkt)
    print(f"  {lbl}: n={ts['n_days']}  BAB {ts['bab_bps']:+.2f} bps (NW t={ts['t_nw']:+.2f}, "
          f"alpha t={ts['alpha_t']:+.2f})")

print("\n# THE TIMER — levered BAB book, costed")
print("  realized turnover x one-way cost x NAV per day; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(book, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/day (cost {tm['cost_bps_per_day']:.2f}/day, "
          f"t={tm['t_net']:+.2f}, NW t={tm['t_net_nw']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, "
          f"~{tm['ann_net_pct']:+.1f}%/yr; avg gross {tm['avg_gross']:.2f}x, "
          f"turnover {tm['avg_turnover']:.3f}/day)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    d0 = st.synthetic_detect(data.synthetic_series(edge=0.0, seed=830 + s_, n_days=2500))
    null_t.append(d0["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: BAB NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
d1 = st.synthetic_detect(data.synthetic_series(edge=0.0006, seed=830, n_days=2500))
print(f"  planted (edge=0.0006, seed 830): BAB NW t = {d1['t_nw']:+.2f}, "
      f"alpha t = {d1['alpha_t']:+.2f}, Sharpe {d1['sharpe']:+.2f}")
