"""Reproducible headline run for Study 905 — Residual Reversal.

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

from resid_reversal import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Residual Reversal — does factor-cleaned weekly reversal beat the raw version?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the "
          "survivorship guard)")
    data.fetch()

panel = data.load_panel()
closes = pd.DataFrame({s: panel[s]["Close"] for s in data.UNIVERSE if s in panel})
print(f"[data] {len(panel)} names, {len(closes)} daily rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(closes)}")
print("  SURVIVORSHIP: current-membership mega-cap panel — magnitudes are an upper "
      "bound (delisted names absent). Named on the Signal axis.")

wret = st.weekly_returns(panel)
liq = st.weekly_dollar_volume(panel)
print(f"  weekly panel: {wret.shape[0]} weeks x {wret.shape[1]} names")

print("\n# THE HEADLINE — long past-week-residual-loser / short-winner, liquidity-screened")
sp_res = st.residual_reversal_spreads(wret, liq, beta_window=52, frac=0.3, liq_frac=0.6)
h = st.reversal_stats(sp_res)
print(f"  residual reversal: {h['spread_bps']:+.2f} bps/wk  NW(8) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}  ({h['n_weeks']} weeks, median "
      f"{int(sp_res['n'].median())} names/wk)")
print(f"  books : loser {h['lo_bps']:+.2f} vs winner {h['hi_bps']:+.2f} bps  "
      f"(Welch t = {h['welch_t']:+.2f})  gross Sharpe {h['gross_sharpe']:.2f}  "
      f"hit {h['hit_rate']*100:.1f}%")

print("\n# THE FOIL — RAW weekly reversal (no factor cleaning), same screen")
sp_raw = st.raw_reversal_spreads(wret, liq, frac=0.3, liq_frac=0.6)
hr = st.reversal_stats(sp_raw)
print(f"  raw reversal     : {hr['spread_bps']:+.2f} bps/wk  NW(8) t = {hr['t_nw']:+.2f}  "
      f"gross Sharpe {hr['gross_sharpe']:.2f}")

print("\n# NO LIQUIDITY SCREEN (residual, all names)")
sp_res_all = st.residual_reversal_spreads(wret, None, beta_window=52, frac=0.3)
ha = st.reversal_stats(sp_res_all)
print(f"  residual reversal, all names: {ha['spread_bps']:+.2f} bps/wk  "
      f"NW t = {ha['t_nw']:+.2f}")

print("\n# PLACEBO — column-permute the forward returns (1,000 permutations)")
pl = st.placebo_pvalue(wret, liq, n_seeds=20, n_draws_per_seed=50)
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp_res[(sp_res.index >= lo) & (sp_res.index < hi)]
    ts = st.reversal_stats(sub)
    print(f"  {lbl}: n={ts['n_weeks']}  spread {ts['spread_bps']:+.2f} bps "
          f"(NW t={ts['t_nw']:+.2f})")

print("\n# THE TIMER — long-loser / short-winner residual book, costed")
print("  2 sides x one-way cost x NAV per weekly rebalance; short pays 50 bps/yr borrow")
for cb in (1.0, 5.0, 10.0):
    tm = st.timer_stats(sp_res, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/wk (cost {tm['cost_bps_per_week']:.2f}/wk, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=905 + s_, n_assets=40, n_days=1600)
    null_t.append(st.synthetic_detect(p0)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: residual spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.35, seed=905, n_assets=40, n_days=2000)
sy = st.synthetic_detect(p1)
print(f"  planted (edge=0.35, seed 905): residual NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}  |  RAW reversal NW t = {sy['raw_t_nw']:+.2f} "
      f"(factor-muddied)")
