"""Reproducible headline run for Study 870 — Industry-Leader Lead-Lag.

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

from leader_lag import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Industry-Leader Lead-Lag — does the biggest name in a sector lead the rest?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the "
          "survivorship guard)")
    data.fetch()

panel = data.load_panel()
closes = pd.DataFrame({s: panel[s]["Close"] for s in data.UNIVERSE if s in panel})
print(f"[data] {len(panel)} names, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(closes)}")
print("  SURVIVORSHIP: current-membership mega-cap panel; LEADERS = today's largest-cap "
      "per sector (static). Named on the Signal axis.")

wret = st.weekly_returns(panel)
print(f"  weekly panel: {wret.shape[0]} weeks x {wret.shape[1]} names; "
      f"{len(data.SECTORS)} sectors, leaders = "
      f"{', '.join(f'{k}:{v}' for k, v in data.LEADERS.items())}")

sp = st.leadlag_spreads(wret, data.SECTORS, data.LEADERS)
h = st.leadlag_stats(sp, wret, data.SECTORS, data.LEADERS)
print("\n# THE HEADLINE — long followers of up-leaders / short followers of down-leaders")
print(f"  spread: {h['spread_bps']:+.2f} bps/week  NW(6) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}  (n = {h['n_weeks']} weeks)")
print(f"  legs  : after up-leader {h['up_bps']:+.2f} vs after down-leader {h['dn_bps']:+.2f} bps "
      f"(Welch t = {h['welch_t']:+.2f})")
sps = sp["spread"].to_numpy()
print(f"  gross weekly Sharpe (no cost): {sps.mean() / sps.std(ddof=1) * np.sqrt(52):.2f}")

print("\n# PLACEBO — shuffle the lead->lag week alignment (1,000 permutations)")
pl = st.placebo_pvalue(wret, data.SECTORS, data.LEADERS, n_seeds=20, n_draws_per_seed=50)
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.leadlag_stats(sub)
    print(f"  {lbl}: n={ts['n_weeks']}  spread {ts['spread_bps']:+.2f} bps "
          f"(NW t={ts['t_nw']:+.2f})")

print("\n# ROBUSTNESS — re-designate leaders by trailing dollar volume (size proxy)")
dv = st.dollar_volume(panel)
dyn = st.designate_leaders(data.SECTORS, dv)
sp_dyn = st.leadlag_spreads(wret, data.SECTORS, dyn)
hd = st.leadlag_stats(sp_dyn, wret, data.SECTORS, dyn)
print(f"  $-vol leaders: {', '.join(f'{k}:{v}' for k, v in dyn.items())}")
print(f"  spread {hd['spread_bps']:+.2f} bps/week (NW t = {hd['t_nw']:+.2f}, n = {hd['n_weeks']})")

print("\n# THE TIMER — long-short followers book, costed")
print("  2 sides x one-way cost x NAV per week; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/week (cost {tm['cost_bps_per_week']:.2f}/wk, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
secs = data.synthetic_sectors()
lds = data.synthetic_leaders()
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=870 + s_, n_weeks=260)
    null_t.append(st.synthetic_detect(p0, secs, lds)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.6, seed=870, n_weeks=320)
sy = st.synthetic_detect(p1, secs, lds)
print(f"  planted (edge=0.6, seed 870): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
