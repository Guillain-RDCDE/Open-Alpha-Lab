"""Reproducible headline run for Study 821 — Turnover Volatility.

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

from turnover_vol import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

WINDOW = 63

print("# Turnover Volatility — do erratic-turnover names go on to earn less?")

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
turn = st.turnover_panel(panel, dollar=False)
for w in (WINDOW,):
    sp = st.cv_spreads(ret, turn, window=w, frac=0.3)
    h = st.cv_stats(sp)
    print(f"\nsort: trailing-{w}d CV-of-turnover, long bottom30% / short top30%, "
          f"{h['n_days']} days (median {int(sp['n'].median())} names/day)")
    print("# THE HEADLINE — long-low-turnover-vol / short-high-turnover-vol spread")
    print(f"  spread: {h['spread_bps']:+.2f} bps/day  NW(10) t = {h['t_nw']:+.2f}  "
          f"one-sample t = {h['t_1s']:+.2f}")
    print(f"  books : low-vol {h['lo_bps']:+.2f} vs high-vol {h['hi_bps']:+.2f} bps  "
          f"(Welch t = {h['welch_t']:+.2f})")
    sps = sp["spread"].to_numpy()
    print(f"  gross spread Sharpe (no cost): "
          f"{sps.mean() / sps.std(ddof=1) * np.sqrt(252):.2f}")

sp = st.cv_spreads(ret, turn, window=WINDOW, frac=0.3)
print("\n# PLACEBO — column-permute the forward returns (1,000 permutations)")
pl = st.placebo_pvalue(ret, turn, window=WINDOW, n_seeds=20, n_draws_per_seed=50)
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")
sig_dist = (pl['obs_bps'] - pl['placebo_mean_bps']) / pl['placebo_sd_bps']
print(f"  observed sits {sig_dist:+.2f} sigma from the placebo mean")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.cv_stats(sub)
    print(f"  {lbl}: n={ts['n_days']}  spread {ts['spread_bps']:+.2f} bps "
          f"(NW t={ts['t_nw']:+.2f})")

print("\n# DOLLAR-VOLUME VARIANT — turnover = Volume x Close")
turn_d = st.turnover_panel(panel, dollar=True)
sp_d = st.cv_spreads(ret, turn_d, window=WINDOW, frac=0.3)
hd = st.cv_stats(sp_d)
print(f"  spread {hd['spread_bps']:+.2f} bps/day  NW t = {hd['t_nw']:+.2f}  (n={hd['n_days']})")

print("\n# THE TIMER — long-low-vol / short-high-vol, costed")
print("  2 sides x one-way cost x NAV per day; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/day (cost {tm['cost_bps_per_day']:.2f}/day, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=821 + s_, n_assets=40, n_days=1200)
    null_t.append(st.synthetic_detect(p0, window=WINDOW)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.0016, seed=821, n_assets=40, n_days=1500)
sy = st.synthetic_detect(p1, window=WINDOW)
print(f"  planted (edge=0.0016, seed 821): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
