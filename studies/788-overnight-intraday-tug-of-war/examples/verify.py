"""Reproducible headline run for Study 788 — Overnight / Intraday Tug of War.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached cross-section panel
under ``_cache/`` (fetching once on a cache miss through the quantlab.universe
survivorship guard), and always runs the synthetic control with no network.

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

from overnight_intraday_tug import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Overnight/Intraday Tug of War — do past-overnight winners keep winning "
      "overnight and reverse intraday?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the "
          "survivorship guard)")
    data.fetch()

panel = data.load_panel()
closes = pd.DataFrame({s: panel[s]["Close"] for s in data.UNIVERSE if s in panel})
opens = pd.DataFrame({s: panel[s]["Open"] for s in data.UNIVERSE if s in panel})
oc = pd.concat([opens.add_suffix("_o"), closes.add_suffix("_c")], axis=1)
print(f"[data] {len(panel)} names, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(OC)={fingerprint(oc)}  fingerprint(Close)={fingerprint(closes)}")
print("  SURVIVORSHIP: current-membership mega-cap panel — magnitudes are an upper "
      "bound (delisted names absent). Named on the Signal axis.")

on, idc = st.leg_panels(panel)
sp = st.tug_spreads(on, idc, window=21, frac=0.3)
print(f"\nsort: trailing-21d overnight, top30%-bot30%, {len(sp)} days "
      f"(median {int(sp['n'].median())} names/day)")

print("\n# THE HEADLINE — legs of the trailing-overnight-sorted H-L spread")
h = st.tug_stats(sp)
print(f"  overnight (persistence): {h['on_mean_bps']:+.2f} bps/day  "
      f"NW(10) t = {h['on_t_nw']:+.2f}  one-sample t = {h['on_t_1s']:+.2f}")
print(f"  intraday  (reversal)   : {h['id_mean_bps']:+.2f} bps/day  "
      f"NW(10) t = {h['id_t_nw']:+.2f}  one-sample t = {h['id_t_1s']:+.2f}")
print(f"  close-close (net)      : {h['cc_mean_bps']:+.2f} bps/day  "
      f"NW(10) t = {h['cc_t_nw']:+.2f}")
print(f"  tug (overnight-intraday): {h['tug_mean_bps']:+.2f} bps/day  "
      f"NW(10) t = {h['tug_t_nw']:+.2f}")
ons = sp["on_spread"].to_numpy()
print(f"  gross overnight-spread Sharpe (no cost): "
      f"{ons.mean() / ons.std(ddof=1) * np.sqrt(252):.2f}")

lp = st.leg_pooled_welch(sp)
print(f"  pooled book legs: overnight top {lp['on_top_bps']:+.2f} vs bot "
      f"{lp['on_bot_bps']:+.2f} (Welch t = {lp['on_welch_t']:+.2f}); intraday top "
      f"{lp['id_top_bps']:+.2f} vs bot {lp['id_bot_bps']:+.2f} (Welch t = {lp['id_welch_t']:+.2f})")

print("\n# PLACEBO — column-permute the forward overnight legs (1,000 permutations)")
pl = st.placebo_pvalue(on, idc, n_seeds=20, n_draws_per_seed=50)
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.tug_stats(sub)
    print(f"  {lbl}: n={ts['n_days']}  overnight {ts['on_mean_bps']:+.2f} bps "
          f"(t={ts['on_t_nw']:+.2f}) | intraday {ts['id_mean_bps']:+.2f} bps "
          f"(t={ts['id_t_nw']:+.2f})")

print("\n# THE TIMER — capture the overnight leg (enter close t-1, exit open t)")
print("  2 legs x 2 sides x one-way cost x NAV per day; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net "
          f"{tm['net_bps']:+.2f} bps/day (round-trip cost {tm['cost_bps_per_day']:.2f}/day, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  the detector must NOT fire on the null (tug=0) and must recover a planted tug.")
null_on = []
for s_ in range(20):
    p0 = data.synthetic_panel(tug=0.0, seed=788 + s_, n_assets=40, n_days=1200)
    null_on.append(st.synthetic_detect(p0)["on_t_nw"])
null_on = np.asarray(null_on)
print(f"  null (tug=0), 20 seeds: overnight NW t mean {null_on.mean():+.2f} "
      f"(sd {null_on.std(ddof=1):.2f}), |t|>=2 in {(abs(null_on) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(tug=0.0020, seed=788, n_assets=40, n_days=1500)
sy = st.synthetic_detect(p1)
print(f"  planted (tug=0.0020, seed 788): overnight NW t = {sy['on_t_nw']:+.2f}, "
      f"intraday NW t = {sy['id_t_nw']:+.2f}")
