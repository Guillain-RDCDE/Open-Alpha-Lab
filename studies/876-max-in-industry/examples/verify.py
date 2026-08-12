"""Reproducible headline run for Study 876 — Industry-Relative MAX.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached cross-section panel under
``_cache/`` (fetching once on a cache miss through the quantlab.universe survivorship guard),
and always runs the synthetic control with no network.

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

from max_industry import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Industry-Relative MAX — does adjusting MAX for its sector sharpen or kill the effect?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the survivorship guard)")
    data.fetch()

panel = data.load_panel()
closes = pd.DataFrame({s: panel[s]["Close"] for s in data.UNIVERSE if s in panel})
print(f"[data] {len(panel)} names, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(closes)}")
print("  SURVIVORSHIP: current-membership mega-cap panel — magnitudes are an upper bound "
      "(delisted names absent). Named on the Signal axis.")

mp = data.build_panel(panel)
mx, fwd, sec = mp["max"], mp["fwd_ret"], mp["sectors"]
n_sec = sec.nunique()
print(f"  monthly MAX panel: {mx.shape[0]} months, {mx.shape[1]} names, {n_sec} GICS sectors")

print("\n# HEAD-TO-HEAD — raw MAX vs industry-relative MAX (Q1 low - Q5 high, next-month)")
res = {}
for tag, adj in [("raw MAX", False), ("industry-relative MAX", True)]:
    r = st.run_sort(mx, fwd, sec, adjusted=adj)
    ss = st.spread_stats(r["spread"])
    res[tag] = (r, ss)
    print(f"  {tag:>22}: spread {ss['mean_bps']:+.1f} bps/mo  NW t = {ss['tstat']:+.2f}  "
          f"one-sample t = {ss['t_1s']:+.2f}  win {ss['win']*100:.0f}%  "
          f"Sharpe {ss['sharpe']:+.2f}  placebo p = {ss['p_placebo']:.3f}  (n={ss['n']})")

# focus on the industry-relative book for the rest
r_adj, ss_adj = res["industry-relative MAX"]
spread_adj = r_adj["spread"]
qs = st.quantile_summary(r_adj["qret"])
print("\n# QUINTILE MONOTONICITY — industry-relative MAX (annualised mean return)")
for q in r_adj["qret"].columns:
    print(f"  {q}: {qs.loc[q, 'mean_ann']*100:+.2f}%/yr  (Sharpe {qs.loc[q, 'sharpe']:+.2f})")

print("\n# PLACEBO — sign-flip null on the industry-relative spread (20,000 draws)")
pl = st.placebo_pvalue(spread_adj)
print(f"  observed {pl['obs_mean']*1e4:+.1f} bps vs placebo mean {pl['placebo_mean']*1e4:+.3f} "
      f"-> p = {pl['p_value']:.4f}")

print("\n# ROBUSTNESS — two eras (split 2018-01-01), industry-relative spread")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = spread_adj[(spread_adj.index >= lo) & (spread_adj.index < hi)]
    ss = st.spread_stats(sub)
    print(f"  {lbl}: n={ss['n']}  spread {ss['mean_bps']:+.1f} bps  NW t = {ss['tstat']:+.2f}")

print("\n# THE TIMER — long low-MAX / short high-MAX (industry-relative), costed")
print("  2 sides x one-way cost x NAV per rebalance; short book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(spread_adj, r_adj["qret"], high=r_adj["high"],
                        cost_bps=cb, borrow_ann_bps=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.1f} -> net {tm['net_bps']:+.1f} "
          f"bps/mo (cost {tm['cost_bps']:.1f}/mo, t={tm['t_net']:+.2f}, "
          f"Sharpe {tm['sharpe_net']:+.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  planted: idiosyncratic MAX priced (negative), sector-wide MAX un-priced")
null_adj = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=876 + s_, n_months=180)
    null_adj.append(st.synthetic_detect(p0, adjusted=True)["t_nw"])
null_adj = np.asarray(null_adj)
print(f"  null (edge=0), 20 seeds [industry-adj]: NW t mean {null_adj.mean():+.2f} "
      f"(sd {null_adj.std(ddof=1):.2f}), |t|>=2 in {(abs(null_adj) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.012, seed=876, n_months=240)
raw_t = st.synthetic_detect(p1, adjusted=False)["t_nw"]
adj_t = st.synthetic_detect(p1, adjusted=True)["t_nw"]
print(f"  planted (edge=0.012, seed 876): raw-MAX NW t = {raw_t:+.2f}  vs  "
      f"industry-relative NW t = {adj_t:+.2f}  (adjustment SHARPENS the planted effect)")
