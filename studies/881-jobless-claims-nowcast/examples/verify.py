"""Reproducible headline run for Study 881 — Jobless-Claims Sector Rotation.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached sector-ETF tape under
``_cache/`` (fetching once on a cache miss via yfinance) and the documented public
4-week-MA claims snapshot, and always runs the synthetic control with no network.

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
from scipy.stats import spearmanr  # noqa: E402

from claims_nowcast import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Jobless-Claims Sector Rotation — does a claims uptick tilt the market to defensives?")

if not data.have_real():
    print("(cache miss — fetching the sector ETFs once via yfinance)")
    data.fetch_etfs()

frame = data.load_real()
fp = data.fingerprint(frame)
print(f"[data] {len(frame)} months  {frame.index.min().date()} -> {frame.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint(sector closes)={fp}")
print("  claims: FRED IC4WSA 4-week-MA (public snapshot; FRED CSV host unreachable here). "
      "SURVIVORSHIP: the four SPDR sector ETFs are continuous since 1998 — no delisting.")

spread = st.cyc_def_spread(frame)
print(f"  cyclical(XLY,XLI) - defensive(XLP,XLU) monthly spread: mean "
      f"{spread.mean()*100:+.3f}%  sd {spread.std()*100:.2f}%")

print("\n# THE HEADLINE — predictive regression  spread_{t+1} ~ dclaims_t  (NW t on slope)")
reg = st.predictive_regression(frame, k=1, lag=1)
print(f"  slope {reg['slope']:+.4f}  NW(6) t = {reg['t_nw']:+.2f}  R2 = {reg['r2']:.4f}  "
      f"corr = {reg['corr']:+.4f}  (n={reg['n']})")
print("  CLAIM needs a NEGATIVE slope (rising claims -> cyclicals under-earn); "
      f"observed sign is {'NEGATIVE' if reg['slope'] < 0 else 'POSITIVE (wrong)'}.")

print("\n# COVID SENSITIVITY — is the fitted slope one 2020 outlier?")
cs = st.covid_sensitivity(frame, k=1, lag=1)
for key, lbl in [("full", "full sample"), ("ex_covid", "ex-COVID 2020"),
                 ("winsor", "winsor x 1/99")]:
    d = cs[key]
    print(f"  {lbl:<14}: slope {d['slope']:+.4f}  NW t = {d['t_nw']:+.2f}  (n={d['n']})")
x, y, _ = st.build_xy(frame, k=1, lag=1)
sp = spearmanr(x, y)
print(f"  Spearman rank-corr (outlier-robust): rho = {sp.statistic:+.4f}  p = {sp.pvalue:.3f}")

print("\n# ROBUSTNESS — two eras (split 2012-01-01)")
er = st.era_regressions(frame, split="2012-01-01")
for key, lbl in [("early", "1999-2011"), ("late", "2012-2026")]:
    d = er[key]
    print(f"  {lbl}: n={d['n']}  slope {d['slope']:+.4f}  NW t = {d['t_nw']:+.2f}")

print("\n# PLACEBO — shuffle the claims change vs the forward spread (2,000 draws)")
pl = st.placebo_pvalue(frame, k=1, lag=1, n_draws=2000)
print(f"  observed slope {pl['obs_slope']:+.4f} vs placebo mean {pl['placebo_mean']:+.5f} "
      f"(sd {pl['placebo_sd']:.4f}) -> two-sided p = {pl['p_value']:.4f}")

print("\n# THE TIMER — long-short cyclical/defensive rotation, costed")
print("  rising claims -> short cyc-def (defensive); falling -> long cyc-def; borrow 50 bps/yr")
for cb in (0.0, 10.0):
    tm = st.rotation_timer(frame, k=1, lag=1, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_ann_pct']:+.2f}%/yr -> net "
          f"{tm['net_ann_pct']:+.2f}%/yr (t_net {tm['net_t']:+.2f}, Sharpe {tm['net_sharpe']:+.2f}, "
          f"{tm['n_switches']} switches)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([st.synthetic_detect(data.synthetic_frame(edge=0.0, seed=881 + s,
                                                            n_months=360))["t_nw"]
                   for s in range(20)])
print(f"  null (edge=0), 20 seeds: slope NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
planted = st.synthetic_detect(data.synthetic_frame(edge=0.5, seed=881, n_months=360))
print(f"  planted (edge=0.5, seed 881): slope {planted['slope']:+.4f}, NW t = {planted['t_nw']:+.2f} "
      f"(recovers the claim's NEGATIVE sign)")
