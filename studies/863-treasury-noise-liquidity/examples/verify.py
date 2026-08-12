"""Reproducible headline run for Study 863 — Treasury Noise Liquidity.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached CMT-yield + risk-ETF tape
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from treasury_noise import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Treasury Noise Liquidity — does a rough yield curve precede weak stocks / wide credit?")

if not data.have_real():
    print("(cache miss — fetching CMT yields + risk ETFs once)")
    data.fetch()

df = data.load_panel()
d = st.build_daily(df)
print(f"[data] {len(df)} daily rows -> {len(d)} signal days  "
      f"{d.index.min().date()} -> {d.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(yields)={data.fingerprint(df)}")
print(f"  noise (RMS residual of the quadratic curve fit): mean {d['noise'].mean():.4f} "
      f"%-pts, sd {d['noise'].std():.4f}, max {d['noise'].max():.4f}")
print("  SURVIVORSHIP: the four CMT indices + liquid ETFs are continuously-listed — no "
      "delisting bias; the noise measure is model-free (fixed quadratic fit).")
print("  RISK-FREE proxied at 0 (named on the Signal axis — shifts the intercept, not the "
      "slope that carries the t-stat).")

for tgt, name in [("ret_spy", "SPY (equity)"), ("ret_credit", "HYG-IEF (credit-excess)")]:
    print(f"\n# THE HEADLINE — forward {name} return regressed on noise_t (per 1σ)")
    for h in (5, 21, 63):
        hd = st.headline(d, target=tgt, horizon=h)
        print(f"  {h:2d}-day: slope {hd['slope_pct']:+6.3f}%/1σ  NW t {hd['t_nw']:+5.2f}  "
              f"OLS t {hd['t_ols']:+5.2f}  R2 {hd['r2']*100:5.2f}%  "
              f"hi-lo noise {hd['hi_minus_lo_pct']:+6.2f}%  n={hd['n']}")

for tgt in ("ret_spy", "ret_credit"):
    print(f"\n# ROBUSTNESS — two eras (split 2016-01-01), target {tgt}")
    for h in (5, 21):
        print(f"  horizon {h}d:")
        print(st.era_cut(d, target=tgt, horizon=h, split="2016-01-01").round(4).to_string())

print("\n# PLACEBO — block-rotate forward returns vs noise (3,000 draws), left-tail (claim: negative)")
for tgt in ("ret_spy", "ret_credit"):
    for h in (5, 21):
        pl = st.placebo_pvalue(d, target=tgt, horizon=h, n_perm=3000)
        sig = (pl["obs_slope"] - pl["placebo_mean"]) / pl["placebo_sd"]
        print(f"  {tgt:10s} {h:2d}d: obs {pl['obs_slope']:+.5f} vs null mean {pl['placebo_mean']:+.5f} "
              f"(sd {pl['placebo_sd']:.5f}) -> {sig:+.2f}σ, left-tail p = {pl['p_value']:.4f}")

print("\n# THE TIMER — own SPY when noise < expanding median, else cash (one-way cost/switch)")
for cb in (1.0, 3.0):
    tm = st.timer_stats(d, cost_bps=cb)
    print(f"  cost {cb:.1f} bps: timer Sharpe {tm['timer_sharpe']:.3f} vs buy-and-hold "
          f"{tm['bh_sharpe']:.3f}; switches/yr {tm['switches_per_yr']:.1f}; "
          f"invested {tm['invested_frac']*100:.0f}%")
    print(f"           timer - buy&hold {tm['spread_bps_day']:+.2f} bps/day (NW t {tm['spread_t']:+.2f})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network (SPY 21d)")
for e in (0.0, 0.03):
    r = st.synthetic_mean_t(data, edge=e, n_seeds=20, target="ret_spy", horizon=21)
    tag = "null   " if e == 0 else "planted"
    print(f"  {tag} (edge={e:.2f}), 20 seeds: slope {r['mean_slope_pct']:+.3f}%/1σ  "
          f"mean NW t {r['mean_t']:+.2f}  R2 {r['mean_r2']*100:.1f}%  "
          f"fires {int(r['fire_frac']*20)}/20")
print("  (credit leg, 20 seeds, 21d:)")
for e in (0.0, 0.05):
    r = st.synthetic_mean_t(data, edge=e, n_seeds=20, target="ret_credit", horizon=21)
    tag = "null   " if e == 0 else "planted"
    print(f"  {tag} (edge={e:.2f}): slope {r['mean_slope_pct']:+.3f}%/1σ  "
          f"mean NW t {r['mean_t']:+.2f}  fires {int(r['fire_frac']*20)}/20")

print("\n# VERDICT: Signal WEAK (right sign, full-tape SPY 5d NW t=-2.20 & placebo-real, "
      "but crisis-era-concentrated and decays post-2016; credit leg fires only in the GFC) "
      "| Tradability MIRAGE | out-of-sample DECAYED.")
