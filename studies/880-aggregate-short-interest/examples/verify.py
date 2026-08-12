"""Reproducible headline run for Study 880 — Aggregate Short Interest.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached FINRA short-interest
panel + SPY under ``_cache/`` (fetching once on a cache miss), and always runs the
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

from quantlab.repro import fingerprint  # noqa: E402

from agg_short import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Aggregate Short Interest — does market-wide short interest predict the market? "
      "(Rapach-Ringgenberg-Zhou 2016)")

if not data.have_real():
    print("(cache miss — fetching the FINRA short-interest panel + SPY once)")
    data.fetch()

real = data.load_real()
idx = real["index"]
print(f"[data] FINRA consolidated short interest, {len(data.PANEL)}-name liquid panel, "
      f"{len(idx)} bi-monthly settlement dates "
      f"{idx.index.min().date()} -> {idx.index.max().date()}  as-of {data.AS_OF}")
print(f"       aggregate index (equal-weight mean days-to-cover): "
      f"last={idx['si_index'].iloc[-1]:.3f}  median names/date={int(idx['n'].median())}  "
      f"fingerprint={fingerprint(idx)}")
print("  SURVIVORSHIP/PROXY: current-membership mega-cap panel; index is a days-to-cover "
      "average (FINRA has no shares-outstanding) not the paper's shares-outstanding ratio. "
      "Named on the Signal axis.")

print("\n# THE HEADLINE — predictive regression of forward SPY return on the detrended "
      "aggregate short-interest index (RRZ predict a NEGATIVE slope)")
print("  forward horizons in bi-monthly settlement periods (~0.5 mo each); one publication "
      "lag (signal at t acted on the next settlement t+1)")
for h in (1, 2, 3, 6):
    fr = st.build_frame(real, horizon=h, lag=1)
    reg = st.predictive_regression(fr, nw_lags=max(4, h + 2))
    print(f"  H={h} (~{h*0.5:.1f} mo): n={reg['n']:3d}  beta={reg['beta']*1e4:+8.1f} bps/z  "
          f"NW t={reg['t_nw']:+.2f}  R2={reg['r2']*100:+.2f}%  fwd mean={reg['fwd_mean_bps']:+.0f} bps")

# Headline horizon = 1 (native bi-monthly forward return)
fr = st.build_frame(real, horizon=1, lag=1)
reg = st.predictive_regression(fr, nw_lags=6)

print("\n# HIGH vs LOW short-interest tercile — forward one-period SPY return")
x = fr["sii"].to_numpy(); y = fr["fwd"].to_numpy()
q1, q2 = np.quantile(x, [1/3, 2/3])
hi = y[x >= q2]; lo = y[x <= q1]
print(f"  low-SI tercile fwd mean  {lo.mean()*1e4:+.0f} bps (n={len(lo)})")
print(f"  high-SI tercile fwd mean {hi.mean()*1e4:+.0f} bps (n={len(hi)})  "
      f"Welch t(high-low) = {st.welch_t(hi, lo):+.2f}")

print("\n# PLACEBO — permute forward returns vs the index (5,000 draws, left-tail on beta<0)")
pl = st.placebo_pvalue(fr, n_draws=5000)
print(f"  observed beta {pl['obs_beta']*1e4:+.2f} bps vs placebo mean {pl['placebo_mean']*1e4:+.3f} "
      f"(sd {pl['placebo_sd']*1e4:.2f}) -> left-tail p = {pl['p_value']:.4f}")

print("\n# ROBUSTNESS — two eras (split 2022-06-01), horizon = 1")
es = st.era_split(fr, "2022-06-01", nw_lags=6)
for lbl, r in [("2017-12 -> 2022-05", es["early"]), ("2022-06 -> 2026-06", es["late"])]:
    print(f"  {lbl}: n={r['n']:3d}  beta={r['beta']*1e4:+7.1f} bps/z  NW t={r['t_nw']:+.2f}  "
          f"R2={r['r2']*100:+.2f}%")

print("\n# THE TIMER — de-risk to cash when shorts are crowded (sii>0); one-way cost x NAV")
for cb in (1.0, 5.0):
    tm = st.timer_stats(fr, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: overlay net {tm['overlay_net_bps']:+.1f} bps/period "
          f"({tm['overlay_ann_pct']:+.1f}%/yr, Sharpe {tm['overlay_sharpe']:.2f}, "
          f"{tm['n_switches']} switches) vs buy-hold {tm['bh_ann_pct']:+.1f}%/yr")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
nt = np.array([st.synthetic_detect(data.synthetic_frame(edge=0.0, seed=880 + s,
              n_periods=200), horizon=1, lag=1)["t_nw"] for s in range(20)])
print(f"  null (edge=0), 20 seeds: NW slope t mean {nt.mean():+.2f} (sd {nt.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(nt) >= 2).sum()}/20 seeds")
dp = st.synthetic_detect(data.synthetic_frame(edge=0.015, seed=880, n_periods=200),
                         horizon=1, lag=1)
print(f"  planted (edge=0.015, seed 880): beta={dp['beta']*1e4:+.1f} bps/z, "
      f"NW slope t = {dp['t_nw']:+.2f}, R2 = {dp['r2']*100:.1f}%")
