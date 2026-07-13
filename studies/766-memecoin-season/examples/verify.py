"""Reproducible headline run for Study 766 — Memecoin-Season.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached BTC/DOGE/SHIB tapes under
``_cache/`` (fetching once on a cache miss). The synthetic control always runs, no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from memecoin_season import data, strategy as st  # noqa: E402

print("# Memecoin-Season — can a momentum rotation harvest the DOGE/SHIB blow-off past BTC?")

if not data.have_real():
    print("(cache miss — fetching BTC/DOGE/SHIB once)")
    data.fetch_all(end="2026-07-01")

wk = data.weekly_prices()
daily = data.load_prices()
print(data_stamp("BTC/DOGE/SHIB weekly close", wk, asof=data.AS_OF))
print(f"  universe window (all three exist): {wk.index.min().date()} -> {wk.index.max().date()} "
      f"({len(wk)} weekly bars). SURVIVORSHIP: DOGE & SHIB are the 2 memecoins that survived "
      "out of thousands — every number below is an ex-post upper bound.")

# --- Did the memecoins actually blow past BTC? (the steelman) --------------------
tot = (daily.iloc[-1] / daily.iloc[0] - 1.0) * 100
print("\n# The steelman — total return over the common window (buy-and-hold each)")
for a in data.ASSETS:
    print(f"  {a:<4}: {tot[a]:+,.0f}%")

# --- The rotation, gross and net -------------------------------------------------
print("\n# Momentum rotation (weekly, top trailing-4w coin, one-week lag) vs benchmarks")
gross = st.run_rotation(wk, lookback=4, cost_bps=0.0)
net = st.run_rotation(wk, lookback=4, cost_bps=30.0)
btc = st.btc_hodl(wk)
ew = st.equal_weight(wk)
sg, sn = st.summarize(gross["net_ret"]), st.summarize(net["net_ret"])
sb, se = st.summarize(btc), st.summarize(ew)
print(f"  rotation GROSS : total {sg['total_pct']:+10,.0f}%  CAGR {sg['cagr_pct']:+7.1f}%  "
      f"Sharpe {sg['sharpe']:+.2f}  maxDD {sg['maxdd_pct']:+.0f}%")
print(f"  rotation NET   : total {sn['total_pct']:+10,.0f}%  CAGR {sn['cagr_pct']:+7.1f}%  "
      f"Sharpe {sn['sharpe']:+.2f}  maxDD {sn['maxdd_pct']:+.0f}%  "
      f"({net['n_switches']} switches, avg turnover {net['avg_turnover']:.2f} legs/wk)")
print(f"  BTC HODL       : total {sb['total_pct']:+10,.0f}%  CAGR {sb['cagr_pct']:+7.1f}%  "
      f"Sharpe {sb['sharpe']:+.2f}  maxDD {sb['maxdd_pct']:+.0f}%")
print(f"  equal-weight   : total {se['total_pct']:+10,.0f}%  CAGR {se['cagr_pct']:+7.1f}%  "
      f"Sharpe {se['sharpe']:+.2f}  maxDD {se['maxdd_pct']:+.0f}%")

# --- Is the edge over BTC real? --------------------------------------------------
ex = st.excess_tstat(net["net_ret"], btc)
print(f"\n# Rotation (net) excess over BTC: mean {ex['mean_excess_pct']:+.3f}%/wk, "
      f"t = {ex['t']:+.2f} (n={ex['n']} weeks)  [REAL bar is |t| >= 2]")

# --- Random-rotation placebo -----------------------------------------------------
print("\n# Random-rotation placebo — pick a coin at random each week (4,000 seeds)")
pl = st.random_rotation_placebo(wk, cost_bps=30.0, n_seeds=4000)
print(f"  momentum net total  {pl['mom_total_pct']:+,.0f}%  (Sharpe {pl['mom_sharpe']:+.2f})")
print(f"  random median total {pl['rand_total_median_pct']:+,.0f}%  "
      f"(Sharpe {pl['rand_sharpe_median']:+.2f})")
print(f"  p(random total  >= momentum) = {pl['p_total']:.3f}")
print(f"  p(random Sharpe >= momentum) = {pl['p_sharpe']:.3f}")

# --- Where does any edge live? ---------------------------------------------------
print("\n# Sub-period split — the 2021 mania vs everything after")
sub = st.subperiod_table(wk, cut="2022-01-01", cost_bps=30.0)
for seg, row in sub.iterrows():
    print(f"  {seg:<16}: rot {row['rot_total_pct']:+8,.0f}% (Sh {row['rot_sharpe']:+.2f})  "
          f"vs BTC {row['btc_total_pct']:+8,.0f}% (Sh {row['btc_sharpe']:+.2f})  "
          f"excess t = {row['excess_t']:+.2f}")

# --- Synthetic positive control --------------------------------------------------
print("\n# Synthetic positive control — momentum rotation must beat equal-weight ONLY when")
print("  momentum persistence is planted. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s in range(20):
    w = data.synthetic_world(persistence=0.0, seed=766 + s)
    null_ts.append(st.momentum_edge_from_returns(w))
null_ts = np.asarray(null_ts)
print(f"  null (persistence=0), 20 seeds: mean excess t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
w = data.synthetic_world(persistence=0.35, seed=766)
planted_t = st.momentum_edge_from_returns(w)
print(f"  planted persistence=0.35 (seed 766): excess t = {planted_t:+.2f}")
