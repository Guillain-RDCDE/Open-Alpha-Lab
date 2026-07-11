"""Reproducible headline run for Study 648 — Grain Seasonality.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached CORN/WEAT/SOYB (ETF) and
ZC=F/ZW=F/ZS=F (futures) tapes under ``_cache/`` (fetching once on a cache miss), and always
runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from grain_seasonality import data, strategy as st  # noqa: E402

print("# Grain Seasonality — does corn/wheat/soy have a spring-high, harvest-low calendar?")
print("\n# The hardcoded USDA crop-progress calendar (facts, no network)")
for g in data.GRAINS:
    cal = data.GRAIN_CALENDAR[g]
    print(f"  {g:>5}: planting {cal['planting']}  weather-scare {cal['weather_scare']}  "
          f"harvest {cal['harvest']}")

if not data.have_real():
    print("(cache miss — fetching CORN/WEAT/SOYB + ZC=F/ZW=F/ZS=F once)")
    data.fetch()

etf, fut = data.load_real()
print(f"\n# Real tape: {data.START} -> {data.AS_OF} (WEAT's 2011-09-19 inception sets the common start)")
for g in data.GRAINS:
    print(data_stamp(f"{data.ETF_TICKERS[g]} adj close", etf[g].to_frame("close"), asof=data.AS_OF))
for g in data.GRAINS:
    print(data_stamp(f"{data.FUT_TICKERS[g]} close (roll-naive spot proxy)",
                      fut[g].to_frame("close"), asof=data.AS_OF))

monthly = {g: st.monthly_log_returns(etf[g]) for g in data.GRAINS}
fut_monthly = {g: st.monthly_log_returns(fut[g]) for g in data.GRAINS}
n_months = {g: len(monthly[g]) for g in data.GRAINS}
print(f"\nmonths on tape per grain: {n_months}")

print("\n# THE HEADLINE — per-month one-sample stats, 3 grains x 12 months = 36 cells")
crit = st.bonferroni_crit_t(36, df=13)
print(f"Bonferroni bar for 36 simultaneous tests (df~13): |t| >= {crit:.2f}")
n_survive = 0
for g in data.GRAINS:
    ms = st.month_stats(monthly[g])
    for m, row in ms.iterrows():
        if abs(row["tstat_hac"]) >= crit:
            n_survive += 1
            print(f"  SURVIVES: {g} month {int(m)}  mean {row['mean']*100:+.2f}%  "
                  f"t_hac {row['tstat_hac']:+.2f}")
print(f"cells clearing the Bonferroni bar: {n_survive}/36")

print("\n# Best / worst calendar month per grain, Welch t vs every other month pooled")
bw = {}
for g in data.GRAINS:
    r = st.best_worst_vs_rest(monthly[g])
    bw[g] = r
    print(f"  {g:>5}: best month {r['best_month']:>2} ({r['best_mean']*100:+.2f}%, "
          f"t={r['best_t']:+.2f}, n={r['best_n']})   "
          f"worst month {r['worst_month']:>2} ({r['worst_mean']*100:+.2f}%, "
          f"t={r['worst_t']:+.2f}, n={r['worst_n']})")

print("\n# Spring (weather-scare) vs harvest, per grain")
sh = {}
for g in data.GRAINS:
    r = st.spring_harvest_tstat(monthly[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g])
    sh[g] = r
    print(f"  {g:>5}: spring {r['spring_mean']*100:+.2f}% (n={r['n_spring']})  vs  "
          f"harvest {r['harvest_mean']*100:+.2f}% (n={r['n_harvest']})   "
          f"spread {r['spread']*100:+.2f}%   Welch t = {r['t']:+.2f}")

print("\n# Pooled spring vs harvest across all 3 grains (the 'old-crop/new-crop' headline)")
pooled = st.pooled_spring_harvest(monthly, data.SPRING_MONTHS, data.HARVEST_MONTHS)
print(f"  spring {pooled['spring_mean']*100:+.2f}% (n={pooled['n_spring']})  vs  "
      f"harvest {pooled['harvest_mean']*100:+.2f}% (n={pooled['n_harvest']})   "
      f"spread {pooled['spread']*100:+.2f}%   Welch t = {pooled['t']:+.2f}")
ci = st.pooled_spread_bootstrap_ci(monthly, data.SPRING_MONTHS, data.HARVEST_MONTHS)
print(f"  circular block-bootstrap 95% CI on the pooled spread "
      f"({ci['n_boot']} draws, 12-month blocks per grain): "
      f"[{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%]")

print("\n# THIRD AXIS — the ETF's own roll vs the roll-naive futures splice ('ETF contango' caveat)")
print("  (etf_mean/fut_mean are unconditional monthly means, bps; drag = etf - fut, "
      "one-sample t of the monthly gap)")
drag = {}
for g in data.GRAINS:
    r = st.roll_drag(monthly[g], fut_monthly[g])
    drag[g] = r
    print(f"  {g:>5}: ETF {r['etf_mean_bps']:+.1f} bps/mo  vs  futures splice {r['fut_mean_bps']:+.1f} "
          f"bps/mo   drag {r['drag_bps']:+.1f} bps/mo   t = {r['t']:+.2f}  (n={r['n']})")

print("\n# Seasonal long/short timer (long spring, short harvest, cash otherwise) vs buy-and-hold")
print("  costs: 4 one-way legs/yr x 10 bps x NAV, spread across the 12 months")
timer_rows = {}
for g in data.GRAINS:
    r = monthly[g]
    timer = st.seasonal_timer(r, data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g])
    net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=10.0)
    s_g, s_n, s_bh = st.summary(timer), st.summary(net), st.summary(r)
    timer_rows[g] = {"gross": s_g, "net": s_n, "bh": s_bh}
    print(f"  {g:>5}: buy&hold Sharpe {s_bh['sharpe']:+.2f} (CAGR {s_bh['cagr']*100:+.1f}%)   "
          f"timer gross Sharpe {s_g['sharpe']:+.2f} (CAGR {s_g['cagr']*100:+.1f}%)   "
          f"timer net Sharpe {s_n['sharpe']:+.2f} (CAGR {s_n['cagr']*100:+.1f}%)")

pooled_bh = sum(monthly[g].reindex(monthly["corn"].index) for g in data.GRAINS) / 3
pooled_timer = sum(st.seasonal_timer(monthly[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g])
                   for g in data.GRAINS) / 3
pooled_net = st.apply_costs(pooled_timer, n_trades_per_year=4, cost_bps_one_way=10.0)
s_pbh, s_pg, s_pn = st.summary(pooled_bh), st.summary(pooled_timer), st.summary(pooled_net)
print(f"  equal-weight 3-grain: buy&hold Sharpe {s_pbh['sharpe']:+.2f} (CAGR {s_pbh['cagr']*100:+.1f}%)   "
      f"timer gross Sharpe {s_pg['sharpe']:+.2f} (CAGR {s_pg['cagr']*100:+.1f}%)   "
      f"timer net Sharpe {s_pn['sharpe']:+.2f} (CAGR {s_pn['cagr']*100:+.1f}%)")

print("\n# 'Beats a coin?' — hit rate on the timer's ACTIVE legs only (excludes cash months)")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / (1 + z2 / n)
    return mid - half, mid + half


import pandas as pd  # noqa: E402

active_legs = []
for g in data.GRAINS:
    timer = st.seasonal_timer(monthly[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g])
    active = timer[timer != 0.0].dropna()
    active_legs.append(active)
    x = active.to_numpy()
    k, n = int((x > 0).sum()), len(x)
    lo, hi = _wilson(k, n)
    print(f"  {g:>5}: active months {n}  hit {k}/{n} = {k/n*100:.1f}%  "
          f"(Wilson [{lo*100:.1f}%, {hi*100:.1f}%])  mean {x.mean()*100:+.2f}%  "
          f"t = {st._one_sample_t(x):+.2f}")
all_active = pd.concat(active_legs).to_numpy()
k, n = int((all_active > 0).sum()), len(all_active)
lo, hi = _wilson(k, n)
print(f"  pooled: active months {n}  hit {k}/{n} = {k/n*100:.1f}%  "
      f"(Wilson [{lo*100:.1f}%, {hi*100:.1f}%])  mean {all_active.mean()*100:+.2f}%  "
      f"t = {st._one_sample_t(all_active):+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (seasonal=0) and must recover a")
print("  planted spring/harvest spread. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    df = data.synthetic_world(seasonal=0.0, seed=648 + s_)
    null_ts.append(st.synthetic_detect(df, (6, 7), (9, 10, 11))["t"])
null_ts = np.asarray(null_ts)
print(f"  null (seasonal=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
df = data.synthetic_world(seasonal=0.06, seed=648)
planted = st.synthetic_detect(df, (6, 7), (9, 10, 11))
print(f"  planted seasonal=+6.0%/yr (seed 648): spring {planted['spring_mean']*100:+.2f}% vs "
      f"harvest {planted['harvest_mean']*100:+.2f}%  Welch t = {planted['t']:+.2f}")
