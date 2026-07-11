"""Reproducible headline run for Study 651 — Sugar-Seasonality.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached CANE (ETF) and SB=F (futures) tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from sugar_seasonality import data, strategy as st  # noqa: E402

print("# Sugar-Seasonality — does raw sugar (No.11) have a Brazil/India crush calendar?")
print("\n# The hardcoded crush calendar (facts, no network)")
print(f"  Brazil Center-South crush: {data.SUGAR_CALENDAR['brazil_crush']}")
print(f"  India crushing season    : {data.SUGAR_CALENDAR['india_crush']}")
print(f"  pre-harvest tight window : {data.SUGAR_CALENDAR['pre_harvest_tight']}")
print(f"  tested TIGHT months (claimed high): {data.TIGHT_MONTHS}")
print(f"  tested CRUSH months (claimed low) : {data.CRUSH_MONTHS}")

if not data.have_real():
    print("(cache miss — fetching CANE + SB=F once)")
    data.fetch()

etf, fut = data.load_real()
print(f"\n# Real tape: {data.START} -> {data.AS_OF} (CANE's 2011-09-19 inception sets the start)")
print(data_stamp(f"{data.ETF_TICKER} adj close", etf.to_frame("close"), asof=data.AS_OF))
print(data_stamp(f"{data.FUT_TICKER} close (roll-naive spot proxy)", fut.to_frame("close"), asof=data.AS_OF))

monthly = st.monthly_log_returns(etf)
fut_monthly = st.monthly_log_returns(fut)
print(f"\nmonths on tape: {len(monthly)} (CANE)  |  {len(fut_monthly)} (SB=F)")

print("\n# THE HEADLINE — per-month one-sample stats, 12 cells")
crit = st.bonferroni_crit_t(12, df=13)
print(f"Bonferroni bar for 12 simultaneous tests (df~13): |t| >= {crit:.2f}")
ms = st.month_stats(monthly)
n_survive = 0
for m, row in ms.iterrows():
    flag = ""
    if abs(row["tstat_hac"]) >= crit:
        n_survive += 1
        flag = "  SURVIVES"
    print(f"  month {int(m):>2}: mean {row['mean']*100:+.2f}%  n={int(row['n']):>2}  "
          f"t_naive={row['tstat']:+.2f}  t_hac={row['tstat_hac']:+.2f}{flag}")
print(f"cells clearing the Bonferroni bar: {n_survive}/12")

print("\n# Best / worst calendar month, Welch t vs every other month pooled")
bw = st.best_worst_vs_rest(monthly)
print(f"  best month {bw['best_month']:>2} ({bw['best_mean']*100:+.2f}%, t={bw['best_t']:+.2f}, "
      f"n={bw['best_n']})   worst month {bw['worst_month']:>2} ({bw['worst_mean']*100:+.2f}%, "
      f"t={bw['worst_t']:+.2f}, n={bw['worst_n']})")

print("\n# Pre-harvest TIGHT vs CRUSH-glut (the decisive number)")
tc = st.tight_crush_tstat(monthly, data.TIGHT_MONTHS, data.CRUSH_MONTHS)
print(f"  tight {tc['tight_mean']*100:+.2f}% (n={tc['n_tight']})  vs  crush "
      f"{tc['crush_mean']*100:+.2f}% (n={tc['n_crush']})   spread {tc['spread']*100:+.2f}%   "
      f"Welch t = {tc['t']:+.2f}")
ci = st.spread_bootstrap_ci(monthly, data.TIGHT_MONTHS, data.CRUSH_MONTHS)
print(f"  circular block-bootstrap 95% CI on the spread ({ci['n_boot']} draws, 12-month blocks): "
      f"[{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%]")

print("\n# THIRD AXIS, part 1 — the ETF's own roll vs the roll-naive futures splice ('contango' caveat)")
drag = st.roll_drag(monthly, fut_monthly)
print(f"  CANE {drag['etf_mean_bps']:+.1f} bps/mo  vs  SB=F splice {drag['fut_mean_bps']:+.1f} bps/mo"
      f"   drag {drag['drag_bps']:+.1f} bps/mo   t = {drag['t']:+.2f}  (n={drag['n']})")

print("\n# THIRD AXIS, part 2 — the seasonal timer (long tight, short crush, cash otherwise) vs buy-and-hold")
print("  costs: 4 one-way legs/yr x 10 bps x NAV, spread across the 12 months")
timer = st.seasonal_timer(monthly, data.TIGHT_MONTHS, data.CRUSH_MONTHS)
net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=10.0)
s_bh, s_g, s_n = st.summary(monthly), st.summary(timer), st.summary(net)
print(f"  buy&hold Sharpe {s_bh['sharpe']:+.2f} (CAGR {s_bh['cagr']*100:+.1f}%)   "
      f"timer gross Sharpe {s_g['sharpe']:+.2f} (CAGR {s_g['cagr']*100:+.1f}%)   "
      f"timer net Sharpe {s_n['sharpe']:+.2f} (CAGR {s_n['cagr']*100:+.1f}%)")

print("\n# 'Beats a coin?' — hit rate on the timer's ACTIVE legs only (excludes cash months)")
active = timer[timer != 0.0].dropna()
x = active.to_numpy()
k, n = int((x > 0).sum()), len(x)
lo, hi = st.wilson_interval(k, n)
print(f"  active months {n}  hit {k}/{n} = {k/n*100:.1f}%  (Wilson [{lo*100:.1f}%, {hi*100:.1f}%])  "
      f"mean {x.mean()*100:+.2f}%  t = {st._one_sample_t(x):+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (seasonal=0) and must recover a")
print("  planted tight/crush spread. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    df = data.synthetic_world(seasonal=0.0, seed=651 + s_)
    null_ts.append(st.synthetic_detect(df, data.TIGHT_MONTHS, data.CRUSH_MONTHS)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (seasonal=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
df = data.synthetic_world(seasonal=0.12, seed=651)
planted = st.synthetic_detect(df, data.TIGHT_MONTHS, data.CRUSH_MONTHS)
print(f"  planted seasonal=+12.0%/yr (seed 651): tight {planted['tight_mean']*100:+.2f}% vs "
      f"crush {planted['crush_mean']*100:+.2f}%  Welch t = {planted['t']:+.2f}")
