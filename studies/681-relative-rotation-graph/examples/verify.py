"""Reproducible headline run for Study 681 — Relative-Rotation-Graph (RRG).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily sector/SPY tape
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

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

from relative_rotation_graph import data, strategy as st  # noqa: E402

print("# Relative-Rotation-Graph — does the JdK quadrant chart rotate you into winning")
print("# S&P sectors, or is it plain momentum wearing a fancier chart?")

if not data.have_real():
    print("(cache miss — fetching 11 sector ETFs + SPY once)")
    data.fetch()

prices = data.load_real()
print(data_stamp("11 sector ETFs + SPY, daily adjusted closes", prices, asof=data.AS_OF))
print(f"RRG windows: RS-Ratio W={data.RS_WINDOW}d (~1 quarter), "
      f"RS-Momentum M={data.MOM_WINDOW}d (~1 month)")

frames = st.rrg_frame(prices, data.TICKERS, data.BENCHMARK, data.RS_WINDOW, data.MOM_WINDOW)
quad = st.monthly_quadrants(frames)
rets = st.monthly_returns(prices, data.TICKERS)
print(f"quadrant panel: {quad.shape[0]} months {quad.index.min()} -> {quad.index.max()} "
      f"x {quad.shape[1]} tickers")
print(f"returns panel: {rets.shape[0]} months {rets.index.min()} -> {rets.index.max()}")

COST_BPS = 5.0
rrg = st.run_rrg_strategy(quad, rets, cost_bps=COST_BPS)
ew = st.equal_weight_returns(rets)
spy_m = st.benchmark_monthly_returns(prices, data.BENCHMARK)
mom = st.run_momentum_strategy(rets, lookback=6, skip=1, k=3, cost_bps=COST_BPS)

print(f"\n# THE HEADLINE — long the Leading quadrant, monthly rebalance, "
      f"{COST_BPS:.0f}bps one-way cost")
rows = [("RRG (Leading quadrant)", st.summarize(rrg["r_net"])),
        ("Equal-weight sector basket", st.summarize(ew)),
        ("SPY buy-and-hold", st.summarize(spy_m)),
        ("Plain 6-1 top-3 momentum", st.summarize(mom["r_net"]))]
for name, s in rows:
    print(f"  {name:<28s} ann {s['ann_ret']*100:+6.2f}%  vol {s['ann_vol']*100:5.2f}%  "
          f"Sharpe {s['sharpe']:5.3f}  maxDD {s['max_drawdown']*100:6.2f}%  "
          f"NW t={s['tstat']:+5.2f}  (n={s['n']})")

print("\n# Holdings & turnover")
vc = rrg["n_leading"].value_counts().sort_index()
print("  sectors held per month:", {int(k): int(v) for k, v in vc.items()})
print(f"  mean sectors held {rrg['n_leading'].mean():.2f} / 11   "
      f"cash months {int((rrg['n_leading']==0).sum())}/{len(rrg)} "
      f"({(rrg['n_leading']==0).mean()*100:.1f}%)")
print(f"  mean one-way monthly turnover {rrg['turnover'].mean()*100:.1f}%")

print("\n# ACTIVE RETURN — does the quadrant rotation beat each control? (Newey-West t)")
for label, ctrl in (("equal-weight basket", ew), ("SPY", spy_m), ("plain 6-1 momentum", mom["r_net"])):
    a = st.active_stats(rrg["r_net"], ctrl)
    print(f"  RRG - {label:<20s}: {a['active_ann']*100:+6.2f}%/yr   t = {a['active_tstat']:+.2f}"
          f"   (n={a['n']})")

matched = st.matched_random_control(quad, rets, n_seeds=200, cost_bps=COST_BPS)
a_rand = st.active_stats(rrg["r_net"], matched)
print(f"  RRG - matched-random (200 seeds, same monthly holding count incl. cash):"
      f" {a_rand['active_ann']*100:+6.2f}%/yr   t = {a_rand['active_tstat']:+.2f}   (n={a_rand['n']})")
print("  -> the matched-random control isolates SELECTION skill from the mechanical")
print("     cash-timing drag of 'go to cash when nothing is Leading'.")

print(f"\n# COST SWEEP (RRG net, one-way bps per leg)")
for cb in (0.0, 5.0, 10.0, 20.0):
    r = st.run_rrg_strategy(quad, rets, cost_bps=cb)
    s = st.summarize(r["r_net"])
    print(f"  {cb:>5.1f} bps: ann {s['ann_ret']*100:+6.2f}%  Sharpe {s['sharpe']:.3f}  "
          f"NW t = {s['tstat']:+.2f}")

print("\n# SUB-PERIOD STABILITY — RRG vs equal-weight basket")
periods = [("1999-2009", "1999-07", "2009-12"), ("2010-2019", "2010-01", "2019-12"),
           ("2020-2026", "2020-01", "2026-06")]
for label, lo, hi in periods:
    m_r = (rrg.index >= lo) & (rrg.index <= hi)
    m_e = (ew.index >= lo) & (ew.index <= hi)
    rs = st.summarize(rrg.loc[m_r, "r_net"])
    es = st.summarize(ew.loc[m_e])
    print(f"  {label}: RRG ann {rs['ann_ret']*100:+6.2f}% Sharpe {rs['sharpe']:.3f} "
          f"t={rs['tstat']:+.2f} (n={rs['n']})  |  EW ann {es['ann_ret']*100:+6.2f}% "
          f"Sharpe {es['sharpe']:.3f} t={es['tstat']:+.2f}")

print("\n# THIRD AXIS (myth-check) — does the chart actually rotate clockwise?")
print(f"  claimed cycle: {' -> '.join(st.CYCLE_ORDER + [st.CYCLE_ORDER[0]])}")
trans = st.quadrant_transition_matrix(quad)
cw = st.clockwise_share(trans)
print("  transition counts (row = month t, col = month t+1):")
print(trans.to_string())
print(f"  pooled clockwise share of quadrant CHANGES: {cw['pooled_clockwise_pct']:.1f}%  "
      f"(random-chance baseline {cw['random_baseline_pct']:.1f}%, n={cw['n_moves']} changes)")
for q, pct in cw["per_quadrant_pct"].items():
    print(f"    from {q:<10s}: {pct:.1f}% of its moves go clockwise-forward")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  the matched-random-control detector must NOT fire on a null world (no planted")
print("  relative drift) and must recover a planted persistent rotation. Null checked")
print("  over 20 seeds (never a single stream).")
sec_cols = [f"SEC{i:02d}" for i in range(9)]
null_ts = []
for s_ in range(20):
    p, _truth = data.synthetic_panel(n_days=6300, n_sectors=9, mom_strength=0.0, seed=681 + s_)
    r = st.synthetic_detect(p, sec_cols, "BENCH", data.RS_WINDOW, data.MOM_WINDOW, n_control_seeds=50)
    null_ts.append(r["active_tstat"])
null_ts = np.asarray(null_ts)
print(f"  null (mom_strength=0), 20 seeds: mean t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")

p, truth = data.synthetic_panel(n_days=6300, n_sectors=9, mom_strength=0.0015, seed=681)
sy = st.synthetic_detect(p, sec_cols, "BENCH", data.RS_WINDOW, data.MOM_WINDOW, n_control_seeds=50)
print(f"  planted persistent relative drift (mom_strength=0.0015, seed=681): "
      f"active {sy['active_ann']*100:+.2f}%/yr vs matched-random,  t = {sy['active_tstat']:+.2f}")
