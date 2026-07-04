"""Reproducible headline run for Study 615 — Yen Safe Haven.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/ysh_tape.csv, built once from yfinance) and always-offline on the synthetic
control.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from yen_safe_haven import data, strategy as st
from quantlab import repro


def show_beta(tag: str, b: dict) -> None:
    print(f"  {tag:<28s} n={b['n']:>5d}  beta_dn={b['beta_dn']:+.3f} "
          f"(HAC t={b['t_dn']:+.2f})  beta_up={b['beta_up']:+.3f} (t={b['t_up']:+.2f})  "
          f"beta_all={b['beta_all']:+.3f} (t={b['t_all']:+.2f})  "
          f"R2={b['r2']:.3f}  NW lags={b['lags']}")


print("# Yen Safe Haven — JPY=X vs SPY daily, yfinance (yen return = -USDJPY pct change)")
if not data.have_real():
    print("(no _cache/ysh_tape.csv — run data.fetch_tape() once to build the cache)")
    sys.exit(0)

tape = data.load_real()
rets = st.to_returns(tape)
m = st.monthly_returns(tape)
print(repro.data_stamp("daily tape", tape.dropna(subset=["usdjpy", "spy"]),
                       cols=["usdjpy", "spy", "irx", "tnx"], asof=data.AS_OF))
print(f"tape: {rets.index.min().date()} -> {rets.index.max().date()}  "
      f"({len(rets):,} joint daily returns, {len(m)} complete months; "
      f"JPY is price-only spot vs USD, SPY total-return)")

print("\n# Conditional (downside) beta — jpy ~ min(spy,0) + max(spy,0), daily, HAC/NW t")
print("  (the safe-haven claim is beta_dn < 0: SPY down-moves push the yen UP)")
B = st.downside_beta(rets)
show_beta("full sample", B)

print("\n# Non-stationarity — the daily signal by era (skeptic check: is it still live?)")
print("  (the full-sample t is carried by the first half; the recent decade is a flat zero)")
for lo, hi in [(1996, 2010), (2011, 2026), (2016, 2026)]:
    sub = rets[(rets.index.year >= lo) & (rets.index.year <= hi)]
    b = st.downside_beta(sub)
    q = st.quintile_ladder(sub)
    print(f"  {lo}-{hi}: n={b['n']:>5,}  beta_dn={b['beta_dn']:+.3f} (HAC t={b['t_dn']:+.2f})  "
          f"Q1={q['q1_bps']:+6.2f} bps/day  Welch t(Q1 vs mid)={q['t_q1_vs_mid']:+.2f}")

print("\n# Quintile ladder — mean JPY return by SPY-day quintile (Q1 = worst SPY days)")
Q = st.quintile_ladder(rets)
for k, (mu, n) in enumerate(zip(Q["means_bps"], Q["ns"]), start=1):
    print(f"  Q{k}: {mu:+6.2f} bps/day  (n={n:,})")
print(f"  SPY mean in Q1 = {Q['spy_q1_bps']:+.1f} bps/day")
print(f"  Welch t (Q1 vs Q5)  = {Q['t_q1_vs_q5']:+.2f}")
print(f"  Welch t (Q1 vs mid) = {Q['t_q1_vs_mid']:+.2f}")

print("\n# Monthly — JPY in SPY down-months vs up-months (Welch t) + worst-decile months")
DM = st.down_month_split(m)
print(f"  down-months (n={DM['n_dn']}): JPY {DM['dn_bps']:+.2f} bps/mo | "
      f"up-months (n={DM['n_up']}): {DM['up_bps']:+.2f} bps/mo | "
      f"Welch t = {DM['t_dn_vs_up']:+.2f}")
print(f"  worst-decile SPY months (n={DM['dec_n']}, SPY {DM['dec_spy_pct']:+.2f}%/mo): "
      f"JPY {DM['dec_jpy_pct']:+.2f}%/mo, up in {DM['dec_hit']*100:.0f}% of them")

print("\n# Event studies — cumulative SPY vs JPY over the named crash windows")
EV = st.event_table(tape)
for e in EV:
    tick = "HEDGED" if e["hedged"] else "FAILED"
    print(f"  {e['label']:<44s} {e['start']} -> {e['end']}: "
          f"SPY {e['spy_pct']:+7.2f}%  JPY {e['jpy_pct']:+7.2f}%  [{tick}]")

print("\n# Regime split A — carry-stock proxy: US 13w bill >= 2% (fat differential) vs < 2% (ZIRP)")
RG = st.carry_regime_split(rets, cut=2.0)
show_beta(f"bill >= 2% ({RG['share_hi']*100:.0f}% of days)", RG["hi"])
show_beta("bill <  2% (ZIRP)", RG["lo"])

print("\n# Regime split B — SPY down-months by 10y-yield direction (growth scare vs rate shock)")
YD = st.yield_direction_split(m)
print(f"  yields FELL (growth scare, n={YD['n_fell']}): JPY {YD['fell_bps']:+.2f} bps/mo")
print(f"  yields ROSE (rate shock,   n={YD['n_rose']}): JPY {YD['rose_bps']:+.2f} bps/mo")
print(f"  Welch t of the gap = {YD['t_gap']:+.2f}")

print("\n# Tradability — the cost of holding the JPY sleeve permanently (excess-vs-excess)")
HC = st.hedge_cost(rets)
print(f"  JPY spot vs USD (price-only): {HC['spot_ann_pct']:+.2f}%/yr over {HC['years']:.1f} years")
print(f"  minus the US bill given up (avg {HC['avg_bill_pct']:.2f}%/yr): "
      f"excess {HC['excess_ann_pct']:+.2f}%/yr")
print(f"  minus FXY-style expense ({HC['expense_bps']:.0f} bps/yr): net {HC['net_ann_pct']:+.2f}%/yr")
print("  (static sleeve held in advance — no timing signal, no lag to apply; entry cost <1 bp/yr amortised)")

print("\n# 2022 focus — the third axis: did the hedge fail exactly when needed?")
ev22 = [e for e in EV if e["start"] == "2022-01-03"][0]
print(f"  2022 bear: SPY {ev22['spy_pct']:+.2f}% top->trough while JPY {ev22['jpy_pct']:+.2f}% "
      f"— the 'hedge' fell ~{abs(ev22['jpy_pct'])/abs(ev22['spy_pct'])*100:.0f}% as much as the thing it was hedging")

print("\n# Synthetic control — machinery proof (never market evidence)")
print("  planted crash-hedge knob; the detector must stay flat on the null and light up on the plant")
for hedge in (0.0, 0.30):
    world = data.synthetic_world(hedge=hedge, seed=615)
    r = st.to_returns(world)
    b = st.downside_beta(r)
    q = st.quintile_ladder(r)
    print(f"  hedge={hedge:.2f}: beta_dn={b['beta_dn']:+.3f} (HAC t={b['t_dn']:+.2f})  "
          f"Q1 mean={q['q1_bps']:+.2f} bps/day  Welch t(Q1 vs Q5)={q['t_q1_vs_q5']:+.2f}")
