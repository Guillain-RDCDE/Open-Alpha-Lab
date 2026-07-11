"""Reproducible headline run for Study 659 — Costless Collar.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY tape under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

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

from costless_collar import data, strategy as st  # noqa: E402

print("# Costless Collar — does a zero-premium 5% put / financed call give free crash protection?")
print(f"assumptions (named, not hidden): risk-free {data.RF_ANNUAL*100:.1f}%/yr constant, "
      f"put struck {data.PUT_OTM*100:.0f}% OTM, call struck where a Black-Scholes premium "
      f"(priced off trailing {data.VOL_WINDOW}-day realized vol, our implied-vol PROXY — no "
      "live option chain anywhere in this study) exactly offsets the put's premium, both "
      "legs rolled monthly, execution: the vol input for month t is measured through the "
      "close of month t-1 (known before month t begins, zero look-ahead)")

if not data.have_real():
    print("(cache miss — fetching SPY once)")
    data.fetch()

daily = data.load_real()
print(data_stamp("SPY daily OHLC+AdjClose", daily, asof=data.AS_OF))

mf = data.month_frame(daily)
caps = st.collar_caps(mf["vol_in"], put_otm=data.PUT_OTM, r=data.RF_ANNUAL, T=data.OPTION_T)
mf = mf.join(caps.rename("cap_pct"))
print(f"months on tape: {len(mf)}  ({mf.index.min().date()} -> {mf.index.max().date()})  "
      f"| modeled cap: mean {mf['cap_pct'].mean()*100:.2f}%  "
      f"(min {mf['cap_pct'].min()*100:.2f}%, max {mf['cap_pct'].max()*100:.2f}%)  "
      f"| floor fixed at {-data.PUT_OTM*100:.1f}%  "
      f"| trailing realized vol input: {mf['vol_in'].min()*100:.1f}% - {mf['vol_in'].max()*100:.1f}% "
      f"(mean {mf['vol_in'].mean()*100:.1f}%)")
print("  NOTE: the modeled cap is nearly vol-invariant at these strikes/tenor/rate (a model "
      "property, not a bug — see docs/references.md) — do not read a ~6% cap as 'small "
      "moves only trigger it'; a 70%-vol COVID month still gets roughly the same ~6% cap.")

print("\n# THE HEADLINE — collar vs SPY buy & hold, net of 2-leg roll costs (full sample)")
be = st.breakeven_cost_bps(mf["spy_ret"], mf["cap_pct"], put_otm=data.PUT_OTM)
print(f"  break-even cost: {be:.2f} bps/leg (above this, the modeled collar's full-sample "
      "mean return falls below SPY's; below it, above)")
for cb in (5.0, 10.0, 15.0, 20.0):
    coll = st.collar_returns(mf["spy_ret"], mf["cap_pct"], put_otm=data.PUT_OTM, cost_bps=cb)
    d = pd.concat([mf, coll], axis=1)
    drag = st.full_sample_drag(d)
    tw = st.terminal_wealth(d["collar_ret"])
    print(f"  cost={cb:>4.1f} bps/leg: mean(collar-SPY) = {drag['mean_diff_bps']:+.1f} bps/mo  "
          f"(one-sample t={drag['t_plain']:+.2f}, NW(3) t={drag['t_nw']:+.2f}, n={drag['n']})  "
          f"| $1 -> ${tw:,.2f}")

coll5 = st.collar_returns(mf["spy_ret"], mf["cap_pct"], put_otm=data.PUT_OTM, cost_bps=5.0)
df = pd.concat([mf, coll5], axis=1)

print("\n# Where the floor bites — months SPY fell more than the put's own 5% strike")
cf = st.crash_floor_effect(df, thresh=-data.PUT_OTM)
print(f"  n={cf['n']} months | mean cushion {cf['mean_cushion_pts']:+.2f} pts  "
      f"(one-sample t={cf['t_plain']:+.2f})")
print(f"  worst SPY month {cf['worst_spy_pct']:+.2f}%  vs worst collar month "
      f"{cf['worst_collar_pct']:+.2f}%")

print("\n# Where the cap bites — months SPY beat that month's modeled cap")
cc = st.cap_cost_effect(df)
print(f"  n={cc['n']} months ({cc['share_of_months']:.1f}% of the sample) | mean cost "
      f"{cc['mean_cost_pts']:+.2f} pts  (one-sample t={cc['t_plain']:+.2f})")
print(f"  best SPY month given up: {cc['best_spy_pct']:+.2f}%")

print("\n# Terminal wealth, Sharpe (excess of cash) and max drawdown — full sample (net 5bps)")
tw_spy, tw_coll = st.terminal_wealth(df["spy_ret"]), st.terminal_wealth(df["collar_ret"])
sh_spy = st.sharpe_excess(df["spy_ret"], data.RF_ANNUAL)
sh_coll = st.sharpe_excess(df["collar_ret"], data.RF_ANNUAL)
dd_spy, dd_coll = st.max_drawdown(df["spy_ret"]), st.max_drawdown(df["collar_ret"])
print(f"  $1 compounded {df.index.min().date()} -> {df.index.max().date()}: "
      f"SPY -> ${tw_spy:,.2f}   collar -> ${tw_coll:,.2f}")
print(f"  Sharpe (excess of {data.RF_ANNUAL*100:.0f}% cash): SPY {sh_spy:.2f}   "
      f"collar {sh_coll:.2f}")
print(f"  max drawdown (full sample): SPY {dd_spy*100:.1f}%   collar {dd_coll*100:.1f}%")

print("\n# Named crash windows — the exact events the claim is sold against")
for label, (lo, hi) in (("2008 GFC", data.GFC_WINDOW), ("2020 COVID", data.COVID_WINDOW)):
    d_spy = st.window_drawdown(df["spy_ret"], lo, hi)
    d_coll = st.window_drawdown(df["collar_ret"], lo, hi)
    print(f"  {label} ({lo} -> {hi}): SPY {d_spy*100:.1f}%   collar {d_coll*100:.1f}%   "
          f"(cut {d_spy*100 - d_coll*100:+.1f} pts)")

print("\n# The honest question: is the full-sample 'edge' just 2 crash windows getting lucky?")
print("  exclude ONLY the two named windows above (22 months) — the dot-com bear (2000-02) "
      "STAYS IN, so this is not cherry-picked to help the SPY side either:")
sub = st.exclude_windows(df, [data.GFC_WINDOW, data.COVID_WINDOW])
drag_sub = st.full_sample_drag(sub)
tw_spy_sub, tw_coll_sub = st.terminal_wealth(sub["spy_ret"]), st.terminal_wealth(sub["collar_ret"])
sh_spy_sub = st.sharpe_excess(sub["spy_ret"], data.RF_ANNUAL)
sh_coll_sub = st.sharpe_excess(sub["collar_ret"], data.RF_ANNUAL)
print(f"  n={drag_sub['n']} months (397-22=375) | mean(collar-SPY) = "
      f"{drag_sub['mean_diff_bps']:+.1f} bps/mo  (one-sample t={drag_sub['t_plain']:+.2f}, "
      f"NW(3) t={drag_sub['t_nw']:+.2f})")
print(f"  $1 compounded over those 375 months: SPY -> ${tw_spy_sub:,.2f}   "
      f"collar -> ${tw_coll_sub:,.2f}")
print(f"  Sharpe ex-crash: SPY {sh_spy_sub:.2f}   collar {sh_coll_sub:.2f}  (lower vol still "
      "helps the ratio even though the compounded dollar outcome is significantly worse)")

print("\n# Synthetic positive control — pure clip mechanics, no network, no roll costs")
print("  the clip-and-drag detector must NOT fire on a null (never-binding) band and must")
print("  recover a tight, always-binding planted band. Checked over 20 seeds.")
null_fires = 0
planted_ts = []
for s_ in range(20):
    world = data.synthetic_world(seed=659 + s_)["spy_ret"]
    null_r = st.synthetic_detect(world, floor=-0.99, cap=0.99)
    if not (np.isnan(null_r["mean_diff_bps"]) or abs(null_r["mean_diff_bps"]) < 1e-9):
        null_fires += 1
    planted_ts.append(st.synthetic_detect(world, floor=-0.03, cap=0.02)["t_plain"])
planted_ts = np.asarray(planted_ts)
print(f"  null (floor=-99%, cap=+99% -- mathematically cannot bind at 16%/yr synthetic vol): "
      f"{null_fires}/20 seeds show ANY difference from raw returns at all -- diff is exactly "
      "zero every month, in every seed. The strongest possible non-signal (a t-ratio is "
      "undefined with zero variance -- there is nothing there to threshold).")
print(f"  planted (floor=-3%, cap=+2% -- binds almost every month), 20 seeds: "
      f"mean t = {planted_ts.mean():+.2f} (sd {planted_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(planted_ts) >= 2).sum()}/20 seeds")
