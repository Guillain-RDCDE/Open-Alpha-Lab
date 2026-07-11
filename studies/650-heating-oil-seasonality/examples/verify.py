"""Reproducible headline run for Study 650 — Heating-Oil-Seasonality.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached HO=F / UHN / ^IRX tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from heating_oil_seasonality import data, strategy as st  # noqa: E402

print("# Heating-Oil-Seasonality — does HO=F really rally into winter on cold-weather demand?")

if not data.have_real():
    print("(cache miss — fetching HO=F / UHN / ^IRX once)")
    data.fetch()

ho, uhn, irx = data.load_real()
print(data_stamp("HO=F raw OHLC", ho, asof=data.AS_OF))
print(data_stamp("UHN adjusted close", uhn, asof=data.AS_OF))
print(data_stamp("^IRX close", irx, asof=data.AS_OF))

ho_ret = data.monthly_returns(ho["Close"], asof=data.AS_OF)
tbill = data.monthly_cash_rate(irx["Close"], ho_ret.index)
print(f"\nHO=F monthly returns: {ho_ret.index.min().date()} -> {ho_ret.index.max().date()} "
      f"({len(ho_ret)} months, month-end daily-close resample, hole-free)")

print("\n# THE HEADLINE — per-calendar-month mean HO=F return (one-sample t vs 0)")
print(f"  (Bonferroni-12 bar: |t| >= ~{st.BONFERRONI_T:.1f}, alpha=0.05/12 two-sided at n~25)")
ms = st.month_stats(ho_ret)
for m in range(1, 13):
    row = ms.loc[m]
    tag = "  <-- clears Bonferroni x12" if abs(row["t"]) >= st.BONFERRONI_T else ""
    print(f"  {data.MONTH_NAMES[m - 1]:4s}: mean {row['mean'] * 100:+6.2f}%  "
          f"t={row['t']:+.2f}  n={int(row['n'])}{tag}")

print("\n# Group tests — autumn-build (Sep-Nov) / winter-draw (Dec-Feb) / heating window (Sep-Feb)"
      "\n#              vs off-season (Mar-Aug), Welch t")
ab = st.group_welch(ho_ret, data.AUTUMN_BUILD_MONTHS, data.OFF_SEASON_MONTHS)
print(f"  autumn-build {ab['mean_a'] * 100:+.2f}%  (n={ab['n_a']})  vs off-season "
      f"{ab['mean_b'] * 100:+.2f}%  (n={ab['n_b']})   Welch t = {ab['t']:+.2f}")
wd = st.group_welch(ho_ret, data.WINTER_DRAW_MONTHS, data.OFF_SEASON_MONTHS)
print(f"  winter-draw  {wd['mean_a'] * 100:+.2f}%  (n={wd['n_a']})  vs off-season "
      f"{wd['mean_b'] * 100:+.2f}%  (n={wd['n_b']})   Welch t = {wd['t']:+.2f}")
hw = st.group_welch(ho_ret, data.HEATING_MONTHS, data.OFF_SEASON_MONTHS)
print(f"  heating (Sep-Feb) {hw['mean_a'] * 100:+.2f}%  (n={hw['n_a']})  vs off-season "
      f"{hw['mean_b'] * 100:+.2f}%  (n={hw['n_b']})   Welch t = {hw['t']:+.2f}")

print("\n# TRADABILITY — seasonal timer (long Sep-Feb, T-bill otherwise) vs buy-and-hold")
print("  (Sharpe = excess of T-bill, both legs; costs = one-way bps x NAV per switch, "
      "2 switches/yr)")
bh = st.buy_hold(ho_ret)
bh_s = st.summary(bh, rf=tbill)
print(f"  buy & hold HO=F              CAGR {bh_s['cagr']:+.2%}  Sharpe {bh_s['sharpe']:.2f}  "
      f"maxDD {bh_s['max_drawdown']:.0%}")
for cb in (0.0, 5.0, 10.0):
    tm = st.seasonal_timer(ho_ret, tbill, data.HEATING_MONTHS, cost_bps=cb)
    s = st.summary(tm, rf=tbill)
    lab = "gross" if cb == 0.0 else f"net {cb:.0f} bps"
    print(f"  seasonal timer ({lab:>8s})     CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  "
          f"maxDD {s['max_drawdown']:.0%}")

print("\n# THIRD AXIS — the real retail vehicle (UHN) vs the HO=F splice, paired per heating "
      "season")
print(f"  UHN traded {uhn.index.min().date()} -> {uhn.index.max().date()} "
      f"(last printed session on Yahoo — USCF wound the fund down; it is not buyable today)")
gap_df = st.uhn_vs_splice(uhn["Close"], ho["Close"], 2008, 2018)
for _, row in gap_df.iterrows():
    print(f"  {row['season']}: UHN {row['uhn_ret'] * 100:+7.2f}%   HO=F splice "
          f"{row['ho_splice_ret'] * 100:+7.2f}%   gap {row['gap'] * 100:+6.2f}%")
gs = st.uhn_gap_stats(gap_df)
print(f"  mean gap (UHN - splice): {gs['mean_gap'] * 100:+.2f}%/season  "
      f"(one-sample t = {gs['t']:+.2f}, n={gs['n']} seasons — small sample, said out loud)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (seasonal=0) and must recover a")
print("  planted heating-season premium. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    world = data.synthetic_world(seasonal=0.0, seed=1000 + s_)
    null_ts.append(st.synthetic_detect(world, data.HEATING_MONTHS, data.OFF_SEASON_MONTHS))
null_ts = np.asarray(null_ts)
print(f"  null (seasonal=0), 20 seeds (base 1000): mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
planted = data.synthetic_world(seasonal=0.15, seed=650)
pt = st.synthetic_detect(planted, data.HEATING_MONTHS, data.OFF_SEASON_MONTHS)
print(f"  planted seasonal=+0.15 (seed 650): Welch t = {pt:+.2f}")
