"""Reproducible headline run for Study 656 — Dragon Portfolio.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY/TLT/GLD/DBC/VXX/SHY/
DBMF tapes under ``_cache/`` (fetching once on a cache miss), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from dragon_portfolio import data, strategy as st  # noqa: E402

print("# Dragon Portfolio — does Cole's 5-sleeve mix survive both inflation and deflation?")

if not data.have_real():
    print("(cache miss — fetching SPY/TLT/GLD/DBC/VXX/SHY/DBMF once)")
    data.fetch()

px = data.prices_frame(tickers=("SPY", "TLT", "GLD", "DBC", "VXX", "SHY"))
print(data_stamp("SPY/TLT/GLD/DBC/VXX/SHY prices", px, asof=data.AS_OF))
for t in ("SPY", "TLT", "GLD", "DBC", "VXX", "SHY"):
    s = px[t].dropna()
    print(f"  {t}: {len(s)} rows  {s.index.min().date()} -> {s.index.max().date()}  "
          f"(inception {data.INCEPTION[t]})")

ret = st.build_returns(px)
rf = ret["SHY"]

print(f"\ncommodity-trend overlay: 12-month DBC signal, warm-up ends -> live from "
      f"{data.CORE_START}")
print(f"core (Dragon-lite) window : {data.CORE_START} -> {data.AS_OF}")
print(f"full (Dragon w/ VXX) window: {data.FULL_START} -> {data.AS_OF}")

# --------------------------------------------------------------------------- #
# Headline portfolios
# --------------------------------------------------------------------------- #
COST = 5.0

dlite = st.blended_portfolio(ret, st.DRAGON_LITE, cost_bps=COST, start=data.CORE_START)
dfull = st.blended_portfolio(ret, st.DRAGON_FULL, cost_bps=COST, start=data.FULL_START)
p6040_core = st.blended_portfolio(ret, st.SIXTY_FORTY, cost_bps=COST, start=data.CORE_START)
p6040_full = st.blended_portfolio(ret, st.SIXTY_FORTY, cost_bps=COST, start=data.FULL_START)
awl_core = st.blended_portfolio(ret, st.ALL_WEATHER_LITE, cost_bps=COST, start=data.CORE_START)
awl_full = st.blended_portfolio(ret, st.ALL_WEATHER_LITE, cost_bps=COST, start=data.FULL_START)
spy_core = st.spy_only(ret, start=data.CORE_START)
spy_full = st.spy_only(ret, start=data.FULL_START)

print("\n# THE HEADLINE — core window (Dragon-lite, no vol sleeve, 2007-> ; sees 2008 AND 2020/2022)")
header = f"{'Portfolio':22s}  {'CAGR':>7}  {'Vol':>6}  {'Sharpe':>7}  {'MaxDD':>7}  {'Worst12m':>9}"
print(header)
print("-" * len(header))
core_rows = [("Dragon-lite (4 sleeve)", dlite), ("60/40 SPY/TLT", p6040_core),
             ("All-Weather-lite", awl_core), ("100% SPY", spy_core)]
core_stats = {}
for name, s in core_rows:
    stt = st.portfolio_stats(s, rf=rf)
    core_stats[name] = stt
    print(f"{name:22s}  {stt['cagr']*100:>6.2f}%  {stt['vol']*100:>5.2f}%  "
          f"{stt['sharpe']:>7.3f}  {stt['max_dd']*100:>6.1f}%  {stt['worst_12m']*100:>8.1f}%")
print(f"  window: {core_stats['Dragon-lite (4 sleeve)']['start'].date()} -> "
      f"{core_stats['Dragon-lite (4 sleeve)']['end'].date()}  "
      f"n_days={core_stats['Dragon-lite (4 sleeve)']['n']}")

print("\n# THE HEADLINE — full window (Dragon-full w/ VXX, 2018-> ; misses 2008 AND most of the")
print("#  product's own 2009-2018 history — yfinance's VXX tape only starts 2018-01-25; sees 2020/2022)")
print(header)
print("-" * len(header))
full_rows = [("Dragon-full (5 sleeve)", dfull), ("60/40 SPY/TLT", p6040_full),
             ("All-Weather-lite", awl_full), ("100% SPY", spy_full)]
full_stats = {}
for name, s in full_rows:
    stt = st.portfolio_stats(s, rf=rf)
    full_stats[name] = stt
    print(f"{name:22s}  {stt['cagr']*100:>6.2f}%  {stt['vol']*100:>5.2f}%  "
          f"{stt['sharpe']:>7.3f}  {stt['max_dd']*100:>6.1f}%  {stt['worst_12m']*100:>8.1f}%")
print(f"  window: {full_stats['Dragon-full (5 sleeve)']['start'].date()} -> "
      f"{full_stats['Dragon-full (5 sleeve)']['end'].date()}  "
      f"n_days={full_stats['Dragon-full (5 sleeve)']['n']}")

# --------------------------------------------------------------------------- #
# Crisis windows — 2020 Q1 crash, 2022 both-down year
# --------------------------------------------------------------------------- #
print("\n# 2020 COVID crash (2020-01-01 -> 2020-04-30 cumulative)")
for name, s in core_rows:
    r = st.window_return(s, "2020-01-01", "2020-04-30")
    print(f"  {name:22s} {r*100:+7.2f}%")
r = st.window_return(dfull, "2020-01-01", "2020-04-30")
print(f"  {'Dragon-full (5 sleeve)':22s} {r*100:+7.2f}%   <- has the VXX sleeve live")

print("\n# 2022 full calendar year (both stocks AND bonds fell)")
for name, s in core_rows:
    r = st.calendar_year_return(s, 2022)
    print(f"  {name:22s} {r*100:+7.2f}%")
r = st.calendar_year_return(dfull, 2022)
print(f"  {'Dragon-full (5 sleeve)':22s} {r*100:+7.2f}%   <- has the VXX sleeve live")

# --------------------------------------------------------------------------- #
# VXX standalone — the honest long-vol diagnostic
# --------------------------------------------------------------------------- #
print("\n# VXX standalone — the 'unproxiable cheaply' diagnostic")
vxx_ret = ret["VXX"].dropna()
vxx_stats = st.portfolio_stats(vxx_ret, rf=rf)
print(f"  CAGR {vxx_stats['cagr']*100:+.1f}%/yr  vol {vxx_stats['vol']*100:.1f}%  "
      f"MaxDD {vxx_stats['max_dd']*100:.1f}%  window {vxx_stats['start'].date()} -> "
      f"{vxx_stats['end'].date()}")
cum = float((1.0 + vxx_ret).prod() - 1.0)
print(f"  cumulative buy-and-hold return since inception: {cum*100:+.1f}%")
covid_spike = st.window_return(vxx_ret, "2020-02-19", "2020-03-23")
print(f"  return during the COVID spike (2020-02-19 -> 2020-03-23, peak-to-trough SPY): "
      f"{covid_spike*100:+.1f}%  <- this is the payoff it exists for")

# --------------------------------------------------------------------------- #
# Inference: HAC t of monthly return gap, bootstrap Sharpe difference
# --------------------------------------------------------------------------- #
print("\n# Inference — is the Dragon's risk-adjusted edge over 60/40 real, or noise?")
hac_lite = st.hac_diff_monthly(dlite, p6040_core)
print(f"  Dragon-lite − 60/40 (core, monthly): mean gap {hac_lite['mean_diff_monthly']*100:+.3f}%/mo  "
      f"NW t = {hac_lite['t']:+.2f}  (n={hac_lite['n_months']} months)")
hac_full = st.hac_diff_monthly(dfull, p6040_full)
print(f"  Dragon-full − 60/40 (full, monthly): mean gap {hac_full['mean_diff_monthly']*100:+.3f}%/mo  "
      f"NW t = {hac_full['t']:+.2f}  (n={hac_full['n_months']} months)")

boot_lite = st.bootstrap_sharpe_diff(dlite, p6040_core, rf=rf, seed=656)
print(f"  bootstrap Sharpe diff, Dragon-lite vs 60/40 (core): {boot_lite['point']:+.3f}  "
      f"CI95=[{boot_lite['ci95'][0]:+.3f}, {boot_lite['ci95'][1]:+.3f}]  "
      f"Dragon wins {boot_lite['frac_a_wins']*100:.0f}% of resamples")
boot_full = st.bootstrap_sharpe_diff(dfull, p6040_full, rf=rf, seed=656)
print(f"  bootstrap Sharpe diff, Dragon-full vs 60/40 (full): {boot_full['point']:+.3f}  "
      f"CI95=[{boot_full['ci95'][0]:+.3f}, {boot_full['ci95'][1]:+.3f}]  "
      f"Dragon wins {boot_full['frac_a_wins']*100:.0f}% of resamples")

boot_awl = st.bootstrap_sharpe_diff(dfull, awl_full, rf=rf, seed=656)
print(f"  bootstrap Sharpe diff, Dragon-full vs All-Weather-lite (full): {boot_awl['point']:+.3f}  "
      f"CI95=[{boot_awl['ci95'][0]:+.3f}, {boot_awl['ci95'][1]:+.3f}]  "
      f"Dragon wins {boot_awl['frac_a_wins']*100:.0f}% of resamples")

# --------------------------------------------------------------------------- #
# Cost sensitivity
# --------------------------------------------------------------------------- #
print("\n# Cost sensitivity (Dragon-full, monthly rebalance)")
for cb in (0.0, 5.0, 10.0):
    s = st.blended_portfolio(ret, st.DRAGON_FULL, cost_bps=cb, start=data.FULL_START)
    stt = st.portfolio_stats(s, rf=rf)
    print(f"  cost={cb:>4.1f} bps one-way: CAGR {stt['cagr']*100:+.2f}%  Sharpe {stt['sharpe']:.3f}")

# --------------------------------------------------------------------------- #
# DBMF side-check — does the DIY trend overlay resemble a real managed-futures fund?
# --------------------------------------------------------------------------- #
real = data.load_real()
if "DBMF" in real:
    dbmf_ret = real["DBMF"].pct_change().dropna()
    trend_ret = ret["TREND"].dropna()
    common = dbmf_ret.index.intersection(trend_ret.index)
    dbmf_c, trend_c = dbmf_ret.loc[common], trend_ret.loc[common]
    corr = float(np.corrcoef(dbmf_c, trend_c)[0, 1])
    dbmf_stats = st.portfolio_stats(dbmf_c)
    trend_stats = st.portfolio_stats(trend_c)
    print(f"\n# DBMF side-check (real managed-futures ETF vs our DBC-trend proxy, "
          f"{common.min().date()} -> {common.max().date()}, n={len(common)} days)")
    print(f"  our TREND proxy : CAGR {trend_stats['cagr']*100:+.1f}%  vol {trend_stats['vol']*100:.1f}%")
    print(f"  real DBMF       : CAGR {dbmf_stats['cagr']*100:+.1f}%  vol {dbmf_stats['vol']*100:.1f}%")
    print(f"  daily return correlation: {corr:+.2f}  <- single-index long/flat is a WEAK proxy for a "
          f"real multi-market trend program if this is low")
else:
    corr = float("nan")
    print("\n# DBMF side-check skipped (no cache)")

# --------------------------------------------------------------------------- #
# Synthetic positive control — the machinery is unbiased
# --------------------------------------------------------------------------- #
print("\n# Synthetic crisis-hedge control — deterministic, no network")
print("  isolates the mechanism: does the Dragon-weighted TREND+VOL sub-sleeve pay off")
print("  MORE in crisis months than normal months? Null (hedge=0) must not fire; a")
print("  planted hedge must. (Testing the hedge sleeve's own crisis-vs-normal gap, not")
print("  'Dragon vs 60/40' — a lower equity weight alone would win any stock crash with")
print("  zero genuine crisis alpha, which would make a naive null fire on a beta artefact.)")
null_ts = []
for s_ in range(20):
    frame, truth = data.synthetic_world(hedge_strength=0.0, seed=656 + s_)
    null_ts.append(st.synthetic_crisis_test(frame, truth)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (hedge=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
frame, truth = data.synthetic_world(hedge_strength=1.0, seed=656)
sy = st.synthetic_crisis_test(frame, truth)
print(f"  planted hedge=1.0 (seed 656): hedge-sleeve crisis-month mean {sy['hedge_crisis_mean']*100:+.2f}% "
      f"vs normal-month mean {sy['hedge_normal_mean']*100:+.2f}%  (n_crisis={sy['n_crisis']})  "
      f"Welch t = {sy['welch_t']:+.2f}")
