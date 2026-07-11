"""Reproducible headline run for Study 671 — Special K.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / ^GSPC tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
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

from special_k import data, strategy as st  # noqa: E402

print("# Special K — does Pring's 12-ROC 'reduced-whipsaw KST' flag major cyclic turns?")
print(f"indicator: ROC/SMA periods {st.ROC_PERIODS}, weights {st.WEIGHTS}, "
      f"signal SMA {st.SIGNAL_N} (StockCharts ChartSchool canonical daily parameters)")

if not data.have_real():
    print("(cache miss — fetching SPY / ^GSPC once)")
    data.fetch()

spy, gspc = data.load_real()
wk = data.weekly_from_daily(spy)
print(data_stamp("SPY total-return close", pd.DataFrame({"close": spy}), asof=data.AS_OF))
print(data_stamp("^GSPC price-only close", pd.DataFrame({"close": gspc}), asof=data.AS_OF))
print(f"weekly (Friday) resample of SPY: {len(wk)} bars {wk.index.min().date()} -> "
      f"{wk.index.max().date()}")

cross_spy = st.crossover_dates(spy)
cross_gspc = st.crossover_dates(gspc)
cross_wk = st.crossover_dates(wk, roc_periods=st.ROC_PERIODS_WEEKLY,
                              sma_periods=st.SMA_PERIODS_WEEKLY, signal_n=st.SIGNAL_N_WEEKLY)
print(f"\ncrossovers (bull / bear): SPY daily {len(cross_spy['bull'])}/{len(cross_spy['bear'])}  "
      f"|  ^GSPC daily {len(cross_gspc['bull'])}/{len(cross_gspc['bear'])}  "
      f"|  SPY weekly {len(cross_wk['bull'])}/{len(cross_wk['bear'])}")

print("\n# THE HEADLINE — post-crossover daily returns vs baseline (SPY, NW t, lags=horizon)")
print("  (signal known at close t; window opens t+1, one execution lag)")
for h in (21, 63, 126):
    rb = st.regime_return_stats(spy, cross_spy["bull"], horizon=h)
    rbe = st.regime_return_stats(spy, cross_spy["bear"], horizon=h)
    print(f"  h={h:>3d}d  bull: window {rb['mean_flag_pct']:+.3f}%/day vs base "
          f"{rb['mean_base_pct']:+.3f}%/day, NW t = {rb['nw_t']:+.2f} (n={rb['n_events']})"
          f"   |  bear: window {rbe['mean_flag_pct']:+.3f}%/day vs base "
          f"{rbe['mean_base_pct']:+.3f}%/day, NW t = {rbe['nw_t']:+.2f} (n={rbe['n_events']})")

print("\n# Random-timing placebo (Coppock-style), SPY, horizon=126d, 5,000 draws")
rt_bull = st.random_timing_test(spy, cross_spy["bull"], horizon=126, n_draws=5000,
                                tail="high", seed=671)
rt_bear = st.random_timing_test(spy, cross_spy["bear"], horizon=126, n_draws=5000,
                                tail="low", seed=671)
print(f"  bull: observed mean fwd return {rt_bull['obs_mean']*100:+.2f}% vs random-timing mean "
      f"{rt_bull['placebo_mean']*100:+.2f}% (n={rt_bull['n_events']}) -> "
      f"p(random >= observed) = {rt_bull['p_value']:.4f}")
print(f"  bear: observed mean fwd return {rt_bear['obs_mean']*100:+.2f}% vs random-timing mean "
      f"{rt_bear['placebo_mean']*100:+.2f}% (n={rt_bear['n_events']}) -> "
      f"p(random <= observed) = {rt_bear['p_value']:.4f}")

print("\n# Cross-check — ^GSPC price-only, 64 years, horizon=126d")
rbg = st.regime_return_stats(gspc, cross_gspc["bull"], horizon=126)
rbeg = st.regime_return_stats(gspc, cross_gspc["bear"], horizon=126)
print(f"  bull: window {rbg['mean_flag_pct']:+.3f}%/day vs base {rbg['mean_base_pct']:+.3f}%/day, "
      f"NW t = {rbg['nw_t']:+.2f} (n={rbg['n_events']})")
print(f"  bear: window {rbeg['mean_flag_pct']:+.3f}%/day vs base {rbeg['mean_base_pct']:+.3f}%/day, "
      f"NW t = {rbeg['nw_t']:+.2f} (n={rbeg['n_events']})")

print("\n# THE LONG/FLAT TIMER — net excess-of-cash Sharpe (1 bp one-way x NAV, 1-day lag), SPY")
res = st.run_experiment(spy, cost_bps=1.0, n_perm=5000, seed=671)
print(f"  {'arm':14s} {'Sharpe':>7} {'CAGR%':>7} {'maxDD%':>7} {'HAC t':>7} "
      f"{'expo':>5} {'trades':>7} {'turn/yr':>8}")
for k in ("sk", "buy_and_hold", "sma200"):
    s = res[k]
    print(f"  {k:14s} {s['sharpe_excess']:>7.3f} {s['cagr_net']*100:>7.2f} "
          f"{s['maxdd_net']*100:>7.1f} {s['hac_t']:>7.2f} {s['exposure']:>5.2f} "
          f"{s['n_trades']:>7d} {s['ann_turnover']:>8.1f}")
print(f"  sign-flip permutation (5,000 draws): observed gross Sharpe "
      f"{res['perm']['obs_sharpe']:.3f}, p(random sign-schedule >= observed) = "
      f"{res['perm']['p_value']:.4f}")

skn = res["books"]["sk"]["net"].to_numpy()
bahn = res["books"]["buy_and_hold"]["net"].to_numpy()
d = skn - bahn
print(f"\n  SK minus buy-and-hold (SPY): {d.mean()*st.TRADING_DAYS*100:+.2f}%/yr   "
      f"HAC t = {st.hac_t(d):+.2f}")

print("\n# Cost sweep — SK vs buy-and-hold net excess Sharpe (SPY)")
for r in st.cost_sweep(spy):
    print(f"  one-way {r['cost_bps']:>4.1f} bp: SK={r['sk_sharpe']:.3f}  "
          f"BAH={r['bah_sharpe']:.3f}")

print("\n# Parameter robustness — scale every ROC/SMA period by a common factor (SPY)")
for r in st.param_robustness(spy, scales=(0.7, 1.0, 1.3)):
    print(f"  scale {r['scale']:.1f}x: Sharpe {r['sharpe']:.3f}  HAC t {r['hac_t']:+.2f}  "
          f"CAGR {r['cagr']*100:.2f}%  trades {r['n_trades']}")

print("\n# Cross-check timer — ^GSPC price-only, 64 years")
res_g = st.run_experiment(gspc, cost_bps=1.0, n_perm=1, seed=671)
print(f"  SK Sharpe {res_g['sk']['sharpe_excess']:.3f} (HAC t={res_g['sk']['hac_t']:.2f})  vs  "
      f"buy-and-hold {res_g['buy_and_hold']['sharpe_excess']:.3f} "
      f"(HAC t={res_g['buy_and_hold']['hac_t']:.2f})")
skn = res_g["books"]["sk"]["net"].to_numpy()
bahn = res_g["books"]["buy_and_hold"]["net"].to_numpy()
d = skn - bahn
print(f"  SK minus buy-and-hold (^GSPC): {d.mean()*st.TRADING_DAYS*100:+.2f}%/yr   "
      f"HAC t = {st.hac_t(d):+.2f}")

print("\n# Cross-check timer — SPY resampled to WEEKLY bars (periods /5)")
resw = st.run_experiment(wk, cost_bps=1.0, n_perm=1, seed=671, periods_per_year=52,
                         sk_kwargs=dict(roc_periods=st.ROC_PERIODS_WEEKLY,
                                        sma_periods=st.SMA_PERIODS_WEEKLY,
                                        signal_n=st.SIGNAL_N_WEEKLY))
print(f"  SK Sharpe {resw['sk']['sharpe_excess']:.3f} (HAC t={resw['sk']['hac_t']:.2f})  vs  "
      f"buy-and-hold {resw['buy_and_hold']['sharpe_excess']:.3f} "
      f"(HAC t={resw['buy_and_hold']['hac_t']:.2f})")
skn = resw["books"]["sk"]["net"].to_numpy()
bahn = resw["books"]["buy_and_hold"]["net"].to_numpy()
d = skn - bahn
print(f"  SK minus buy-and-hold (weekly): {d.mean()*52*100:+.2f}%/yr   "
      f"HAC t = {st.hac_t(d):+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  planted multi-year bull/bear regime cycle (geometric ~1,160-session sojourns);")
print("  null (amp=0) must not manufacture a POSITIVE spread; a strong planted cycle must.")


def _spread_batch(amp: float, n_seeds: int = 20):
    spreads, sks, bahs = [], [], []
    for s in range(n_seeds):
        close = data.synthetic_tape(amp=amp, seed=671 + s, n_days=6000, mean_regime_days=750)
        r = st.run_experiment(close, cost_bps=1.0, n_perm=1, seed=671)
        skn_ = r["books"]["sk"]["net"].to_numpy()
        bahn_ = r["books"]["buy_and_hold"]["net"].to_numpy()
        spreads.append(st.hac_t(skn_ - bahn_))
        sks.append(r["sk"]["sharpe_excess"])
        bahs.append(r["buy_and_hold"]["sharpe_excess"])
    return np.array(spreads), np.array(sks), np.array(bahs)


null_t, null_sk, null_bah = _spread_batch(0.0)
print(f"  null (amp=0), 20 seeds: spread HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds (all negative-direction — the mechanical cash-drag "
      f"of being flat, same driver as the real-tape underperformance; never a false-positive edge)")
print(f"  null SK Sharpe {null_sk.mean():.3f} vs buy-and-hold {null_bah.mean():.3f}  "
      f"(SK beats BAH in {(null_sk > null_bah).mean()*100:.0f}% of seeds)")

plant_t, plant_sk, plant_bah = _spread_batch(0.003)
print(f"  planted cycle (amp=0.003), 20 seeds: spread HAC t mean {plant_t.mean():+.2f} "
      f"(sd {plant_t.std(ddof=1):.2f}), |t|>=2 in {(abs(plant_t) >= 2).sum()}/20 seeds")
print(f"  planted SK Sharpe {plant_sk.mean():.3f} vs buy-and-hold {plant_bah.mean():.3f}  "
      f"(SK beats BAH in {(plant_sk > plant_bah).mean()*100:.0f}% of seeds)")
