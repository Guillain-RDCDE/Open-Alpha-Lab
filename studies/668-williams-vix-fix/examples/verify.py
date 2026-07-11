"""Reproducible headline run for Study 668 — Williams VIX Fix.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached eight-ticker basket under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from williams_vix_fix import data, strategy as st  # noqa: E402

print("# Williams VIX Fix — does a price-only 'synthetic VIX' spike mark capitulation bottoms?")
print(f"basket: {len(data.TICKERS)} tickers {data.TICKERS} "
      f"({len(data.ETF_TICKERS)} index ETFs, no survivorship + "
      f"{len(data.TICKERS) - len(data.ETF_TICKERS)} long-history single names, "
      "survivorship named)")

if not data.have_real():
    print("(cache miss — fetching the basket once)")
    data.fetch()

real = data.load_real()
for t in data.TICKERS:
    print(data_stamp(t, real[t], asof=data.AS_OF))

frames = {t: st.ticker_frame(real[t]) for t in data.TICKERS}

print("\n# THE HEADLINE — WVF-spike-onset forward return vs unconditional, by horizon")
print("  (spike = WVF >= 20-day mean + 2 std of itself; one entry per capitulation episode;")
print("   execution: signal known at close t, enter t+1 open, exit h sessions later's close)")
headline = {}
for h in (5, 10, 20):
    pooled_spike, pooled_rest, nw_ts = [], [], []
    for t in data.TICKERS:
        fr = frames[t]
        f = fr["onset_wvf"].to_numpy(bool)
        y = fr[f"fwd_{h}"].to_numpy(float)
        pooled_spike.append(y[f])
        pooled_rest.append(y[~f])
        nw_ts.append(st.nw_dummy_stats(fr, h, "onset_wvf"))
    ps = np.concatenate(pooled_spike)
    pr = np.concatenate(pooled_rest)
    ps_v = ps[~np.isnan(ps)]
    k_up = int((ps_v > 0).sum())
    lo, hi = st.wilson_interval(k_up, len(ps_v))
    nw_ts = np.asarray(nw_ts)
    headline[h] = dict(
        n_spike=len(ps_v), spike_bps=float(np.nanmean(ps)) * 1e4,
        rest_bps=float(np.nanmean(pr)) * 1e4,
        gap_bps=(float(np.nanmean(ps)) - float(np.nanmean(pr))) * 1e4,
        welch_t=st.welch_t(ps, pr), hit_rate=k_up / len(ps_v), hit_lo=lo, hit_hi=hi,
        nw_mean=float(nw_ts.mean()), nw_min=float(nw_ts.min()), nw_max=float(nw_ts.max()),
    )
    s = headline[h]
    print(f"  h={h:2d}d: spike {s['spike_bps']:+7.1f} bps  vs  rest {s['rest_bps']:+7.1f} bps  "
          f"gap {s['gap_bps']:+7.1f} bps   Welch t = {s['welch_t']:+.2f}   "
          f"hit {s['hit_rate']*100:.1f}% (Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")
    print(f"          per-ticker Newey-West t (lags={h}): mean {s['nw_mean']:+.2f}, "
          f"range [{s['nw_min']:+.2f}, {s['nw_max']:+.2f}]  (n={s['n_spike']} spike episodes pooled)")

print("\n# Per-ticker detail, h=10d (the primary horizon for the timer below)")
per_ticker = {}
for t in data.TICKERS:
    s = st.spike_stats(frames[t], 10, "onset_wvf")
    s["nw_t"] = st.nw_dummy_stats(frames[t], 10, "onset_wvf")
    per_ticker[t] = s
    print(f"  {t:5s} n={s['n_spike']:3d}  spike {s['spike_mean']*1e4:+7.1f} bps  "
          f"rest {s['rest_mean']*1e4:+7.1f} bps  Welch t={s['welch_t']:+.2f}  NW t={s['nw_t']:+.2f}  "
          f"hit {s['hit_rate']*100:.1f}%")

print("\n# Random-calendar placebo (h=10, pooled): 20 seeds x 500 draws of "
      f"{headline[10]['n_spike']} random non-spike days")
pool_all, spike_all = [], []
for t in data.TICKERS:
    fr = frames[t]
    f = fr["onset_wvf"].to_numpy(bool)
    y = fr["fwd_10"].to_numpy(float)
    pool_all.append(y[~f])
    spike_all.append(y[f])
pool_all = np.concatenate(pool_all)
spike_all = np.concatenate(spike_all)
spike_all = spike_all[~np.isnan(spike_all)]
obs = float(np.nanmean(spike_all))
pl = st.placebo_pvalue(pool_all, obs, len(spike_all), n_draws_per_seed=500, n_seeds=20,
                       base_seed=668)
print(f"  observed spike-day mean {pl['obs']*1e4:+.1f} bps vs placebo mean "
      f"{pl['placebo_mean']*1e4:+.1f} bps (sd {pl['placebo_sd']*1e4:.1f}) over "
      f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.3f}  "
      "(right-tail: share of random calendars beating the observed mean)")

print("\n# THIRD AXIS — is WVF more than a plain close-only drawdown proxy? (h=10)")
print("  two-dummy HAC regression: fwd_10 ~ a + b*onset(WVF) + c*onset(drawdown-proxy)")
wm_rows = {}
for t in data.TICKERS:
    w = st.wick_marginal_stats(frames[t], 10)
    wm_rows[t] = w
    print(f"  {t:5s} overlap(WVF also trips drawdown)={w['overlap_of_wvf']*100:5.1f}%   "
          f"marginal WVF t={w['t_wvf_marginal']:+.2f}   marginal drawdown t={w['t_dd_marginal']:+.2f}   "
          f"WVF-only mean {w['mean_wvf_only']*1e4:+.1f} bps (n={w['n_wvf_only']})  vs  "
          f"drawdown-only mean {w['mean_dd_only']*1e4:+.1f} bps (n={w['n_dd_only']})  "
          f"Welch t(diff)={w['welch_t_wvf_vs_dd_only']:+.2f}")
avg_t_wvf = float(np.mean([w["t_wvf_marginal"] for w in wm_rows.values()]))
avg_t_dd = float(np.mean([w["t_dd_marginal"] for w in wm_rows.values()]))
avg_overlap = float(np.mean([w["overlap_of_wvf"] for w in wm_rows.values()]))
print(f"  basket average: marginal WVF t = {avg_t_wvf:+.2f}   marginal drawdown t = {avg_t_dd:+.2f}   "
      f"overlap = {avg_overlap*100:.1f}%")

print("\n# THE TIMER — buy every WVF-spike onset, hold 10 sessions, one round trip = "
      "2 x one-way cost x NAV")
for cb in (5.0, 10.0):
    ledgers = [st.timer_ledger(frames[t], 10, cb, "onset_wvf") for t in data.TICKERS]
    pooled = pd.concat(ledgers, ignore_index=True)
    s = st.summarize_ledger(pooled, "ret_net")
    print(f"  cost={cb:>4.1f} bps: n={s['n_trades']}  net {s['mean_bps']:+.1f} bps/trade  "
          f"win {s['win_rate']*100:.1f}%  Sharpe/trade {s['sharpe']:.3f}  HAC t = {s['tstat']:+.2f}")
ledgers = [st.timer_ledger(frames[t], 10, 5.0, "onset_wvf") for t in data.TICKERS]
pooled = pd.concat(ledgers, ignore_index=True)
g = st.summarize_ledger(pooled, "ret_gross")
print(f"  gross (no cost): n={g['n_trades']}  {g['mean_bps']:+.1f} bps/trade  "
      f"HAC t = {g['tstat']:+.2f}")
print("  per-ticker net (5 bps):")
for t in data.TICKERS:
    led = st.timer_ledger(frames[t], 10, 5.0, "onset_wvf")
    s = st.summarize_ledger(led, "ret_net")
    print(f"    {t:5s} n={s['n_trades']:3d}  net {s['mean_bps']:+7.1f} bps/trade  "
          f"win {s['win_rate']*100:.1f}%  HAC t={s['tstat']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network, horizon=10")
print("  the detector must NOT fire on a null world (bounce=0) and must recover a planted")
print("  post-crash bounce. Null checked over 20 seeds (never a single stream); primary")
print("  detector statistic is the Newey-West t (accounts for the h-day overlap).")
null_w, null_n = [], []
for s_ in range(20):
    sdf = data.synthetic_world(bounce=0.0, seed=668 + s_)
    r = st.synthetic_detect(sdf, horizon=10)
    null_w.append(r["welch_t"])
    null_n.append(r["nw_t"])
null_w = np.asarray(null_w)
null_n = np.asarray(null_n)
print(f"  null (bounce=0), 20 seeds: Welch t mean {null_w.mean():+.2f} (sd {null_w.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_w) >= 2).sum()}/20 seeds")
print(f"                              NW t    mean {null_n.mean():+.2f} (sd {null_n.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_n) >= 2).sum()}/20 seeds  <- primary detector")
sdf = data.synthetic_world(bounce=0.002, seed=668)
sy = st.synthetic_detect(sdf, horizon=10)
print(f"  planted bounce=+0.002/day for 10 sessions post-crash (seed 668): "
      f"spike mean {sy['spike_mean']*1e4:+.1f} bps vs rest {sy['rest_mean']*1e4:+.1f} bps  "
      f"Welch t = {sy['welch_t']:+.2f}   NW t = {sy['nw_t']:+.2f}   n={sy['n_spike']}")
