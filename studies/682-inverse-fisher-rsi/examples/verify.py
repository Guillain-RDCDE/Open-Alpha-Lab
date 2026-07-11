"""Reproducible headline run for Study 682 — Inverse-Fisher-RSI.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached total-return tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from inverse_fisher_rsi import data, strategy as st  # noqa: E402

print("# Inverse-Fisher-RSI — does Ehlers' IFT sharpen RSI into a cleaner reversal signal?")

if not data.have_real():
    print("(cache miss — fetching total-return closes once)")
    data.fetch()

real = data.load_real()
for t in data.TICKERS:
    print(data_stamp(t, real[t].to_frame(), asof=data.AS_OF))

signals = st.basket_signals(real)
n_ift = sum(int(signals[t]["ift_bull"].sum()) for t in data.TICKERS)
n_ift_bear = sum(int(signals[t]["ift_bear"].sum()) for t in data.TICKERS)
n_rsi14 = sum(int(signals[t]["rsi14_bull"].sum()) for t in data.TICKERS)
n_rsi2 = sum(int(signals[t]["rsi2_bull"].sum()) for t in data.TICKERS)
print(f"\nsignal counts (pooled, {len(data.TICKERS)} tickers, {data.START} -> {data.AS_OF}):")
print(f"  IFT-RSI bullish cross (-0.5)  : {n_ift}")
print(f"  IFT-RSI bearish cross (+0.5)  : {n_ift_bear}")
print(f"  plain RSI(14) cross up 30     : {n_rsi14}")
print(f"  plain RSI(2) cross up 10      : {n_rsi2}")

print("\n# THE HEADLINE — IFT-RSI bullish cross forward return vs unconditional")
for h in st.HORIZONS:
    s = st.headline_stats(real, signals, "ift_bull", h)
    print(f"  h={h:>2d}d: signal {s['sig_mean_bps']:+7.1f} bps (n={s['n_sig']})  "
          f"vs unconditional {s['base_mean_bps']:+7.1f} bps (n={s['n_base']})  "
          f"gap {s['gap_bps']:+7.1f} bps  Welch t={s['welch_t']:+.2f}  NW t={s['nw_t']:+.2f}  "
          f"hit {s['hit_rate']*100:.1f}% (Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")

print("\n# Bearish-cross check — IFT-RSI cross DOWN through +0.5 (should predict weaker returns)")
for h in st.HORIZONS:
    s = st.headline_stats(real, signals, "ift_bear", h)
    print(f"  h={h:>2d}d: signal {s['sig_mean_bps']:+7.1f} bps (n={s['n_sig']})  "
          f"vs unconditional {s['base_mean_bps']:+7.1f} bps  gap {s['gap_bps']:+7.1f} bps  "
          f"Welch t={s['welch_t']:+.2f}  NW t={s['nw_t']:+.2f}")

print("\n# Random-signal placebo (20 seeds x 200 draws), h=10d, IFT-RSI bullish cross")
pl = st.random_signal_placebo(real, signals, "ift_bull", 10)
print(f"  observed {pl['obs']*1e4:+.1f} bps vs placebo mean {pl['placebo_mean']*1e4:+.1f} bps "
      f"(sd {pl['placebo_sd']*1e4:.1f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# THE COMPARISON — does IFT-RSI beat plain RSI(14) / RSI(2) reversal? (h=10d)")
for name in ("ift_bull", "rsi14_bull", "rsi2_bull"):
    s = st.headline_stats(real, signals, name, 10)
    print(f"  {name:<11s}: n={s['n_sig']:>4d}  gap {s['gap_bps']:+7.1f} bps  "
          f"Welch t={s['welch_t']:+.2f}  NW t={s['nw_t']:+.2f}  hit {s['hit_rate']*100:.1f}%")

print("\n# THIRD AXIS — a timer with costs (long-flat, IFT-RSI enter -0.5 / exit +0.5)")
for cb in (5.0, 10.0):
    tm = st.timer_with_costs(real, signals, "ift_bull", "ift_bear", cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: net Sharpe {tm['sharpe_net']:+.2f}  "
          f"(buy&hold {tm['sharpe_bh']:+.2f})  ann.ret net {tm['ann_ret_net_pct']:+.2f}%  "
          f"(buy&hold {tm['ann_ret_bh_pct']:+.2f}%)  exposure {tm['exposure']*100:.1f}%")
tm5 = st.timer_with_costs(real, signals, "ift_bull", "ift_bear", cost_bps=5.0)
rnd = st.random_exposure_control(real, tm5["exposure"], cost_bps=5.0)
z = (tm5["sharpe_net"] - rnd.mean()) / rnd.std(ddof=1)
print(f"  random-exposure control (20 seeds, matched {tm5['exposure']*100:.1f}% time-in-market): "
      f"mean Sharpe {rnd.mean():+.2f} (sd {rnd.std(ddof=1):.2f})  z = {z:+.2f}  "
      f"real Sharpe beats {int((tm5['sharpe_net'] > rnd).sum())}/20 random seeds")

print("\n# RSI(14) timer, for comparison (enter cross up 30 / exit cross down 70)")
for cb in (5.0, 10.0):
    tm14 = st.timer_with_costs(real, signals, "rsi14_bull", "rsi14_bear", cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: net Sharpe {tm14['sharpe_net']:+.2f}  "
          f"ann.ret net {tm14['ann_ret_net_pct']:+.2f}%  exposure {tm14['exposure']*100:.1f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the AR(1) reversion knob operates at a 1-day lag, so the machinery check runs at")
print("  h=1 (where the planted effect actually lives); the null (rho=0) must NOT fire across")
print("  20 seeds and a planted rho=0.6 must light up.")
import numpy as np  # noqa: E402

null_ts = []
for s_ in range(20):
    close = data.synthetic_world(rho=0.0, seed=682 + s_)
    null_ts.append(st.synthetic_detect(close, h=1)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (rho=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
close = data.synthetic_world(rho=0.6, seed=682)
sy = st.synthetic_detect(close, h=1)
print(f"  planted rho=0.6 (seed 682): signal mean {sy['sig_mean_bps']:+.1f} bps "
      f"vs base {sy['base_mean_bps']:+.1f} bps  Welch t = {sy['welch_t']:+.2f}  (n={sy['n_sig']})")
