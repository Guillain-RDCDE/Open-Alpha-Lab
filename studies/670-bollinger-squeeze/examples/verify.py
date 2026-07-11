"""Reproducible headline run for Study 670 — Bollinger-Squeeze.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily basket tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
from quantlab.repro import data_stamp  # noqa: E402

from bollinger_squeeze import data, strategy as st  # noqa: E402

print("# Bollinger-Squeeze — does a BB-inside-KC squeeze predict a profitable directional breakout?")

if not data.have_cache():
    print("(cache miss — fetching the basket once)")
    data.fetch_all()

baskets = data.load_basket()
for t, bars in baskets.items():
    print(data_stamp(t, bars[["close"]], asof=data.AS_OF))
bfp = data.basket_fingerprint(baskets)
print(f"basket fingerprint: {bfp}  ({len(baskets)} tickers: {', '.join(baskets)})")

frames = {t: st.squeeze_frame(b) for t, b in baskets.items()}
events = {t: st.squeeze_release_events(f, min_run=5) for t, f in frames.items()}
n_ev_total = sum(len(e) for e in events.values())
print(f"\nsqueeze-release events (min run 5 bars, BB(20,2.0) inside KC(20,1.5xATR20)): "
      f"{n_ev_total} pooled across {len(baskets)} tickers")
for t in baskets:
    print(f"  {t}: {len(events[t])} events, {int((frames[t]['squeeze']).sum())} squeeze bars "
          f"of {len(frames[t])} ({(frames[t]['squeeze']).mean()*100:.1f}%)")

# --------------------------------------------------------------------------- #
print("\n# TEST 1 — does volatility genuinely EXPAND after the release "
      "(more than a random day, not just vs. the squeeze itself)?")
K = 10
vol_rows = {}
for t in baskets:
    vol_rows[t] = st.vol_expansion_stats(frames[t], events[t], k=K)
pooled_fwd = np.concatenate([vol_rows[t]["fwd"] for t in baskets])
pooled_rand = np.concatenate([vol_rows[t]["fwd_rand"] for t in baskets])
for t in baskets:
    v = vol_rows[t]
    print(f"  {t}: fwd-{K}d vol {v['fwd_vol_mean']*100:.3f}%/day vs trailing (during squeeze) "
          f"{v['trail_vol_mean']*100:.3f}%/day (ratio {v['ratio_mean']:.2f}x, "
          f"t-vs-1 = {v['ratio_t_vs_1']:+.2f})  |  vs random day "
          f"{v['fwd_rand_mean']*100:.3f}%/day, Welch t = {v['welch_t_vs_random']:+.2f} "
          f"(n={v['n_events']})")
pooled_welch = st.welch_t(pooled_fwd, pooled_rand)
print(f"  POOLED: fwd-{K}d vol after release {np.nanmean(pooled_fwd)*100:.3f}%/day vs random-day "
      f"{np.nanmean(pooled_rand)*100:.3f}%/day  ->  Welch t = {pooled_welch:+.2f}  "
      f"(n={len(pooled_fwd)} events)")

# --------------------------------------------------------------------------- #
print(f"\n# TEST 2 — the directional breakout trade (hold {K} days, enter next open, "
      "5/10 bps one-way costs, matched random-direction-mix control)")
for cb in (5.0, 10.0):
    sig_all, ctrl_all = [], []
    for t in baskets:
        led = st.breakout_ledger(baskets[t], events[t], hold_days=K, cost_bps=cb)
        ctrl = st.random_control_ledger(baskets[t], frames[t], events[t], hold_days=K, cost_bps=cb,
                                         borrow_bps_annual=20.0, seed=670)
        sig_all.append(led["ret_net"].to_numpy())
        ctrl_all.append(ctrl["ret_net"].to_numpy())
    sig = np.concatenate(sig_all)
    ctrl = np.concatenate(ctrl_all)
    wt = st.welch_t(sig, ctrl)
    ht = st.hac_t(sig)
    print(f"  cost={cb:>4.1f} bps: signal {sig.mean()*1e4:+.1f} bps/trade (HAC t={ht:+.2f}, "
          f"win {float((sig>0).mean())*100:.1f}%)  vs random-mix control "
          f"{ctrl.mean()*1e4:+.1f} bps/trade   Welch t (signal - control) = {wt:+.2f}  "
          f"(n={len(sig)})")

sig_all, ctrl_all = [], []
for t in baskets:
    led = st.breakout_ledger(baskets[t], events[t], hold_days=K, cost_bps=5.0)
    ctrl = st.random_control_ledger(baskets[t], frames[t], events[t], hold_days=K, cost_bps=5.0,
                                     borrow_bps_annual=20.0, seed=670)
    sig_all.append(led["ret_net"].to_numpy())
    ctrl_all.append(ctrl["ret_net"].to_numpy())
sig = np.concatenate(sig_all)
ctrl = np.concatenate(ctrl_all)
print(f"  headline (5 bps): directional breakout {sig.mean()*1e4:+.1f} bps/trade net, "
      f"win rate {float((sig>0).mean())*100:.1f}%, HAC t vs 0 = {st.hac_t(sig):+.2f}  |  "
      f"vs matched random control: Welch t = {st.welch_t(sig, ctrl):+.2f}")

# --------------------------------------------------------------------------- #
print("\n# TEST 3 — per-ticker breakdown (hold 10d, 5 bps)")
for t in baskets:
    led = st.breakout_ledger(baskets[t], events[t], hold_days=K, cost_bps=5.0)
    ctrl = st.random_control_ledger(baskets[t], frames[t], events[t], hold_days=K, cost_bps=5.0,
                                     borrow_bps_annual=20.0, seed=670)
    s = st.summarize(led)
    wt = st.welch_t(led["ret_net"].to_numpy(), ctrl["ret_net"].to_numpy())
    print(f"  {t}: n={s['n_trades']}, mean {s['mean_bps']:+.1f} bps, win {s['win_rate']*100:.1f}%, "
          f"HAC t={s['hac_t']:+.2f}  |  vs random Welch t={wt:+.2f}")

# --------------------------------------------------------------------------- #
print("\n# TEST 4 — parameter robustness sweep (BB std x KC mult x hold days, pooled, 5 bps)")
sweep = st.param_sweep(baskets)
n_clear = int((sweep["welch_t"].abs() >= 2).sum())
print(f"  {len(sweep)} combos tested; |Welch t| >= 2 in {n_clear}/{len(sweep)} "
      f"({n_clear/len(sweep)*100:.0f}%)")
print(f"  Welch t range: [{sweep['welch_t'].min():+.2f}, {sweep['welch_t'].max():+.2f}], "
      f"mean {sweep['welch_t'].mean():+.2f}")
best = sweep.loc[sweep["welch_t"].abs().idxmax()]
print(f"  best |t| combo: BB={best['bb_std']}, KC={best['kc_mult']}, hold={int(best['hold_days'])}d "
      f"-> t={best['welch_t']:+.2f} (n={int(best['n_events'])})")

# --------------------------------------------------------------------------- #
print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (continuation=0) across 10 seeds, "
      "and must recover a planted directional-continuation effect.")
null_ts = []
for s_ in range(10):
    bars, truth = data.synthetic_daily(continuation=0.0, seed=670 + s_)
    r = st.synthetic_detect(bars)
    null_ts.append(r["welch_t"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (continuation=0), 10 seeds: mean Welch t = {np.nanmean(null_ts):+.2f} "
      f"(sd {np.nanstd(null_ts, ddof=1):.2f}), |t|>=2 in "
      f"{int((np.abs(null_ts) >= 2).sum())}/10 seeds")
bars, truth = data.synthetic_daily(continuation=1.0, seed=670)
r = st.synthetic_detect(bars)
print(f"  planted continuation=1.0 (seed 670): signal {r['mean_bps']:+.1f} bps vs control "
      f"{r['ctrl_mean_bps']:+.1f} bps  Welch t = {r['welch_t']:+.2f}  (n={r['n_events']} events)")
