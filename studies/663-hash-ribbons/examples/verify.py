"""Reproducible headline run for Study 663 — Hash-Ribbons.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached BTC-USD tape under ``_cache/``
(fetching once on a cache miss); the hashrate path is a pure function of the hardcoded
month-end table, no network. The synthetic control always runs, no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from hash_ribbons import data, strategy as st  # noqa: E402

print("# Hash-Ribbons — does a hashrate-capitulation-recovery signal buy Bitcoin dips?")

hr = data.hashrate_daily()
print(f"hashrate path: {len(hr)} interpolated daily points {hr.index.min().date()} -> "
      f"{hr.index.max().date()} (linear interpolation of the hardcoded month-end anchors — "
      "identical table to sibling study 292-bitcoin-hashrate)")

if not data.have_real():
    print("(cache miss — fetching BTC-USD once)")
    data.fetch_btc(end="2026-07-01")

btc = data.load_btc()
print(data_stamp("BTC-USD daily close", btc.to_frame(), asof=data.AS_OF))

print("\n# THE SIGNAL — genuine Hash-Ribbons buy signals on the (interpolated) tape")
print("  (SMA30 crosses back above SMA60 after >=21 days below it AND >=8% peak-to-trough "
      "hashrate decline during the capitulation stretch)")
sig = st.capitulation_signals(hr)
print(f"  n = {len(sig)} signals over {hr.index.min().year}-{hr.index.max().year}: "
      f"n is TINY, cap every claim below accordingly.")
for _, row in sig.iterrows():
    print(f"  {row['signal_date'].date()}  capitulation {row['capitulation_days']}d "
          f"({row['window_start'].date()} -> {row['window_end'].date()}), hashrate fell "
          f"{row['decline_pct']:+.1f}%")

print("\n# Forward BTC returns after each signal vs the unconditional distribution")
fr = st.forward_return_stats(btc, sig["signal_date"])
for h, row in fr.iterrows():
    print(f"  +{h:>3d}d: signal mean {row['signal_mean_pct']:+6.1f}% (median "
          f"{row['signal_median_pct']:+6.1f}%, n={int(row['n'])})  vs unconditional "
          f"{row['uncond_mean_pct']:+6.2f}% (sd {row['uncond_sd_pct']:5.2f}%)   "
          f"Welch t = {row['welch_t']:+.2f}  |  placebo p = {row['placebo_p']:.4f} "
          f"({int(row['n_draws']):,} draws)")

print("\n# Timer (long 180d after each signal, else cash) vs continuous buy-and-hold,")
print("  same [first signal -> tape end] window, one-day entry lag, 10 bps one-way x 2/episode")
tb = st.timer_backtest(btc, sig["signal_date"], hold_days=180, cost_bps=10.0)
print(f"  {tb['n_signals']} signals -> {tb['n_entries']} distinct holding episodes "
      f"(overlapping windows merge) over {tb['years']:.1f} years")
print(f"  timer      : total {tb['strat_total_pct']:+8.1f}%  CAGR {tb['strat_cagr_pct']:+6.2f}%  "
      f"Sharpe {tb['strat_sharpe']:+.2f}  (exposed {tb['exposure_pct']:.1f}% of the time)")
print(f"  buy & hold : total {tb['bh_total_pct']:+8.1f}%  CAGR {tb['bh_cagr_pct']:+6.2f}%  "
      f"Sharpe {tb['bh_sharpe']:+.2f}  (exposed 100.0% of the time)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT fire on a null world (effect=0) and must recover a planted")
print("  post-signal drift. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s in range(20):
    r, sidx = data.synthetic_world(effect=0.0, seed=663 + s)
    null_ts.append(st.synthetic_detect(r, sidx))
null_ts = np.asarray(null_ts)
print(f"  null (effect=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
r, sidx = data.synthetic_world(effect=15.0, seed=663)
planted_t = st.synthetic_detect(r, sidx)
print(f"  planted effect=+15%/yr post-signal drift (seed 663): Welch t = {planted_t:+.2f}")
