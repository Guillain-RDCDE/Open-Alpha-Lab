"""Reproducible headline run for Study 680 — Disparity Index.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + basket daily tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
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

from disparity_index import data, strategy as st  # noqa: E402

WINDOW, OVERSOLD, OVERBOUGHT, HOLD = 10, 95.0, 105.0, 5

print("# Disparity Index — is DI(10) > 105 / < 95 a mean-reversion gauge?")
print(f"universe: {', '.join(data.UNIVERSE)}  |  window={WINDOW}  thresholds="
      f"[{OVERSOLD:.0f}, {OVERBOUGHT:.0f}]  hold={HOLD}d")

if not data.have_real():
    print("(cache miss — fetching daily OHLC once)")
    data.fetch()

bars_map = data.load_real()
for t, b in bars_map.items():
    print(data_stamp(f"{t} OHLC", b, asof=data.AS_OF))

print("\n# THE HEADLINE — forward return conditioned on the DI reading (pooled universe)")
c = st.pooled_conditional(bars_map, window=WINDOW, oversold=OVERSOLD, overbought=OVERBOUGHT,
                          h=HOLD)
print(f"  oversold  (DI<{OVERSOLD:.0f}, n={c['n_low']:,}): {c['mean_low_bps']:+.2f} bps/{HOLD}d  "
      f"Welch t = {c['welch_t_low']:+.2f}  NW({HOLD}) t = {c['nw_t_low']:+.2f}")
print(f"  neutral   (n={c['n_mid']:,}): {c['mean_mid_bps']:+.2f} bps/{HOLD}d")
print(f"  overbought(DI>{OVERBOUGHT:.0f}, n={c['n_high']:,}): {c['mean_high_bps']:+.2f} bps/{HOLD}d  "
      f"Welch t = {c['welch_t_high']:+.2f}  NW({HOLD}) t = {c['nw_t_high']:+.2f}")
print("  (contrarian claim: oversold should beat neutral (+); overbought should trail (-))")

print("\n# THE TIMER — zone-trigger trade ledger (enter next open, hold "
      f"{HOLD}d, pooled) vs a random-direction control on identical entries")
for cb in (5.0, 10.0):
    strat_cb, rand_cb = st.pooled_trade_ledger(bars_map, window=WINDOW, oversold=OVERSOLD,
                                               overbought=OVERBOUGHT, hold_days=HOLD,
                                               cost_bps=cb, seed=680)
    ss = st.summarize(strat_cb, "ret_net", lags=HOLD)
    rs = st.summarize(rand_cb, "ret_net", lags=HOLD)
    print(f"  cost={cb:>4.1f} bps: DI timer n={ss['n_trades']:,} win={ss['win_rate']*100:.1f}% "
          f"mean={ss['mean_bps']:+.2f} bps/trade (HAC t={ss['tstat']:+.2f})  |  "
          f"coin n={rs['n_trades']:,} win={rs['win_rate']*100:.1f}% mean={rs['mean_bps']:+.2f} "
          f"bps/trade (HAC t={rs['tstat']:+.2f})")

strat5, rand5 = st.pooled_trade_ledger(bars_map, window=WINDOW, oversold=OVERSOLD,
                                       overbought=OVERBOUGHT, hold_days=HOLD, cost_bps=5.0,
                                       seed=680)
ss5, rs5 = st.summarize(strat5, "ret_net", lags=HOLD), st.summarize(rand5, "ret_net", lags=HOLD)
gap_bps = ss5["mean_bps"] - rs5["mean_bps"]
welch_vs_coin = st.welch_t(strat5["ret_net"].to_numpy(), rand5["ret_net"].to_numpy())
print(f"  delta (DI timer - coin) at 5 bps: {gap_bps:+.2f} bps/trade  Welch t = "
      f"{welch_vs_coin:+.2f}")

print("\n# THE DRIFT CHECK — does the surviving leg beat plain buy-any-random-day (same "
      "ticker, same direction)?")
db = st.random_day_baseline(bars_map, window=WINDOW, oversold=OVERSOLD, overbought=OVERBOUGHT,
                            hold_days=HOLD, cost_bps=5.0, seed=680)
for leg, label in (("low", "buy-oversold (long)"), ("high", "short-overbought")):
    r = db[leg]
    print(f"  {label}: n={r['n']:,} mean={r['mean_bps']:+.2f} bps/trade (HAC t={r['tstat']:+.2f})  "
          f"vs random-day n={r['n']:,} mean={r['rand_mean_bps']:+.2f} bps/trade "
          f"(HAC t={r['rand_tstat']:+.2f})  |  delta={r['delta_bps']:+.2f} bps  "
          f"Welch t = {r['welch_t_vs_random_day']:+.2f}")

print("\n# Parameter robustness — window x (oversold, overbought) grid, "
      f"conditional-split Welch t, hold={HOLD}d")
grid = st.param_grid(bars_map, h=HOLD)
for _, row in grid.iterrows():
    print(f"  window={int(row['window']):>2d}  zone=[{row['oversold']:.0f},{row['overbought']:.0f}]  "
          f"n_low={int(row['n_low']):>5d} t_low={row['welch_t_low']:+.2f}   "
          f"n_high={int(row['n_high']):>5d} t_high={row['welch_t_high']:+.2f}")

print("\n# \"Just short-term reversal?\" diagnostic — DI vs trailing N-day return")
dc = st.di_return_correlation(bars_map, window=WINDOW)
print(f"  pooled Pearson corr(DI-100, trailing {WINDOW}d return), n={dc['n']:,}: "
      f"r = {dc['corr']:.4f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the conditional-split detector must NOT fire on a null world (reversal=0) and")
print("  must recover a planted short-horizon reversal. Each seed pools a 6-series basket")
print("  (mirroring the real 6-ticker universe); null checked over 20 seeds. The forward")
print("  windows overlap (h=5), so Welch t alone over-fires on a single autocorrelated")
print("  path; the calibrated detector here is the NW(h) t (the same cross-check used on")
print("  the headline split) — both are reported for transparency.")
null_lo_w, null_hi_w, null_lo_nw, null_hi_nw = [], [], [], []
for s_ in range(20):
    basket = data.synthetic_basket(reversal=0.0, seed=680 + s_)
    r = st.pooled_conditional(basket, window=WINDOW, oversold=OVERSOLD, overbought=OVERBOUGHT,
                              h=HOLD)
    null_lo_w.append(r["welch_t_low"])
    null_hi_w.append(r["welch_t_high"])
    null_lo_nw.append(r["nw_t_low"])
    null_hi_nw.append(r["nw_t_high"])
null_lo_w, null_hi_w = np.asarray(null_lo_w), np.asarray(null_hi_w)
null_lo_nw, null_hi_nw = np.asarray(null_lo_nw), np.asarray(null_hi_nw)
print(f"  Welch  — t_low mean {null_lo_w.mean():+.2f} (sd {null_lo_w.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_lo_w) >= 2).sum()}/20 seeds  |  "
      f"t_high mean {null_hi_w.mean():+.2f} (sd {null_hi_w.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_hi_w) >= 2).sum()}/20 seeds")
print(f"  NW({HOLD}) — t_low mean {null_lo_nw.mean():+.2f} (sd {null_lo_nw.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_lo_nw) >= 2).sum()}/20 seeds  |  "
      f"t_high mean {null_hi_nw.mean():+.2f} (sd {null_hi_nw.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_hi_nw) >= 2).sum()}/20 seeds")
planted_basket = data.synthetic_basket(reversal=0.006, seed=680)
rp = st.pooled_conditional(planted_basket, window=WINDOW, oversold=OVERSOLD,
                           overbought=OVERBOUGHT, h=HOLD)
print(f"  planted reversal=+0.006/day (seed 680, 6-series basket): "
      f"Welch t_low = {rp['welch_t_low']:+.2f}, t_high = {rp['welch_t_high']:+.2f}  |  "
      f"NW t_low = {rp['nw_t_low']:+.2f}, t_high = {rp['nw_t_high']:+.2f}")
