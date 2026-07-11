"""Reproducible headline run for Study 679 — Psychological Line.

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

from psychological_line import data, strategy as st  # noqa: E402

WINDOW, OVERSOLD, OVERBOUGHT, HOLD = 12, 25.0, 75.0, 5

print("# Psychological Line — is PSY(12) > 75 / < 25 an overbought/oversold gauge?")
print(f"universe: {', '.join(data.UNIVERSE)}  |  window={WINDOW}  thresholds="
      f"[{OVERSOLD:.0f}, {OVERBOUGHT:.0f}]  hold={HOLD}d")

if not data.have_real():
    print("(cache miss — fetching daily OHLC once)")
    data.fetch()

bars_map = data.load_real()
for t, b in bars_map.items():
    print(data_stamp(f"{t} OHLC", b, asof=data.AS_OF))

print("\n# THE HEADLINE — forward return conditioned on the PSY reading (pooled universe)")
c = st.pooled_conditional(bars_map, window=WINDOW, oversold=OVERSOLD, overbought=OVERBOUGHT,
                          h=HOLD)
print(f"  oversold  (PSY<{OVERSOLD:.0f}, n={c['n_low']:,}): {c['mean_low_bps']:+.2f} bps/{HOLD}d  "
      f"Welch t = {c['welch_t_low']:+.2f}  NW({HOLD}) t = {c['nw_t_low']:+.2f}")
print(f"  neutral   (n={c['n_mid']:,}): {c['mean_mid_bps']:+.2f} bps/{HOLD}d")
print(f"  overbought(PSY>{OVERBOUGHT:.0f}, n={c['n_high']:,}): {c['mean_high_bps']:+.2f} bps/{HOLD}d  "
      f"Welch t = {c['welch_t_high']:+.2f}  NW({HOLD}) t = {c['nw_t_high']:+.2f}")
print("  (contrarian claim: oversold should beat neutral (+); overbought should trail (-))")

print("\n# THE TIMER — zone-trigger trade ledger (enter next open, hold "
      f"{HOLD}d, pooled) vs a random-direction control on identical entries")
strat, rand = st.pooled_trade_ledger(bars_map, window=WINDOW, oversold=OVERSOLD,
                                     overbought=OVERBOUGHT, hold_days=HOLD, cost_bps=5.0,
                                     seed=679)
for cb in (5.0, 10.0):
    strat_cb, rand_cb = st.pooled_trade_ledger(bars_map, window=WINDOW, oversold=OVERSOLD,
                                               overbought=OVERBOUGHT, hold_days=HOLD,
                                               cost_bps=cb, seed=679)
    ss = st.summarize(strat_cb, "ret_net", lags=HOLD)
    rs = st.summarize(rand_cb, "ret_net", lags=HOLD)
    print(f"  cost={cb:>4.1f} bps: PSY timer n={ss['n_trades']:,} win={ss['win_rate']*100:.1f}% "
          f"mean={ss['mean_bps']:+.2f} bps/trade (HAC t={ss['tstat']:+.2f})  |  "
          f"coin n={rs['n_trades']:,} win={rs['win_rate']*100:.1f}% mean={rs['mean_bps']:+.2f} "
          f"bps/trade (HAC t={rs['tstat']:+.2f})")

strat5, rand5 = st.pooled_trade_ledger(bars_map, window=WINDOW, oversold=OVERSOLD,
                                       overbought=OVERBOUGHT, hold_days=HOLD, cost_bps=5.0,
                                       seed=679)
ss5, rs5 = st.summarize(strat5, "ret_net", lags=HOLD), st.summarize(rand5, "ret_net", lags=HOLD)
gap_bps = ss5["mean_bps"] - rs5["mean_bps"]
welch_vs_coin = st.welch_t(strat5["ret_net"].to_numpy(), rand5["ret_net"].to_numpy())
print(f"  delta (PSY timer - coin) at 5 bps: {gap_bps:+.2f} bps/trade  Welch t = "
      f"{welch_vs_coin:+.2f}")

print("\n# Parameter robustness — window x (oversold, overbought) grid, "
      f"conditional-split Welch t, hold={HOLD}d")
grid = st.param_grid(bars_map, h=HOLD)
for _, row in grid.iterrows():
    print(f"  window={int(row['window']):>2d}  zone=[{row['oversold']:.0f},{row['overbought']:.0f}]  "
          f"n_low={int(row['n_low']):>5d} t_low={row['welch_t_low']:+.2f}   "
          f"n_high={int(row['n_high']):>5d} t_high={row['welch_t_high']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the conditional-split detector must NOT fire on a null world (mean_rev=0) and")
print("  must recover a planted short-horizon reversal. Each seed pools a 6-series basket")
print("  (mirroring the real 6-ticker universe); null checked over 20 seeds.")
null_lo, null_hi = [], []
for s_ in range(20):
    basket = data.synthetic_basket(reversal=0.0, seed=679 + s_)
    r = st.pooled_conditional(basket, window=WINDOW, oversold=OVERSOLD, overbought=OVERBOUGHT,
                              h=HOLD)
    null_lo.append(r["welch_t_low"])
    null_hi.append(r["welch_t_high"])
null_lo, null_hi = np.asarray(null_lo), np.asarray(null_hi)
print(f"  null (reversal=0), 20 seeds: t_low mean {null_lo.mean():+.2f} (sd {null_lo.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_lo) >= 2).sum()}/20 seeds")
print(f"                              t_high mean {null_hi.mean():+.2f} (sd {null_hi.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_hi) >= 2).sum()}/20 seeds")
planted_basket = data.synthetic_basket(reversal=0.006, seed=679)
rp = st.pooled_conditional(planted_basket, window=WINDOW, oversold=OVERSOLD,
                           overbought=OVERBOUGHT, h=HOLD)
print(f"  planted reversal=+0.006/day (seed 679, 6-series basket): t_low = {rp['welch_t_low']:+.2f}   "
      f"t_high = {rp['welch_t_high']:+.2f}")
