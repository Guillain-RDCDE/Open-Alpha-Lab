"""Headline run for Study 224 (Monday Effect) on the real total-return tape.

Measures the per-weekday mean daily return on SPY (total return), the Monday-vs-rest
contrast with HAC t-stats, the pre-2000 vs post-2000 split with a test of the change,
and literal day-of-week timers net of costs vs buy-and-hold.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from monday_effect import data, strategy  # noqa: E402

AS_OF = "2026-06-16"
COST_BPS = 1.0


def main() -> None:
    frame = data.load_real("SPY", mode="total_return")
    frame = frame.loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    print(f"[data] SPY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    wm = strategy.weekday_means(close)
    print("")
    print("--- Per-weekday mean daily return (bps), HAC t ---")
    for day, r in wm.iterrows():
        print(f"  {day}  mean {r['mean_bps']:+6.2f} bps   n {int(r['n']):5d}   HAC t {r['hac_t']:+5.2f}")

    mon = strategy.contrast(close, 0)
    tue = strategy.contrast(close, 1)
    print("")
    print("--- Headline contrasts ---")
    print(f"  Monday  - rest: {mon['diff_bps']:+6.2f} bps   HAC t {mon['hac_t']:+5.2f}")
    print(f"  Tuesday - rest: {tue['diff_bps']:+6.2f} bps   HAC t {tue['hac_t']:+5.2f}")

    mon_sp = strategy.subperiod_effect(close, 0, cut="2000-01-01")
    print("")
    print("--- Pre-2000 vs post-2000 Monday effect (with HAC test of the change) ---")
    print(f"  Monday  pre {mon_sp['pre_diff_bps']:+.2f}  post {mon_sp['post_diff_bps']:+.2f}  "
          f"change {mon_sp['change_bps']:+.2f} bps  HAC t(change) {mon_sp['hac_t_change']:+.2f}")

    bh = strategy.buy_and_hold(close)
    mon_only = strategy.backtest(close, strategy.monday_only_position(close), cost_bps=COST_BPS)
    skip_mon = strategy.backtest(close, strategy.skip_monday_position(close), cost_bps=COST_BPS)

    def row(name, d):
        print(f"  {name:<22} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
              f"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%  "
              f"TiM {d['time_in_market']*100:4.0f}%  switches {d['switches']}")

    print("")
    print(f"--- Literal day-of-week timers, cost = {COST_BPS:.0f} bp/switch ---")
    row("Buy Monday only", mon_only)
    row("Skip Monday", skip_mon)
    row("Buy and hold", bh)
    print("")
    print(f"  Buy-Monday CAGR    - BH CAGR   = {(mon_only['cagr']-bh['cagr'])*100:+.2f} pts/yr")
    print(f"  Skip-Monday CAGR   - BH CAGR   = {(skip_mon['cagr']-bh['cagr'])*100:+.2f} pts/yr")
    print(f"  Buy-Monday Sharpe  - BH Sharpe = {mon_only['sharpe']-bh['sharpe']:+.3f}")
    print(f"  Skip-Monday Sharpe - BH Sharpe = {skip_mon['sharpe']-bh['sharpe']:+.3f}")


if __name__ == "__main__":
    main()
