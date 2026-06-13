"""Headline run for Study 90 (Weekend / day-of-week effect) on the real total-return tape.

Measures the per-weekday mean daily return on SPY (total return), the Monday-vs-rest and
Tuesday-vs-rest contrasts with HAC t-stats, the pre-2000 vs post-2000 split *with a test of
the change*, and a literal "buy Tuesday" / "skip Monday" timer net of costs vs buy-and-hold.
Prints the numbers the README front-card and the notebooks quote, and stamps the as-of date +
content fingerprint. Re-run to reproduce; match the fingerprint to confirm you hold the same
tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from weekend import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 1.0


def main() -> None:
    frame = data.load_real("SPY", mode="total_return")
    frame = frame.loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    print(f"[data] SPY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    # --- Per-weekday means + HAC t ---
    wm = strategy.weekday_means(close)
    print("\n--- Per-weekday mean daily return (bps), HAC t ---")
    for day, r in wm.iterrows():
        print(f"  {day}  mean {r['mean_bps']:+6.2f} bps   n {int(r['n']):5d}   HAC t {r['hac_t']:+5.2f}")

    # --- Contrasts ---
    mon = strategy.contrast(close, 0)
    tue = strategy.contrast(close, 1)
    print("\n--- Contrasts (whole sample) ---")
    print(f"  Monday  - rest: {mon['diff_bps']:+6.2f} bps   HAC t {mon['hac_t']:+5.2f}")
    print(f"  Tuesday - rest: {tue['diff_bps']:+6.2f} bps   HAC t {tue['hac_t']:+5.2f}")

    # --- Sub-period split with a test of the change ---
    mon_sp = strategy.subperiod_effect(close, 0, cut="2000-01-01")
    tue_sp = strategy.subperiod_effect(close, 1, cut="2000-01-01")
    print("\n--- Pre-2000 vs post-2000 (with HAC test of the *change*) ---")
    print(f"  Monday  effect: pre {mon_sp['pre_diff_bps']:+6.2f}  post {mon_sp['post_diff_bps']:+6.2f}  "
          f"change {mon_sp['change_bps']:+6.2f} bps  HAC t(change) {mon_sp['hac_t_change']:+5.2f}")
    print(f"  Tuesday effect: pre {tue_sp['pre_diff_bps']:+6.2f}  post {tue_sp['post_diff_bps']:+6.2f}  "
          f"change {tue_sp['change_bps']:+6.2f} bps  HAC t(change) {tue_sp['hac_t_change']:+5.2f}")

    # --- Literal timers vs buy-and-hold, net of costs ---
    bh = strategy.buy_and_hold(close)
    tue_only = strategy.backtest(close, strategy.tuesday_only_position(close), cost_bps=COST_BPS)
    skip_mon = strategy.backtest(close, strategy.skip_monday_position(close), cost_bps=COST_BPS)

    def row(name, d):
        print(f"  {name:<22} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
              f"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%  "
              f"TiM {d['time_in_market']*100:4.0f}%  switches {d['switches']}")

    print(f"\n--- Literal day-of-week timers, cost = {COST_BPS:.0f} bp/switch ---")
    row("Buy Tuesday only", tue_only)
    row("Skip Monday", skip_mon)
    row("Buy & hold", bh)
    print(f"\n  Buy-Tuesday Sharpe  - BH Sharpe = {tue_only['sharpe']-bh['sharpe']:+.3f}")
    print(f"  Skip-Monday Sharpe  - BH Sharpe = {skip_mon['sharpe']-bh['sharpe']:+.3f}")
    print(f"  Buy-Tuesday CAGR    - BH CAGR   = {(tue_only['cagr']-bh['cagr'])*100:+.2f} pts/yr")
    print(f"  Skip-Monday CAGR    - BH CAGR   = {(skip_mon['cagr']-bh['cagr'])*100:+.2f} pts/yr")


if __name__ == "__main__":
    main()
