"""Headline run for Study 98 (High-Noon) on the real total-return tape.

Tests the folk rule *"never buy at an all-time high"* literally: it splits SPY (total
return) days into *at-ATH* (within 1% of the running high) vs *not-at-ATH* and compares
the forward 1/3/12-month returns (means with HAC t-stats, win-rates with Wilson CIs, and
a HAC test of the difference), then runs the *avoid-the-highs* timer against buy-and-hold.
Prints the numbers the README front-card and the notebooks quote, and stamps the as-of
date + content fingerprint. Re-run to reproduce; match the fingerprint to confirm you
hold the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from high_noon import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
NEAR_PCT = 0.01
COST_BPS = 5.0
HORIZONS = [(strategy.MONTH, "1-month"), (3 * strategy.MONTH, "3-month"), (12 * strategy.MONTH, "12-month")]


def main() -> None:
    frame = data.load_real("SPY", mode="total_return")
    frame = frame.loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    flag = strategy.at_ath(close, NEAR_PCT)
    print(f"[data] SPY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")
    print(f"  days within {NEAR_PCT*100:.0f}% of the running all-time high: "
          f"{flag.mean()*100:.1f}%")

    print("\n--- Forward returns: at-ATH vs not-at-ATH (1% band) ---")
    print(f"  {'horizon':<9} {'arm':<9} {'n':>6} {'mean%':>8} {'win%':>7} "
          f"{'Wilson95':>15} {'HAC t':>7}")
    for h, label in HORIZONS:
        st = strategy.conditional_forward_stats(close, h, NEAR_PCT)
        for arm, name in [(st["at"], "at-ATH"), (st["not_at"], "not-ATH")]:
            print(f"  {label:<9} {name:<9} {arm['n']:>6} {arm['mean']*100:>8.2f} "
                  f"{arm['win_rate']*100:>7.1f} "
                  f"[{arm['win_lo']*100:5.1f},{arm['win_hi']*100:5.1f}] {arm['hac_t']:>7.2f}")
        print(f"  {label:<9} {'DIFF':<9} {'':>6} {st['diff_mean']*100:>8.2f} "
              f"{'':>7} {'':>15} {st['diff_hac_t']:>7.2f}   (at minus not-at)")

    print("\n--- Avoid-the-highs timer vs buy-and-hold, cost = 5 bps/switch ---")
    pos = strategy.avoid_highs_position(close, NEAR_PCT, lag=1)
    timer = strategy.backtest(close, pos, cost_bps=COST_BPS)
    bh = strategy.buy_and_hold(close)

    def row(name, d):
        print(f"  {name:<22} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
              f"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%  "
              f"TiM {d['time_in_market']*100:4.0f}%  switches {d['switches']}")

    row("Avoid-highs timer", timer)
    row("Buy & hold", bh)
    print(f"\n  Timer CAGR - BH CAGR     = {(timer['cagr']-bh['cagr'])*100:+.2f} pts/yr")
    print(f"  Timer Sharpe - BH Sharpe = {timer['sharpe']-bh['sharpe']:+.3f}")
    print(f"  Final wealth ratio       = {timer['final']/bh['final']:.2f}x buy-and-hold")


if __name__ == "__main__":
    main()
