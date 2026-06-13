"""Headline run for Study 89 (Turn-of-the-Month) on the real total-return tape.

Defines the turn-of-month (TOM) window by trading-day-of-month — last trading day of
month t-1 through trading day +3 of month t — a *calendar-known* rule (no execution lag).
Measures the TOM-vs-rest daily-return gap (HAC t), the share of the total cumulative
return earned inside the window, a TOM-only timer vs buy-and-hold (raw and per-day-
invested), and an early-vs-late decay test. Prints the numbers the README front-card and
notebooks quote, and stamps the as-of date + content fingerprint.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from turn_of_the_month import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 5.0
LAST, FIRST = 1, 3
SPLIT = "2010-01-01"  # justified midpoint-ish split for the decay test (see results.md)


def main() -> None:
    frame = data.load_real("SPY", mode="total_return")
    # Drop any in-progress month: keep only complete trading through the as-of date.
    frame = frame.loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    print(f"[data] SPY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    wc = strategy.window_contrast(close, last=LAST, first=FIRST)
    print("\n--- TOM window [-1, +3] vs rest-of-month (SPY total return) ---")
    print(f"  TOM  mean daily return  = {wc['tom_mean_bps']:+.2f} bps/day  "
          f"({wc['n_tom']:,} days)")
    print(f"  rest mean daily return  = {wc['rest_mean_bps']:+.2f} bps/day  "
          f"({wc['n_rest']:,} days)")
    print(f"  difference (TOM - rest) = {wc['diff_bps']:+.2f} bps/day   "
          f"HAC t = {wc['t_diff']:+.2f}")
    print(f"  TOM window is {wc['frac_tom_days']*100:.1f}% of trading days, "
          f"earns {wc['share_of_total_logret']*100:.1f}% of total cumulative return")

    pos = strategy.tom_position(close.index, last=LAST, first=FIRST)
    timer = strategy.backtest(close, pos, cost_bps=COST_BPS)
    bh = strategy.buy_and_hold(close)

    def row(name, d):
        print(f"  {name:<20} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
              f"Sharpe {d['sharpe']:.3f}  Sharpe(inv) {d['sharpe_invested']:.3f}  "
              f"maxDD {d['max_dd']*100:6.1f}%  TiM {d['time_in_market']*100:4.0f}%  "
              f"switches {d['switches']}")

    print(f"\n--- TOM-only timer vs buy-and-hold, cost = {COST_BPS:.0f} bps/switch ---")
    row("TOM-only timer", timer)
    row("Buy & hold", bh)
    print(f"\n  Timer CAGR - BH CAGR        = {(timer['cagr']-bh['cagr'])*100:+.2f} pts/yr")
    print(f"  Timer Sharpe - BH Sharpe    = {timer['sharpe']-bh['sharpe']:+.3f} (whole-period)")
    print(f"  Timer Sharpe(inv) vs BH     = {timer['sharpe_invested']-bh['sharpe']:+.3f} "
          f"(per-day-invested minus BH)")
    print(f"  Drawdown vs BH              = {(timer['max_dd']-bh['max_dd'])*100:+.1f} pts")

    sp = strategy.subperiod_contrast(close, split=SPLIT, last=LAST, first=FIRST)
    print(f"\n--- Decay test: TOM-minus-rest daily premium, split at {SPLIT} ---")
    print(f"  early ({sp['n_early']:,} days): {sp['early_diff_bps']:+.2f} bps/day  "
          f"HAC t = {sp['early_t']:+.2f}")
    print(f"  late  ({sp['n_late']:,} days): {sp['late_diff_bps']:+.2f} bps/day  "
          f"HAC t = {sp['late_t']:+.2f}")
    print(f"  change (late - early)      = {sp['delta_bps']:+.2f} bps/day   "
          f"block-bootstrap p = {sp['p_delta']:.3f}")


if __name__ == "__main__":
    main()
