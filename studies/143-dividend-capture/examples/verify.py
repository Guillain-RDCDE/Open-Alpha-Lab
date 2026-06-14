"""Real-tape verification — Study 143 (Dividend-Capture). Regenerates docs/results.md numbers.

Reads (or fetches) unadjusted daily OHLCV + dividends for the default basket of 8 tickers,
computes the ex-date drop ratio and capture trade P&L, and prints the numbers that populate
docs/results.md. Network is touched only with --fetch.

    python studies/143-dividend-capture/examples/verify.py            # cache-only
    python studies/143-dividend-capture/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dividend_capture import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    events = data.fetch_events(fetch=fetch)
    if events.empty:
        print("No events — run with --fetch to populate the cache.")
        return

    print(f"=== Dividend-Capture — Real Tape ===")
    print(f"Events: {len(events)} | "
          f"Date range: {events.ex_date.min().date()} – {events.ex_date.max().date()}")
    print(f"Tickers: {', '.join(events.ticker.unique())}")
    print(f"Fingerprint: {data.fingerprint(events)}")
    print()

    # --- Drop ratio ---
    drop = st.drop_ratio_stats(events)
    print("=== Ex-date price drop vs dividend ===")
    print(f"  n events            : {drop['n']}")
    print(f"  mean drop/div       : {drop['mean_ratio']:+.4f}  (1.0 = fully efficient)")
    print(f"  median drop/div     : {drop['median_ratio']:+.4f}")
    print(f"  std drop/div        : {drop['std_ratio']:.4f}")
    print(f"  frac drop < div     : {drop['frac_below_one']:.3f}  (fraction where gross capture > 0)")
    print(f"  HAC t vs 1.0        : {drop['tstat_vs_one']:+.2f}  (H0: ratio = 1.0)")
    print(f"  mean drop (bps)     : {drop['mean_drop_bps']:+.2f}")
    print(f"  mean div  (bps)     : {drop['mean_div_bps']:+.2f}")
    print()

    # --- Capture P&L ---
    cap = st.capture_trade_stats(events, cost_bps=5.0)
    print("=== Capture trade (buy t-1 close, sell t+1 close, collect div) ===")
    print(f"  n trades            : {cap['n']}")
    print(f"  events per year     : {cap['events_per_year']:.1f}")
    print(f"  mean gross (bps)    : {cap['mean_gross_bps']:+.2f}")
    print(f"  win rate (gross)    : {cap['win_rate_gross']:.3f}")
    print(f"  skew (gross)        : {cap['skew_gross']:+.3f}")
    print(f"  HAC t-stat (gross)  : {cap['tstat_gross']:+.2f}")
    print(f"  mean net (bps, 5bp) : {cap['mean_net_bps']:+.2f}")
    print(f"  HAC t-stat (net)    : {cap['tstat_net']:+.2f}")
    print()

    # --- Buy-and-hold baseline ---
    bah = st.buy_and_hold_stats(events)
    print("=== Buy-and-hold baseline (same 2-day window, no div) ===")
    print(f"  mean BAH (bps)      : {bah['mean_bah_bps']:+.2f}")
    print(f"  win rate BAH        : {bah['win_rate_bah']:.3f}")
    print()

    # --- Per-ticker ---
    print("=== Per-ticker breakdown ===")
    for t in events.ticker.unique():
        ev = events[events.ticker == t]
        d2 = st.drop_ratio_stats(ev)
        c2 = st.capture_trade_stats(ev, cost_bps=5.0)
        print(
            f"  {t:5s} n={d2['n']:3d} "
            f"drop/div={d2['mean_ratio']:+.3f} t_vs1={d2['tstat_vs_one']:+.2f}  "
            f"gross={c2['mean_gross_bps']:+7.2f}bps t_g={c2['tstat_gross']:+.2f}"
        )
    print()

    # --- Cost sweep ---
    print("=== Cost sweep (gross + net at various round-trip costs) ===")
    for cost in [0.0, 5.0, 10.0, 20.0, 50.0]:
        c3 = st.capture_trade_stats(events, cost_bps=cost)
        print(f"  cost={cost:5.1f}bps  net={c3['mean_net_bps']:+.2f}bps  t={c3['tstat_net']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
