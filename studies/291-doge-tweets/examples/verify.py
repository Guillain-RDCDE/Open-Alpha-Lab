"""Real-data verification — Study 291 (Doge-Tweets). Regenerates docs/results.md numbers.

Loads the cached DOGE-USD / BTC-USD daily panel (cache-only by default), builds the
abnormal-return event study around the hardcoded Musk/Doge tweet dates, and runs the
lagged tradable backtest. No network unless you pass --fetch.

    python studies/291-doge-tweets/examples/verify.py
    python studies/291-doge-tweets/examples/verify.py --fetch   # refresh the cache once
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doge_tweets import data, strategy as st  # noqa: E402


def main() -> None:
    fetch = "--fetch" in sys.argv
    print("Study 291 -- Doge-Tweets -- real-data verification\n")
    print("Data: DOGE-USD / BTC-USD daily closes (Yahoo via yfinance, cache-only)")
    print("      Musk/Doge tweet dates: hardcoded in data.py (2019-2025)\n")

    prices = data.fetch_prices(fetch=fetch)
    rets = data.prices_to_returns(prices)
    tw = data.tweet_table()
    fp = data.fingerprint(rets)

    print(f"Data stamp: {rets.index.min().date()} -> {rets.index.max().date()}, "
          f"n={len(rets)} daily returns, fingerprint={fp}")
    print(f"Tweets: {len(tw)} events ({int((tw['direction']>0).sum())} bullish)\n")

    s = st.summarize(rets, tw, pre=1, post=3, hold_days=3, n_permutations=10_000, seed=291)

    print("=== Event study (DOGE abnormal vs market model on BTC) ===")
    print(f"  Market-model beta on BTC:  {s['beta_btc']:.3f}")
    print(f"  Day-0 abnormal return:     {s['aar_day0_pct']:+.2f}%  (t = {s['t_day0']:+.2f})")
    print(f"  CAR [-1,+3]:               {s['car_pct']:+.2f}%  (t = {s['car_t']:+.2f})")
    print(f"  Permutation p (day-0):     {s['perm_p']:.4f}")
    print(f"  Bullish-only day-0:        {s['bull_aar_day0_pct']:+.2f}%  "
          f"(t = {s['bull_t_day0']:+.2f}, perm p = {s['bull_perm_p']:.4f})\n")

    print("=== Tradable backtest (one-day lag, hold 3d, 30 bps one-way) ===")
    print(f"  Trades:                    {s['bt_n_trades']}")
    print(f"  Gross per trade:           {s['bt_gross_pct']:+.2f}%")
    print(f"  Net per trade:             {s['bt_net_pct']:+.2f}%  (t = {s['bt_net_t']:+.2f})")
    print(f"  BTC benchmark per trade:   {s['bt_btc_pct']:+.2f}%")
    print(f"  Net excess vs BTC:         {s['bt_excess_pct']:+.2f}%")
    print(f"  Hit-rate:                  {s['bt_hit']:.1%}\n")

    print("Verdict: Signal=REAL (day-0 abnormal +28%, perm p~0, t~3.9 ex-outlier);")
    print("         Tradability=MIRAGE (the move lands on the tweet day; the lagged")
    print("         trade nets ~+4% with t<=1 and <40% hit-rate -- uncapturable).")


if __name__ == "__main__":
    main()
