"""Real-data verification -- Study 294 (Coinbase-Rank). Regenerates docs/results.md numbers.

Loads the cached BTC-USD / ETH-USD daily panel (cache-only by default), builds the
forward abnormal-return event study around the hardcoded Coinbase rank-spike dates,
and runs the lagged tradable short backtest. No network unless you pass --fetch.

    python studies/294-coinbase-rank/examples/verify.py
    python studies/294-coinbase-rank/examples/verify.py --fetch   # refresh the cache once
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coinbase_rank import data, strategy as st  # noqa: E402


def main() -> None:
    fetch = "--fetch" in sys.argv
    print("Study 294 -- Coinbase-Rank -- real-data verification\n")
    print("Data: BTC-USD / ETH-USD daily closes (Yahoo via yfinance, cache-only)")
    print("      Coinbase App-Store rank spikes: hardcoded in data.py (2017-2025)\n")

    prices = data.fetch_prices(fetch=fetch)
    rets = data.prices_to_returns(prices)
    tw = data.rank_spike_table()
    fp = data.fingerprint(rets)

    print(f"Data stamp: {rets.index.min().date()} -> {rets.index.max().date()}, "
          f"n={len(rets)} daily returns, fingerprint={fp}")
    print(f"Rank spikes: {len(tw)} events (all +1 euphoria markers)\n")

    s = st.summarize(rets, tw, pre=1, post=10, fwd=5, hold_days=5,
                     n_permutations=10_000, seed=294)

    print("=== Forward event study (BTC abnormal vs market model on ETH) ===")
    print(f"  Market-model beta on ETH:  {s['beta_eth']:.3f}")
    print(f"  Day-0 abnormal return:     {s['aar_day0_pct']:+.2f}%  (spike-day reaction)")
    print(f"  Forward CAR[+1,+5]:        {s['fwd_car_pct']:+.2f}%  (t = {s['fwd_car_t']:+.2f})")
    print(f"  Permutation p (forward):   {s['perm_p']:.4f}")
    print(f"  Negative events:           {s['n_neg']}/{s['n_events']}\n")

    print("=== Tradable short backtest (one-day lag, hold 5d, 20 bps one-way + 5 bps/day borrow) ===")
    print(f"  Trades:                    {s['bt_n_trades']}")
    print(f"  Gross per trade (short):   {s['bt_gross_pct']:+.2f}%")
    print(f"  Net per trade (short):     {s['bt_net_pct']:+.2f}%  (t = {s['bt_net_t']:+.2f})")
    print(f"  BTC long benchmark:        {s['bt_btc_pct']:+.2f}%")
    print(f"  Hit-rate:                  {s['bt_hit']:.1%}\n")

    print("Verdict: Signal=WEAK (forward CAR -5.2%, right direction, perm p~0.001, BUT")
    print("         cross-sectional t -1.69 < |2|, 8/15 negative, fragile to one crash);")
    print("         Tradability=MIRAGE (lagged short nets ~+2.4% with t~0.8 and ~53% hit --")
    print("         a coin flip after fees and borrow on a structurally up-trending asset).")


if __name__ == "__main__":
    main()
